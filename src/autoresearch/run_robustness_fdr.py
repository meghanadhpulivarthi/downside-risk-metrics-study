"""
Multiple-testing correction + small-sample robustness — the P1 fixes from peer review.

(1) BH-FDR across the two significance families the paper reports:
      - partial family : partial rho(metric, drawdown | vol)   (Table A1, 9 metrics x 5 beds)
      - head-to-head   : rho(metric, dd) - rho(vol, dd)        (Table 2,  9 metrics x 5 beds)
    For each cell we compute a paired block-bootstrap two-sided p-value from the per-date
    series, then apply Benjamini-Hochberg at q=0.05 WITHIN each family and report which
    "CI-excludes-0" stars survive.

(2) Small-sample robustness on the crypto beds (7-9 dates):
      - leave-one-date-out (LOO) jackknife range for each metric's partial rho
      - minimum detectable effect (MDE) at 80% power per bed
        MDE = (z_.975 + z_.80) * sd(per-date series) / sqrt(n_dates)

We rebuild the per-date cross-sectional series with the SAME machinery as run_mechanism.py
(compute_metrics, forward_drawdown at 90d, trailing 180d) so numbers match Table A1 / Table 2.

Run: uv run --project <repo> python -u src/autoresearch/run_robustness_fdr.py
"""

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import evaluate as ev
import left_tail as lt
from run_multi_testbed import compute_metrics, METRICS, TRAILING_DAYS, FORWARD_DAYS
from run_mechanism import build_beds, _spearman, _partial_spearman, MIN_PAIRS

OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"
COMPETING = [m for m in METRICS if m != "volatility"]
Q = 0.05
BLOCK = 3
N_BOOT = 3000
Z_975, Z_80 = 1.959964, 0.841621


def per_date_series(close, market_ret, formation_dates, universe_fn):
    """Per date: partial rho(metric|vol) and head-to-head diff(metric - vol) for each metric."""
    partial = {m: [] for m in COMPETING}
    headtohead = {m: [] for m in COMPETING}
    for fdate in formation_dates:
        rows = []
        for asset in universe_fn(fdate):
            if asset not in close.columns:
                continue
            r = close.loc[fdate - pd.Timedelta(days=TRAILING_DAYS):fdate, asset].dropna().pct_change().dropna()
            if len(r) < 60:
                continue
            m = market_ret.reindex(r.index).fillna(0.0).values
            feats = compute_metrics(r.values, m)
            if feats is None:
                continue
            fwd = close.loc[fdate:fdate + pd.Timedelta(days=FORWARD_DAYS), asset]
            dd = lt.forward_drawdown(fwd, fdate + pd.Timedelta(days=FORWARD_DAYS))
            if np.isnan(dd):
                continue
            feats["fwd_dd"] = dd
            rows.append(feats)
        g = pd.DataFrame(rows)
        if len(g) < MIN_PAIRS:
            print(f"    skip {fdate.date()}: {len(g)} assets")
            continue
        vol = g["volatility"].values
        dd = g["fwd_dd"].values
        r_vol_dd = _spearman(vol, dd)
        for metric in COMPETING:
            mv = g[metric].values
            r_m_dd = _spearman(mv, dd)
            r_m_vol = _spearman(mv, vol)
            partial[metric].append(_partial_spearman(r_m_dd, r_m_vol, r_vol_dd))
            headtohead[metric].append((r_m_dd - r_vol_dd) if np.isfinite(r_m_dd) and np.isfinite(r_vol_dd) else np.nan)
    return partial, headtohead


def block_boot_means(d):
    """Block-bootstrap distribution of the mean of a per-date series."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return None
    rng = np.random.default_rng(0)
    nb = int(np.ceil(n / BLOCK))
    starts_max = max(1, n - BLOCK + 1)
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        starts = rng.integers(0, starts_max, nb)
        means[i] = np.concatenate([d[s:s + BLOCK] for s in starts])[:n].mean()
    return means


def cell_stats(series):
    """Return mean, two-sided bootstrap p-value, n, sd for a per-date series."""
    d = np.asarray(series, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return {"mean": np.nan, "p": np.nan, "n": n, "sd": np.nan}
    means = block_boot_means(d)
    frac_pos = float(np.mean(means >= 0))
    frac_neg = float(np.mean(means <= 0))
    p = min(1.0, 2.0 * min(frac_pos, frac_neg))
    return {"mean": float(np.mean(d)), "p": p, "n": n, "sd": float(np.std(d, ddof=1))}


def bh_fdr(pvals, q=Q):
    """Benjamini-Hochberg. pvals: dict key->p. Returns set of keys that survive at level q."""
    items = [(k, p) for k, p in pvals.items() if np.isfinite(p)]
    m = len(items)
    if m == 0:
        return set(), np.nan
    items.sort(key=lambda kv: kv[1])
    thresh_p = 0.0
    k_max = 0
    for i, (k, p) in enumerate(items, start=1):
        if p <= (i / m) * q:
            k_max = i
            thresh_p = p
    survivors = {k for k, p in items if p <= thresh_p} if k_max > 0 else set()
    return survivors, thresh_p


def main():
    now = datetime.datetime.now()
    print("=" * 74)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : BH-FDR q={Q}; block={BLOCK}; n_boot={N_BOOT}")
    print("=" * 74)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_robustness_fdr"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    beds = build_beds()
    partial_cells, h2h_cells = {}, {}
    per_bed_series = {}
    for name, (close, mkt, dates, univ) in beds.items():
        print(f"\n[{name}] collecting per-date series ...")
        partial, h2h = per_date_series(close, mkt, dates, univ)
        per_bed_series[name] = {"partial": partial, "h2h": h2h}
        for metric in COMPETING:
            partial_cells[(name, metric)] = cell_stats(partial[metric])
            h2h_cells[(name, metric)] = cell_stats(h2h[metric])

    # --- BH-FDR within each family ---
    partial_p = {k: v["p"] for k, v in partial_cells.items()}
    h2h_p = {k: v["p"] for k, v in h2h_cells.items()}
    partial_survivors, p_thresh_partial = bh_fdr(partial_p)
    h2h_survivors, p_thresh_h2h = bh_fdr(h2h_p)

    def sign_word(cell, survivors):
        if not np.isfinite(cell["mean"]) or not np.isfinite(cell["p"]):
            return "n/e"
        key_survives = cell["p"]  # placeholder; membership checked by caller
        return None

    print("\n" + "=" * 74)
    print(f"PARTIAL family (Table A1): {len([p for p in partial_p.values() if np.isfinite(p)])} tests, "
          f"BH q={Q}, survivor p-threshold={p_thresh_partial:.4f}")
    print("=" * 74)
    print(f"{'metric':16s}" + "".join(f"{n.split('_')[0][:7]:>10s}" for n in beds))
    for metric in COMPETING:
        cells = []
        for name in beds:
            c = partial_cells[(name, metric)]
            if not np.isfinite(c["mean"]):
                cells.append("  n/e")
                continue
            surv = (name, metric) in partial_survivors
            mark = "*" if surv else ("." if c["p"] < 0.05 else " ")  # * survives FDR; . nominal only
            cells.append(f"{c['mean']:+.2f}{mark}")
        print(f"{metric:16s}" + "".join(f"{x:>10s}" for x in cells))
    print("  * = survives BH-FDR;  . = nominally p<.05 but does NOT survive FDR;  blank = ns")

    print("\n" + "=" * 74)
    print(f"HEAD-TO-HEAD family (Table 2): BH q={Q}, survivor p-threshold={p_thresh_h2h:.4f}")
    print("=" * 74)
    print(f"{'metric':16s}" + "".join(f"{n.split('_')[0][:7]:>10s}" for n in beds))
    for metric in COMPETING:
        cells = []
        for name in beds:
            c = h2h_cells[(name, metric)]
            if not np.isfinite(c["mean"]):
                cells.append("  n/e"); continue
            surv = (name, metric) in h2h_survivors
            direction = "+" if c["mean"] > 0 else "-"
            mark = ("*" + direction) if surv else "  "
            cells.append(f"{c['mean']:+.2f}{mark}")
        print(f"{metric:16s}" + "".join(f"{x:>11s}" for x in cells))
    print("  *+ beats vol (survives FDR);  *- worse than vol (survives FDR);  blank = ns after FDR")

    # --- Small-sample robustness on crypto beds ---
    crypto_beds = [b for b in beds if b.startswith("crypto")]
    print("\n" + "=" * 74)
    print("SMALL-SAMPLE ROBUSTNESS (crypto beds): MDE@80% power and LOO jackknife for VaR partial")
    print("=" * 74)
    robustness = {}
    for name in crypto_beds:
        c = partial_cells[(name, "var5")]
        n, sd = c["n"], c["sd"]
        mde = (Z_975 + Z_80) * sd / np.sqrt(n) if n >= 2 and np.isfinite(sd) else np.nan
        # LOO jackknife on VaR partial series
        s = np.asarray(per_bed_series[name]["partial"]["var5"], dtype=float)
        s = s[np.isfinite(s)]
        loo = [np.mean(np.delete(s, i)) for i in range(len(s))] if len(s) >= 2 else []
        robustness[name] = {"var5_partial_mean": round(c["mean"], 3), "n_dates": n,
                            "mde_80pct": round(mde, 3) if np.isfinite(mde) else None,
                            "loo_min": round(float(np.min(loo)), 3) if loo else None,
                            "loo_max": round(float(np.max(loo)), 3) if loo else None}
        print(f"  {name:20s} VaR partial={c['mean']:+.3f}  n={n}  MDE@80%={mde:+.3f}  "
              f"LOO range=[{robustness[name]['loo_min']}, {robustness[name]['loo_max']}]")

    out = {
        "partial": {f"{n}|{m}": partial_cells[(n, m)] for n in beds for m in COMPETING},
        "head_to_head": {f"{n}|{m}": h2h_cells[(n, m)] for n in beds for m in COMPETING},
        "partial_fdr_survivors": [f"{n}|{m}" for (n, m) in partial_survivors],
        "h2h_fdr_survivors": [f"{n}|{m}" for (n, m) in h2h_survivors],
        "partial_fdr_p_threshold": p_thresh_partial,
        "h2h_fdr_p_threshold": p_thresh_h2h,
        "crypto_robustness": robustness,
    }
    with open(output_dir / "robustness_fdr.json", "w") as h:
        json.dump(out, h, indent=2, default=float)
    print(f"\nResults saved: {output_dir / 'robustness_fdr.json'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
