"""
The honest "better metric": a PRE-REGISTERED, zero-tuning composite.

The mechanism (run_mechanism.py) says tail MAGNITUDE adds a small increment beyond
volatility, and VaR is the magnitude metric with the most consistent incremental signal
(significant on all 5 beds in M1). So the single, theory-motivated composite to test is:

    comp = z(volatility) + z(VaR5)      (equal weight, cross-sectional z per date)

There is NO fitting and NO free parameter, so this is not data-snooping — it is one
pre-specified model, evaluated once per test bed and once on the locked panel test split.
Expectation from the mechanism: a small positive edge over volatility in the fat-tailed
crypto beds, ~none in equity. Reported either way.

Run: uv run --project <repo> python -u src/autoresearch/run_composite_honest.py
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
from run_multi_testbed import compute_metrics, TRAILING_DAYS, FORWARD_DAYS
from run_multi_testbed_v2 import block_bootstrap_diff
from run_mechanism import build_beds, _spearman, MIN_PAIRS

OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"


def zscore(x):
    """Cross-sectional z-score, NaN-aware (NaNs stay NaN)."""
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.full_like(x, np.nan)
    return (x - mu) / sd


def bed_diff(close, market_ret, formation_dates, universe_fn):
    """Per-date rho(composite, dd) - rho(vol, dd); block-bootstrap mean diff + CI."""
    diffs, rho_comp_list, rho_vol_list = [], [], []
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
            rows.append({"vol": feats["volatility"], "var5": feats["var5"], "dd": dd})
        g = pd.DataFrame(rows)
        if len(g) < MIN_PAIRS:
            print(f"    skip {fdate.date()}: {len(g)} assets (<{MIN_PAIRS})")
            continue
        comp = zscore(g["vol"].values) + zscore(g["var5"].values)
        rho_comp = _spearman(comp, g["dd"].values)
        rho_vol = _spearman(g["vol"].values, g["dd"].values)
        if np.isfinite(rho_comp) and np.isfinite(rho_vol):
            diffs.append(rho_comp - rho_vol)
            rho_comp_list.append(rho_comp)
            rho_vol_list.append(rho_vol)
    mean, lo, hi = block_bootstrap_diff(diffs)
    return {"rho_vol": round(float(np.mean(rho_vol_list)), 3),
            "rho_comp": round(float(np.mean(rho_comp_list)), 3),
            "diff": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)],
            "sig_beats_vol": bool(np.isfinite(lo) and lo > 0),
            "sig_worse": bool(np.isfinite(hi) and hi < 0), "n_dates": len(diffs)}


def panel_locked_test():
    """Single pre-specified comparison on the LOCKED test split: vol vs z(vol)+z(var5)."""
    panel = ev.load_panel()
    test = panel[panel["split"] == "test"].dropna(subset=["vol", "var5", "fwd_drawdown"]).copy()
    # cross-sectional z per formation date, then equal-weight sum
    comp = np.full(len(test), np.nan)
    for _, idx in test.groupby("formation_date").groups.items():
        sub = test.loc[idx]
        comp[test.index.get_indexer(idx)] = zscore(sub["vol"].values) + zscore(sub["var5"].values)
    test["_comp"] = comp

    vol_rhos = ev._per_date_spearman(test["vol"].values, test)
    comp_rhos = ev._per_date_spearman(test["_comp"].values, test)
    # paired per-date difference where both exist (same dates, same order)
    diff = comp_rhos - vol_rhos if len(comp_rhos) == len(vol_rhos) else np.array([])
    return {"n_test_dates": int(len(vol_rhos)),
            "vol_test_mean": round(float(np.mean(vol_rhos)), 3),
            "comp_test_mean": round(float(np.mean(comp_rhos)), 3),
            "diff_mean": round(float(np.mean(diff)), 3) if len(diff) else None,
            "diff_newey_west_t": round(ev.newey_west_t(diff), 2) if len(diff) else None}


def main():
    now = datetime.datetime.now()
    print("=" * 70)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print("Composite   : z(vol) + z(VaR5), equal weight, NO fitting (pre-registered)")
    print("=" * 70)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_composite_honest"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    beds = build_beds()
    per_bed = {}
    print("\n5-bed head-to-head (composite vs volatility):")
    for name, (close, mkt, dates, univ) in beds.items():
        print(f"\n[{name}]")
        res = bed_diff(close, mkt, dates, univ)
        per_bed[name] = res
        flag = " ** beats vol **" if res["sig_beats_vol"] else (" (worse)" if res["sig_worse"] else " (n.s.)")
        print(f"    vol rho={res['rho_vol']:+.3f}  comp rho={res['rho_comp']:+.3f}  "
              f"diff={res['diff']:+.3f} CI{res['ci']}{flag}")

    print("\n" + "=" * 70)
    print("LOCKED PANEL — single pre-specified test-split comparison")
    print("=" * 70)
    locked = panel_locked_test()
    print(f"    vol  test mean rho = {locked['vol_test_mean']:+.3f}")
    print(f"    comp test mean rho = {locked['comp_test_mean']:+.3f}  "
          f"(diff {locked['diff_mean']:+.3f}, NW t={locked['diff_newey_west_t']}, "
          f"n_dates={locked['n_test_dates']})")

    out = {"per_bed": per_bed, "locked_panel": locked}
    with open(output_dir / "composite_honest.json", "w") as h:
        json.dump(out, h, indent=2)
    print(f"\nResults saved: {output_dir / 'composite_honest.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
