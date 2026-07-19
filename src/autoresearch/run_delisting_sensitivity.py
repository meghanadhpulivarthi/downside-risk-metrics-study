"""
Delisting-coding SENSITIVITY (reviewer P1-#2).

The magnitude story ("VaR/ES/downside-dev add incremental signal beyond volatility")
lives mostly in the crypto beds, where forward max-drawdown codes a mid-window
delisting as a terminal -100% loss (left_tail.forward_drawdown returns 1.0). A
reviewer's worry: because own-asset dispersion (vol/VaR) mechanically tracks death
risk, the "drawdown-forecast" magnitude edge might just be *death prediction*, making
it non-independent from the blow-up result and partly an artifact of the -100% coding.

We test how much of the magnitude increment survives when death is NOT coded as the
worst-possible drawdown, under three drawdown definitions:

  baseline  : delisting => drawdown = 1.0  (the paper's coding)
  lastprice : no forced 1.0; standard peak-to-trough over whatever data exists
              (i.e., value the asset at its last traded price, not at zero)
  exclude   : drop assets that died mid-window; skill among SURVIVORS only

For each bed and each own-asset MAGNITUDE metric we recompute the partial Spearman
rho(metric, fwd_dd | volatility). If the increment collapses under lastprice/exclude,
the edge was death-prediction; if it persists, it is genuine drawdown-magnitude skill.

Reuses build_beds / _spearman / _partial_spearman from run_mechanism and the paired
block bootstrap from run_multi_testbed_v2.

Run: uv run python -u src/autoresearch/run_delisting_sensitivity.py
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
from run_mechanism import build_beds, _spearman, _partial_spearman, MIN_PAIRS

# Config — edit these directly
OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"
# Own-asset magnitude metrics whose increment we stress-test (the paper's "winners").
MAGNITUDE_METRICS = ["var5", "es5", "downside_dev"]
DEATH_GAP_DAYS = 14           # same as left_tail default; how early data must end to count as "died"
NEVER_FORCE_GAP = 50_000      # ~137 yr: large enough that forward_drawdown never forces the -100%
                              # code, but within pandas Timedelta's ~106,751-day limit


def _dd_variants(fwd, window_end):
    """Return (dd_baseline, dd_lastprice, died) for one asset's forward window.

    dd_baseline : delisting coded as 1.0 (paper's forward_drawdown).
    dd_lastprice: peak-to-trough over available data; the -100% override is disabled.
    died        : True if the asset delisted / crashed to dust mid-window.
    """
    dd_base = lt.forward_drawdown(fwd, window_end, death_gap_days=DEATH_GAP_DAYS)
    dd_last = lt.forward_drawdown(fwd, window_end, death_gap_days=NEVER_FORCE_GAP)
    died = lt.died_in_window(fwd, window_end, death_gap_days=DEATH_GAP_DAYS)
    return dd_base, dd_last, died


def per_date_variants(close, market_ret, formation_dates, universe_fn):
    """Per formation date, build the cross-sectional frame once (metrics + all three
    drawdown codings + death flag), then compute the partial rho(metric | vol) for each
    magnitude metric under each of the three variants. Returns per-variant series dicts."""
    variants = ["baseline", "lastprice", "exclude"]
    series = {v: {m: [] for m in MAGNITUDE_METRICS} for v in variants}
    death_rate = []          # fraction of the cross-section that died, per date (for context)
    n_dates_used = 0

    for fdate in formation_dates:
        rows = []
        for asset in universe_fn(fdate):
            if asset not in close.columns:
                continue
            trailing = close.loc[fdate - pd.Timedelta(days=TRAILING_DAYS):fdate, asset].dropna()
            r = trailing.pct_change().dropna()
            if len(r) < 60:
                continue
            m = market_ret.reindex(r.index).fillna(0.0).values
            feats = compute_metrics(r.values, m)
            if feats is None:
                continue
            fwd = close.loc[fdate:fdate + pd.Timedelta(days=FORWARD_DAYS), asset]
            dd_base, dd_last, died = _dd_variants(fwd, fdate + pd.Timedelta(days=FORWARD_DAYS))
            if np.isnan(dd_base) or np.isnan(dd_last):
                continue
            feats["dd_base"] = dd_base
            feats["dd_last"] = dd_last
            feats["died"] = bool(died)
            rows.append(feats)

        g = pd.DataFrame(rows)
        if len(g) < MIN_PAIRS:
            print(f"    skip {fdate.date()}: only {len(g)} assets (<{MIN_PAIRS})")
            continue
        n_dates_used += 1
        death_rate.append(float(g["died"].mean()))

        vol = g["volatility"].values
        # Frames per variant. exclude => survivors only (baseline==lastprice there, no deaths).
        frames = {
            "baseline": (vol, g["dd_base"].values, g),
            "lastprice": (vol, g["dd_last"].values, g),
        }
        surv = g[~g["died"]]
        if len(surv) >= MIN_PAIRS:
            frames["exclude"] = (surv["volatility"].values, surv["dd_last"].values, surv)
        else:
            # Not enough survivors to trust a cross-sectional rho on this date — skip, loudly.
            print(f"    {fdate.date()}: only {len(surv)} survivors (<{MIN_PAIRS}); exclude-variant skips this date")

        for variant, (v, dd, frame) in frames.items():
            r_vol_dd = _spearman(v, dd)
            for metric in MAGNITUDE_METRICS:
                mv = frame[metric].values
                r_m_dd = _spearman(mv, dd)
                r_m_vol = _spearman(mv, v)
                series[variant][metric].append(_partial_spearman(r_m_dd, r_m_vol, r_vol_dd))

    return series, n_dates_used, death_rate


def summarize(series, n_dates, death_rate):
    out = {"n_dates": n_dates,
           "mean_death_rate": round(float(np.mean(death_rate)), 3) if death_rate else 0.0,
           "variants": {}}
    for variant, per_metric in series.items():
        out["variants"][variant] = {}
        for metric, vals in per_metric.items():
            usable = [x for x in vals if np.isfinite(x)]
            if len(usable) < 3:
                # Too few usable dates for this variant/metric — record NaN, do not fake a CI.
                print(f"    {variant}/{metric}: only {len(usable)} usable dates; reporting NaN")
                out["variants"][variant][metric] = {"mean": None, "ci": [None, None], "n": len(usable)}
                continue
            mean, lo, hi = block_bootstrap_diff(usable)
            out["variants"][variant][metric] = {
                "mean": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)],
                "n": len(usable), "sig_pos": bool(np.isfinite(lo) and lo > 0)}
    return out


def main():
    now = datetime.datetime.now()
    print("=" * 70)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : magnitude={MAGNITUDE_METRICS} DEATH_GAP={DEATH_GAP_DAYS}d "
          f"TRAILING={TRAILING_DAYS}d FORWARD={FORWARD_DAYS}d MIN_PAIRS={MIN_PAIRS}")
    print("=" * 70)

    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_delisting_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    beds = build_beds()
    results = {}
    for name, (close, mkt, dates, univ) in beds.items():
        print(f"\n[{name}] scoring {len(dates)} formation dates ...")
        series, n_dates, death_rate = per_date_variants(close, mkt, dates, univ)
        if n_dates < 3:
            print(f"  SKIP bed {name}: only {n_dates} usable dates")
            continue
        results[name] = summarize(series, n_dates, death_rate)
        print(f"  used {n_dates} dates; mean death rate {results[name]['mean_death_rate']:.1%}")

    # --- Side-by-side partial rho(metric | vol) across drawdown codings ---
    print("\n" + "=" * 70)
    print("Partial Spearman rho(metric, fwd_dd | volatility) under 3 drawdown codings")
    print("  baseline = death coded -100% ; lastprice = last traded price ; exclude = survivors only")
    print("=" * 70)
    for metric in MAGNITUDE_METRICS:
        print(f"\n{metric}")
        print(f"  {'bed':22s}{'baseline':>18s}{'lastprice':>18s}{'exclude':>18s}")
        for name in results:
            cells = []
            for variant in ["baseline", "lastprice", "exclude"]:
                rec = results[name]["variants"][variant][metric]
                if rec["mean"] is None:
                    cells.append("n/e")
                else:
                    flag = "*" if rec.get("sig_pos") else " "
                    cells.append(f"{rec['mean']:+.3f}{flag} n={rec['n']}")
            print(f"  {name:22s}" + "".join(f"{c:>18s}" for c in cells))
    print("\n  * = 95% block-bootstrap CI excludes 0")

    out_path = output_dir / "delisting_sensitivity.json"
    with open(out_path, "w") as h:
        json.dump(results, h, indent=2)
    print(f"\nResults saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
