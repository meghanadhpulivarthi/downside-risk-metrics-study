"""
Exp 3 --- Disciplines ablation: which shape metric would you FALSELY crown as
beating volatility if you dropped each of the paper's three fair-field disciplines?

  (1) No survivorship-free data  [the new experiment, Option A]
      Rerun the crypto head-to-head battery on a SURVIVORS-ONLY universe: at each
      formation date keep only coins that still have price data at the dataset end
      (the ones you'd get by downloading "today's" coins and backfilling). Compare
      which metrics beat volatility under survivors-only vs the full survivorship-free
      universe, and whether VaR's magnitude edge shrinks.

  (2) No multiple-testing correction  [re-summary of run_robustness_fdr.py]
      Head-to-head winners at nominal p<.05 that do NOT survive Benjamini-Hochberg.

  (3) No locked holdout  [re-summary of the snooping run, if present]
      The validation-selected metric's inflated VAL score vs its honest locked-TEST score.

Head-to-head "beats vol" on a bed = mean per-date [rho(metric,dd) - rho(vol,dd)] > 0.
Survivorship is Option A (survives to dataset end), agreed with the author.

Run: uv run --project <repo> python -u src/autoresearch/run_disciplines_ablation.py
"""

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from run_multi_testbed import METRICS
from run_mechanism import build_beds, per_date_series
from run_multi_testbed_v2 import block_bootstrap_diff

# Config --- edit these directly
OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent / "outputs"
SURVIVE_TOL_DAYS = 30          # a coin "survives" if its last price is within this of the panel end
COMPETING = [m for m in METRICS if m != "volatility"]
MAGNITUDE = ["downside_dev", "es5", "var5"]
SHAPE = ["vn_ratio", "hill_tail", "down_semibeta", "down_beta", "ltd_crash", "gda_voldown"]


def survivor_set(close, tol_days, dust_fraction=0.05):
    """
    Option A survivors: coins you would still see if you built the universe from
    names that 'made it' to the dataset end. In this panel, dead coins keep
    reporting a dust price rather than terminating, so survivorship is about
    crashing to dust, not the series ending. A coin survives iff it neither
    terminated early (>tol_days before the panel end) NOR ended below
    dust_fraction of its all-time peak (i.e. it did not crater to ~zero).
    """
    end = close.index.max()
    survivors = set()
    for c in close.columns:
        s = close[c].dropna()
        if len(s) < 2:
            continue
        terminated = s.index.max() < end - pd.Timedelta(days=tol_days)
        to_dust = float(s.iloc[-1]) < dust_fraction * float(s.max())
        if not terminated and not to_dust:
            survivors.add(c)
    return survivors, end


def head_to_head(series):
    """Per-metric mean per-date [rho(metric,dd) - rho(vol,dd)] with block-bootstrap 95% CI.
    series is the dict returned by run_mechanism.per_date_series (raw::* lists, date-aligned)."""
    vol = np.asarray(series["raw::volatility"], dtype=float)
    out = {}
    for metric in COMPETING:
        mvals = np.asarray(series[f"raw::{metric}"], dtype=float)
        n = min(len(vol), len(mvals))
        diff = (mvals[:n] - vol[:n])
        diff = diff[np.isfinite(diff)]
        if len(diff) < 3:
            out[metric] = {"mean": np.nan, "ci": [np.nan, np.nan], "beats_vol": False, "n": len(diff)}
            continue
        mean, lo, hi = block_bootstrap_diff(list(diff))
        out[metric] = {"mean": round(float(mean), 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
                       "beats_vol": bool(mean > 0), "sig_beats": bool(np.isfinite(lo) and lo > 0),
                       "n": int(len(diff))}
    return out


def var_partial_mean(series):
    s = np.asarray(series["partial::var5"], dtype=float)
    s = s[np.isfinite(s)]
    return round(float(np.mean(s)), 3) if len(s) else np.nan


def load_no_bh_arm():
    """From the latest robustness_fdr run: head-to-head winners nominally p<.05 that do NOT survive BH."""
    runs = sorted(OUTPUT_ROOT.glob("*_robustness_fdr/robustness_fdr.json"))
    if not runs:
        print("  (no robustness_fdr run found — skipping no-BH arm)")
        return None
    data = json.loads(runs[-1].read_text())
    h2h = data["head_to_head"]
    survivors = set(data.get("h2h_fdr_survivors", []))
    nominal_winners, false_crowns = [], []
    for key, cell in h2h.items():
        mean, p = cell.get("mean"), cell.get("p")
        if mean is None or p is None or not np.isfinite(mean) or not np.isfinite(p):
            continue
        if mean > 0 and p < 0.05:
            nominal_winners.append(key)
            if key not in survivors:
                false_crowns.append(key)
    return {"source": runs[-1].parent.name,
            "nominal_positive_winners_p<.05": sorted(nominal_winners),
            "vanish_under_BH": sorted(false_crowns),
            "shape_false_crowns": sorted([k for k in false_crowns if k.split("|")[1] in SHAPE])}


def load_no_locked_arm():
    """From the latest snooping run: val-selected metric's VAL score vs its locked-TEST score."""
    runs = sorted(OUTPUT_ROOT.glob("*_snooping_robustness/*.json"))
    if not runs:
        print("  (no snooping_robustness run found — no-locked arm referenced from paper instead)")
        return None
    data = json.loads(runs[-1].read_text())
    return {"source": runs[-1].parent.name, "keys": sorted(list(data.keys()))[:20]}


def main():
    now = datetime.datetime.now()
    print("=" * 74)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : Option A survivors, SURVIVE_TOL_DAYS={SURVIVE_TOL_DAYS}")
    print("=" * 74)
    output_dir = OUTPUT_ROOT / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_disciplines_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    beds = build_beds()
    crypto_beds = {n: v for n, v in beds.items() if n.startswith("crypto")}

    survivorship = {}
    for name, (close, mkt, dates, univ) in crypto_beds.items():
        survivors, end = survivor_set(close, SURVIVE_TOL_DAYS)
        n_full_univ = len(set().union(*[set(univ(d)) for d in dates]))
        n_surv_univ = len(set().union(*[set(univ(d)) for d in dates]) & survivors)
        print(f"\n[{name}] panel end {end.date()}; union universe {n_full_univ} coins, "
              f"of which {n_surv_univ} survive to end ({SURVIVE_TOL_DAYS}d tol)")

        print(f"  full (survivorship-free) universe ...")
        series_full, n_full = per_date_series(close, mkt, dates, univ)
        surv_univ = lambda d, _u=univ: [c for c in _u(d) if c in survivors]
        print(f"  survivors-only universe ...")
        series_surv, n_surv = per_date_series(close, mkt, dates, surv_univ)

        h2h_full = head_to_head(series_full)
        h2h_surv = head_to_head(series_surv)
        # metrics that beat vol under survivors-only but NOT under the full survivorship-free bed
        false_crowns = [m for m in COMPETING
                        if h2h_surv[m]["beats_vol"] and not h2h_full[m]["beats_vol"]]
        survivorship[name] = {
            "n_dates_full": n_full, "n_dates_surv": n_surv,
            "n_coins_union_full": n_full_univ, "n_coins_union_surv": n_surv_univ,
            "var_partial_full": var_partial_mean(series_full),
            "var_partial_surv": var_partial_mean(series_surv),
            "h2h_full": h2h_full, "h2h_surv": h2h_surv,
            "false_crowns_any": false_crowns,
            "false_crowns_shape": [m for m in false_crowns if m in SHAPE],
        }
        print(f"    VaR partial: full {survivorship[name]['var_partial_full']:+.3f}  "
              f"-> survivors-only {survivorship[name]['var_partial_surv']:+.3f}")
        print(f"    metrics beating vol only under survivors-only: {false_crowns or 'NONE'}")
        print(f"      of which SHAPE metrics: {survivorship[name]['false_crowns_shape'] or 'NONE'}")

    print("\n" + "=" * 74)
    print("(2) NO multiple-testing correction")
    no_bh = load_no_bh_arm()
    if no_bh:
        print(f"  nominal h2h winners p<.05: {no_bh['nominal_positive_winners_p<.05'] or 'NONE'}")
        print(f"  vanish under BH: {no_bh['vanish_under_BH'] or 'NONE'}")
        print(f"  SHAPE false crowns without BH: {no_bh['shape_false_crowns'] or 'NONE'}")

    print("\n(3) NO locked holdout")
    no_locked = load_no_locked_arm()

    out = {"config": {"survive_tol_days": SURVIVE_TOL_DAYS, "option": "A"},
           "no_survivorship_free": survivorship, "no_bh": no_bh, "no_locked": no_locked}
    with open(output_dir / "disciplines_ablation.json", "w") as h:
        json.dump(out, h, indent=2, default=float)
    print(f"\nResults saved: {output_dir / 'disciplines_ablation.json'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
