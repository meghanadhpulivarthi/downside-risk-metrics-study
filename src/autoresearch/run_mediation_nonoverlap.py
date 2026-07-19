"""
Non-overlapping mediation — the P0 fix from peer review.

The original mediation (run_mechanism_depth.py) measured forward volatility and forward
drawdown over the SAME forward window, so the forward-vol->drawdown link is contemporaneous
/ mechanical and the "sufficient statistic" reading is partly definitional. Here we measure
the mediator on a STRICTLY EARLIER, DISJOINT window than the outcome, so the mediator is
genuinely PREDICTIVE:

    trailing vol : [t-180, t]        (predictor)
    early fwd vol: [t,     t+30]     (mediator; days 1-30)
    drawdown     : [t+30,  t+120]    (outcome; days 31-120, a 90d window AFTER the vol window)

Mediation test: partial rho(trailing_vol, drawdown | early_fwd_vol). If it still collapses
toward 0 on equity, the "forward volatility (near-)mediates trailing volatility" claim
stands on a genuinely predictive design, not an accounting identity.

For direct comparison we also recompute the OVERLAP design (vol and dd both on [t, t+90]).

Run: uv run --project <repo> python -u src/autoresearch/run_mediation_nonoverlap.py
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
from run_multi_testbed import compute_metrics, TRAILING_DAYS
from run_multi_testbed_v2 import block_bootstrap_diff
from run_mechanism import build_beds, _spearman, _partial_spearman, MIN_PAIRS

# Config — edit these directly
VOL_H = 30    # mediator: early forward-vol window length (days)
DD_H = 90     # outcome: drawdown window length (days)
OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"


def _fwd_vol(close, asset, start, end):
    w = close.loc[start:end, asset].dropna().pct_change().dropna()
    return float(np.std(w.values)) if len(w) >= 5 else np.nan


def run_bed(close, market_ret, formation_dates, universe_fn, overlap):
    """Per-date mediation partials for one bed.

    overlap=True : vol & dd both on [t, t+DD_H]   (contemporaneous, the old design)
    overlap=False: vol on [t, t+VOL_H], dd on [t+VOL_H, t+VOL_H+DD_H]  (disjoint, predictive)
    """
    tv_given_fv, fv_given_tv, raw_tv_dd, raw_fv_dd = [], [], [], []
    n_used = 0
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
            if overlap:
                vol_end = fdate + pd.Timedelta(days=DD_H)
                dd_start, dd_end = fdate, fdate + pd.Timedelta(days=DD_H)
                fvol = _fwd_vol(close, asset, fdate, vol_end)
            else:
                fvol = _fwd_vol(close, asset, fdate, fdate + pd.Timedelta(days=VOL_H))
                dd_start = fdate + pd.Timedelta(days=VOL_H)
                dd_end = fdate + pd.Timedelta(days=VOL_H + DD_H)
            dd = lt.forward_drawdown(close.loc[dd_start:dd_end, asset], dd_end)
            if np.isnan(dd) or np.isnan(fvol):
                continue
            rows.append({"tvol": feats["volatility"], "fvol": fvol, "dd": dd})
        g = pd.DataFrame(rows)
        if len(g) < MIN_PAIRS:
            print(f"    skip {fdate.date()}: {len(g)} assets (<{MIN_PAIRS})")
            continue
        n_used += 1
        tv, fv, dd = g["tvol"].values, g["fvol"].values, g["dd"].values
        r_tv_dd = _spearman(tv, dd)
        r_fv_dd = _spearman(fv, dd)
        r_tv_fv = _spearman(tv, fv)
        raw_tv_dd.append(r_tv_dd)
        raw_fv_dd.append(r_fv_dd)
        tv_given_fv.append(_partial_spearman(r_tv_dd, r_tv_fv, r_fv_dd))
        fv_given_tv.append(_partial_spearman(r_fv_dd, r_tv_fv, r_tv_dd))

    def agg(s):
        mean, lo, hi = block_bootstrap_diff(s)
        return {"mean": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)],
                "sig_pos": bool(np.isfinite(lo) and lo > 0)}

    return {"n_dates": n_used,
            "trailing_given_forward": agg(tv_given_fv),
            "forward_given_trailing": agg(fv_given_tv),
            "raw_trailing_dd": agg(raw_tv_dd),
            "raw_forward_dd": agg(raw_fv_dd)}


def main():
    now = datetime.datetime.now()
    print("=" * 74)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : VOL_H={VOL_H}d  DD_H={DD_H}d  (non-overlap: vol[0,{VOL_H}] dd[{VOL_H},{VOL_H+DD_H}])")
    print("=" * 74)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_mediation_nonoverlap"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    beds = build_beds()
    results = {"overlap": {}, "nonoverlap": {}}
    for design, overlap in [("overlap", True), ("nonoverlap", False)]:
        print(f"\n### DESIGN = {design} ###")
        for name, (close, mkt, dates, univ) in beds.items():
            print(f"[{name}]")
            results[design][name] = run_bed(close, mkt, dates, univ, overlap)

    print("\n" + "=" * 74)
    print("MEDIATION  rho(trailing_vol, drawdown | forward_vol)   [~0 => forward vol mediates]")
    print("=" * 74)
    print(f"{'bed':22s} {'OVERLAP (mechanical)':>24s}   {'NON-OVERLAP (predictive)':>26s}")
    for name in beds:
        o = results["overlap"][name]["trailing_given_forward"]
        n = results["nonoverlap"][name]["trailing_given_forward"]
        nd = results["nonoverlap"][name]["n_dates"]
        print(f"{name:22s} {o['mean']:+.3f} CI{str(o['ci']):>14s}   "
              f"{n['mean']:+.3f} CI{str(n['ci']):>14s}  (n={nd})")

    print("\nfor reference, rho(forward_vol, drawdown | trailing_vol) NON-OVERLAP:")
    for name in beds:
        n = results["nonoverlap"][name]["forward_given_trailing"]
        print(f"  {name:22s} {n['mean']:+.3f} CI{n['ci']}")

    with open(output_dir / "mediation_nonoverlap.json", "w") as h:
        json.dump(results, h, indent=2)
    print(f"\nResults saved: {output_dir / 'mediation_nonoverlap.json'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
