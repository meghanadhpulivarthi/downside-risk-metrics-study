"""
Mechanism DEPTH: sharpen the volatility-persistence result three ways.

run_mechanism.py established (M2) that trailing vol -> forward vol -> drawdown, and
(M1) that only tail MAGNITUDE adds signal beyond vol. This script deepens that:

  (1) MEDIATION (reference horizon h=90). Is forward volatility a *sufficient statistic*
      for trailing volatility? Partial Spearman rho(trailing_vol, dd | forward_vol) should
      collapse toward 0, while rho(forward_vol, dd | trailing_vol) stays high. If so,
      trailing vol forecasts drawdown ONLY through its forecast of forward vol.
      NB: forward_vol->dd is contemporaneous/mechanical (both from the forward window);
      the *predictive* content lives entirely in trailing->forward vol.

  (2) HORIZON DECAY. For h in {14,30,45,60,90}: does drawdown skill decay in lock-step
      with volatility persistence as the horizon grows? Mechanism => yes.

  (3) PLACEBO. Permute forward drawdown within each date; the trailing-vol correlation
      must collapse to ~0 (calibrates the block-bootstrap CIs).

Also emits per-bed heterogeneity inputs (blow-up rate, trailing excess kurtosis) for the
"increment scales with tail fatness" figure.

Run: uv run --project <repo> python -u src/autoresearch/run_mechanism_depth.py
"""

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

import evaluate as ev
import left_tail as lt
from run_multi_testbed import compute_metrics, TRAILING_DAYS
from run_multi_testbed_v2 import block_bootstrap_diff
from run_mechanism import build_beds, _spearman, _partial_spearman, MIN_PAIRS

# Config — edit these directly
HORIZONS = [14, 30, 45, 60, 90]   # forward days
REF_H = 90                        # reference horizon for mediation / heterogeneity
BLOWUP_DD = 0.8                   # fwd_dd >= this counts as a blow-up (near-total loss)
PLACEBO_SEED = 7
OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"


def per_date_rows(close, market_ret, fdate, universe_fn):
    """One row per asset for a formation date: trailing vol/var5/kurtosis + forward
    drawdown and forward realized vol at every horizon in HORIZONS."""
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
        row = {"volatility": feats["volatility"], "var5": feats["var5"],
               "kurt": float(kurtosis(r.values, fisher=True, bias=False)) if len(r) > 8 else np.nan}
        for h in HORIZONS:
            fwd = close.loc[fdate:fdate + pd.Timedelta(days=h), asset]
            dd = lt.forward_drawdown(fwd, fdate + pd.Timedelta(days=h))
            fwd_r = fwd.dropna().pct_change().dropna()
            row[f"dd_{h}"] = dd
            row[f"fvol_{h}"] = float(np.std(fwd_r.values)) if len(fwd_r) >= 5 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_bed(close, market_ret, formation_dates, universe_fn):
    """Return per-date series dicts for mediation, horizon, placebo + heterogeneity scalars."""
    med_tv_given_fv, med_fv_given_tv = [], []          # mediation partials @ REF_H
    hz_persist = {h: [] for h in HORIZONS}             # rho(trailing_vol, forward_vol_h)
    hz_skill = {h: [] for h in HORIZONS}               # rho(trailing_vol, dd_h)
    hz_var_incr = {h: [] for h in HORIZONS}            # partial rho(var5, dd_h | vol)
    placebo = []
    blowup_num, blowup_den, kurts = 0, 0, []
    rng = np.random.default_rng(PLACEBO_SEED)

    for fdate in formation_dates:
        g = per_date_rows(close, market_ret, fdate, universe_fn)
        if len(g) < MIN_PAIRS:
            print(f"    skip {fdate.date()}: {len(g)} assets (<{MIN_PAIRS})")
            continue

        vol = g["volatility"].values
        var5 = g["var5"].values

        # --- mediation @ REF_H ---
        dd = g[f"dd_{REF_H}"].values
        fvol = g[f"fvol_{REF_H}"].values
        r_tv_dd = _spearman(vol, dd)
        r_fv_dd = _spearman(fvol, dd)
        r_tv_fv = _spearman(vol, fvol)
        med_tv_given_fv.append(_partial_spearman(r_tv_dd, r_tv_fv, r_fv_dd))
        med_fv_given_tv.append(_partial_spearman(r_fv_dd, r_tv_fv, r_tv_dd))

        # --- horizon decay ---
        for h in HORIZONS:
            ddh = g[f"dd_{h}"].values
            fvh = g[f"fvol_{h}"].values
            hz_persist[h].append(_spearman(vol, fvh))
            hz_skill[h].append(_spearman(vol, ddh))
            r_var_dd = _spearman(var5, ddh)
            r_var_vol = _spearman(var5, vol)
            r_vol_dd = _spearman(vol, ddh)
            hz_var_incr[h].append(_partial_spearman(r_var_dd, r_var_vol, r_vol_dd))

        # --- placebo: permute dd within the date ---
        dd_valid = dd[np.isfinite(dd)]
        vol_valid = vol[np.isfinite(dd)]
        if len(dd_valid) >= MIN_PAIRS:
            placebo.append(_spearman(vol_valid, rng.permutation(dd_valid)))

        # --- heterogeneity accumulation ---
        blowup_num += int(np.sum(dd >= BLOWUP_DD))
        blowup_den += int(np.sum(np.isfinite(dd)))
        kurts.extend(g["kurt"].dropna().tolist())

    def agg(series):
        mean, lo, hi = block_bootstrap_diff(series)
        return {"mean": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)],
                "sig_pos": bool(np.isfinite(lo) and lo > 0),
                "sig_neg": bool(np.isfinite(hi) and hi < 0)}

    return {
        "mediation": {"trailing_given_forward": agg(med_tv_given_fv),
                      "forward_given_trailing": agg(med_fv_given_tv)},
        "horizon": {str(h): {"persistence": agg(hz_persist[h]),
                             "skill": agg(hz_skill[h]),
                             "var5_incr": agg(hz_var_incr[h])} for h in HORIZONS},
        "placebo": agg(placebo),
        "heterogeneity": {"blowup_rate": round(blowup_num / blowup_den, 3) if blowup_den else None,
                          "mean_trailing_kurt": round(float(np.mean(kurts)), 2) if kurts else None},
    }


def main():
    now = datetime.datetime.now()
    print("=" * 70)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : horizons={HORIZONS} ref_h={REF_H} blowup>={BLOWUP_DD}")
    print("=" * 70)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_mechanism_depth"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    beds = build_beds()
    results = {}
    for name, (close, mkt, dates, univ) in beds.items():
        print(f"\n[{name}] {len(dates)} formation dates ...")
        results[name] = run_bed(close, mkt, dates, univ)

    print("\n" + "=" * 70)
    print("MEDIATION @ h=90  (trailing vol acts only through forward vol?)")
    print("=" * 70)
    for name, res in results.items():
        med = res["mediation"]
        print(f"[{name}]")
        print(f"    rho(trailing_vol, dd | forward_vol) = {med['trailing_given_forward']['mean']:+.3f} "
              f"CI{med['trailing_given_forward']['ci']}   <- expect ~0 (mediated)")
        print(f"    rho(forward_vol,  dd | trailing_vol)= {med['forward_given_trailing']['mean']:+.3f} "
              f"CI{med['forward_given_trailing']['ci']}   <- expect high")

    print("\n" + "=" * 70)
    print("HORIZON DECAY  (persistence and drawdown-skill should track)")
    print("=" * 70)
    for name, res in results.items():
        print(f"[{name}]  " + "  ".join(
            f"h{h}: persist={res['horizon'][str(h)]['persistence']['mean']:+.2f} "
            f"skill={res['horizon'][str(h)]['skill']['mean']:+.2f}" for h in HORIZONS))

    print("\n" + "=" * 70)
    print("PLACEBO (permuted dd -> ~0) and HETEROGENEITY")
    print("=" * 70)
    for name, res in results.items():
        p = res["placebo"]
        het = res["heterogeneity"]
        print(f"[{name}] placebo={p['mean']:+.3f} CI{p['ci']}  |  "
              f"blowup_rate={het['blowup_rate']}  mean_kurt={het['mean_trailing_kurt']}")

    out_path = output_dir / "mechanism_depth.json"
    with open(out_path, "w") as h:
        json.dump(results, h, indent=2)
    print(f"\nResults saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
