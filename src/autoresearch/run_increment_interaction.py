"""
Exp 4 --- Is the VaR tail-magnitude increment larger where tails are fatter?

Replaces the descriptive n=5 cross-bed scatter (paper App: cross-bed heterogeneity)
with a real test. We pool per-date observations across all five beds and regress the
per-date VaR increment on the per-date tail-fatness of the cross-section:

    var_partial_date  ~  tailfat_date

where
  var_partial_date = per-date partial Spearman rho(VaR, fwd_dd | volatility)
  tailfat_date     = cross-sectional median trailing excess kurtosis on that date
                     (robustness: cross-sectional blow-up rate = share with fwd_dd >= 0.8)

Significance comes from a date-level nonparametric bootstrap on the pooled slope, plus
a bed-clustered bootstrap (resample whole beds) as a conservative robustness variant.
Success = a positive slope whose bootstrap CI excludes zero.

Reuses the exact machinery behind Table 1/2 (build_beds, compute_metrics, the 90d
forward drawdown, trailing 180d) so the numbers are comparable to the paper.

Run: uv run --project <repo> python -u src/autoresearch/run_increment_interaction.py
"""

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

import left_tail as lt
from run_multi_testbed import compute_metrics, TRAILING_DAYS, FORWARD_DAYS
from run_mechanism import build_beds, _spearman, _partial_spearman, MIN_PAIRS

# Config --- edit these directly
OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent / "outputs"
BLOWUP_CUTOFF = 0.80          # fwd_dd >= this counts as a blow-up (for the blow-up-rate variant)
N_BOOT = 5000                 # bootstrap resamples for the slope CI
MIN_TRAIL_OBS = 60            # min trailing daily returns to compute a metric (matches pipeline)


def per_date_records(close, market_ret, formation_dates, universe_fn, bed_name):
    """One record per usable formation date: VaR increment + tail-fatness of the cross-section."""
    records = []
    for fdate in formation_dates:
        var_vals, vol_vals, dd_vals, kurt_vals = [], [], [], []
        for asset in universe_fn(fdate):
            if asset not in close.columns:
                continue
            trailing = close.loc[fdate - pd.Timedelta(days=TRAILING_DAYS):fdate, asset].dropna()
            r = trailing.pct_change().dropna()
            if len(r) < MIN_TRAIL_OBS:
                continue
            m = market_ret.reindex(r.index).fillna(0.0).values
            feats = compute_metrics(r.values, m)
            if feats is None:
                continue
            fwd = close.loc[fdate:fdate + pd.Timedelta(days=FORWARD_DAYS), asset]
            dd = lt.forward_drawdown(fwd, fdate + pd.Timedelta(days=FORWARD_DAYS))
            if np.isnan(dd):
                continue
            var_vals.append(feats["var5"])
            vol_vals.append(feats["volatility"])
            dd_vals.append(dd)
            kurt_vals.append(float(kurtosis(r.values, fisher=True, bias=False)))

        n = len(dd_vals)
        if n < MIN_PAIRS:
            # too few assets to trust a per-date cross-sectional rho — skip, loudly
            print(f"    [{bed_name}] skip {fdate.date()}: only {n} assets (<{MIN_PAIRS})")
            continue

        var_arr = np.asarray(var_vals); vol_arr = np.asarray(vol_vals); dd_arr = np.asarray(dd_vals)
        r_var_dd = _spearman(var_arr, dd_arr)
        r_var_vol = _spearman(var_arr, vol_arr)
        r_vol_dd = _spearman(vol_arr, dd_arr)
        var_partial = _partial_spearman(r_var_dd, r_var_vol, r_vol_dd)
        if not np.isfinite(var_partial):
            print(f"    [{bed_name}] skip {fdate.date()}: VaR partial not finite")
            continue

        records.append({
            "bed": bed_name,
            "date": str(fdate.date()),
            "var_partial": float(var_partial),
            "median_kurt": float(np.median(kurt_vals)),
            "blowup_rate": float(np.mean(dd_arr >= BLOWUP_CUTOFF)),
            "n_assets": int(n),
        })
    return records


def _slope(x, y):
    """OLS slope of y on x (finite pairs)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def bootstrap_slope(frame, xcol, ycol, cluster_col=None, n_boot=N_BOOT):
    """Bootstrap CI + one-sided p for the pooled slope. If cluster_col given, resample
    whole clusters (beds); otherwise resample individual date-records."""
    rng = np.random.default_rng(0)
    point = _slope(frame[xcol].values, frame[ycol].values)
    slopes = np.empty(n_boot)
    if cluster_col is None:
        idx = np.arange(len(frame))
        for i in range(n_boot):
            take = rng.choice(idx, size=len(idx), replace=True)
            sub = frame.iloc[take]
            slopes[i] = _slope(sub[xcol].values, sub[ycol].values)
    else:
        clusters = frame[cluster_col].unique()
        for i in range(n_boot):
            chosen = rng.choice(clusters, size=len(clusters), replace=True)
            sub = pd.concat([frame[frame[cluster_col] == c] for c in chosen], ignore_index=True)
            slopes[i] = _slope(sub[xcol].values, sub[ycol].values)
    slopes = slopes[np.isfinite(slopes)]
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    frac_le0 = float(np.mean(slopes <= 0))
    return {"slope": round(point, 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "p_one_sided": round(2 * min(frac_le0, 1 - frac_le0), 4), "n_boot": int(len(slopes))}


def main():
    now = datetime.datetime.now()
    print("=" * 74)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : blowup_cutoff={BLOWUP_CUTOFF}  n_boot={N_BOOT}  trailing={TRAILING_DAYS}d forward={FORWARD_DAYS}d")
    print("=" * 74)
    output_dir = OUTPUT_ROOT / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_increment_interaction"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    beds = build_beds()
    all_records = []
    for name, (close, mkt, dates, univ) in beds.items():
        print(f"\n[{name}] collecting per-date records ...")
        recs = per_date_records(close, mkt, dates, univ, name)
        print(f"  usable dates: {len(recs)}")
        all_records.extend(recs)

    frame = pd.DataFrame(all_records)
    print(f"\nPooled date-level observations: {len(frame)} across {frame['bed'].nunique()} beds")
    if len(frame) < 10:
        raise SystemExit("Too few pooled observations to fit an interaction — aborting loudly.")

    results = {"n_obs": int(len(frame)), "n_beds": int(frame["bed"].nunique())}
    # Primary: tail-fatness = median trailing excess kurtosis
    results["kurt_date_bootstrap"] = bootstrap_slope(frame, "median_kurt", "var_partial")
    results["kurt_bed_clustered"] = bootstrap_slope(frame, "median_kurt", "var_partial", cluster_col="bed")
    # Robustness: tail-fatness = cross-sectional blow-up rate
    results["blowup_date_bootstrap"] = bootstrap_slope(frame, "blowup_rate", "var_partial")
    results["blowup_bed_clustered"] = bootstrap_slope(frame, "blowup_rate", "var_partial", cluster_col="bed")

    # Per-bed means for context (matches the descriptive scatter we are replacing)
    per_bed = frame.groupby("bed").agg(
        n_dates=("date", "size"), var_partial=("var_partial", "mean"),
        median_kurt=("median_kurt", "mean"), blowup_rate=("blowup_rate", "mean")).round(3)
    results["per_bed"] = per_bed.reset_index().to_dict(orient="records")

    print("\n" + "=" * 74)
    print("Per-bed means (context):")
    print(per_bed.to_string())
    print("\nPooled interaction  VaR_partial ~ tail-fatness:")
    for key in ["kurt_date_bootstrap", "kurt_bed_clustered", "blowup_date_bootstrap", "blowup_bed_clustered"]:
        r = results[key]
        print(f"  {key:24s} slope={r['slope']:+.4f}  95% CI {r['ci']}  p={r['p_one_sided']}")
    print("=" * 74)

    frame.to_csv(output_dir / "per_date_records.csv", index=False)
    with open(output_dir / "increment_interaction.json", "w") as h:
        json.dump(results, h, indent=2)
    print(f"\nResults saved: {output_dir / 'increment_interaction.json'}")
    print(f"Records saved: {output_dir / 'per_date_records.csv'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
