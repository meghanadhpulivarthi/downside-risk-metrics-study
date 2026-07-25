"""
P0-1 --- Target-confound test (the Devil's Advocate CRITICAL).

Charge: "magnitude beats shape" may be baked into forecasting max drawdown, which is a
magnitude-dominated functional of volatility (E[MDD] ~ sigma*sqrt(T)). So scoring metrics
by rank correlation against drawdown structurally favors magnitude metrics.

Test: re-score all ten metrics against SHAPE-SENSITIVE forward targets and see whether
shape metrics beat volatility there. If volatility still wins even on shape's home turf,
"magnitude, not shape" is not a target artifact. If shape metrics win on the shape targets,
the law must be rescoped to magnitude-like targets (drawdown).

Forward targets (all: higher = more downside/shape risk):
  maxdd        : forward 90d max drawdown (baseline, magnitude-dominated)
  neg_skew     : -skew(forward daily returns)  (left-asymmetry; pure shape, scale-free)
  downside_frac: downside_dev(fwd) / std(fwd)  (asymmetry ratio; shape, magnitude-normalized)
  tail_freq    : fraction of forward days below the trailing 5th percentile (tail-exceedance
                 frequency beyond trailing VaR; tail-shape, partly magnitude-normalized)

For each bed/target we report the per-date mean Spearman(metric, target) and the head-to-head
vs volatility (mean per-date [rho(metric,target) - rho(vol,target)]), and list which metrics
beat volatility. Reuses build_beds / compute_metrics / horizons from the paper's pipeline.

Run: uv run --project <repo> python -u src/autoresearch/run_target_sensitivity.py
"""

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import skew

import left_tail as lt
from run_multi_testbed import compute_metrics, METRICS, TRAILING_DAYS, FORWARD_DAYS
from run_mechanism import build_beds, _spearman, MIN_PAIRS

OUTPUT_ROOT = Path(__file__).resolve().parent.parent.parent / "outputs"
MIN_FWD_OBS = 20            # min forward daily returns to compute skew/tail targets
SHAPE_METRICS = {"vn_ratio", "hill_tail", "down_semibeta", "down_beta", "ltd_crash", "gda_voldown"}
TARGETS = ["maxdd", "neg_skew", "downside_frac", "tail_freq"]


def forward_targets(fwd_prices, fdate, trailing_returns):
    """Compute the four forward targets from a forward price window. NaN if not computable."""
    out = {t: np.nan for t in TARGETS}
    out["maxdd"] = lt.forward_drawdown(fwd_prices, fdate + pd.Timedelta(days=FORWARD_DAYS))
    fr = fwd_prices.dropna().pct_change().dropna().values
    if len(fr) >= MIN_FWD_OBS:
        out["neg_skew"] = -float(skew(fr))
        sd = float(np.std(fr))
        downside = fr[fr < 0]
        dd = float(np.sqrt(np.mean(downside ** 2))) if len(downside) else 0.0
        out["downside_frac"] = (dd / sd) if sd > 0 else np.nan
        if len(trailing_returns) >= 40:
            q05 = float(np.quantile(trailing_returns, 0.05))
            out["tail_freq"] = float(np.mean(fr <= q05))
    return out


def per_date_rhos(close, market_ret, formation_dates, universe_fn):
    """Per date and target: Spearman(metric, target) for every metric. Returns
    {target: {metric: [per-date rho, ...]}}."""
    series = {t: {m: [] for m in METRICS} for t in TARGETS}
    n_used = 0
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
            tgt = forward_targets(fwd, fdate, r.values)
            if np.isnan(tgt["maxdd"]):
                continue
            feats.update({f"tgt_{k}": v for k, v in tgt.items()})
            rows.append(feats)
        g = pd.DataFrame(rows)
        if len(g) < MIN_PAIRS:
            print(f"    skip {fdate.date()}: {len(g)} assets (<{MIN_PAIRS})")
            continue
        n_used += 1
        for target in TARGETS:
            tvals = g[f"tgt_{target}"].values
            for metric in METRICS:
                series[target][metric].append(_spearman(g[metric].values, tvals))
    return series, n_used


def summarize_bed(series):
    """Per target: mean Spearman per metric, head-to-head vs volatility, and beats-vol lists."""
    out = {}
    for target in TARGETS:
        vol = np.asarray(series[target]["volatility"], dtype=float)
        per_metric, h2h = {}, {}
        for metric in METRICS:
            mv = np.asarray(series[target][metric], dtype=float)
            per_metric[metric] = round(float(np.nanmean(mv)), 3) if np.isfinite(mv).any() else None
            n = min(len(vol), len(mv))
            diff = mv[:n] - vol[:n]
            diff = diff[np.isfinite(diff)]
            h2h[metric] = round(float(np.mean(diff)), 3) if len(diff) else None
        beats = [m for m in METRICS if m != "volatility" and h2h[m] is not None and h2h[m] > 0]
        shape_beats = [m for m in beats if m in SHAPE_METRICS]
        out[target] = {"mean_rho": per_metric, "h2h_vs_vol": h2h,
                       "beats_vol": beats, "shape_beats_vol": shape_beats}
    return out


def main():
    now = datetime.datetime.now()
    print("=" * 78)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : targets={TARGETS} trailing={TRAILING_DAYS}d forward={FORWARD_DAYS}d")
    print("=" * 78)
    output_dir = OUTPUT_ROOT / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_target_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    beds = build_beds()
    results = {}
    for name, (close, mkt, dates, univ) in beds.items():
        print(f"\n[{name}] scoring {len(dates)} dates over {len(TARGETS)} targets ...")
        series, n_used = per_date_rhos(close, mkt, dates, univ)
        if n_used < 3:
            print(f"  SKIP {name}: {n_used} usable dates")
            continue
        results[name] = {"n_dates": n_used, "targets": summarize_bed(series)}
        print(f"  used {n_used} dates")

    # Report: on each target, does volatility still win, and do shape metrics beat it?
    print("\n" + "=" * 78)
    print("WHICH METRICS BEAT VOLATILITY, BY TARGET (head-to-head mean per-date rho diff > 0)")
    print("=" * 78)
    for target in TARGETS:
        tag = "MAGNITUDE target" if target == "maxdd" else "SHAPE target"
        print(f"\n--- {target}  ({tag}) ---")
        for name in results:
            t = results[name]["targets"][target]
            volrho = t["mean_rho"]["volatility"]
            print(f"  [{name:20s}] vol rho={volrho:+.3f} | beats vol: {t['beats_vol'] or 'NONE'}")
            print(f"       shape metrics beating vol: {t['shape_beats_vol'] or 'NONE'}")

    with open(output_dir / "target_sensitivity.json", "w") as h:
        json.dump(results, h, indent=2, default=float)
    print(f"\nResults saved: {output_dir / 'target_sensitivity.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
