"""
RQ2 — formal ablation of the autocorrelation "predictive" term.

H2: the (1-|ρ|) adjustment (Eq 7) adds no robust out-of-sample predictive power
over the base UPM/LPM ratio (Eq 4). Phase 1 previewed this (Δrank-corr ≤ 0.024);
here we make it rigorous by asking: does ANY variant of the term help?

We vary, on the paper's 3 windows × 4 investor types:
  - base (Eq 4)  vs  adjusted (Eq 7)
  - lag ∈ {1, 2, 3}                    (paper uses lag 1)
  - form ∈ {linear 1-|ρ|, squared 1-ρ²}(paper uses linear)
  - ρ method ∈ {correlation, covariance}(Eq 5 literally says covariance)

Outcome metric: OOS Spearman corr (metric vs 1y holding return). We report the
DELTA (adjusted − base) and a bootstrap-over-stocks 95% CI for the paper's exact
variant (lag1, linear, correlation). A CI straddling 0 => the term adds nothing.

Run: uv run python -u src/run_rq2_ablation.py
"""

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import partial_moments as pm
import data_pipeline as dp
from run_biased_arm import (
    WINDOWS, BENCHMARK_TICKER, RISKFREE_TICKER, PERIODS_PER_YEAR, MIN_OBS,
    INVESTOR_TYPES, build_double_benchmark_target, holding_mean_returns,
)

# ---------------------------------------------------------------- Config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"

LAGS   = [1, 2, 3]
FORMS  = {"linear": lambda rho: 1.0 - np.abs(rho), "squared": lambda rho: 1.0 - rho ** 2}
METHODS = ["correlation", "covariance"]
N_BOOTSTRAP = 1000
PAPER_VARIANT = ("lag1", "linear", "correlation")  # Eq 7 as written


def lag_autocorr(returns, lag, method):
    """Lag-k serial dependence; correlation (bounded) or covariance (Eq 5 literal)."""
    current, lagged = returns[lag:], returns[:-lag]
    if len(current) < 3:
        return 0.0
    if method == "correlation":
        if np.std(current) == 0 or np.std(lagged) == 0:
            return 0.0
        return float(np.corrcoef(current, lagged)[0, 1])
    return float(np.cov(current, lagged, ddof=0)[0, 1])


def per_stock_row(stock_returns, target_series):
    """Base ratio per investor + rho for every (lag, method); None if < MIN_OBS."""
    aligned = pd.concat({"r": stock_returns, "target": target_series}, axis=1).dropna()
    if len(aligned) < MIN_OBS:
        return None
    returns, target = aligned["r"].values, aligned["target"].values

    row = {}
    for type_name, deg in INVESTOR_TYPES.items():
        try:
            row[f"base_{type_name}"] = pm.explanatory_metric(returns, target, q=deg["q"], n=deg["n"])
        except ZeroDivisionError:
            return None
    for lag in LAGS:
        for method in METHODS:
            row[f"rho_lag{lag}_{method}"] = lag_autocorr(returns, lag, method)
    return row


def adjusted_metric(metrics_frame, investor, lag, method, form_name):
    """Eq-7-style adjusted metric column for one variant."""
    base = metrics_frame[f"base_{investor}"]
    rho = metrics_frame[f"rho_lag{lag}_{method}"]
    return base * FORMS[form_name](rho)


def bootstrap_delta_ci(base_metric, adj_metric, holding, n_boot):
    """Bootstrap-over-stocks 95% CI for Δ = spearman(adj,hold) − spearman(base,hold)."""
    paired = pd.concat({"base": base_metric, "adj": adj_metric, "hold": holding}, axis=1).dropna()
    if len(paired) < 10:
        return np.nan, np.nan, np.nan
    base_v, adj_v, hold_v = paired["base"].values, paired["adj"].values, paired["hold"].values
    point = spearmanr(adj_v, hold_v)[0] - spearmanr(base_v, hold_v)[0]
    # Proper with-replacement bootstrap over stocks; fixed seed for reproducibility.
    rng = np.random.default_rng(42)
    n = len(paired)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = spearmanr(adj_v[idx], hold_v[idx])[0] - spearmanr(base_v[idx], hold_v[idx])[0]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print("=" * 60)
    output_dir = OUTPUT_ROOT / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_rq2_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir  : {output_dir}")

    tickers = dp.get_current_sp500_tickers(DATA_DIR / "sp500_current.csv")
    prices = dp.download_adjusted_close(tickers, "1978-01-01", "2011-01-01",
                                        DATA_DIR / "yf_sp500_prices_1978_2011.parquet")
    bench = dp.download_adjusted_close([BENCHMARK_TICKER], "1978-01-01", "2011-01-01",
                                       DATA_DIR / "yf_benchmark_1978_2011.parquet")
    rf = dp.download_adjusted_close([RISKFREE_TICKER], "1978-01-01", "2011-01-01",
                                    DATA_DIR / "yf_riskfree_1978_2011.parquet")

    grid_rows = []
    ci_rows = []
    for window in WINDOWS:
        expl_start, expl_end = window["expl"]
        bench_returns = dp.to_monthly_returns(bench.loc[expl_start:expl_end])[BENCHMARK_TICKER]
        rf_rate = rf.loc[expl_start:expl_end][RISKFREE_TICKER].resample("ME").last()
        target_series = build_double_benchmark_target(
            bench_returns, dp.annualized_pct_to_periodic_rate(rf_rate, PERIODS_PER_YEAR))
        expl_returns = dp.to_monthly_returns(prices.loc[expl_start:expl_end])

        records = {}
        for ticker in expl_returns.columns:
            row = per_stock_row(expl_returns[ticker], target_series)
            if row is not None:
                records[ticker] = row
        metrics_frame = pd.DataFrame.from_dict(records, orient="index")
        holding = holding_mean_returns(prices, window["hold_start"], window["hold_1y"]).reindex(metrics_frame.index)
        print(f"  {window['name']}: {len(metrics_frame)} stocks")

        for investor in INVESTOR_TYPES:
            base_metric = metrics_frame[f"base_{investor}"]
            base_corr = spearmanr(*pd.concat({"m": base_metric, "h": holding}, axis=1).dropna().values.T)[0]
            for lag in LAGS:
                for method in METHODS:
                    for form_name in FORMS:
                        adj = adjusted_metric(metrics_frame, investor, lag, method, form_name)
                        paired = pd.concat({"m": adj, "h": holding}, axis=1).dropna()
                        adj_corr = spearmanr(paired["m"], paired["h"])[0]
                        grid_rows.append({
                            "window": window["name"], "investor": investor,
                            "lag": lag, "method": method, "form": form_name,
                            "base_corr": base_corr, "adjusted_corr": adj_corr,
                            "delta": adj_corr - base_corr,
                        })
            # Bootstrap CI for the paper's exact variant
            adj_paper = adjusted_metric(metrics_frame, investor, 1, "correlation", "linear")
            point, lo, hi = bootstrap_delta_ci(base_metric, adj_paper, holding, N_BOOTSTRAP)
            ci_rows.append({
                "window": window["name"], "investor": investor,
                "delta": point, "ci_lo": lo, "ci_hi": hi,
                "ci_includes_zero": bool(lo <= 0 <= hi) if not np.isnan(lo) else None,
            })

    grid = pd.DataFrame(grid_rows)
    ci = pd.DataFrame(ci_rows)
    grid.to_csv(output_dir / "ablation_grid.csv", index=False)
    ci.to_csv(output_dir / "paper_variant_bootstrap_ci.csv", index=False)
    with open(output_dir / "config.json", "w") as h:
        json.dump({"lags": LAGS, "forms": list(FORMS), "methods": METHODS,
                   "n_bootstrap": N_BOOTSTRAP, "paper_variant": PAPER_VARIANT}, h, indent=2)

    print("\n" + "=" * 60)
    print("RQ2 — does ANY variant of the autocorr term help? |delta| summary:")
    print(f"  max |delta| over ALL variants/windows/investors: {grid['delta'].abs().max():.4f}")
    print(f"  mean |delta|: {grid['delta'].abs().mean():.4f}")
    best = grid.loc[grid["delta"].idxmax()]
    print(f"  best-case single improvement: delta={best['delta']:.4f} "
          f"({best['window']}, {best['investor']}, lag{best['lag']}, {best['method']}, {best['form']})")
    print("\nPaper's exact variant (lag1, linear, correlation) — bootstrap 95% CI of delta:")
    print(ci.round(4).to_string(index=False))
    n_zero = ci["ci_includes_zero"].sum()
    print(f"\n{n_zero}/{len(ci)} cells have a CI straddling 0 => term adds no significant OOS power.")
    print("=" * 60)


if __name__ == "__main__":
    main()
