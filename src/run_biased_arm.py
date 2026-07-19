"""
Phase 1 — biased-arm replication, across the paper's THREE windows.

Reproduce Viole & Nawrocki (2016)'s claim on a deliberately SURVIVING universe
(today's S&P 500 back-filled via yfinance), at the paper's ~monthly frequency,
over its three overlapping 11-year windows. The biased numbers here become the
baseline that Phase 2's survivorship-bias-free arm is measured against.

Robustness question: does the out-of-sample rank correlation's sign/strength
hold across windows, or is it regime-specific (a §4 soft spot)?

Run: uv run python -u src/run_biased_arm.py
"""

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

import partial_moments as pm
import data_pipeline as dp

# ---------------------------------------------------------------- Config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"

# One download spanning all windows; monthly frequency mirrors the paper.
FULL_DOWNLOAD_START = "1978-01-01"
FULL_DOWNLOAD_END   = "2011-01-01"

# The paper's three overlapping 11-year windows; holding does NOT overlap its own
# explanatory window (fixes §4.3 within each window; cross-window overlap is the
# paper's design, kept here on purpose for like-for-like replication).
WINDOWS = [
    {"name": "1978-1989", "expl": ["1978-01-01", "1989-01-01"],
     "hold_start": "1989-01-01", "hold_1y": "1990-01-01", "hold_2y": "1991-01-01"},
    {"name": "1988-1999", "expl": ["1988-01-01", "1999-01-01"],
     "hold_start": "1999-01-01", "hold_1y": "2000-01-01", "hold_2y": "2001-01-01"},
    {"name": "1998-2009", "expl": ["1998-01-01", "2009-01-01"],
     "hold_start": "2009-01-01", "hold_1y": "2010-01-01", "hold_2y": "2011-01-01"},
]

BENCHMARK_TICKER = "^GSPC"   # systemic proxy (paper used CRSP mkt-cap index)
RISKFREE_TICKER  = "^IRX"    # 13-week T-bill, annualized percent
PERIODS_PER_YEAR = 12        # monthly
MIN_OBS          = 100       # monthly obs; paper: moments stabilize ~100+ obs
RHO_METHOD       = "correlation"  # covariance variant is audited in RQ2
INVESTOR_TYPES   = pm.INVESTOR_DEGREES

CONFIG = {
    "windows": WINDOWS, "benchmark": BENCHMARK_TICKER, "riskfree": RISKFREE_TICKER,
    "return_frequency": "monthly", "periods_per_year": PERIODS_PER_YEAR,
    "min_obs": MIN_OBS, "rho_method": RHO_METHOD, "investor_types": INVESTOR_TYPES,
    "universe": "current_sp500_survivors",
}


def build_double_benchmark_target(bench_returns, rf_periodic):
    """Elementwise max(benchmark return, risk-free) aligned by date -> Series."""
    joined = pd.concat({"bench": bench_returns, "rf": rf_periodic}, axis=1).dropna()
    target = np.maximum(joined["bench"].values, joined["rf"].values)
    return pd.Series(target, index=joined.index, name="target")


def compute_stock_metrics(stock_returns, target_series):
    """All partial-moment metrics for one stock; None (logged by caller) if < MIN_OBS."""
    aligned = pd.concat({"r": stock_returns, "target": target_series}, axis=1).dropna()
    if len(aligned) < MIN_OBS:
        return None

    returns = aligned["r"].values
    target = aligned["target"].values
    rho = pm.lag1_autocorr(returns, rho_method=RHO_METHOD)

    metrics = {
        "n_obs": len(aligned), "rho": rho,
        "sharpe": float(np.mean(returns) / np.std(returns)),
        "explanatory_mean_return": float(np.mean(returns)),
    }
    for type_name, degrees in INVESTOR_TYPES.items():
        q, n = degrees["q"], degrees["n"]
        try:
            ratio = pm.explanatory_metric(returns, target, q=q, n=n)
        except ZeroDivisionError as exc:
            print(f"    LPM=0 for investor={type_name}: {exc}")
            return None
        metrics[f"explanatory_{type_name}"] = ratio
        metrics[f"predictive_{type_name}"] = ratio * (1.0 - abs(rho))
    return metrics


def holding_mean_returns(prices, start, end):
    """Mean MONTHLY return per ticker over a holding window; NaN if no data."""
    window_returns = dp.to_monthly_returns(prices.loc[start:end])
    return window_returns.mean(axis=0, skipna=True)


def rank_correlation_rows(metrics_frame, holding_1y, holding_2y):
    """Four rank correlations per investor type + the Sharpe foil."""
    rows = []

    def add(metric_col, target_name, target_series):
        paired = pd.concat({"m": metrics_frame[metric_col], "t": target_series}, axis=1).dropna()
        if len(paired) < 3:
            print(f"    skip {metric_col} vs {target_name}: <3 paired points")
            return
        rho_s, p_value = spearmanr(paired["m"], paired["t"])
        rows.append({
            "metric": metric_col, "target": target_name,
            "spearman_rho": float(rho_s), "p_value": float(p_value), "n": int(len(paired)),
        })

    targets = {
        "explanatory_mean_return": metrics_frame["explanatory_mean_return"],
        "holding_1y_mean_return": holding_1y,
        "holding_2y_mean_return": holding_2y,
    }
    for type_name in INVESTOR_TYPES:
        for metric_col in (f"explanatory_{type_name}", f"predictive_{type_name}"):
            for target_name, target_series in targets.items():
                add(metric_col, target_name, target_series)
    add("sharpe", "holding_1y_mean_return", holding_1y)
    add("sharpe", "holding_2y_mean_return", holding_2y)
    return rows


def run_window(prices, bench, rf, window, output_dir):
    """Full metric + rank-correlation pass for one window. Returns tagged rows."""
    expl_start, expl_end = window["expl"]
    print(f"\n--- window {window['name']}: explanatory {expl_start}..{expl_end} ---")

    bench_returns = dp.to_monthly_returns(bench.loc[expl_start:expl_end])[BENCHMARK_TICKER]
    rf_rate = rf.loc[expl_start:expl_end][RISKFREE_TICKER].resample("ME").last()
    rf_periodic = dp.annualized_pct_to_periodic_rate(rf_rate, PERIODS_PER_YEAR)
    target_series = build_double_benchmark_target(bench_returns, rf_periodic)
    print(f"  double-benchmark target: {len(target_series)} months")

    expl_returns = dp.to_monthly_returns(prices.loc[expl_start:expl_end])
    records = {}
    skipped = 0
    for ticker in expl_returns.columns:
        result = compute_stock_metrics(expl_returns[ticker], target_series)
        if result is None:
            skipped += 1
            continue
        records[ticker] = result
    print(f"  metrics for {len(records)} tickers; skipped {skipped} (< {MIN_OBS} obs)")
    if len(records) < 3:
        print(f"  WARNING: too few tickers with history in {window['name']} — skipping window")
        return []

    metrics_frame = pd.DataFrame.from_dict(records, orient="index")
    metrics_frame.index.name = "ticker"
    metrics_frame.to_csv(output_dir / f"per_stock_metrics_{window['name']}.csv")

    holding_1y = holding_mean_returns(prices, window["hold_start"], window["hold_1y"]).reindex(metrics_frame.index)
    holding_2y = holding_mean_returns(prices, window["hold_start"], window["hold_2y"]).reindex(metrics_frame.index)

    rows = rank_correlation_rows(metrics_frame, holding_1y, holding_2y)
    for row in rows:
        row["window"] = window["name"]
        row["n_stocks"] = len(records)
    return rows


def main():
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Windows     : {[w['name'] for w in WINDOWS]}")
    print("=" * 60)

    output_dir = OUTPUT_ROOT / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_biased_arm_multiwindow"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir  : {output_dir}")

    # --- Load universe + prices once over the full span (restartable via caches)
    tickers = dp.get_current_sp500_tickers(DATA_DIR / "sp500_current.csv")
    prices = dp.download_adjusted_close(
        tickers, FULL_DOWNLOAD_START, FULL_DOWNLOAD_END, DATA_DIR / "yf_sp500_prices_1978_2011.parquet"
    )
    bench = dp.download_adjusted_close(
        [BENCHMARK_TICKER], FULL_DOWNLOAD_START, FULL_DOWNLOAD_END, DATA_DIR / "yf_benchmark_1978_2011.parquet"
    )
    rf = dp.download_adjusted_close(
        [RISKFREE_TICKER], FULL_DOWNLOAD_START, FULL_DOWNLOAD_END, DATA_DIR / "yf_riskfree_1978_2011.parquet"
    )
    print(f"Loaded prices for {prices.shape[1]}/{len(tickers)} tickers")

    all_rows = []
    for window in tqdm(WINDOWS, desc="windows"):
        all_rows.extend(run_window(prices, bench, rf, window, output_dir))

    results = pd.DataFrame(all_rows)
    with open(output_dir / "config.json", "w") as handle:
        json.dump(CONFIG, handle, indent=2)
    results.to_csv(output_dir / "rank_correlations_all_windows.csv", index=False)
    print(f"\nSaved config + rank_correlations_all_windows.csv -> {output_dir}")

    # --- Robustness view: predictive metric vs 1y holding return, window × investor type
    print("\n" + "=" * 60)
    print("ROBUSTNESS: predictive metric vs 1y holding return (spearman rho)")
    headline = results[
        results["metric"].str.startswith("predictive_")
        & (results["target"] == "holding_1y_mean_return")
    ].copy()
    headline["investor"] = headline["metric"].str.replace("predictive_", "", regex=False)
    pivot = headline.pivot(index="investor", columns="window", values="spearman_rho")
    print(pivot.round(3).to_string())
    print("\nSharpe foil vs 1y holding return (spearman rho):")
    sharpe = results[(results["metric"] == "sharpe") & (results["target"] == "holding_1y_mean_return")]
    print(sharpe[["window", "spearman_rho", "p_value", "n"]].to_string(index=False))
    print("\nStock counts per window:")
    print(results.groupby("window")["n_stocks"].first().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
