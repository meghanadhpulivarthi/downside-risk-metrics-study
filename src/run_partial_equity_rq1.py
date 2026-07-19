"""
Phase 2a / RQ1 — directional survivorship test on equity (LOWER BOUND).

Free survivorship-free equity prices are not obtainable (Kaggle misses the
failure tail — see context/gotchas.md). So this is a directional test, not a
full bias-free replication:

  Universe A (biased)      = point-in-time S&P 500 members 1998-2009 that are
                             STILL in the index today (survivors).
  Universe B (less biased) = A + the members that LEFT the index and that Kaggle
                             still retains (the partial delisted tail we can get).

If the predictive metric's out-of-sample rank correlation shrinks / changes from
A to B, that is evidence for H1 (survivorship inflates the edge) — and because B
only adds the delisted names Kaggle happens to keep, the measured shift is a
LOWER BOUND on the true bias. All prices come from Kaggle for consistency;
benchmark/rf reuse the yfinance caches. Coverage is printed loudly.

Run: uv run python -u src/run_partial_equity_rq1.py
"""

import datetime
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

import data_pipeline as dp
# Reuse the exact Phase-1 metric machinery so A/B are computed identically.
from run_biased_arm import (
    BENCHMARK_TICKER, RISKFREE_TICKER, PERIODS_PER_YEAR, INVESTOR_TYPES,
    build_double_benchmark_target, compute_stock_metrics,
    holding_mean_returns, rank_correlation_rows,
)

# ---------------------------------------------------------------- Config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"
KAGGLE_STOCKS = DATA_DIR / "kaggle_huge_stock" / "Data" / "Stocks"

EXPL_START, EXPL_END = "1998-01-01", "2009-01-01"
HOLD_START = "2009-01-01"
HOLD_1Y_END, HOLD_2Y_END = "2010-01-01", "2011-01-01"


def predictive_vs_holding(results, universe_label):
    """Extract predictive-metric vs 1y-holding rows, tagged with the universe."""
    sub = results[
        results["metric"].str.startswith("predictive_")
        & (results["target"] == "holding_1y_mean_return")
    ].copy()
    sub["investor"] = sub["metric"].str.replace("predictive_", "", regex=False)
    sub["universe"] = universe_label
    return sub[["universe", "investor", "spearman_rho", "p_value", "n"]]


def main():
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Window      : explanatory {EXPL_START}..{EXPL_END}, holding -> {HOLD_2Y_END}")
    print("=" * 60)

    output_dir = OUTPUT_ROOT / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_partial_equity_rq1"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir  : {output_dir}")

    # --- Point-in-time universe (fja05680) split into survivors vs left-index
    hist = pd.read_csv(DATA_DIR / "sp500_historical_constituents.csv")
    members = dp.pit_members_in_window(hist, EXPL_START, "2009-12-31")
    today = {t.strip().upper() for t in pd.read_csv(DATA_DIR / "sp500_current.csv")["wiki_symbol"]}
    survivors = sorted(members & today)
    left_index = sorted(members - today)
    print(f"PIT members 1998-2009: {len(members)} | survivors={len(survivors)} | left index={len(left_index)}")

    # --- Kaggle prices for both groups
    print("Loading survivor prices from Kaggle:")
    surv_prices, surv_matched, _ = dp.load_kaggle_prices(survivors, KAGGLE_STOCKS)
    print("Loading left-index (delisted tail) prices from Kaggle:")
    left_prices, left_matched, _ = dp.load_kaggle_prices(left_index, KAGGLE_STOCKS)
    print(f"COVERAGE: survivors {len(surv_matched)}/{len(survivors)}, "
          f"delisted tail {len(left_matched)}/{len(left_index)} "
          f"(the missing tail is the lower-bound caveat)")

    prices = pd.concat([surv_prices, left_prices], axis=1)

    # --- Benchmark + risk-free (reuse yfinance caches) -> monthly double benchmark
    bench = dp.download_adjusted_close([BENCHMARK_TICKER], EXPL_START, HOLD_2Y_END,
                                       DATA_DIR / "yf_benchmark_1978_2011.parquet")
    rf = dp.download_adjusted_close([RISKFREE_TICKER], EXPL_START, HOLD_2Y_END,
                                    DATA_DIR / "yf_riskfree_1978_2011.parquet")
    bench_returns = dp.to_monthly_returns(bench.loc[EXPL_START:EXPL_END])[BENCHMARK_TICKER]
    rf_rate = rf.loc[EXPL_START:EXPL_END][RISKFREE_TICKER].resample("ME").last()
    rf_periodic = dp.annualized_pct_to_periodic_rate(rf_rate, PERIODS_PER_YEAR)
    target_series = build_double_benchmark_target(bench_returns, rf_periodic)

    # --- Metrics for every ticker in the pooled universe
    expl_returns = dp.to_monthly_returns(prices.loc[EXPL_START:EXPL_END])
    records = {}
    skipped = 0
    for ticker in expl_returns.columns:
        result = compute_stock_metrics(expl_returns[ticker], target_series)
        if result is None:
            skipped += 1
            continue
        records[ticker] = result
    metrics_frame = pd.DataFrame.from_dict(records, orient="index")
    metrics_frame.index.name = "ticker"
    print(f"Metrics computed for {len(metrics_frame)} tickers (skipped {skipped} < min obs)")

    # --- Holding returns. For firms that delisted mid-holding, Kaggle data stops;
    #     the mean over available months is used and the count of such firms is flagged.
    holding_1y = holding_mean_returns(prices, HOLD_START, HOLD_1Y_END).reindex(metrics_frame.index)
    holding_2y = holding_mean_returns(prices, HOLD_START, HOLD_2Y_END).reindex(metrics_frame.index)

    surv_set = set(surv_matched) & set(metrics_frame.index)
    left_set = set(left_matched) & set(metrics_frame.index)
    left_missing_hold = sum(1 for t in left_set if pd.isna(holding_1y.get(t)))

    # Delisting-sensitivity check (reviewer W3): a true survivorship-free test would
    # assign a -100% terminal return to firms that STOP TRADING within the holding
    # window. Count how many universe-B firms actually end inside [HOLD_START, HOLD_2Y_END].
    delisted_in_window = 0
    for ticker in left_set:
        last_obs = prices[ticker].dropna().index.max()
        if pd.Timestamp(HOLD_START) <= last_obs <= pd.Timestamp(HOLD_2Y_END):
            delisted_in_window += 1
    print(f"Universe A (survivors) n={len(surv_set)} | Universe B adds {len(left_set)} delisted "
          f"({left_missing_hold} lack 1y holding data)")
    print(f"Delisting-sensitivity check: {delisted_in_window}/{len(left_set)} universe-B firms "
          f"stop trading WITHIN the holding window. If 0, the -100% terminal-return convention "
          f"cannot change RQ1 — the free source retains no in-window deaths (the survivorship gap).")

    # --- A vs B rank correlations
    frame_a = metrics_frame.loc[sorted(surv_set)]
    frame_b = metrics_frame.loc[sorted(surv_set | left_set)]
    rows_a = rank_correlation_rows(frame_a, holding_1y.reindex(frame_a.index), holding_2y.reindex(frame_a.index))
    rows_b = rank_correlation_rows(frame_b, holding_1y.reindex(frame_b.index), holding_2y.reindex(frame_b.index))
    results_a = pd.DataFrame(rows_a)
    results_b = pd.DataFrame(rows_b)

    comparison = pd.concat([
        predictive_vs_holding(results_a, "A_survivors"),
        predictive_vs_holding(results_b, "B_plus_delisted"),
    ])

    # --- Save
    with open(output_dir / "config.json", "w") as handle:
        json.dump({
            "window_expl": [EXPL_START, EXPL_END], "holding_1y_end": HOLD_1Y_END,
            "coverage_survivors": [len(surv_matched), len(survivors)],
            "coverage_delisted_tail": [len(left_matched), len(left_index)],
            "delisted_in_holding_window": delisted_in_window,
            "note": "directional lower-bound RQ1; delisted coverage partial",
        }, handle, indent=2)
    metrics_frame.to_csv(output_dir / "per_stock_metrics.csv")
    comparison.to_csv(output_dir / "A_vs_B_predictive_vs_holding.csv", index=False)

    # --- Report
    print("\n" + "=" * 60)
    print("H1 DIRECTIONAL TEST — predictive metric vs 1y holding, A vs B:")
    pivot = comparison.pivot(index="investor", columns="universe", values="spearman_rho")
    pivot["shift_B_minus_A"] = pivot["B_plus_delisted"] - pivot["A_survivors"]
    print(pivot.round(3).to_string())
    print("\nShift != 0 => adding the (partial) delisted tail changes the edge (H1 support).")
    print("Magnitude is a LOWER BOUND: the full failure tail is missing from free data.")
    print("=" * 60)


if __name__ == "__main__":
    main()
