"""
Reviewer-requested robustness checks (revision round 1):
  (A) Total-return benchmark: re-run the predictive-vs-1y-holding rank correlations
      with ^SP500TR (total return) instead of ^GSPC (price index), for the windows
      ^SP500TR covers (from 1988). Shows whether the price-vs-total-return benchmark
      mismatch moves any conclusion. (Methodology W, Devil's Advocate m2)
  (B) Multiple-testing: Benjamini-Hochberg FDR across the full family of reported
      rank correlations from the 3-window biased run. (Methodology W4)

Run: uv run python -u src/run_robustness_checks.py
"""

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import data_pipeline as dp
from run_biased_arm import (
    WINDOWS, BENCHMARK_TICKER, RISKFREE_TICKER, PERIODS_PER_YEAR, INVESTOR_TYPES,
    build_double_benchmark_target, compute_stock_metrics, holding_mean_returns,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"
TR_BENCHMARK = "^SP500TR"


def predictive_holding_corrs(prices, bench_prices, bench_col, rf, window):
    """Predictive-metric vs 1y-holding rank corr per investor type, for one benchmark."""
    expl_start, expl_end = window["expl"]
    bench_returns = dp.to_monthly_returns(bench_prices.loc[expl_start:expl_end])[bench_col]
    rf_rate = rf.loc[expl_start:expl_end][RISKFREE_TICKER].resample("ME").last()
    target = build_double_benchmark_target(
        bench_returns, dp.annualized_pct_to_periodic_rate(rf_rate, PERIODS_PER_YEAR))
    expl_returns = dp.to_monthly_returns(prices.loc[expl_start:expl_end])

    records = {}
    for ticker in expl_returns.columns:
        row = compute_stock_metrics(expl_returns[ticker], target)
        if row is not None:
            records[ticker] = row
    metrics = pd.DataFrame.from_dict(records, orient="index")
    holding = holding_mean_returns(prices, window["hold_start"], window["hold_1y"]).reindex(metrics.index)

    out = {}
    for investor in INVESTOR_TYPES:
        paired = pd.concat({"m": metrics[f"predictive_{investor}"], "h": holding}, axis=1).dropna()
        out[investor] = spearmanr(paired["m"], paired["h"])[0] if len(paired) >= 10 else np.nan
    return out


def benjamini_hochberg(pvals, alpha=0.05):
    """Return (n_significant_raw, n_significant_fdr) at level alpha."""
    pvals = np.asarray(pvals, dtype=float)
    pvals = pvals[~np.isnan(pvals)]
    m = len(pvals)
    order = np.argsort(pvals)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    passed = pvals[order] <= thresholds
    n_fdr = 0
    if passed.any():
        n_fdr = np.max(np.where(passed)[0]) + 1  # largest k passing
    return int((pvals < alpha).sum()), int(n_fdr)


def main():
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Robustness checks — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    output_dir = OUTPUT_ROOT / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_robustness_checks"
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = dp.get_current_sp500_tickers(DATA_DIR / "sp500_current.csv")
    prices = dp.download_adjusted_close(tickers, "1978-01-01", "2011-01-01",
                                        DATA_DIR / "yf_sp500_prices_1978_2011.parquet")
    gspc = dp.download_adjusted_close([BENCHMARK_TICKER], "1978-01-01", "2011-01-01",
                                      DATA_DIR / "yf_benchmark_1978_2011.parquet")
    rf = dp.download_adjusted_close([RISKFREE_TICKER], "1978-01-01", "2011-01-01",
                                    DATA_DIR / "yf_riskfree_1978_2011.parquet")
    sp500tr = dp.download_adjusted_close([TR_BENCHMARK], "1988-01-01", "2011-01-01",
                                         DATA_DIR / "yf_sp500tr_1988_2011.parquet")

    # (A) Total-return benchmark comparison on windows SP500TR covers (post-1988).
    print("\n(A) Benchmark: ^GSPC (price) vs ^SP500TR (total return), predictive vs 1y holding")
    tr_rows = []
    for window in WINDOWS:
        if pd.Timestamp(window["expl"][0]) < pd.Timestamp("1988-01-01"):
            print(f"  {window['name']}: skipped (before ^SP500TR coverage)")
            continue
        gspc_corrs = predictive_holding_corrs(prices, gspc, BENCHMARK_TICKER, rf, window)
        tr_corrs = predictive_holding_corrs(prices, sp500tr, TR_BENCHMARK, rf, window)
        for investor in INVESTOR_TYPES:
            tr_rows.append({"window": window["name"], "investor": investor,
                            "gspc": round(gspc_corrs[investor], 3),
                            "sp500tr": round(tr_corrs[investor], 3),
                            "abs_diff": round(abs(gspc_corrs[investor] - tr_corrs[investor]), 3)})
    tr_frame = pd.DataFrame(tr_rows)
    print(tr_frame.to_string(index=False))
    max_diff = tr_frame["abs_diff"].max() if len(tr_frame) else np.nan
    print(f"  MAX |Δ rank-corr| from switching to total-return benchmark: {max_diff}")

    # (B) FDR across the reported rank-correlation family from the latest multiwindow run.
    print("\n(B) Multiple-testing (Benjamini-Hochberg FDR) over reported rank correlations")
    multiwindow = sorted(OUTPUT_ROOT.glob("*_biased_arm_multiwindow/rank_correlations_all_windows.csv"))
    fdr_summary = {}
    if multiwindow:
        family = pd.read_csv(multiwindow[-1])
        n_raw, n_fdr = benjamini_hochberg(family["p_value"].values, alpha=0.05)
        fdr_summary = {"family_size": int(len(family)), "significant_raw_p<0.05": n_raw,
                       "significant_after_BH_FDR": n_fdr, "source": str(multiwindow[-1].parent.name)}
        print(f"  family size {len(family)}: {n_raw} significant at raw p<.05, "
              f"{n_fdr} survive BH-FDR at q<.05")
    else:
        print("  (no multiwindow results found)")

    tr_frame.to_csv(output_dir / "benchmark_total_return_comparison.csv", index=False)
    with open(output_dir / "summary.json", "w") as h:
        json.dump({"max_abs_diff_total_return": None if pd.isna(max_diff) else float(max_diff),
                   "fdr": fdr_summary}, h, indent=2)
    print(f"\nSaved -> {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
