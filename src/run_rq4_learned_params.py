"""
RQ4 — are the (q, n) degree exponents learnable, and do they generalize?

The paper hand-sets q,n by "investor type." RQ4 asks: if we instead LEARN the
(q,n) that best fits the data in one era, does that choice generalize to a
different era? If the metric carries real signal, learned params should hold up
out of sample; if the framework is curve-fitting, in-sample fit will NOT survive
the jump to a new window (and won't beat the hand-set presets).

Procedure (equity survivor universe, monthly, the paper's 3 windows):
  1. On a TRAIN window, grid-search (q,n) maximizing Spearman(explanatory metric,
     1y holding return).
  2. Apply the learned (q,n) to a later TEST window (true out-of-sample).
  3. Compare to (a) the best hand-set preset on the test window and (b) the
     test-window grid-optimal (the in-sample ceiling for that window).

Overfitting signature: test_corr(learned) << train_corr(learned), and no better
than presets.

Run: uv run python -u src/run_rq4_learned_params.py
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"

# Includes the four preset degrees (0.25, 0.44, 1.0, 2.0) so presets are comparable on-grid.
GRID = [0.1, 0.25, 0.44, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
# Train->test walk-forward pairs (earlier era learns, later era tests).
PAIRS = [("1978-1989", "1988-1999"), ("1988-1999", "1998-2009")]


def window_metrics(window, prices, bench, rf):
    """Per-stock explanatory metric for every (q,n) on the grid + 1y holding return."""
    expl_start, expl_end = window["expl"]
    bench_returns = dp.to_monthly_returns(bench.loc[expl_start:expl_end])[BENCHMARK_TICKER]
    rf_rate = rf.loc[expl_start:expl_end][RISKFREE_TICKER].resample("ME").last()
    target_series = build_double_benchmark_target(
        bench_returns, dp.annualized_pct_to_periodic_rate(rf_rate, PERIODS_PER_YEAR))
    expl_returns = dp.to_monthly_returns(prices.loc[expl_start:expl_end])

    records = {}
    for ticker in expl_returns.columns:
        aligned = pd.concat({"r": expl_returns[ticker], "target": target_series}, axis=1).dropna()
        if len(aligned) < MIN_OBS:
            continue
        returns, target = aligned["r"].values, aligned["target"].values
        upm_by_q = {q: pm.upm(returns, target, q) for q in GRID}
        lpm_by_n = {n: pm.lpm(returns, target, n) for n in GRID}
        row = {}
        for q in GRID:
            for n in GRID:
                lower = lpm_by_n[n]
                row[(q, n)] = upm_by_q[q] / lower if lower > 0 else np.nan
        records[ticker] = row
    metrics_frame = pd.DataFrame.from_dict(records, orient="index")
    holding = holding_mean_returns(prices, window["hold_start"], window["hold_1y"]).reindex(metrics_frame.index)
    return metrics_frame, holding


def corr_for(metrics_frame, holding, q, n):
    paired = pd.concat({"m": metrics_frame[(q, n)], "h": holding}, axis=1).dropna()
    if len(paired) < 10:
        return np.nan
    return spearmanr(paired["m"], paired["h"])[0]


def best_grid(metrics_frame, holding):
    """(q,n) maximizing POSITIVE Spearman with holding return (framework's directional claim)."""
    best_qn, best_corr = None, -np.inf
    for q in GRID:
        for n in GRID:
            c = corr_for(metrics_frame, holding, q, n)
            if not np.isnan(c) and c > best_corr:
                best_qn, best_corr = (q, n), c
    return best_qn, best_corr


def best_preset(metrics_frame, holding):
    """Best hand-set preset on this window (upper bound of the paper's fixed choices)."""
    best_name, best_corr = None, -np.inf
    for name, deg in INVESTOR_TYPES.items():
        c = corr_for(metrics_frame, holding, deg["q"], deg["n"])
        if not np.isnan(c) and c > best_corr:
            best_name, best_corr = name, c
    return best_name, best_corr


def main():
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print("=" * 60)
    output_dir = OUTPUT_ROOT / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_rq4_learned_params"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir  : {output_dir}")

    tickers = dp.get_current_sp500_tickers(DATA_DIR / "sp500_current.csv")
    prices = dp.download_adjusted_close(tickers, "1978-01-01", "2011-01-01",
                                        DATA_DIR / "yf_sp500_prices_1978_2011.parquet")
    bench = dp.download_adjusted_close([BENCHMARK_TICKER], "1978-01-01", "2011-01-01",
                                       DATA_DIR / "yf_benchmark_1978_2011.parquet")
    rf = dp.download_adjusted_close([RISKFREE_TICKER], "1978-01-01", "2011-01-01",
                                    DATA_DIR / "yf_riskfree_1978_2011.parquet")

    cache = {}
    for window in WINDOWS:
        cache[window["name"]] = window_metrics(window, prices, bench, rf)
        print(f"  cached metrics for {window['name']}: {len(cache[window['name']][0])} stocks")

    rows = []
    for train_name, test_name in PAIRS:
        train_frame, train_hold = cache[train_name]
        test_frame, test_hold = cache[test_name]

        learned_qn, train_corr = best_grid(train_frame, train_hold)
        test_corr_learned = corr_for(test_frame, test_hold, *learned_qn)
        preset_name, test_corr_preset = best_preset(test_frame, test_hold)
        _, test_ceiling = best_grid(test_frame, test_hold)

        rows.append({
            "train": train_name, "test": test_name,
            "learned_q": learned_qn[0], "learned_n": learned_qn[1],
            "train_corr_insample": round(train_corr, 3),
            "test_corr_learned_OOS": round(test_corr_learned, 3),
            "test_corr_best_preset": round(test_corr_preset, 3),
            "best_preset": preset_name,
            "test_corr_ceiling_insample": round(test_ceiling, 3),
            "generalization_gap": round(train_corr - test_corr_learned, 3),
        })

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "rq4_walk_forward.csv", index=False)
    with open(output_dir / "config.json", "w") as h:
        json.dump({"grid": GRID, "pairs": PAIRS, "objective": "maximize positive Spearman vs 1y holding"}, h, indent=2)

    print("\n" + "=" * 60)
    print("RQ4 — learned (q,n) generalization (walk-forward):")
    print(results.to_string(index=False))
    print("\nRead: if test_corr_learned_OOS << train_corr_insample (big gap) and not")
    print("above test_corr_best_preset, the learned params DON'T generalize —")
    print("consistent with curve-fitting rather than a learnable structural signal.")
    print("=" * 60)


if __name__ == "__main__":
    main()
