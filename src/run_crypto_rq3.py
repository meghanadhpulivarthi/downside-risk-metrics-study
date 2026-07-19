"""
RQ3 (crypto clean-regime arm) — risk-forecasting vs return-forecasting.

The genuinely survivorship-free test the equity arm could not be: universe is the
top-N coins by market cap AS OF a past date (point-in-time), then tracked through
their actual outcomes — including the 2018 crash — because CoinMarketCap retained
crashed-but-listed coins (context/findings.md). Daily returns (crypto has no
monthly-frequency tradition and a short history).

H3: partial-moment (downside) metrics forecast future DOWNSIDE (drawdown) better
than variance, but do NOT forecast future RETURNS better than Sharpe — i.e. the
value is in risk forecasting, not return forecasting.

  Explanatory 2016-07..2017-06  ->  select universe at 2017-06-30
  Holding     2017-07..2018-06  (bull then crash: the downside stress test)

Run: uv run python -u src/run_crypto_rq3.py
"""

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import partial_moments as pm
import data_pipeline as dp

# ---------------------------------------------------------------- Config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"
CRYPTO_CSV   = DATA_DIR / "kaggle_crypto" / "crypto-markets.csv"

EXPL_START, EXPL_END = "2016-07-01", "2017-06-30"
HOLD_START, HOLD_END = "2017-07-01", "2018-06-30"
AS_OF        = "2017-06-30"     # point-in-time universe selection date
TOP_N        = 100
MIN_OBS      = 200              # daily obs in the explanatory window
RISK_NEUTRAL = pm.INVESTOR_DEGREES["risk_neutral"]  # q=n=1 for the ratio


def spearman_dropna(x, y):
    paired = pd.concat({"x": x, "y": y}, axis=1).dropna()
    if len(paired) < 10:
        return np.nan, np.nan, len(paired)
    rho, p = spearmanr(paired["x"], paired["y"])
    return float(rho), float(p), int(len(paired))


def paired_bootstrap_diff(pred_a, pred_b, target, n_boot=2000):
    """
    Bootstrap 95% CI for [spearman(pred_a,target) - spearman(pred_b,target)],
    resampling coins PAIRED (same draw for both predictors, since they share the
    target vector). Fixed seed for reproducibility. Reviewer-requested (RQ3 CI must
    be committed, not computed ad hoc).
    """
    frame = pd.concat({"a": pred_a, "b": pred_b, "t": target}, axis=1).dropna()
    if len(frame) < 10:
        return np.nan, np.nan, np.nan
    a, b, t = frame["a"].values, frame["b"].values, frame["t"].values
    point = spearmanr(a, t)[0] - spearmanr(b, t)[0]
    rng = np.random.default_rng(42)
    n = len(frame)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = spearmanr(a[idx], t[idx])[0] - spearmanr(b[idx], t[idx])[0]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def decile_long_short(predictor, outcome, k=10):
    """Top-decile-minus-bottom-decile mean outcome — a minimal economic-significance
    panel (rank correlation is not tradable value; this checks the extremes separate)."""
    frame = pd.concat({"p": predictor, "o": outcome}, axis=1).dropna()
    if len(frame) < 2 * k:
        return np.nan
    ranked = frame.sort_values("p")
    bottom = ranked.head(len(frame) // k)["o"].mean()
    top = ranked.tail(len(frame) // k)["o"].mean()
    return float(top - bottom)


def main():
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Explanatory {EXPL_START}..{EXPL_END} -> holding {HOLD_START}..{HOLD_END}")
    print("=" * 60)
    output_dir = OUTPUT_ROOT / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_crypto_rq3"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir  : {output_dir}")

    close, market_cap = dp.load_crypto_panel(CRYPTO_CSV)
    print(f"Loaded crypto panel: {close.shape[1]} coins, {close.index.min().date()} -> {close.index.max().date()}")

    universe = dp.point_in_time_top_n(market_cap, AS_OF, TOP_N)
    print(f"Point-in-time universe at {AS_OF}: top {len(universe)} by market cap")

    # Benchmark = cap-weighted market daily return over the FULL universe panel;
    # risk-free = 0 (crypto has no risk-free). Double benchmark target = max(mkt, 0).
    expl_returns = dp.to_daily_returns(close.loc[EXPL_START:EXPL_END])
    # Cap-weighted market index return = pct-change of total market cap.
    mcap_total = market_cap.loc[EXPL_START:EXPL_END].sum(axis=1, min_count=1)
    mkt_return = mcap_total.pct_change().iloc[1:]

    records = {}
    skipped = 0
    for coin in universe:
        if coin not in expl_returns.columns:
            skipped += 1
            continue
        aligned = pd.concat({"r": expl_returns[coin], "mkt": mkt_return}, axis=1).dropna()
        if len(aligned) < MIN_OBS:
            skipped += 1
            continue
        returns = aligned["r"].values
        target = np.maximum(aligned["mkt"].values, 0.0)  # rf=0 crypto double benchmark
        try:
            ratio = pm.explanatory_metric(returns, target, q=RISK_NEUTRAL["q"], n=RISK_NEUTRAL["n"])
        except ZeroDivisionError:
            skipped += 1
            continue
        records[coin] = {
            "n_obs": len(aligned),
            # downside predictors (explanatory window)
            "lpm2_downside": pm.lpm(returns, target, n=2),   # partial-moment downside risk
            "variance": float(np.var(returns)),              # symmetric risk (the foil)
            # return predictor
            "upm_lpm_ratio": ratio,
            "sharpe": float(np.mean(returns) / np.std(returns)),
        }
    metrics_frame = pd.DataFrame.from_dict(records, orient="index")
    print(f"Metrics for {len(metrics_frame)} coins (skipped {skipped})")

    # --- Holding-window outcomes: future return and future DOWNSIDE (max drawdown)
    hold_close = close.loc[HOLD_START:HOLD_END]
    hold_returns = dp.to_daily_returns(hold_close)
    holding_return = hold_returns.mean(axis=0, skipna=True).reindex(metrics_frame.index)
    holding_drawdown = pd.Series(
        {coin: dp.max_drawdown(hold_close[coin]) for coin in metrics_frame.index if coin in hold_close.columns}
    ).reindex(metrics_frame.index)

    # --- The RQ3 comparisons
    results = []
    # DOWNSIDE forecasting: does explanatory downside-LPM beat variance at predicting holding drawdown?
    for predictor in ["lpm2_downside", "variance"]:
        rho, p, n = spearman_dropna(metrics_frame[predictor], holding_drawdown)
        results.append({"question": "downside_forecast", "predictor": predictor,
                        "target": "holding_max_drawdown", "spearman_rho": rho, "p_value": p, "n": n})
    # RETURN forecasting: does explanatory UPM/LPM ratio beat Sharpe at predicting holding return?
    for predictor in ["upm_lpm_ratio", "sharpe"]:
        rho, p, n = spearman_dropna(metrics_frame[predictor], holding_return)
        results.append({"question": "return_forecast", "predictor": predictor,
                        "target": "holding_mean_return", "spearman_rho": rho, "p_value": p, "n": n})
    results = pd.DataFrame(results)

    # --- Paired bootstrap CI for the downside-forecast difference (LPM2 vs variance).
    dd_point, dd_lo, dd_hi = paired_bootstrap_diff(
        metrics_frame["lpm2_downside"], metrics_frame["variance"], holding_drawdown)
    # --- Minimal economic-significance panel: decile long/short spreads.
    ls_return = decile_long_short(metrics_frame["upm_lpm_ratio"], holding_return)
    ls_drawdown = decile_long_short(metrics_frame["lpm2_downside"], holding_drawdown)

    metrics_frame.to_csv(output_dir / "per_coin_metrics.csv")
    results.to_csv(output_dir / "rq3_forecasting.csv", index=False)
    bootstrap_summary = {
        "downside_diff_lpm2_minus_variance": dd_point,
        "downside_diff_ci95": [dd_lo, dd_hi],
        "downside_diff_ci_includes_zero": bool(dd_lo <= 0 <= dd_hi),
        "decile_ls_ratio_vs_return": ls_return,
        "decile_ls_lpm2_vs_drawdown": ls_drawdown,
    }
    with open(output_dir / "config.json", "w") as h:
        json.dump({"expl": [EXPL_START, EXPL_END], "hold": [HOLD_START, HOLD_END],
                   "as_of": AS_OF, "top_n": TOP_N, "min_obs": MIN_OBS,
                   "universe_size": len(metrics_frame), **bootstrap_summary}, h, indent=2)
    with open(output_dir / "bootstrap_and_economic.json", "w") as h:
        json.dump(bootstrap_summary, h, indent=2)

    print("\n" + "=" * 60)
    print("RQ3 (crypto, survivorship-free universe) — forecasting rank correlations:")
    print(results.round(3).to_string(index=False))
    print(f"\nDownside-forecast difference (LPM2 - variance vs drawdown): {dd_point:+.3f} "
          f"95% CI [{dd_lo:+.3f}, {dd_hi:+.3f}]  (includes 0: {dd_lo <= 0 <= dd_hi})")
    print(f"Economic panel (decile long/short): ratio->return spread = {ls_return:+.4f}/day, "
          f"LPM2->drawdown spread = {ls_drawdown:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
