"""
Death-aware left-tail risk on the survivorship-free crypto testbed — SOLIDIFIED.

Thesis: survivorship selection MASKS the incremental value of left-tail risk.
Under naive survivor-selected evaluation, ES5 adds nothing beyond volatility for
forecasting future drawdown; under death-aware point-in-time evaluation, it does.

This version hardens the result:
  - MONTHLY formation dates (more observations).
  - PAIRED test of the design DIFFERENCE (death-aware minus naive), on common dates.
  - Newey-West (HAC) t-stats — forward windows overlap, inducing autocorrelation.
  - Moving-block bootstrap CI on the mean design-difference.
  - Robustness grid over trailing/forward horizons and the tail cutoff q; VaR5 too.

Designs share identical formation dates:
  NAIVE       universe = top-N by market cap as of a recent well-covered date (survivor pick).
  DEATH-AWARE universe = top-N by market cap AS OF the formation date (point-in-time),
              forward outcomes score in-window deaths at terminal -100%.

Run: uv run python -u src/run_death_aware_crypto.py
"""

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import data_pipeline as dp
import left_tail as lt

# ---------------------------------------------------------------- Config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"
CRYPTO_CSV   = DATA_DIR / "kaggle_crypto" / "crypto-markets.csv"

TOP_N = 100
FORMATION_DATES = pd.date_range("2016-01-31", "2018-08-31", freq="ME")  # monthly
BASE = {"trailing": 180, "forward": 90, "q": 0.05}
# Robustness grid (focused).
GRID = [{"trailing": t, "forward": f, "q": q}
        for t in (120, 180, 252) for f in (90, 180) for q in (0.05, 0.10)]


def partial_spearman(frame, a, b, c):
    """Partial Spearman corr of a,b controlling for c."""
    sub = frame[[a, b, c]].dropna()
    if len(sub) < 20:
        return np.nan
    rab, rac, rbc = (spearmanr(sub[a], sub[b])[0], spearmanr(sub[a], sub[c])[0],
                     spearmanr(sub[b], sub[c])[0])
    denom = np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))
    return float((rab - rac * rbc) / denom) if denom > 0 else np.nan


def cross_section(close, market_cap, formation_date, as_of, trailing, forward, q):
    """One formation-date cross-section for the top-N universe selected as of `as_of`."""
    trailing_start = formation_date - pd.Timedelta(days=trailing)
    forward_end = formation_date + pd.Timedelta(days=forward)
    try:
        universe = dp.point_in_time_top_n(market_cap.loc[:as_of], as_of, TOP_N)
    except ValueError:
        return None
    trailing_returns = dp.to_daily_returns(close.loc[trailing_start:formation_date])
    rows = {}
    for coin in universe:
        if coin not in trailing_returns.columns or coin not in close.columns:
            continue
        r = trailing_returns[coin].dropna().values
        es = lt.expected_shortfall(r, q=q)
        if np.isnan(es):
            continue
        forward_prices = close.loc[formation_date:forward_end][coin]
        drawdown = lt.forward_drawdown(forward_prices, forward_end)
        if np.isnan(drawdown):
            continue
        rows[coin] = {"es5": es, "var5": lt.value_at_risk(r, q=q),
                      "vol": lt.realized_volatility(r), "fwd_drawdown": drawdown,
                      "died": lt.died_in_window(forward_prices, forward_end)}
    if len(rows) < 20:
        return None
    return pd.DataFrame.from_dict(rows, orient="index")


def date_row(frame):
    """Per-date statistics for one design's cross-section."""
    return {
        "n": len(frame),
        "mean_dd": float(frame["fwd_drawdown"].mean()),
        "es5_beyond_vol": partial_spearman(frame, "es5", "fwd_drawdown", "vol"),
        "var5_beyond_vol": partial_spearman(frame, "var5", "fwd_drawdown", "vol"),
        "vol_beyond_es5": partial_spearman(frame, "vol", "fwd_drawdown", "es5"),
    }


def newey_west_t(series, lag):
    """HAC t-stat for the mean of an (autocorrelated) series."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    var = np.mean(e * e)
    for l in range(1, min(lag, n - 1) + 1):
        weight = 1 - l / (lag + 1)
        var += 2 * weight * np.mean(e[l:] * e[:-l])
    se = np.sqrt(var / n)
    return float(x.mean() / se) if se > 0 else np.nan


def block_bootstrap_ci(series, block, n_boot=2000):
    """Moving-block bootstrap 95% CI for the mean of an autocorrelated series."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    starts_max = n - block + 1
    for i in range(n_boot):
        starts = rng.integers(0, starts_max, n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        means[i] = sample.mean()
    return tuple(np.percentile(means, [2.5, 97.5]))


def run_config(close, market_cap, naive_as_of, cfg):
    """Per-date paired stats for both designs on COMMON formation dates."""
    records = []
    for formation_date in FORMATION_DATES:
        naive = cross_section(close, market_cap, formation_date, naive_as_of,
                              cfg["trailing"], cfg["forward"], cfg["q"])
        death = cross_section(close, market_cap, formation_date, formation_date,
                              cfg["trailing"], cfg["forward"], cfg["q"])
        if naive is None or death is None:
            continue  # require BOTH valid so the difference is paired
        n_row, d_row = date_row(naive), date_row(death)
        records.append({
            "formation_date": formation_date.date().isoformat(),
            "es5bv_naive": n_row["es5_beyond_vol"], "es5bv_death": d_row["es5_beyond_vol"],
            "var5bv_naive": n_row["var5_beyond_vol"], "var5bv_death": d_row["var5_beyond_vol"],
            "meandd_naive": n_row["mean_dd"], "meandd_death": d_row["mean_dd"],
        })
    perdate = pd.DataFrame(records)
    if len(perdate) > 0:
        perdate["es5bv_diff"] = perdate["es5bv_death"] - perdate["es5bv_naive"]
        perdate["meandd_diff"] = perdate["meandd_death"] - perdate["meandd_naive"]
    return perdate


def summarize(perdate, cfg):
    """Fama-MacBeth means with Newey-West t-stats; the paired design difference is the headline."""
    lag = int(np.ceil(cfg["forward"] / 30)) + 1  # HAC lag ~ forward-window overlap in months
    out = {"trailing": cfg["trailing"], "forward": cfg["forward"], "q": cfg["q"],
           "n_dates": len(perdate)}
    if len(perdate) == 0:
        return out
    for col in ["es5bv_naive", "es5bv_death", "es5bv_diff", "var5bv_death",
                "meandd_naive", "meandd_death", "meandd_diff"]:
        out[col] = round(float(perdate[col].mean()), 4)
        out[col + "_t"] = round(newey_west_t(perdate[col].values, lag), 2)
    lo, hi = block_bootstrap_ci(perdate["es5bv_diff"].values, block=max(2, lag))
    out["es5bv_diff_ci95"] = [round(lo, 4), round(hi, 4)]
    return out


def main():
    now = datetime.datetime.now()
    print("=" * 64)
    print(f"Death-aware left-tail risk (crypto) — SOLIDIFIED — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)
    output_dir = OUTPUT_ROOT / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_death_aware_crypto_solid"
    output_dir.mkdir(parents=True, exist_ok=True)

    close, market_cap = dp.load_crypto_panel(CRYPTO_CSV)
    coverage = market_cap.notna().sum(axis=1)
    naive_as_of = coverage[coverage >= 500].index.max()
    print(f"Panel {close.shape[1]} coins {close.index.min().date()}->{close.index.max().date()}; "
          f"naive as-of {naive_as_of.date()}; {len(FORMATION_DATES)} monthly formation dates")

    # --- Base config: full detail
    perdate = run_config(close, market_cap, naive_as_of, BASE)
    perdate.to_csv(output_dir / "per_date_base.csv", index=False)
    base = summarize(perdate, BASE)
    print(f"\nBASE config {BASE}  ({base['n_dates']} paired dates; Newey-West t):")
    print(f"  ES5 beyond vol  — naive : {base['es5bv_naive']}  (t={base['es5bv_naive_t']})")
    print(f"  ES5 beyond vol  — death : {base['es5bv_death']}  (t={base['es5bv_death_t']})")
    print(f"  DESIGN DIFFERENCE (death-naive): {base['es5bv_diff']}  (t={base['es5bv_diff_t']}, "
          f"95% CI {base['es5bv_diff_ci95']})   <-- headline")
    print(f"  VaR5 beyond vol — death : {base['var5bv_death']}  (t={base['var5bv_death_t']})")
    print(f"  mean drawdown   — naive {base['meandd_naive']} vs death {base['meandd_death']} "
          f"(diff {base['meandd_diff']}, t={base['meandd_diff_t']})")

    # --- Robustness grid
    grid_rows = []
    for cfg in GRID:
        grid_rows.append(summarize(run_config(close, market_cap, naive_as_of, cfg), cfg))
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(output_dir / "robustness_grid.csv", index=False)
    with open(output_dir / "base_summary.json", "w") as h:
        json.dump(base, h, indent=2)

    print("\nROBUSTNESS GRID (death ES5-beyond-vol, and design difference; Newey-West t):")
    show = grid[["trailing", "forward", "q", "n_dates", "es5bv_death", "es5bv_death_t",
                 "es5bv_diff", "es5bv_diff_t"]]
    print(show.to_string(index=False))
    pos = int(((grid["es5bv_death_t"] > 1.96) & (grid["es5bv_diff"] > 0)).sum())
    print(f"\nConfigs with death-aware ES5-beyond-vol significant AND positive difference: "
          f"{pos}/{len(grid)}")
    print(f"Saved -> {output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
