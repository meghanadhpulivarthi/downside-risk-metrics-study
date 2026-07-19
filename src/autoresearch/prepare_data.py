"""
FROZEN data prep for the autoresearch metric-discovery loop (Karpathy-style).

Analogous to autoresearch/prepare.py: builds the fixed dataset ONCE and never
changes during the search. Produces a long feature panel (one row per
formation-date × coin) on the survivorship-free crypto testbed, with a forward
drawdown label and a TIME-BASED train/val/test split. The locked TEST split is
what protects the whole exercise from the loop's own data-snooping.

Splits (time-based, non-overlapping formation dates):
  TRAIN 2016-01..2017-03  — fit any within-metric parameters (e.g. death hazard)
  VAL   2017-04..2018-01  — the loop optimizes the metric here
  TEST  2018-02..2018-08  — LOCKED; touched exactly once, at the very end

Run: uv run python -u src/autoresearch/prepare_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_pipeline as dp
import left_tail as lt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CRYPTO_CSV = DATA_DIR / "kaggle_crypto" / "crypto-markets.csv"
OUT_PANEL = DATA_DIR / "autoresearch_feature_panel.parquet"

TOP_N = 100
TRAILING_DAYS = 180
FORWARD_DAYS = 90
FORMATION_DATES = pd.date_range("2016-01-31", "2018-08-31", freq="ME")

SPLITS = {  # (inclusive start, inclusive end) on formation date
    "train": ("2016-01-01", "2017-03-31"),
    "val":   ("2017-04-01", "2018-01-31"),
    "test":  ("2018-02-01", "2018-08-31"),
}


def assign_split(formation_date):
    for name, (start, end) in SPLITS.items():
        if pd.Timestamp(start) <= formation_date <= pd.Timestamp(end):
            return name
    return None


def coin_features(returns, prices_trailing, mcap_value, age_days):
    """All trailing features for one coin at one formation date."""
    r = returns.dropna().values
    if len(r) < 60:
        return None
    es5 = lt.expected_shortfall(r, q=0.05)
    if np.isnan(es5):
        return None
    downside = r[r < 0]
    return {
        "vol": float(np.std(r)),
        "es5": es5,
        "es10": lt.expected_shortfall(r, q=0.10),
        "var5": lt.value_at_risk(r, q=0.05),
        "lpm2": float(np.mean(np.maximum(0.0, -r) ** 2)),        # semivariance-like
        "downside_dev": float(np.sqrt(np.mean(downside ** 2))) if len(downside) else 0.0,
        "trailing_dd": dp.max_drawdown(prices_trailing),
        "cum_ret": float(prices_trailing.dropna().iloc[-1] / prices_trailing.dropna().iloc[0] - 1),
        "skew": float(skew(r)) if len(r) > 2 else 0.0,
        "kurt": float(kurtosis(r)) if len(r) > 3 else 0.0,
        "mean_ret": float(np.mean(r)),
        "log_mcap": float(np.log(mcap_value)) if mcap_value and mcap_value > 0 else np.nan,
        "age_days": float(age_days),
    }


def main():
    print("Loading crypto panel ...")
    close, market_cap = dp.load_crypto_panel(CRYPTO_CSV)
    first_obs = close.apply(lambda c: c.dropna().index.min())

    rows = []
    for formation_date in FORMATION_DATES:
        split = assign_split(formation_date)
        if split is None:
            continue
        trailing_start = formation_date - pd.Timedelta(days=TRAILING_DAYS)
        forward_end = formation_date + pd.Timedelta(days=FORWARD_DAYS)
        # DEATH-AWARE point-in-time universe (includes coins that later crater).
        try:
            universe = dp.point_in_time_top_n(market_cap.loc[:formation_date], formation_date, TOP_N)
        except ValueError:
            continue
        trailing_returns = dp.to_daily_returns(close.loc[trailing_start:formation_date])
        n_added = 0
        for coin in universe:
            if coin not in trailing_returns.columns or coin not in close.columns:
                continue
            prices_trailing = close.loc[trailing_start:formation_date][coin]
            mcap_value = market_cap.loc[:formation_date][coin].dropna()
            mcap_value = mcap_value.iloc[-1] if len(mcap_value) else np.nan
            age = (formation_date - first_obs[coin]).days if pd.notna(first_obs[coin]) else np.nan
            feats = coin_features(trailing_returns[coin], prices_trailing, mcap_value, age)
            if feats is None:
                continue
            forward_prices = close.loc[formation_date:forward_end][coin]
            fwd_dd = lt.forward_drawdown(forward_prices, forward_end)
            if np.isnan(fwd_dd):
                continue
            feats.update({
                "formation_date": formation_date.date().isoformat(),
                "split": split, "coin": coin,
                "fwd_drawdown": fwd_dd,
                "died": lt.died_in_window(forward_prices, forward_end),
            })
            rows.append(feats)
            n_added += 1
        print(f"  {formation_date.date()} [{split}]: {n_added} coins")

    panel = pd.DataFrame(rows)
    panel.to_parquet(OUT_PANEL)
    print(f"\nSaved feature panel: {len(panel)} rows -> {OUT_PANEL}")
    print("Rows by split:")
    print(panel.groupby("split").size().to_string())
    print(f"Overall death rate: {panel['died'].mean():.3f}")
    print(f"Features: {[c for c in panel.columns if c not in ('formation_date','split','coin','fwd_drawdown','died')]}")


if __name__ == "__main__":
    main()
