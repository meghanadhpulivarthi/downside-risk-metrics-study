"""
FROZEN evaluator for the autoresearch loop. The search may NOT edit this file.

A "metric" here is either a single trailing feature or a linear model over a
feature subset (fit on TRAIN only). Its score on a split = mean, across that
split's formation dates, of the cross-sectional Spearman rank correlation between
the metric and realized forward drawdown (higher = better downside forecaster).

The TEST split is only ever scored via `final_test_eval`, which the loop driver
calls exactly once after the metric is frozen. Because TEST is locked, its score
is an honest out-of-sample estimate no matter how hard VAL was searched.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PANEL_PATH = PROJECT_ROOT / "data" / "autoresearch_feature_panel.parquet"

ALL_FEATURES = ["vol", "es5", "es10", "var5", "lpm2", "downside_dev", "trailing_dd",
                "cum_ret", "skew", "kurt", "mean_ret", "log_mcap", "age_days"]
MIN_COINS_PER_DATE = 20


def load_panel():
    return pd.read_parquet(PANEL_PATH)


def _per_date_spearman(score_values, split_df):
    frame = split_df.assign(_score=score_values)
    rhos = []
    for _, group in frame.groupby("formation_date"):
        sub = group[["_score", "fwd_drawdown"]].dropna()
        if len(sub) >= MIN_COINS_PER_DATE:
            rhos.append(spearmanr(sub["_score"], sub["fwd_drawdown"])[0])
    return np.array(rhos)


def newey_west_t(series, lag=3):
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


def score_single_feature(panel, feature, split):
    """Score a single trailing feature (no fitting) on a split."""
    split_df = panel[panel["split"] == split]
    return _per_date_spearman(split_df[feature].values, split_df)


def _fit_linear(panel, features):
    train = panel[panel["split"] == "train"].dropna(subset=features + ["fwd_drawdown"])
    scaler = StandardScaler().fit(train[features].values)
    model = LinearRegression().fit(scaler.transform(train[features].values), train["fwd_drawdown"].values)
    return model, scaler


def score_model(panel, features, split):
    """Fit a linear metric on TRAIN over `features`; score it on `split`."""
    model, scaler = _fit_linear(panel, features)
    split_df = panel[panel["split"] == split].dropna(subset=features)
    preds = model.predict(scaler.transform(split_df[features].values))
    return _per_date_spearman(preds, split_df), (model, scaler)


def death_adjusted_es(panel):
    """
    The hand-designed DES baseline: DES = p*(1) + (1-p)*es5, where p is a death
    hazard (logistic on features) fit on TRAIN. Added as a column on the panel.
    """
    from sklearn.linear_model import LogisticRegression
    feats = ["trailing_dd", "vol", "es5", "cum_ret", "log_mcap", "age_days"]
    train = panel.dropna(subset=feats)
    train_rows = train[train["split"] == "train"]
    scaler = StandardScaler().fit(train_rows[feats].values)
    hazard = LogisticRegression(class_weight="balanced", max_iter=1000)
    hazard.fit(scaler.transform(train_rows[feats].values), train_rows["died"].astype(int).values)
    p = np.full(len(panel), np.nan)
    ok = panel[feats].notna().all(axis=1).values
    p[ok] = hazard.predict_proba(scaler.transform(panel.loc[ok, feats].values))[:, 1]
    return p * 1.0 + (1 - p) * panel["es5"].values


def baseline_scores(panel):
    """VAL/TEST mean rank-corr for the key baselines the loop must beat."""
    panel = panel.copy()
    panel["des"] = death_adjusted_es(panel)
    out = {}
    for name, col in [("volatility", "vol"), ("ES5", "es5"), ("DES", "des")]:
        val = score_single_feature(panel, col, "val")
        test = score_single_feature(panel, col, "test")
        out[name] = {"val_mean": float(np.mean(val)), "val_t": newey_west_t(val),
                     "test_mean": float(np.mean(test)), "test_t": newey_west_t(test)}
    return out


if __name__ == "__main__":
    panel = load_panel()
    print("Baseline downside-forecast scores (mean per-date Spearman; Newey-West t):")
    for name, s in baseline_scores(panel).items():
        print(f"  {name:11s}  VAL {s['val_mean']:.3f} (t={s['val_t']:.2f})   "
              f"TEST {s['test_mean']:.3f} (t={s['test_t']:.2f})")
