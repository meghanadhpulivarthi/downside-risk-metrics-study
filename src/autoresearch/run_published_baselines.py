"""
Published-metric baselines on the survivorship-free locked test.

Reviewer-critical: benchmark the ACTUAL published downside/tail-risk metrics — not
just volatility — on the same locked crypto test, for two tasks:
  (a) forecasting forward 90d max drawdown (per-date Spearman), and
  (b) forecasting blow-up (fwd 90d return <= -80%) (ROC-AUC).

Published metrics computed per coin over the trailing window:
  - VN_ratio   : Viole-Nawrocki UPM/LPM explanatory ratio (q=n=1) vs a double
                 benchmark max(crypto-market return, 0). LOWER ratio = more downside.
  - left_tail_ES5 / VaR5 : Atilgan-Bali-Demirtas-Gunaydin (2020) left-tail risk.
  - hill_tail  : Kelly-Jiang (2014) style Hill tail exponent of the loss tail.
  - down_semibeta : Bollerslev-Patton-Quaedvlieg (2022) style negative-negative
                 semibeta vs the crypto market.
  - volatility, downside_dev : simple baselines.

Run: uv run --project <repo> python -u src/autoresearch/run_published_baselines.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import evaluate as ev
import data_pipeline as dp
import partial_moments as pm
from run_blowup_predictor import forward_return, FORWARD_DAYS, BLOWUP_THRESHOLD

TRAILING_DAYS = 180
CRYPTO_CSV = ev.PROJECT_ROOT / "data" / "kaggle_crypto" / "crypto-markets.csv"
OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"


def hill_tail(returns, q=0.05):
    """Kelly-Jiang-style Hill estimator on the loss tail (higher = fatter tail)."""
    losses = -returns[returns < 0]
    if len(losses) < 15:
        return np.nan
    u = np.quantile(losses, 1 - q)
    exceed = losses[losses >= u]
    if len(exceed) < 5 or u <= 0:
        return np.nan
    return float(np.mean(np.log(exceed / u)))


def down_semibeta(r, m):
    """BPQ-style negative-negative semibeta vs market m."""
    both_down = (r < 0) & (m < 0)
    denom = np.sum(m[m < 0] ** 2)
    if denom <= 0 or both_down.sum() < 5:
        return np.nan
    return float(np.sum(r[both_down] * m[both_down]) / denom)


def vn_ratio(r, target):
    """Viole-Nawrocki explanatory UPM/LPM ratio; return -ratio so higher = more downside."""
    try:
        return -pm.explanatory_metric(r, target, q=1, n=1)
    except ZeroDivisionError:
        return np.nan


def main():
    import datetime
    now = datetime.datetime.now()
    print("=" * 64)
    print(f"Published-metric baselines — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_published_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = ev.load_panel()
    close, mcap = dp.load_crypto_panel(CRYPTO_CSV)
    market_ret = mcap.sum(axis=1, min_count=1).pct_change()

    rows = []
    for _, row in panel.iterrows():
        coin, fdate = row["coin"], pd.Timestamp(row["formation_date"])
        ts = fdate - pd.Timedelta(days=TRAILING_DAYS)
        if coin not in close.columns:
            continue
        r = close.loc[ts:fdate, coin].dropna().pct_change().dropna()
        if len(r) < 60:
            continue
        m = market_ret.reindex(r.index).fillna(0.0).values
        rv = r.values
        target = np.maximum(m, 0.0)  # double benchmark, rf=0
        blowret = forward_return(close, coin, fdate, fdate + pd.Timedelta(days=FORWARD_DAYS))
        rows.append({
            "split": row["split"], "fwd_drawdown": row["fwd_drawdown"],
            "blowup": np.nan if np.isnan(blowret) else float(blowret <= BLOWUP_THRESHOLD),
            "VN_ratio": vn_ratio(rv, target),
            "left_tail_ES5": row["es5"], "VaR5": row["var5"],
            "hill_tail": hill_tail(rv), "down_semibeta": down_semibeta(rv, m),
            "volatility": row["vol"], "downside_dev": row["downside_dev"],
        })
    df = pd.DataFrame(rows)
    print(f"Rows with published metrics: {len(df)}")

    metrics = ["VN_ratio", "left_tail_ES5", "VaR5", "hill_tail", "down_semibeta",
               "downside_dev", "volatility"]

    # Score per-date Spearman using the panel's formation_date grouping.
    df["formation_date"] = panel.loc[df.index, "formation_date"].values

    def score_spearman(metric, split):
        sub = df[df["split"] == split]
        rhos = []
        for _, g in sub.groupby("formation_date"):
            gg = g[[metric, "fwd_drawdown"]].dropna()
            if len(gg) >= ev.MIN_COINS_PER_DATE:
                rhos.append(spearmanr(gg[metric], gg["fwd_drawdown"])[0])
        return float(np.mean(rhos)) if rhos else np.nan

    def score_auc(metric, split):
        sub = df[df["split"] == split][[metric, "blowup"]].dropna()
        if sub["blowup"].nunique() < 2:
            return np.nan
        return float(roc_auc_score(sub["blowup"], sub[metric]))

    print("\nHead-to-head on the LOCKED TEST (higher = better forecaster):")
    print(f"{'metric':16s} {'VAL rho':>8s} {'TEST rho':>9s} {'TEST AUC':>9s}")
    result = {}
    for metric in metrics:
        vr, tr, ta = score_spearman(metric, "val"), score_spearman(metric, "test"), score_auc(metric, "test")
        result[metric] = {"val_spearman": round(vr, 4), "test_spearman": round(tr, 4), "test_blowup_auc": round(ta, 4)}
        star = "  <- baseline" if metric == "volatility" else ""
        print(f"{metric:16s} {vr:8.3f} {tr:9.3f} {ta:9.3f}{star}")

    vol_test = result["volatility"]["test_spearman"]
    beats = [m for m in metrics if m != "volatility" and result[m]["test_spearman"] > vol_test]
    print(f"\nPublished/other metrics beating volatility on TEST drawdown-forecast: "
          f"{beats if beats else 'NONE'}")
    with open(output_dir / "published_baselines.json", "w") as h:
        json.dump(result, h, indent=2)
    print(f"Saved -> {output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
