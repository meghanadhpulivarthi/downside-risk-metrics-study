"""
Ablations for the headline claim ("volatility is hard to beat OOS, survivorship-free").

A reviewer will ask whether "volatility wins" is an artifact of design choices. We
hold the trailing signals fixed (they don't depend on the forward outcome) and sweep:
  - forward HORIZON in {30, 90, 180} days  (drawdown-forecast Spearman on locked TEST)
  - blow-up THRESHOLD in {-50%, -70%, -90%} (blow-up ROC-AUC on locked TEST)
For each config we score volatility and the published metrics and report whether any
metric beats volatility on the locked test.

Run: uv run --project <repo> python -u src/autoresearch/run_ablations.py
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
import left_tail as lt
from run_published_baselines import hill_tail, down_semibeta, vn_ratio

TRAILING_DAYS = 180
CRYPTO_CSV = ev.PROJECT_ROOT / "data" / "kaggle_crypto" / "crypto-markets.csv"
OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"
HORIZONS = [30, 90, 180]
THRESHOLDS = [-0.5, -0.7, -0.9]
METRICS = ["VN_ratio", "left_tail_ES5", "VaR5", "hill_tail", "down_semibeta", "downside_dev", "volatility"]


def main():
    import datetime
    now = datetime.datetime.now()
    print("=" * 64)
    print(f"Ablations — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_ablations"
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = ev.load_panel()
    close, mcap = dp.load_crypto_panel(CRYPTO_CSV)
    market_ret = mcap.sum(axis=1, min_count=1).pct_change()

    # --- Trailing (horizon-independent) metrics per row, computed once ---
    recs = []
    for _, row in panel.iterrows():
        coin, fdate = row["coin"], pd.Timestamp(row["formation_date"])
        if coin not in close.columns:
            continue
        r = close.loc[fdate - pd.Timedelta(days=TRAILING_DAYS):fdate, coin].dropna().pct_change().dropna()
        if len(r) < 60:
            continue
        m = market_ret.reindex(r.index).fillna(0.0).values
        rv = r.values
        recs.append({
            "split": row["split"], "coin": coin, "formation_date": row["formation_date"],
            "VN_ratio": vn_ratio(rv, np.maximum(m, 0.0)),
            "left_tail_ES5": row["es5"], "VaR5": row["var5"],
            "hill_tail": hill_tail(rv), "down_semibeta": down_semibeta(rv, m),
            "downside_dev": row["downside_dev"], "volatility": row["vol"],
        })
    df = pd.DataFrame(recs)
    print(f"Rows: {len(df)}\n")

    def test_spearman(metric, dd_col):
        sub = df[df["split"] == "test"]
        rhos = []
        for _, g in sub.groupby("formation_date"):
            gg = g[[metric, dd_col]].dropna()
            if len(gg) >= ev.MIN_COINS_PER_DATE:
                rhos.append(spearmanr(gg[metric], gg[dd_col])[0])
        return float(np.mean(rhos)) if rhos else np.nan

    def test_auc(metric, label_col):
        sub = df[df["split"] == "test"][[metric, label_col]].dropna()
        if sub[label_col].nunique() < 2:
            return np.nan
        return float(roc_auc_score(sub[label_col], sub[metric]))

    results = {"horizon_drawdown": {}, "blowup_threshold": {}}

    # --- Horizon ablation: forward max drawdown at each horizon ---
    print("HORIZON ablation — TEST drawdown-forecast Spearman (best metric per horizon):")
    for h in HORIZONS:
        dd = []
        for _, row in df.iterrows():
            coin, fdate = row["coin"], pd.Timestamp(row["formation_date"])
            fwd = close.loc[fdate:fdate + pd.Timedelta(days=h), coin]
            dd.append(lt.forward_drawdown(fwd, fdate + pd.Timedelta(days=h)))
        df[f"dd_{h}"] = dd
        scores = {mm: test_spearman(mm, f"dd_{h}") for mm in METRICS}
        best = max(scores, key=lambda k: -1 if np.isnan(scores[k]) else scores[k])
        vol = scores["volatility"]
        beats = [mm for mm in METRICS if mm != "volatility" and scores[mm] > vol]
        results["horizon_drawdown"][h] = {k: round(v, 4) for k, v in scores.items()}
        print(f"  h={h:3d}d: best={best} ({scores[best]:.3f}); volatility={vol:.3f}; "
              f"beats-vol={beats if beats else 'NONE'}")

    # --- Blow-up threshold ablation: forward return <= threshold, TEST AUC ---
    print("\nBLOW-UP THRESHOLD ablation — TEST AUC (best metric per threshold):")
    from run_blowup_predictor import forward_return, FORWARD_DAYS
    fret = []
    for _, row in df.iterrows():
        coin, fdate = row["coin"], pd.Timestamp(row["formation_date"])
        fret.append(forward_return(close, coin, fdate, fdate + pd.Timedelta(days=FORWARD_DAYS)))
    df["fret"] = fret
    for thr in THRESHOLDS:
        df["blow"] = (df["fret"] <= thr).astype(float)
        df.loc[df["fret"].isna(), "blow"] = np.nan
        scores = {mm: test_auc(mm, "blow") for mm in METRICS}
        best = max(scores, key=lambda k: -1 if np.isnan(scores[k]) else scores[k])
        vol = scores["volatility"]
        beats = [mm for mm in METRICS if mm != "volatility" and not np.isnan(scores[mm]) and scores[mm] > vol]
        rate = float(df[df.split == "test"]["blow"].mean())
        results["blowup_threshold"][thr] = {k: round(v, 4) for k, v in scores.items()}
        print(f"  thr={thr:+.0%} (test rate {rate:.3f}): best={best} ({scores[best]:.3f}); "
              f"volatility={vol:.3f}; beats-vol={beats if beats else 'NONE'}")

    with open(output_dir / "ablations.json", "w") as h:
        json.dump(results, h, indent=2)
    print(f"\nSaved -> {output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
