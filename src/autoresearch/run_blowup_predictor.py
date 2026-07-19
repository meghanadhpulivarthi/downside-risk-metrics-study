"""
Blow-up / death predictor — the honest useful metric.

Forecasting marginal drawdown-rank beyond volatility barely works (see the sweep).
But predicting which assets CRATER is a task with real signal and real utility. We
build a survivorship-free blow-up risk score and hold it to the same discipline:
fit on TRAIN, select the model on VAL, evaluate ONCE on the LOCKED TEST.

Label: blow_up = forward 90d return <= -80% (delisting scored as -100%).
Features: the 13 trailing signals in the panel.
Success = the multi-feature model beats VOLATILITY-ALONE at ranking blow-ups on the
locked test (ROC-AUC, average precision, and top-decile lift).

Run: uv run --project <repo> python -u src/autoresearch/run_blowup_predictor.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

import evaluate as ev
import data_pipeline as dp

FORWARD_DAYS = 90
BLOWUP_THRESHOLD = -0.80          # >= 80% loss over the forward window = blow-up
DEATH_GAP_DAYS = 14
CRYPTO_CSV = ev.PROJECT_ROOT / "data" / "kaggle_crypto" / "crypto-markets.csv"
OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"


def forward_return(close, coin, formation_date, forward_end):
    """Formation->horizon return; delisting mid-window scored as -100%."""
    window = close.loc[formation_date:forward_end][coin].dropna() if coin in close.columns else pd.Series(dtype=float)
    if len(window) < 2:
        return np.nan
    last_obs = window.index.max()
    if last_obs < pd.Timestamp(forward_end) - pd.Timedelta(days=DEATH_GAP_DAYS):
        return -1.0  # delisted / stopped trading -> total loss
    return float(window.loc[last_obs] / window.iloc[0] - 1.0)


def build_labels(panel):
    close, _ = dp.load_crypto_panel(CRYPTO_CSV)
    labels = np.full(len(panel), np.nan)
    for i, (_, row) in enumerate(panel.iterrows()):
        fdate = pd.Timestamp(row["formation_date"])
        fret = forward_return(close, row["coin"], fdate, fdate + pd.Timedelta(days=FORWARD_DAYS))
        if not np.isnan(fret):
            labels[i] = float(fret <= BLOWUP_THRESHOLD)
    return labels


def top_decile_lift(y_true, scores, frac=0.10):
    """Lift: blow-up rate among the top-`frac` riskiest vs the base rate."""
    n = len(y_true)
    k = max(1, int(n * frac))
    order = np.argsort(-scores)
    top_rate = y_true[order[:k]].mean()
    base = y_true.mean()
    return float(top_rate / base) if base > 0 else np.nan, float(top_rate), float(base)


def evaluate_scores(y, scores, label):
    auc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)
    lift, top_rate, base = top_decile_lift(y, scores)
    print(f"  {label:22s} AUC {auc:.3f}  AP {ap:.3f}  top-decile blow-up rate {top_rate:.2f} "
          f"(base {base:.2f}, lift {lift:.1f}x)")
    return {"auc": round(auc, 4), "ap": round(ap, 4), "top_decile_lift": round(lift, 2)}


def main():
    import datetime
    now = datetime.datetime.now()
    print("=" * 64)
    print(f"Blow-up / death predictor — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_blowup_predictor"
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = ev.load_panel()
    print("Building blow-up labels (forward 90d return <= -80%) ...")
    panel = panel.copy()
    panel["blowup"] = build_labels(panel)
    panel = panel.dropna(subset=["blowup"] + ev.ALL_FEATURES)
    for split in ["train", "val", "test"]:
        s = panel[panel["split"] == split]
        print(f"  {split}: {len(s)} rows, blow-up rate {s['blowup'].mean():.3f}")

    feats = ev.ALL_FEATURES
    tr = panel[panel["split"] == "train"]
    va = panel[panel["split"] == "val"]
    te = panel[panel["split"] == "test"]
    scaler = StandardScaler().fit(tr[feats].values)
    Xtr, Xva, Xte = (scaler.transform(d[feats].values) for d in (tr, va, te))
    ytr, yva, yte = (d["blowup"].values for d in (tr, va, te))

    # Candidate models fit on TRAIN; select by VAL AUC.
    models = {
        "logistic": LogisticRegression(class_weight="balanced", max_iter=2000),
        "grad_boost": GradientBoostingClassifier(random_state=0),
    }
    val_auc = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        val_auc[name] = roc_auc_score(yva, model.predict_proba(Xva)[:, 1])
    chosen = max(val_auc, key=val_auc.get)
    print(f"\nModel selection by VAL AUC: {val_auc} -> chosen: {chosen}")

    # Baselines: single trailing features as risk scores (higher = riskier).
    print("\nVALIDATION (model selection):")
    for name in models:
        evaluate_scores(yva, models[name].predict_proba(Xva)[:, 1], f"model:{name}")
    for feat in ["vol", "trailing_dd"]:
        evaluate_scores(yva, va[feat].values, f"baseline:{feat}")

    print("\n" + "-" * 64)
    print("LOCKED-TEST EVALUATION (touched once):")
    test_scores = models[chosen].predict_proba(Xte)[:, 1]
    model_res = evaluate_scores(yte, test_scores, f"model:{chosen}")
    base_res = {}
    for feat in ["vol", "trailing_dd", "downside_dev"]:
        base_res[feat] = evaluate_scores(yte, te[feat].values, f"baseline:{feat}")
    best_base_auc = max(b["auc"] for b in base_res.values())
    beats = model_res["auc"] > best_base_auc
    print(f"\n  => blow-up model beats best single-feature baseline on locked-TEST AUC: {beats}")
    print(f"     ({model_res['auc']} vs {best_base_auc})")

    result = {"blowup_threshold": BLOWUP_THRESHOLD, "forward_days": FORWARD_DAYS,
              "chosen_model": chosen, "val_auc": {k: round(v, 4) for k, v in val_auc.items()},
              "test_model": model_res, "test_baselines": base_res,
              "beats_baseline_on_test_auc": bool(beats),
              "blowup_rate": {s: round(float(panel[panel.split == s]["blowup"].mean()), 3)
                              for s in ["train", "val", "test"]}}
    with open(output_dir / "blowup_result.json", "w") as h:
        json.dump(result, h, indent=2)
    print(f"\nSaved -> {output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
