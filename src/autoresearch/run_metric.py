"""
run_metric.py — FROZEN runner. The agent must NOT edit this.

Loads the metric from metric.py, scores it on the VALIDATION split only, and prints
a machine-parseable results block. TRAIN keeps its label (so metrics may fit on it);
VAL and TEST labels are withheld, and TEST rows are never even passed here.
"""

import numpy as np

import evaluate as ev
import metric


def main():
    panel = ev.load_panel()
    # TRAIN + VAL only (TEST excluded). The METRIC sees VAL with its label withheld;
    # the SCORER uses the true VAL label.
    truth = panel[panel["split"].isin(["train", "val"])].copy()
    metric_input = truth.copy()
    metric_input.loc[metric_input["split"] == "val", "fwd_drawdown"] = np.nan

    scores = np.asarray(metric.build_scores(metric_input), dtype=float)
    val_mask = (truth["split"] == "val").values
    val_rhos = ev._per_date_spearman(scores[val_mask], truth[val_mask])

    print("---")
    print(f"val_score: {float(np.mean(val_rhos)):.6f}")
    print(f"val_t: {ev.newey_west_t(val_rhos):.4f}")
    print(f"n_val_dates: {len(val_rhos)}")
    print(f"description: {getattr(metric, 'DESCRIPTION', '')}")


if __name__ == "__main__":
    main()
