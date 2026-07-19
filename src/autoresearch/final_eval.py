"""
final_eval.py — LOCKED-TEST evaluation. Run EXACTLY ONCE, after the loop ends and
metric.py is frozen. This is the only script that scores the TEST split.

The TEST score is the honest out-of-sample estimate regardless of how many VAL
experiments the loop ran — that is the anti-snooping guarantee.
"""

import numpy as np

import evaluate as ev
import metric


def main():
    panel = ev.load_panel()
    # The METRIC sees only TRAIN labels (val/test withheld); the SCORER uses true labels.
    metric_input = panel.copy()
    metric_input.loc[metric_input["split"] != "train", "fwd_drawdown"] = np.nan
    scores = np.asarray(metric.build_scores(metric_input), dtype=float)

    test_mask = (panel["split"] == "test").values
    test_rhos = ev._per_date_spearman(scores[test_mask], panel[test_mask])
    val_mask = (panel["split"] == "val").values
    val_rhos = ev._per_date_spearman(scores[val_mask], panel[val_mask])

    baselines = ev.baseline_scores(panel)
    best_base_test = max(b["test_mean"] for b in baselines.values())

    print("=" * 60)
    print("LOCKED-TEST EVALUATION (touched once)")
    print(f"  metric: {getattr(metric, 'DESCRIPTION', '')}")
    print(f"  VAL  score {float(np.mean(val_rhos)):.4f}")
    print(f"  TEST score {float(np.mean(test_rhos)):.4f}  (Newey-West t={ev.newey_west_t(test_rhos):.2f})")
    print("  baselines on TEST:")
    for name, b in baselines.items():
        print(f"    {name:11s} {b['test_mean']:.4f} (t={b['test_t']:.2f})")
    print(f"  => discovered metric beats all baselines on locked TEST: "
          f"{float(np.mean(test_rhos)) > best_base_test}")
    print("=" * 60)


if __name__ == "__main__":
    main()
