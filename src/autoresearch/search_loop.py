"""
The autoresearch search loop (Karpathy-style, with anti-snooping guardrails).

Analogous to the overnight loop that edits train.py to lower val_bpb: here the loop
searches the space of linear downside-risk metrics (feature subsets) to MAXIMISE the
validation downside-forecast score, keeping improvements greedily and logging every
trial. When it converges, the metric is FROZEN and scored on the LOCKED TEST split
exactly once.

Guardrails that make this honest (unlike naive metric-mining):
  - TRAIN fits parameters; VAL is the only split the search optimises; TEST is locked.
  - Every evaluated hypothesis is counted (search intensity) for transparency/deflation.
  - The locked-TEST score is an honest OOS estimate regardless of search intensity —
    that is the whole point of the holdout.

This file is the SEARCHER; it may propose metrics but may not edit evaluate.py.
Run: uv run python -u src/autoresearch/search_loop.py
"""

import json
from pathlib import Path

import numpy as np

import evaluate as ev

OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"
EPSILON = 0.005          # minimum VAL improvement to accept a feature
MAX_FEATURES = 6


def greedy_forward_select(panel, pool):
    selected, best_val, trials, log = [], -np.inf, 0, []
    while len(selected) < MAX_FEATURES:
        step_best_feat, step_best_val = None, best_val
        for feature in pool:
            if feature in selected:
                continue
            rhos, _ = ev.score_model(panel, selected + [feature], "val")
            trials += 1
            mean_val = float(np.mean(rhos))
            if mean_val > step_best_val:
                step_best_val, step_best_feat = mean_val, feature
        if step_best_feat is None or step_best_val - best_val < EPSILON:
            break
        selected.append(step_best_feat)
        best_val = step_best_val
        log.append({"step": len(selected), "added": step_best_feat,
                    "val_mean": round(best_val, 4), "cumulative_trials": trials})
        print(f"  step {len(selected)}: +{step_best_feat:13s} VAL={best_val:.4f}  (trials={trials})")
    return selected, best_val, trials, log


def main():
    import datetime
    now = datetime.datetime.now()
    print("=" * 64)
    print(f"Autoresearch metric-discovery loop — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_autoresearch_loop"
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = ev.load_panel()
    panel = panel.copy()
    panel["des"] = ev.death_adjusted_es(panel)
    pool = ev.ALL_FEATURES + ["des"]

    baselines = ev.baseline_scores(panel)
    best_baseline_val = max(b["val_mean"] for b in baselines.values())
    print(f"Baselines (VAL): " + ", ".join(f"{k}={v['val_mean']:.3f}" for k, v in baselines.items()))
    print(f"Best baseline VAL score to beat: {best_baseline_val:.4f}\n")

    print("SEARCH (greedy forward selection, optimising VAL):")
    selected, best_val, trials, log = greedy_forward_select(panel, pool)
    print(f"\nDiscovered metric = linear({selected})")
    print(f"VAL score {best_val:.4f} over {trials} trials "
          f"(vs best baseline {best_baseline_val:.4f})")

    # ---- FINAL: touch the LOCKED TEST split exactly once ----
    test_rhos, _ = ev.score_model(panel, selected, "test")
    test_mean, test_t = float(np.mean(test_rhos)), ev.newey_west_t(test_rhos)
    print("\n" + "-" * 64)
    print("LOCKED-TEST evaluation (touched once, honest OOS regardless of search):")
    print(f"  discovered metric : TEST {test_mean:.3f}  (Newey-West t={test_t:.2f})")
    for name, b in baselines.items():
        print(f"  baseline {name:11s}: TEST {b['test_mean']:.3f}  (t={b['test_t']:.2f})")
    beats = test_mean > max(b["test_mean"] for b in baselines.values())
    print(f"  => discovered metric beats all baselines on locked TEST: {beats}")

    result = {
        "selected_features": selected, "val_score": round(best_val, 4),
        "search_trials": trials, "epsilon": EPSILON,
        "best_baseline_val": round(best_baseline_val, 4),
        "test_score": round(test_mean, 4), "test_t": round(test_t, 2),
        "baselines": baselines, "beats_baselines_on_test": bool(beats),
        "selection_log": log,
    }
    with open(output_dir / "result.json", "w") as h:
        json.dump(result, h, indent=2)
    print(f"\nSaved -> {output_dir}")
    print("Note: TEST is locked; its score is honest OOS. Search intensity "
          f"({trials} trials) is logged for deflation/transparency.")
    print("=" * 64)


if __name__ == "__main__":
    main()
