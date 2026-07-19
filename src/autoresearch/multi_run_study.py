"""
Multi-run study — the paper's core evidence on search-induced overfitting.

A single search is an anecdote. Here we characterise the DISTRIBUTION: generate many
nonlinear candidate metrics, and simulate many independent "searches" (each picks the
metric with the best VALIDATION score, as the loop does), then ask what happens on the
LOCKED TEST split. Quantifies: (i) the val->test overfitting gap, (ii) whether val-
selection reliably beats plain volatility out-of-sample, (iii) how the gap grows with
search intensity (best-of-N). This is the White (2000) reality-check logic.

Metrics are parameter-free (random but fixed) nonlinear functions of per-date feature
ranks — so each candidate is a distinct hypothesis in an open-ended, nonlinear space.

Run: uv run --project <repo> python -u src/autoresearch/multi_run_study.py
"""

import json
from pathlib import Path

import numpy as np

import evaluate as ev

OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"
N_CANDIDATES = 4000
POOL_SIZE = 15          # experiments per simulated "search" (matches the LLM loop length)
N_POOLS = 250           # independent simulated searches
SEED = 12345


def per_date_ranks(panel):
    """Per-formation-date percentile rank of each feature (nonlinear building blocks)."""
    ranks = {}
    for feat in ev.ALL_FEATURES:
        ranks[feat] = panel.groupby("formation_date")[feat].rank(pct=True).values
    return ranks


def sample_metric(rng, ranks, panel):
    """Sample one random nonlinear metric -> per-row score. Returns (scores, description)."""
    feats = ev.ALL_FEATURES
    kind = rng.choice(["single", "consensus", "ratio", "product", "threshold", "wsum", "interaction"])
    pick = lambda k: list(rng.choice(feats, size=k, replace=False))
    if kind == "single":
        f = pick(1)[0]
        return ranks[f], f"single:{f}"
    if kind == "consensus":
        subset = pick(rng.integers(2, 5))
        return np.mean([ranks[f] for f in subset], axis=0), f"consensus:{'+'.join(subset)}"
    if kind == "ratio":
        a, b = pick(2)
        return ranks[a] / (ranks[b] + 0.1), f"ratio:{a}/{b}"
    if kind == "product":
        a, b = pick(2)
        return ranks[a] * ranks[b], f"product:{a}*{b}"
    if kind == "threshold":
        a, b = pick(2)
        return ranks[a] * (ranks[b] > 0.5), f"threshold:{a}|{b}>med"
    if kind == "wsum":
        subset = pick(rng.integers(2, 5))
        weights = rng.normal(size=len(subset))
        return np.sum([w * ranks[f] for w, f in zip(weights, subset)], axis=0), f"wsum:{'+'.join(subset)}"
    # interaction (the LLM-winner family): consensus boosted by a runup-like term
    subset = pick(rng.integers(2, 4)); g = pick(1)[0]
    base = np.mean([ranks[f] for f in subset], axis=0)
    return base * (1.0 + 0.5 * ranks[g]), f"interaction:({'+'.join(subset)})x{g}"


def main():
    import datetime
    now = datetime.datetime.now()
    print("=" * 64)
    print(f"Multi-run search-overfitting study — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_multirun_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = ev.load_panel()
    ranks = per_date_ranks(panel)
    val_mask = (panel["split"] == "val").values
    test_mask = (panel["split"] == "test").values
    val_df, test_df = panel[val_mask], panel[test_mask]

    base = ev.baseline_scores(panel)
    vol_val, vol_test = base["volatility"]["val_mean"], base["volatility"]["test_mean"]
    print(f"Baseline volatility: VAL {vol_val:.3f}, TEST {vol_test:.3f}\n")

    # ---- Generate candidate metrics; score each on VAL and TEST ----
    rng = np.random.default_rng(SEED)
    cand_val, cand_test, descs = [], [], []
    for _ in range(N_CANDIDATES):
        scores, desc = sample_metric(rng, ranks, panel)
        v = float(np.mean(ev._per_date_spearman(scores[val_mask], val_df)))
        t = float(np.mean(ev._per_date_spearman(scores[test_mask], test_df)))
        cand_val.append(v); cand_test.append(t); descs.append(desc)
    cand_val, cand_test = np.array(cand_val), np.array(cand_test)
    finite = np.isfinite(cand_val) & np.isfinite(cand_test)
    cand_val, cand_test = cand_val[finite], cand_test[finite]
    print(f"Generated {len(cand_val)} valid candidate metrics.")

    # ---- (i) Overfitting gap among val-winners ----
    val_winners = cand_val > vol_val
    gap = cand_val[val_winners] - cand_test[val_winners]
    print(f"\n(i) Candidates beating vol on VAL: {val_winners.sum()}/{len(cand_val)}")
    print(f"    of those, mean VAL {cand_val[val_winners].mean():.3f} vs mean TEST "
          f"{cand_test[val_winners].mean():.3f}  (mean val-test gap {gap.mean():+.3f})")
    reliability = float((cand_test[val_winners] > vol_test).mean())
    print(f"    fraction that ALSO beat vol on TEST: {reliability:.1%}")

    # ---- (ii) Simulate many independent searches: pick best-VAL in each pool ----
    rng2 = np.random.default_rng(SEED + 1)
    n = len(cand_val)
    sel_val, sel_test = [], []
    for _ in range(N_POOLS):
        idx = rng2.choice(n, size=POOL_SIZE, replace=False)
        winner = idx[np.argmax(cand_val[idx])]
        sel_val.append(cand_val[winner]); sel_test.append(cand_test[winner])
    sel_val, sel_test = np.array(sel_val), np.array(sel_test)
    beat_test = float((sel_test > vol_test).mean())
    print(f"\n(ii) {N_POOLS} simulated {POOL_SIZE}-experiment searches (pick best-VAL):")
    print(f"     selected metric mean VAL {sel_val.mean():.3f} (beats vol by {sel_val.mean()-vol_val:+.3f})")
    print(f"     selected metric mean TEST {sel_test.mean():.3f} (vs vol {vol_test:.3f}, "
          f"diff {sel_test.mean()-vol_test:+.3f})")
    print(f"     fraction of searches whose winner beats vol on locked TEST: {beat_test:.1%}")

    # ---- (iii) Best-of-N: val inflates with search intensity, test does not ----
    print(f"\n(iii) Best-of-N (search intensity -> val inflation vs test):")
    curve = []
    for N in [1, 5, 10, 25, 50, 100, 500, 2000]:
        if N > n:
            break
        vsel, tsel = [], []
        for _ in range(200):
            idx = rng2.choice(n, size=N, replace=False)
            w = idx[np.argmax(cand_val[idx])]
            vsel.append(cand_val[w]); tsel.append(cand_test[w])
        curve.append({"N": N, "val": round(float(np.mean(vsel)), 3), "test": round(float(np.mean(tsel)), 3)})
        print(f"     N={N:5d}: best-val {np.mean(vsel):.3f}  ->  its test {np.mean(tsel):.3f}")

    result = {
        "n_candidates": int(len(cand_val)), "vol_val": vol_val, "vol_test": vol_test,
        "val_winners": int(val_winners.sum()),
        "mean_val_test_gap_among_winners": round(float(gap.mean()), 4),
        "reliability_val_winner_beats_vol_on_test": reliability,
        "sim_searches": {"pool_size": POOL_SIZE, "n_pools": N_POOLS,
                         "sel_mean_val": round(float(sel_val.mean()), 4),
                         "sel_mean_test": round(float(sel_test.mean()), 4),
                         "frac_winners_beat_vol_on_test": beat_test},
        "best_of_N_curve": curve,
        "val_test_correlation": round(float(np.corrcoef(cand_val, cand_test)[0, 1]), 3),
    }
    with open(output_dir / "study_result.json", "w") as h:
        json.dump(result, h, indent=2)
    print(f"\nval-test correlation across candidates: {result['val_test_correlation']}")
    print(f"Saved -> {output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
