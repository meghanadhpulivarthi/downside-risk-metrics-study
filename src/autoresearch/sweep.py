"""
Directed massive sweep — loop over DIRECTIONS (parameterized metric families),
sweeping each over thousands of deterministic configs, instead of proposing metrics
one at a time. This is the efficient, high-intensity version of the search: the
creative layer picks families; code sweeps them at scale.

For the paper it is also the strongest anti-snooping stress test: it drives search
intensity far higher than the one-at-a-time loop, so if the locked-TEST score still
plateaus at the volatility baseline while VAL keeps inflating, the point is made
decisively (White 2000 reality-check logic at scale).

Each family is a parameterized function of per-date feature RANKS (nonlinear via
products, gates, ratios, interactions). We record every config's VAL and TEST score,
then report the best-of-total-N curve, per-family bests, and val-selection reliability.

Run: uv run --project <repo> python -u src/autoresearch/sweep.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import evaluate as ev

OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"
CONFIGS_PER_FAMILY = 4000
SEED = 20260712
DOWNSIDE = ["var5", "es10", "lpm2", "downside_dev", "es5", "trailing_dd"]


def per_date_ranks(panel):
    return {f: panel.groupby("formation_date")[f].rank(pct=True).values for f in ev.ALL_FEATURES}


def fast_scorer(panel, split):
    """Return score(full_scores)->mean per-date Spearman for `split`, precomputed for speed."""
    split_df = panel[panel["split"] == split]
    groups = []
    for _, g in split_df.groupby("formation_date"):
        if len(g) >= ev.MIN_COINS_PER_DATE:
            groups.append((g.index.values, g["fwd_drawdown"].values))
    row_pos = {idx: i for i, idx in enumerate(panel.index.values)}

    def score(full_scores):
        rhos = []
        for idx, dd in groups:
            pos = [row_pos[i] for i in idx]
            rhos.append(spearmanr(full_scores[pos], dd)[0])
        return float(np.mean(rhos))
    return score


# ------------------ DIRECTIONS (parameterized metric families) ------------------

def fam_weighted_consensus(ranks, rng):
    k = rng.integers(2, len(DOWNSIDE) + 1)
    subset = list(rng.choice(DOWNSIDE, size=k, replace=False))
    w = rng.random(k)
    score = sum(wi * ranks[f] for wi, f in zip(w, subset))
    return score, f"wcons:{'+'.join(subset)}"


def fam_consensus_boost(ranks, rng):
    k = rng.integers(2, 5)
    subset = list(rng.choice(DOWNSIDE, size=k, replace=False))
    g = rng.choice(["cum_ret", "mean_ret"])
    b = rng.uniform(0, 1.2)
    base = np.mean([ranks[f] for f in subset], axis=0)
    return base * (1.0 + b * ranks[g]), f"boost:({'+'.join(subset)})x{b:.2f}{g}"


def fam_ratio(ranks, rng):
    a = rng.choice(DOWNSIDE)
    b = rng.choice(ev.ALL_FEATURES)
    c = rng.uniform(0, 2)
    return ranks[a] - c * ranks[b], f"ratio:{a}-{c:.2f}{b}"


def fam_gated(ranks, rng):
    k = rng.integers(2, 5)
    subset = list(rng.choice(DOWNSIDE, size=k, replace=False))
    g = rng.choice(ev.ALL_FEATURES)
    q = rng.uniform(0.3, 0.7)
    s = rng.uniform(0, 1.5)
    base = np.mean([ranks[f] for f in subset], axis=0)
    return base * (1.0 + s * (ranks[g] > q)), f"gate:({'+'.join(subset)})|{g}>{q:.2f}"


def fam_tilt_blend(ranks, rng):
    k = rng.integers(2, 5)
    subset = list(rng.choice(DOWNSIDE, size=k, replace=False))
    base = np.mean([ranks[f] for f in subset], axis=0)
    a, b, c, d = rng.uniform(0.3, 1), rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)
    sc = (a * base + b * (1 - ranks["log_mcap"]) + c * (1 - ranks["age_days"])
          + d * (1 - ranks["log_mcap"]) * (1 - ranks["age_days"]))
    return sc, f"tilt:({'+'.join(subset)})"


FAMILIES = {"weighted_consensus": fam_weighted_consensus, "consensus_boost": fam_consensus_boost,
            "ratio": fam_ratio, "gated": fam_gated, "tilt_blend": fam_tilt_blend}


def main():
    import datetime
    now = datetime.datetime.now()
    print("=" * 64)
    print(f"Directed massive sweep — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_directed_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = ev.load_panel()
    ranks = per_date_ranks(panel)
    score_val, score_test = fast_scorer(panel, "val"), fast_scorer(panel, "test")
    base = ev.baseline_scores(panel)
    vol_val, vol_test = base["volatility"]["val_mean"], base["volatility"]["test_mean"]
    print(f"Baseline volatility: VAL {vol_val:.3f}, TEST {vol_test:.3f}")
    print(f"Directions: {list(FAMILIES)} x {CONFIGS_PER_FAMILY} configs each\n")

    rng = np.random.default_rng(SEED)
    all_val, all_test, fam_of = [], [], []
    per_family_best = {}
    for fname, fn in FAMILIES.items():
        best_v, best_t = -np.inf, None
        for _ in range(CONFIGS_PER_FAMILY):
            scores, _ = fn(ranks, rng)
            scores = np.asarray(scores, dtype=float)
            v = score_val(scores)
            if not np.isfinite(v):
                continue
            t = score_test(scores)
            all_val.append(v); all_test.append(t); fam_of.append(fname)
            if v > best_v:
                best_v, best_t = v, t
        per_family_best[fname] = {"best_val": round(best_v, 4), "its_test": round(best_t, 4)}
        print(f"  {fname:20s}: best VAL {best_v:.4f} -> TEST {best_t:.4f}")

    all_val, all_test = np.array(all_val), np.array(all_test)
    n = len(all_val)
    gbest = int(np.argmax(all_val))
    print(f"\nTotal configs: {n}")
    print(f"GLOBAL best VAL {all_val[gbest]:.4f} (family {fam_of[gbest]}) -> locked TEST {all_test[gbest]:.4f}")
    print(f"  vs baselines TEST: vol {vol_test:.3f}, DES {base['DES']['test_mean']:.3f}")

    # Best-of-N curve (search intensity -> val inflation vs test plateau)
    rng2 = np.random.default_rng(SEED + 1)
    print("\nBest-of-N (search intensity):")
    curve = []
    for N in [1, 10, 100, 1000, 5000, n]:
        if N > n:
            N = n
        vsel, tsel = [], []
        for _ in range(200):
            idx = rng2.choice(n, size=N, replace=False)
            w = idx[np.argmax(all_val[idx])]
            vsel.append(all_val[w]); tsel.append(all_test[w])
        curve.append({"N": int(N), "val": round(float(np.mean(vsel)), 3), "test": round(float(np.mean(tsel)), 3)})
        print(f"  N={N:6d}: best-val {np.mean(vsel):.3f} -> its test {np.mean(tsel):.3f}")

    winners = all_val > vol_val
    reliability = float((all_test[winners] > vol_test).mean()) if winners.any() else float("nan")
    result = {
        "n_configs": n, "vol_val": vol_val, "vol_test": vol_test,
        "global_best_val": round(float(all_val[gbest]), 4),
        "global_best_val_test": round(float(all_test[gbest]), 4),
        "global_best_family": fam_of[gbest],
        "per_family_best": per_family_best, "best_of_N": curve,
        "val_test_corr": round(float(np.corrcoef(all_val, all_test)[0, 1]), 3),
        "val_winners_beating_vol_on_test": reliability,
    }
    with open(output_dir / "sweep_result.json", "w") as h:
        json.dump(result, h, indent=2)
    print(f"\nval-test corr: {result['val_test_corr']}; "
          f"val-winners beating vol on TEST: {reliability:.1%}")
    print(f"Saved -> {output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
