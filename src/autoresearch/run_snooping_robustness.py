"""
Data-snooping robustness across PANELS and SEEDS (reviewer P2-#5).

sweep.py / multi_run_study.py establish the search-overfitting result on a SINGLE
locked panel (one fixed val/test split) with a SINGLE search seed. A reviewer asks:
is the "validation inflates, locked test plateaus at volatility" result a property of
that particular holdout, or does it hold across panels and seeds?

The original split is strictly chronological (train 2016, val 2017 bull, test 2018
crash), so the locked test is genuinely OUT-OF-TIME across a regime shift. We re-run
the overfitting study over a grid of panels x search seeds, using TWO panel families:

  * OUT-OF-TIME (faithful) — slide a fixed-length test window through the timeline with
    validation always chronologically BEFORE its test. This preserves the regime-shift
    structure the original claim is about and tests robustness to WHICH future window
    we lock away. This is the design that strengthens the claim.
  * RANDOM (contrast) — repartition the non-train dates into val/test ignoring time.
    This destroys the regime shift, making val and test exchangeable. Reported to show
    that random cross-validation UNDERSTATES the snooping risk that out-of-time testing
    reveals (a methodological caution, not the main result).

Search seeds are independent RNGs for the candidate-metric generator.

For every (panel x seed) run we record, on that panel's own volatility baseline:
  - val_test_corr : rank corr of val vs test score across candidates
  - win_rate      : fraction of val-winners (beat vol on VAL) that ALSO beat vol on TEST
  - val_inflation : best-val candidate's VAL score minus vol (how much search inflates val)
  - test_gap      : that same best-val candidate's TEST score minus vol (the plateau)
Then we aggregate (mean / std / min / max) across all runs. The claim is supported iff
test_gap stays ~0 (plateau) with small dispersion while val_inflation is large and
positive, and win_rate stays near a coin flip, on EVERY panel.

Reuses sample_metric / per_date_ranks from multi_run_study and _per_date_spearman from
evaluate. Ranks depend only on formation_date, so they are computed once.

Run: uv run python -u src/autoresearch/run_snooping_robustness.py
"""

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import evaluate as ev
from multi_run_study import sample_metric, per_date_ranks

# Config — edit these directly
OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"
N_SEARCH_SEEDS = 3           # independent candidate-generator RNGs per panel
N_CANDIDATES = 1500          # candidate metrics generated per run
N_VAL_DATES = 10             # match the original split sizes
N_TEST_DATES = 7
N_RANDOM_PANELS = 6          # random (order-ignoring) repartitions, for contrast
OOT_STARTS = [10, 13, 16, 19, 22]  # sliding test-window start indices (out-of-time panels)
PANEL_SEED_BASE = 70000
SEARCH_SEED_BASE = 80000


def build_panels(all_dates, nontrain):
    """Return a list of (panel_type, panel_id, val_dates, test_dates).

    all_dates : chronologically sorted array of every formation date.
    nontrain  : chronologically sorted array of non-train dates (val+test pool).
    """
    panels = []
    # OUT-OF-TIME: fixed-length test window slid through the timeline; val strictly before.
    for start in OOT_STARTS:
        if start < N_VAL_DATES or start + N_TEST_DATES > len(all_dates):
            print(f"  OOT start {start} infeasible (need {N_VAL_DATES} before, "
                  f"{N_TEST_DATES} after within {len(all_dates)} dates); skipping")
            continue
        val = set(all_dates[start - N_VAL_DATES:start])
        test = set(all_dates[start:start + N_TEST_DATES])
        panels.append(("out_of_time", f"oot@{start}", val, test))
    # RANDOM: order-ignoring repartition of the non-train pool (the original val+test dates).
    for p in range(N_RANDOM_PANELS):
        rng = np.random.default_rng(PANEL_SEED_BASE + p)
        order = rng.permutation(len(nontrain))
        val = set(nontrain[order[:N_VAL_DATES]])
        test = set(nontrain[order[N_VAL_DATES:N_VAL_DATES + N_TEST_DATES]])
        panels.append(("random", f"rand{p}", val, test))
    return panels


def _split_score(scores, mask, split_df):
    """Mean per-date cross-sectional Spearman of `scores[mask]` vs drawdown on split_df."""
    return float(np.mean(ev._per_date_spearman(scores[mask], split_df)))


def run_one(panel, ranks, val_mask, test_mask, val_df, test_df, vol_val, vol_test, search_seed):
    """One search on a fixed panel: generate candidates, score val+test, summarize."""
    rng = np.random.default_rng(search_seed)
    cand_val, cand_test = [], []
    for _ in range(N_CANDIDATES):
        scores, _ = sample_metric(rng, ranks, panel)
        scores = np.asarray(scores, dtype=float)
        v = _split_score(scores, val_mask, val_df)
        t = _split_score(scores, test_mask, test_df)
        if np.isfinite(v) and np.isfinite(t):
            cand_val.append(v)
            cand_test.append(t)
    cand_val, cand_test = np.array(cand_val), np.array(cand_test)
    if len(cand_val) < 50:
        # Too few finite candidates to characterize a distribution — fail loud, skip run.
        print(f"      search_seed {search_seed}: only {len(cand_val)} finite candidates; skipping")
        return None

    winners = cand_val > vol_val
    if winners.sum() == 0:
        print(f"      search_seed {search_seed}: NO candidate beat vol on val; win_rate=NaN")
        win_rate = float("nan")
    else:
        win_rate = float((cand_test[winners] > vol_test).mean())

    best = int(np.argmax(cand_val))   # the metric a val-driven search would select at max intensity
    return {
        "search_seed": search_seed,
        "n_candidates": int(len(cand_val)),
        "val_test_corr": round(float(np.corrcoef(cand_val, cand_test)[0, 1]), 3),
        "win_rate": None if np.isnan(win_rate) else round(win_rate, 3),
        "val_inflation": round(float(cand_val[best] - vol_val), 3),
        "test_gap": round(float(cand_test[best] - vol_test), 3),
        "n_val_winners": int(winners.sum()),
    }


def main():
    now = datetime.datetime.now()
    print("=" * 70)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : out-of-time starts {OOT_STARTS} + {N_RANDOM_PANELS} random panels, "
          f"x {N_SEARCH_SEEDS} seeds x {N_CANDIDATES} candidates | val/test={N_VAL_DATES}/{N_TEST_DATES}")
    print("=" * 70)

    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_snooping_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    panel = ev.load_panel()
    ranks = per_date_ranks(panel)  # per-formation-date; independent of the val/test split
    fdates = panel["formation_date"]
    all_dates = np.array(sorted(panel["formation_date"].unique()))
    nontrain = np.array(sorted(panel.loc[panel["split"].isin(["val", "test"]),
                                         "formation_date"].unique()))
    print(f"Loaded panel: {len(panel)} rows, {len(all_dates)} formation dates "
          f"({len(nontrain)} non-train)")

    panels = build_panels(all_dates, nontrain)
    # Prepend the EXACT original chronological split (val 2017 bull -> test 2018 crash) as a
    # labeled reference panel, so the crash-holdout number lives in the same artifact.
    orig_val = set(panel.loc[panel["split"] == "val", "formation_date"].unique())
    orig_test = set(panel.loc[panel["split"] == "test", "formation_date"].unique())
    panels = [("out_of_time", "orig_crash", orig_val, orig_test)] + panels

    runs = []
    for panel_idx, (ptype, pid, val_dates, test_dates) in enumerate(panels):
        val_mask = fdates.isin(val_dates).values
        test_mask = fdates.isin(test_dates).values
        val_df, test_df = panel[val_mask], panel[test_mask]
        vol_val = _split_score(panel["vol"].values, val_mask, val_df)
        vol_test = _split_score(panel["vol"].values, test_mask, test_df)
        print(f"\n[{ptype} {pid}] vol baseline: VAL {vol_val:.3f}  TEST {vol_test:.3f}")

        for s in range(N_SEARCH_SEEDS):
            rec = run_one(panel, ranks, val_mask, test_mask, val_df, test_df,
                          vol_val, vol_test, SEARCH_SEED_BASE + panel_idx * 100 + s)
            if rec is None:
                continue
            rec.update({"panel_type": ptype, "panel_id": pid,
                        "vol_val": round(vol_val, 3), "vol_test": round(vol_test, 3)})
            runs.append(rec)
            wr = "NaN" if rec["win_rate"] is None else f"{rec['win_rate']:.2f}"
            print(f"      seed {rec['search_seed']}: val_test_corr {rec['val_test_corr']:+.2f}  "
                  f"win_rate {wr}  val_inflation {rec['val_inflation']:+.3f}  "
                  f"test_gap {rec['test_gap']:+.3f}")

    # --- Aggregate, separately per panel family ---
    def agg(subset, key):
        vals = [r[key] for r in subset if r[key] is not None]
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        arr = np.array(vals, dtype=float)
        return {"mean": round(float(arr.mean()), 3), "std": round(float(arr.std()), 3),
                "min": round(float(arr.min()), 3), "max": round(float(arr.max()), 3), "n": len(vals)}

    summary = {"n_candidates": N_CANDIDATES, "by_type": {}}
    print("\n" + "=" * 70)
    print("AGGREGATE by panel family")
    print("=" * 70)
    for ptype in ["out_of_time", "random"]:
        subset = [r for r in runs if r["panel_type"] == ptype]
        summary["by_type"][ptype] = {"n_runs": len(subset)}
        print(f"\n[{ptype}]  ({len(subset)} runs)")
        for key in ["val_test_corr", "win_rate", "val_inflation", "test_gap"]:
            a = agg(subset, key)
            summary["by_type"][ptype][key] = a
            if a["mean"] is not None:
                print(f"  {key:16s}: mean {a['mean']:+.3f}  std {a['std']:.3f}  "
                      f"range [{a['min']:+.3f}, {a['max']:+.3f}]  (n={a['n']})")
    print("\n  Claim strengthened iff OUT-OF-TIME test_gap ~ 0 (small std) while val_inflation")
    print("  is large +, on every panel; RANDOM shows the contrast (positive gap / high win_rate).")

    out = {"summary": summary, "runs": runs}
    with open(output_dir / "snooping_robustness.json", "w") as h:
        json.dump(out, h, indent=2)
    print(f"\nResults saved: {output_dir / 'snooping_robustness.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
