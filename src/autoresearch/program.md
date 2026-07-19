# program.md — Autoresearch: discover a downside-risk metric

Adapted from Karpathy's autoresearch `program.md`. The agent autonomously invents
and refines a **downside-risk metric** to maximize an out-of-sample validation score,
keeping improvements and logging every experiment. Anti-snooping guardrails are added
because an unconstrained search over metrics is a data-snooping engine.

## Goal

**Maximize `val_score`** = the mean, across validation formation dates, of the
cross-sectional Spearman rank correlation between the metric and realized forward
drawdown. Higher is better. (This is a downside *forecasting* score.)

## In-scope files

- `README` / panel schema — read to understand the features. Read-only.
- `evaluate.py` — the frozen scorer. **NEVER edit.**
- `run_metric.py` — the frozen runner (scores VAL only). **NEVER edit.**
- `metric.py` — **THE ONLY FILE YOU EDIT.** It defines `build_scores(panel)`.

## The metric — OPEN-ENDED (not linear)

`metric.py::build_scores(panel) -> array` returns one score per row (higher = more
predicted downside). It may be **any Python**: nonlinear transforms, ratios (e.g.
UPM/LPM), thresholds, regime conditioning, mixtures (e.g. DES = p·1+(1-p)·ES),
interactions, or a small model fit **on the TRAIN split only**. You are NOT limited
to a linear combination and NOT limited to the provided features — you may derive
new features from them. Available feature columns per row:
`vol, es5, es10, var5, lpm2, downside_dev, trailing_dd, cum_ret, skew, kurt,
mean_ret, log_mcap, age_days`.

## Loop protocol (repeat each iteration)

1. Read `results.tsv` (the notebook of past attempts) and the current `metric.py`.
2. Edit `metric.py` with ONE new idea (set the `DESCRIPTION` string).
3. Run `uv run run_metric.py > run.log 2>&1`.
4. Extract `val_score` from the `---` results block (grep). Empty = crash → read the
   last ~30 lines of `run.log`, fix trivial errors, re-run; skip if fundamentally broken.
5. If `val_score` improved over the running best, KEEP (this becomes the new base);
   else DISCARD and revert `metric.py` to the best version.
6. Append a row to `results.tsv`: `iter, val_score, status(keep/discard/crash), description`.

## Rules

- Edit ONLY `metric.py`. Never touch `evaluate.py`, `run_metric.py`, or the data.
- **Never read the TEST split or the `fwd_drawdown` of val/test rows.** `run_metric.py`
  withholds them; do not try to reconstruct them. Fit parameters on the TRAIN split only.
- Prefer simpler metrics when scores are comparable (deleting complexity for equal/better
  score is a win).

## Anti-snooping guardrails (the point of the paper)

- The **TEST split is LOCKED**. It is scored exactly once, by `final_eval.py`, AFTER the
  loop ends and the metric is frozen. Its score is the honest out-of-sample result no
  matter how many val experiments were run.
- Every experiment is logged, so **search intensity** is countable and reported.
- Expectation to test: val gains from aggressive search will NOT transfer to the locked
  test — that is the result, not a failure.
