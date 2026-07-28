# Pre-registration record

This file documents the confirmatory design of the study *"Magnitude, Not Shape:
Volatility Beats Downside- and Tail-Risk Metrics at Forecasting Drawdown."* It records
the tests, decision rules, and the composite that were fixed as the confirmatory core of
the study, separately from the slices explored afterward. Everything below was frozen
before the locked-panel analysis was run; the locked panel is scored exactly once.

Recorded: 2026-07-28 (consolidated from `research_problem.md` §5 and §8, which state the
hypotheses and success criteria set at the study's outset).

## 1. Pre-registered (confirmatory) tests

1. **The verdict.** Whether any of the nine downside- or tail-risk metrics beats plain
   trailing volatility at forecasting 90-day forward maximum drawdown in the
   cross-section, across the five test beds.
2. **The magnitude-versus-shape increments.** The partial rank correlation
   `rho(metric, forward drawdown | volatility)` for each metric on each bed, with the
   prediction that only magnitude-bearing metrics carry a positive increment.
3. **The composite.** The equal-weight, untuned composite `z(vol) + z(VaR)`, evaluated on
   the locked panel.

## 2. Generalizability bar (fixed in advance)

An edge counts as **generalizable** only if a metric BH-significantly beats volatility
(q = 0.05) on **at least one crypto and one equity bed**, i.e., survives a change of
asset class. This bar was fixed before the confirmatory analysis to avoid a moving
goalpost.

## 3. Composite construction (fixed before the locked panel was opened)

- Form: equal-weight sum of cross-sectional z-scores, `z(vol) + z(VaR)`, no fitted
  weights.
- Second term: **VaR was fixed as the composite's second term before the locked panel was
  opened.** It was chosen because it was the only metric whose partial increment survived
  BH correction on all five *exploratory* beds. Because the weights are untuned and the
  term was fixed before opening the locked panel, the locked panel is fully
  out-of-sample.
- Locked panel: train on 2017, score once on the 2018 crash.

## 4. Exploratory (not pre-registered)

The finer slices are exploratory and are reported as suggestive only: the split of the
VaR increment by market stress and tail-fatness, the horizon sweep (14–90 days), the
survivorship / multiple-testing / locked-holdout ablations, and the agentic-search
robustness grid (out-of-time and random repartitions).

## Caveat on verifiability

The public repository's initial commit postdates the raw result logs, so the commit
history does not by itself timestamp the freeze relative to the confirmatory analyses.
This document is the design record; it is offered as the pre-registration artifact, not
as cryptographic proof of ordering.
