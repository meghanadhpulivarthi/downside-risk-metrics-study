# Volatility Is Hard to Beat: A Survivorship-Free Benchmark of Downside-Risk Metrics and the Limits of LLM Metric Discovery

**Working paper — draft, revision 2 (2026-07-13). Target: ACM ICAIF (short paper).**
All results reproduce from `src/` and `outputs/`; every number traces to
`context/findings.md`. Published-metric implementations are daily-data adaptations,
not exact replications, and several published metrics are *repurposed* here as
drawdown forecasters (see §3, §6). References verified against publisher records.

## Abstract

Machine-driven quantitative research — LLM agents that write and test their own
signals — makes it trivially cheap to search a vast space of risk metrics. We show,
on a downside-risk case study, that this cheap search is a data-snooping engine that
a locked holdout is *required* to expose, and that the metrics it (and the published
literature) produce do not reliably beat plain volatility. Concretely: (i) an
autonomous LLM autoresearch loop and a 20,000-config directed search inflate
*validation* skill monotonically with search intensity while the *locked-test* score
plateaus at the volatility baseline; validation-selected metrics beat volatility
out-of-sample only ~43–54% of the time (a coin flip). (ii) Benchmarking ten metrics
(eight published, two classical) across **five test beds** spanning calm and crash
regimes in **crypto (survivorship-free, dead coins retained) and equity**, with
**bootstrap confidence intervals** on every comparison: the only statistically
significant wins over volatility are two volatility-*variants* (downside deviation,
VaR) on a single crypto-bull bed, and they **reverse** on equity; the genuinely
different, return-*pricing* metrics (semibetas, downside beta, tail dependence) —
repurposed here as drawdown forecasters, a use their authors did not claim — are
significantly *worse* than volatility. (iii) The one robust positive: volatility is a
strong survivorship-free *blow-up* predictor (ROC-AUC up to 0.95; 2.5× top-decile
lift), which a gradient-boosting model fails to improve on. The takeaway is a
methodological one for AI-in-finance: automated metric discovery demands anti-snooping
guardrails — a locked, ideally survivorship-free, holdout is not optional.

## 1. Introduction

LLM agents are now wired into quantitative-research loops that propose, code, and
score candidate signals autonomously. This collapses the cost of specification search
to near zero — and thereby makes the classical data-snooping problem (White, 2000;
Harvey, Liu & Zhu, 2016) acute in a new way: an agent can try thousands of metric
variants unsupervised. We ask what discipline is required for such loops to produce
trustworthy signals, using downside-risk forecasting as a concrete, well-populated
case study.

The case study is motivated by auditing Viole & Nawrocki (2016): their
partial-moment "predictive" metric proved to have an inert autocorrelation term and a
survivorship-inflated edge. That raised the fair question — on a level field
(survivorship-free data, a locked holdout, multiple regimes, significance tests) —
does *any* downside-risk metric, published or machine-discovered, beat plain
volatility at forecasting tail risk?

**Contributions.** (1) Evidence that autonomous metric discovery inflates validation
but not a locked-test score (search-induced overfitting), and that this is exposed
only by a locked, survivorship-free holdout — a cautionary result for LLM-driven quant
research, with a released guarded harness. (2) A significance-tested, multi-regime,
survivorship-free benchmark of ten downside-risk metrics showing none consistently
beats volatility. (3) A useful positive: volatility as a survivorship-free blow-up
predictor. We deliberately report a *benchmark and a cautionary tale*, not a new
metric — our attempts to build one, including the discovery loop, failed to beat
volatility once the guardrails were in place.

## 2. Test beds and protocol

**Five test beds, calm and crash, two asset classes.** *Crypto (survivorship-free):*
a CoinMarketCap daily panel (2013–2018) that retains crashed/delisted coins, so a
point-in-time top-100-by-market-cap universe includes assets that later crater; we use
three regimes — **2016 (calm)**, **2017 (bull)**, **2018 (crash)**. *Equity:* S&P 500
constituents (yfinance), two regimes — **1994–1999 (calm bull, n=66 months)** and
**2005–2009 (GFC, n=54)**. The equity beds are survivorship-*biased* by construction
(free equity retains almost no true deaths — itself a finding); they test only
cross-asset generality of the ranking, **not** the survivorship claim, which rests on
the crypto beds.

**Skill measure and significance.** From a 180-day trailing window we forecast 90-day
forward maximum drawdown (delisting scored −100%); skill is the per-formation-date
cross-sectional Spearman correlation, averaged over the bed. Crucially, we place a
**paired block-bootstrap 95% CI on the difference (metric − volatility)** for every
comparison; a metric "beats volatility" only if that CI excludes zero. (This corrects
a bare point-estimate ranking.)

**Machine discovery** (§4.2) uses a strict train/validation/**locked-test** split: the
search optimizes validation; the test is scored once, after the metric is frozen.

## 3. Metrics

We group ten metrics by what they were *designed* to do, because most were not built
to forecast drawdown and are **repurposed** here:
- **Volatility family (risk magnitude):** volatility; downside deviation; 5% VaR; 5%
  ES (Atilgan et al., 2020). These are near-transforms of trailing dispersion.
- **Return-pricing / tail-comovement measures (repurposed):** Viole–Nawrocki UPM/LPM
  ratio; Bollerslev–Patton–Quaedvlieg negative semibeta; Ang–Chen–Xing / Levi–Welch
  downside beta; Chabi-Yo et al. lower-tail dependence; Farago–Tédongap
  volatility-downside. These were built to price the cross-section of *returns* or
  premia, not to rank an asset's forward drawdown.

We drop Kelly–Jiang tail risk from the cross-sectional table: it is an *aggregate*
time-series estimator (pooled Hill exponent predicting the market premium), not a
per-asset measure, and does not have a faithful per-asset analogue. All
implementations are daily adaptations (an appendix documents each estimator and its
fidelity trade-off); results for the pricing measures should be read as "does this
construct, operationalized comparably, forecast drawdown," not as a verdict on the
originals' pricing claims.

## 4. Results

### 4.1 Automated discovery inflates validation, not the locked test (lead result)

We built a Karpathy-style autoresearch loop: an LLM agent freely rewrites a single
`metric.py`; a frozen scorer evaluates it on validation; improvements are kept; the
discovery is scored once on the locked test. A 20,000-config directed search makes the
mechanism unambiguous (Table 1): the best *validation* score rises monotonically with
search intensity while the *locked-test* score plateaus at the volatility baseline.

| Search intensity N | Best validation | Its locked test |
|---|---|---|
| 1 | 0.27 | 0.31 |
| 100 | 0.35 | 0.37 |
| 20,000 | **0.356** | **0.373** (vol baseline 0.376) |

*Table 1. Best-of-N. Validation inflates with search; locked test does not.*
Validation-selected metrics beat volatility on the locked test only **~43–54%** of the
time — a coin flip (42.7% in the 20k directed sweep; 53.6% in an independent
multi-run study of shorter searches). An open-ended 50-experiment LLM loop reached a locked-test 0.42
(a real but small gain that did not survive as a *significant* improvement and was not
reproduced by an independent greedy search), reinforcing that searching harder buys
nothing reliable out of sample.

**Novelty (honest).** The *statistical* phenomenon is classical data-snooping (White,
2000). Our contribution is that it now occurs via an **autonomous agent** proposing
thousands of code-level metrics unsupervised, collapsing search cost to near zero — so
the locked-holdout guardrail becomes non-optional. Notably, a **non-LLM** random-metric
sweep reproduces the same plateau, which is the point: the danger is the *cheap
search*, not the language model per se, and the remedy is the holdout.

### 4.2 No downside-risk metric consistently beats volatility (significance-tested)

Table 2 reports out-of-sample drawdown-forecast rank correlations across the five test
beds, with significance vs volatility from paired block-bootstrap CIs.

| Metric | crypto-16 calm | crypto-17 bull | crypto-18 crash | equity 94–99 | equity 05–09 |
|---|---|---|---|---|---|
| **volatility** (baseline) | 0.623 | 0.355 | 0.376 | 0.606 | 0.564 |
| downside deviation | 0.615 | **0.379\*** | 0.410 | 0.582 † | 0.535 † |
| VaR5 | 0.611 † | **0.400\*** | 0.362 | 0.580 † | 0.540 † |
| ES5 (Atilgan) | 0.623 | 0.351 | 0.327 † | 0.581 † | 0.522 † |
| semibeta (BPQ) | 0.400 † | 0.257 † | 0.309 † | 0.487 † | 0.462 † |
| downside beta (ACX/LW) | 0.014 † | 0.093 † | 0.151 † | 0.242 † | 0.344 † |
| lower-tail dep. (Chabi-Yo) | −0.376 † | −0.154 † | −0.285 † | −0.065 † | −0.046 † |
| vol-downside (Farago–Téd.) | −0.055 † | −0.105 † | −0.050 † | −0.243 † | −0.321 † |
| UPM/LPM (Viole–Nawrocki) | −0.549 † | −0.276 † | −0.346 † | −0.426 † | −0.424 † |

*Table 2. OOS drawdown-forecast Spearman. \* = significantly beats volatility (95%
block-bootstrap CI on the per-date difference excludes 0); † = significantly worse;
unmarked = not distinguishable from volatility. Volatility's own skill is highest in
the calm crypto regime (0.623).*

The **only** significant wins over volatility, on any bed, are two volatility-variants
(downside deviation, VaR5) on the single crypto-bull bed — and they **reverse on
equity**, where volatility significantly beats them. Every genuinely different
(pricing) metric is significantly *worse* than volatility, as expected when a
return-pricing measure is repurposed to forecast drawdown. There is thus **no metric
that consistently or generalizably beats volatility** across regimes and asset
classes. Note this is a narrower and more defensible claim than "nothing beats
volatility": small, regime-specific edges exist for volatility's own variants but do
not survive a change of asset class.

### 4.3 The useful positive: volatility predicts blow-ups

Forecasting *which* assets crater (forward return ≤ −80%) has real signal. On the
locked crypto test, plain volatility attains ROC-AUC 0.68 (0.95 at a −90% threshold)
and a 2.5× top-decile lift; a gradient-boosting model **overfits and loses to
volatility** (AUC 0.59). This is the paper's most decision-relevant result — though it,
too, is a ranking measure, not a cost-aware P&L (see §6).

### 4.4 Robustness (ablations)

The pattern is robust to forecast horizon (30, 90 days) and to the blow-up threshold
(−50/−70/−90%: volatility is the best blow-up predictor at every threshold). A 180-day
horizon is infeasible on this data (forward windows exceed the 2018 panel end).

## 5. Discussion

Two disciplines the source literature usually omits — a survivorship-free universe and
a locked holdout with significance tests — flip the picture from "we found a better
metric" to "nothing reliably beats volatility, and automated search only manufactures
validation mirages." The autoresearch result is the transferable lesson for
AI-in-finance: as LLM agents make signal search nearly free and unsupervised, a
locked, ideally survivorship-free holdout is the minimum guardrail before trusting a
discovered metric. As an integrity note, an earlier version of this project advanced a
"survivorship masks downside value" thesis that a hardened significance test refuted;
we retracted it, and this paper reports only the claims that survived stress-testing.

## 6. Limitations

The skill measure is a rank correlation, **not** a cost-aware / tradable P&L; a
decile-sorted panel exists in the artifact but a turnover/cost-adjusted result is
future work, and conclusions are about forecast ordering, not net value. Published
pricing metrics are **repurposed** as drawdown forecasters and implemented as daily
adaptations (e.g., lower-tail dependence as a quintile co-exceedance) — their negative
results are about this repurposing, not the originals' pricing claims. The equity beds
are survivorship-biased and cannot speak to the survivorship thesis (which rests on
crypto). Five test beds are not all markets. These bound generality, but the direction
— volatility is hard to beat, and search doesn't help — is consistent across every bed.
Finally, we evaluate raw cross-sectional drawdown-forecast rank correlation; an earlier
exploration found ES beats volatility on a partial-correlation (incremental-beyond-vol)
criterion, but that stricter/different task is superseded here by the head-to-head
ranking, on which ES is not distinguishable from (or worse than) volatility.

## 7. Conclusion

On a survivorship-free, multi-regime, significance-tested benchmark, no downside-risk
metric — published or machine-discovered — consistently beats plain volatility at
forecasting tail risk; the only significant wins are small, regime-specific volatility
variants that reverse across asset classes. Volatility is a strong survivorship-free
blow-up predictor. And automated, LLM-driven metric discovery inflates validation
while a locked holdout stays flat — a cautionary result for the practice of AI-in-finance.

## Appendix — reproducibility

`src/autoresearch/`: frozen `prepare_data.py`, agent-edited `metric.py`, frozen
`run_metric.py`/`evaluate.py`, locked `final_eval.py`, `search_loop.py`/
`multi_run_study.py`/`sweep.py` (discovery), `run_blowup_predictor.py`,
`run_multi_testbed_v2.py` (significance + calm regimes), `run_ablations.py`,
`analysis.ipynb`. Data: CoinMarketCap crypto dump; yfinance S&P 500 + `^GSPC`;
`fja05680/sp500` point-in-time membership. Results in `outputs/`; decision trail in
`context/`.

## References

Ang, Chen & Xing (2006, RFS); Ang & Chua (1979); Atilgan, Bali, Demirtas & Gunaydin
(2020, JFE); Bawa (1975); Bollerslev, Patton & Quaedvlieg (2022, JFE); Brown,
Goetzmann, Ibbotson & Ross (1992); Chabi-Yo, Ruenzi & Weigert (2018, JFQA); Farago &
Tédongap (2018, JFE); Fishburn (1977); Harvey, Liu & Zhu (2016, RFS); Kelly & Jiang
(2014, RFS); Levi & Welch (2020, RFS); Novy-Marx & Velikov (2016, RFS); Sortino & van
der Meer (1991); Viole & Nawrocki (2016, JMF); Welch & Goyal (2008, RFS); White (2000,
Econometrica).
