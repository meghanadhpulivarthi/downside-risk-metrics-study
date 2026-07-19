# Is the Partial-Moment "Predictive Edge" Real? A Replication and Adversarial Audit of Viole & Nawrocki (2016)

**Working paper — draft, revision 2 (2026-07-12).** All numerical results are
reproducible from the code in `src/` and the logged results in `outputs/`; every
figure is traceable to `context/findings.md`. Revision 2 responds to a simulated
peer-review panel: it leads with the load-bearing result, engages the anchor
paper's own Appendix A defence, adds a total-return-benchmark robustness check, an
FDR multiplicity correction, an economic-significance panel, and the downside-risk
/ survivorship / out-of-sample-predictability literature. *All references were verified
against publisher records in a literature-review pass (2026-07-12); see
`writeup/literature_review.md`.*

---

## Abstract

Viole & Nawrocki (2016) propose that a partial-moment risk metric — an
Upper/Lower Partial Moment (UPM/LPM) ratio, discounted by a one-period
autocorrelation term — predicts out-of-sample equity performance where classical
metrics cannot, tested on ~300 *surviving* S&P 500 firms. We re-implement the
metric exactly from the source equations and audit the claim along four lines.
**The load-bearing result (RQ2):** the `(1−|ρ|)` autocorrelation adjustment — the
paper's headline "predictive" innovation — cannot change the cross-sectional
ranking. Because the estimated lag-1 autocorrelation is small (median |ρ| ≈ 0.08),
the discount `(1−|ρ|) ≈ 0.92` is a near-constant multiplier; across 144
configurations (lags, functional forms, the covariance-vs-correlation reading,
three windows, four investor types) it moves the out-of-sample rank correlation by
at most 0.045, and for the paper's exact specification all twelve bootstrap CIs
include zero. The marketed cross-sectional improvement is structurally
near-impossible, not merely absent. Three corroborating results follow: (RQ1) even
a partial survivorship correction shifts the predictive out-of-sample correlation
*down* in every configuration — directly engaging the anchor's Appendix A claim
that survivorship inflates only the *explanatory* correlations; (RQ3) in a
genuinely survivorship-free crypto regime the partial-moment downside metric does
not robustly beat plain variance at forecasting drawdown (difference +0.066, 95%
CI [−0.065, +0.211]); and (RQ4) data-learned degree parameters do not reliably
generalise across eras. Robustness checks (total-return benchmark; FDR
correction) do not change the conclusions. A secondary, co-equal contribution is
methodological: genuinely survivorship-bias-free equity price data is not freely
obtainable — free sources retain point-in-time *membership* but systematically
omit the go-to-zero *price* tail — which is part of why such claims go
under-audited. We conclude that the cross-sectional *rank-predictive* content of
the metric, as specified, does not survive audit; we do not test tradable
(cost-and-turnover) economic value, which is a distinct question.

---

## 1. Introduction

Re-deriving a textbook risk metric is routine. Auditing a *published predictive
claim* for a bias capable of manufacturing the entire result is not. Viole &
Nawrocki (2016, *Journal of Mathematical Finance*) argue that classical
benefit/cost metrics (Sharpe, Sortino, Treynor, Jensen's alpha, VaR) are
"ill-conceived" because they use a single observation as both the risk and the
return term, and propose a partial-moment replacement they report to have genuine
out-of-sample predictive power.

**Positioning.** This work sits in three literatures. First, the *downside-risk /
partial-moment* tradition that the anchor builds on: below-target risk (Bawa,
1975; Fishburn, 1977; Fishburn & Kochenberger, 1979; Holthausen, 1981), semivariance
(Markowitz, 1959), the Sortino ratio (Sortino & van der Meer, 1991), and
loss-aversion (Kahneman & Tversky, 1979), whose λ ≈ 2.25 motivates the "prospect"
exponent. Second, the *survivorship-bias* literature that quantified how
conditioning on survival inflates performance measurement (Brown, Goetzmann,
Ibbotson & Ross, 1992; Brown, Goetzmann & Ross, 1995; Elton, Gruber & Blake, 1996;
Carhart, Carpenter, Lynch & Musto, 2002). Third, the *out-of-sample predictability
and data-mining* literature that supplies the tools for our audit: out-of-sample
R² and the in-sample/OOS gap as an overfitting diagnostic (Welch & Goyal, 2008;
Campbell & Thompson, 2008), multiple-testing discipline (White, 2000; Harvey, Liu &
Zhu, 2016), and the gap between statistical predictability and tradable value
after costs (Novy-Marx & Velikov, 2016).

**The anchor's own defence.** Viole & Nawrocki anticipate the survivorship
critique in their Appendix A ("Survivorship/Systemic Biases"). They write: *"We
actually want this bias present in our data because we are analyzing the
characteristics of the survivors… If anything, the survivorship bias in our study
would pad the performance of the explanatory metrics generating higher ex post
rank correlations. However, this does not offer an acceptable counterpoint to why
these metrics completely fail out-of-sample."* Their position is therefore precise:
survivorship inflates the *explanatory* (ex-post) correlations but, they argue,
cannot explain the behaviour of the *predictive* (out-of-sample) metric. Our RQ1 is
designed to test *that specific claim*, not the strawman that the authors were
unaware of survivorship.

**Research questions.**
- **RQ1 (survivorship):** does a survivorship correction move the *predictive*
  out-of-sample correlation, as Appendix A says it should not?
- **RQ2 (the autocorrelation term):** does `(1 − |ρ|)` add out-of-sample power over
  the base ratio? (Load-bearing.)
- **RQ3 (asymmetry vs variance):** do partial-moment metrics forecast future
  *downside* better than variance, out of sample, in a survivorship-free regime?
- **RQ4 (learned parameters):** can the degree exponents `(q, n)` be learned and
  generalise?

Our aim is a defensible verdict, not a positive one.

## 2. The metric

Implemented directly from the paper's equations (`src/partial_moments.py`, 12
hand-checked unit tests):

- LPM: `LPM(n,h,i) = (1/T) Σ max(0, h − R_it)ⁿ` (Eq 1)
- UPM: `UPM(q,l,i) = (1/T) Σ max(0, R_it − l)^q` (Eq 2); `h = l` (footnote 3).
- Explanatory metric: `UPM(q,y,x) / LPM(n,y,x)` (Eq 4), with target `y` the
  **double non-stationary benchmark**: at each date, the elementwise maximum of the
  asset-class benchmark return and the risk-free rate (p. 905).
- Predictive metric: `(UPM/LPM) · (1 − |ρ(x)|)` (Eq 7).
- Degree presets encode "investor type": Risk-Averse (q=0.25), Prospect (q=0.44 ≈
  1/2.25, the inverse Kahneman–Tversky loss-aversion coefficient), Risk-Neutral
  (q=1), Risk-Seeking (q=2), with n=1.

**Two specification issues.** (i) Eq 5 literally defines `ρ(x) = Cov(x_t, x_{t−1})`
— a covariance — but the narrative (entry at ρ=0, full exit at |ρ|=1, the discount
`1−|ρ|`) is coherent only for a bounded correlation; under the literal covariance
reading `1−|ρ|` can go negative and the metric sign-flips. We judge the correlation
reading to be the only internally coherent one and treat Eq 5 as a notational error
for the lag-1 autocorrelation *coefficient*; we nonetheless test the covariance
variant for completeness (it never helps; §4.1). (ii) By footnote 10, `ρ` is
estimated over the *entire* explanatory period, so the "dynamic positioning"
narrative (cutting exposure as ρ rises intra-period) is not what is actually tested.

## 3. Data and method

A survivorship-bias-free equity test cannot be run on free data — itself a finding
(§4.6). We use three universes:

1. **Survivor equity** (biased baseline): today's S&P 500 back-filled via yfinance,
   monthly returns, over the paper's three 11-year windows. Benchmark: `^GSPC`; a
   total-return robustness check with `^SP500TR` is in §4.7. Risk-free: `^IRX`.
2. **Partial-delisted equity** (RQ1): point-in-time membership from
   `fja05680/sp500` (1996–2019), priced from the Kaggle "Huge Stock Market" dump.
3. **Survivorship-free crypto** (RQ3): the CoinMarketCap "crypto-markets" dump
   (2013–2018, 2,071 coins), which retains crashed-but-listed coins, allowing
   genuine point-in-time selection.

**Outcome metric and inference.** Following the anchor's own rank-correlation
design, the outcome throughout is the Spearman rank correlation between an
explanatory-window metric and a holding-window outcome, with bootstrap CIs where a
difference is claimed. Two caveats bind the *inference*: (a) stock returns are
cross-sectionally dependent within a window, so reported p-values are descriptive,
not strict inferential statements, and the stock-level bootstrap understates CI
width; (b) we report an FDR multiple-testing correction in §4.7. Frequency is
monthly for equity (matching the anchor's ~132-observation windows) and daily for
crypto. An early diagnostic constrains the analysis: at daily frequency median |ρ|
≈ 0.04, so `(1−|ρ|) ≈ 0.96` and the predictive term is inert by construction; we
therefore evaluate the equity claim at its own monthly frequency.

## 4. Results

### 4.1 RQ2 — the autocorrelation term cannot change the ranking (load-bearing)

At monthly frequency the estimated lag-1 autocorrelation is small: median |ρ| ≈
0.076, so the discount `(1−|ρ|) ≈ 0.92` acts as a *near-constant multiplier* across
stocks. A near-constant multiplier cannot materially re-order a cross-section — so
the paper's marketed cross-sectional "predictive" improvement is structurally
near-impossible, not merely empirically absent. The ablation confirms this: across
144 configurations (lags {1,2,3} × forms {linear `1−|ρ|`, squared `1−ρ²`} × methods
{correlation, covariance} × 3 windows × 4 investor types), the maximum absolute
change in out-of-sample rank correlation from adding the term is **0.045** (mean
0.004). For the paper's exact specification (lag-1, linear, correlation), a
stock-level bootstrap (1,000 resamples) gives an adjusted-minus-base difference
whose 95% CI **includes zero in all 12 window×investor cells**. This is our
highest-confidence result and, uniquely, requires no delisted data.

*Scope.* This refutes the *cross-sectional rank* claim, which the anchor makes via
its Tables 1–6 (the ρ-adjusted "predictive" metric is reported there as a
standalone predictive column). The anchor also gestures (its Figure 3) at a
distinct *defensive-timing* role for ρ — attenuating exposure ahead of systemic
events. That is a time-series claim a cross-sectional rank correlation cannot
capture; we do not test it, and flag it as future work. Our result should be read
as: the term adds no cross-sectional predictive content and, given ρ's magnitude,
structurally cannot.

### 4.2 No robust predictive edge

On the survivor universe the predictive metric's out-of-sample rank correlation
with the one-year holding return is sign-unstable: for the risk-averse profile it
is +0.14, −0.11, −0.42 across the 1978–89, 1988–99, 1998–2009 windows, and it flips
between risk-averse (−0.42) and risk-seeking (+0.28) within 1998–2009. The Sharpe
foil is likewise unstable (+0.13, −0.12, −0.30). The anchor argues (p. 917) that
inconsistency across investor types is *expected* because it offers a per-investor
framework, not a universal optimum. We grant this; our point is narrower and
survives it: there is no *consistent positive* out-of-sample correlation for *any*
fixed investor type across windows, and the classical foil is no worse — so the
paper's "significant predictive, MPT-sparse" pattern does not replicate robustly.

### 4.3 RQ1 — survivorship moves the predictive correlation (engaging Appendix A)

Appendix A concedes survivorship inflates the *explanatory* correlations but claims
it cannot explain the *out-of-sample* behaviour. We test that: adding to the
survivor universe the point-in-time members that later *left the index* (those the
free data retains) shifts the *predictive* out-of-sample rank correlation **down**
in every investor type (risk-neutral +0.16 → +0.02; prospect −0.24 → −0.36;
risk-averse −0.36 → −0.45; risk-seeking +0.30 → +0.24). The direction contradicts
the Appendix A claim that survivorship leaves the predictive result untouched.

Two honesty constraints bound this. First, it is a **lower bound**: free equity
sources retain essentially no firms that actually stopped trading (§4.6), so the
added names are index-*leavers that survived*, not go-to-zero failures. Second, we
confirmed that **0 of 104** added firms stop trading within the holding window, so
a −100%-terminal-return convention for in-window deaths cannot change the result —
there are no such deaths in the free data to score. The true survivorship effect is
therefore larger than measured and, on free equity data, unmeasurable in full.

### 4.4 RQ3 — no downside-forecasting advantage over variance (clean regime)

In the survivorship-free crypto universe (top-100 by market cap as of 2017-06-30,
n=69 with sufficient history; explanatory 2016-07→2017-06, holding 2017-07→2018-06,
spanning the 2017 bull and 2018 crash), the explanatory downside-LPM forecasts
holding-period maximum drawdown with rank correlation 0.40, versus 0.34 for plain
variance. The paired-bootstrap difference is **+0.066, 95% CI [−0.065, +0.211]** —
straddling zero (computed and saved in `run_crypto_rq3.py`). The partial-moment
downside metric does not robustly beat variance at its supposed specialty; the two
are near-collinear in a single drawdown regime, which is itself part of the point.
A minimal economic-significance panel (decile long/short) is consistent: a
LPM-sorted book separates future drawdown strongly (top-minus-bottom +0.22), but so
would any volatility proxy. Contrary to H3, the UPM/LPM ratio *does* correlate with
holding returns here (ρ=0.31; decile return spread +0.8%/day) — we attribute this
to momentum in the 2017–18 regime, not a risk/return separation. Confidence is
low-to-medium: one extreme window, n=69.

### 4.5 RQ4 — learned parameters do not reliably generalise

Grid-searching `(q, n)` to maximise in-sample rank correlation on a training window
and applying it to a later window: training 1978–89 → testing 1988–99 learned
`(0.5, 2.0)` with in-sample +0.15 but out-of-sample **−0.28** (a sign inversion),
worse than the best hand-set preset (+0.27); training 1988–99 → testing 1998–2009
learned `(0.75, 0.1)` and did generalise (+0.33 → +0.30). The learned optimum is
unstable across eras (n=2.0 vs n=0.1). With only two walk-forward pairs and one
inversion, we state this as *suggestive of curve-fitting rather than a stable
learnable structure*; we note the alternative reading — genuine regime-dependence,
which the anchor itself embraces — is not excluded by two pairs.

### 4.6 A structural finding: the audit ecosystem paywalls the failure tail

Genuinely survivorship-bias-free equity price data is a paid product (CRSP,
Norgate). Free reconstruction fails at the *price* layer, not the *membership*
layer: point-in-time S&P 500 membership is freely recoverable (`fja05680/sp500`),
but free price sources systematically omit firms that went to zero — the Kaggle dump
retains none of Lehman, WaMu, Bear Stearns, Enron, Fannie, or Freddie, and only
15–27% of index-removed firms in any era; yfinance silently drops delisted tickers.
The observations most needed to falsify a downside-risk claim are precisely the ones
paywalled or dropped, so the cost of a rigorous audit is anti-correlated with the
claim's checkability. This is a portable lesson for budget-constrained replication
and a co-equal contribution of this paper.

### 4.7 Robustness

- **Total-return benchmark.** Re-running the predictive-vs-holding rank
  correlations with `^SP500TR` (total return) instead of `^GSPC` (price index),
  for the windows `^SP500TR` covers (1988–99, 1998–2009), moves every correlation
  by **≤ 0.027** (max across all cells). The price-vs-total-return mismatch does not
  affect any conclusion.
- **Multiple testing.** Applying a Benjamini–Hochberg FDR correction across the
  78-correlation family from the three-window run: 51 are significant at raw p<.05
  and **all 51 survive** at q<.05. The significant correlations (both the positive
  survivor-universe ones and the negative ones) are robust to multiplicity; the RQ2
  null is, if anything, strengthened by having searched 144 configurations.
- **Cross-sectional dependence.** We treat reported p-values as descriptive; the
  stock-level bootstrap ignores within-window return dependence and thus understates
  CI width. This does not affect the RQ2 verdict (a null made *more* conservative by
  wider CIs) but means the positive survivor-universe correlations should be read as
  effect sizes, not strict significance claims.

## 5. Discussion and limitations

The four results point one way, with clearly unequal weight. RQ2 is decisive and
robust: the marketed autocorrelation innovation cannot change the cross-sectional
ranking. RQ1, RQ3, and RQ4 corroborate — survivorship moves the predictive
correlation, the metric does not beat variance at downside forecasting in a clean
regime, and its parameters do not reliably generalise — but each is individually
qualified (a lower bound; a single small crypto window; two walk-forward pairs).

**What we do not claim.** Every result is a rank correlation. We therefore establish
that the metric's cross-sectional *rank-predictive* content does not survive audit;
we do *not* test tradable economic value net of turnover and transaction costs,
which can diverge from rank correlation in both directions (Novy-Marx & Velikov,
2016) and is a distinct question. We also do not test ρ's defensive-timing claim
(§4.1). **Fairness of the audit.** RQ2 is a design-internal ablation — it holds the
data fixed and toggles the term — and so is immune to the benchmark-proxy objection;
the `^GSPC`/CRSP substitution (shown immaterial in §4.7) and the today's-survivors
universe bear only on RQ1/§4.2, which we already frame as directional. Remaining
limitations: RQ1 is a lower bound because free equity data lacks the failure tail;
RQ3 rests on one extreme regime; and inference ignores cross-sectional dependence.
A paid-data RQ1, additional crypto windows for RQ3, and a defensive-timing test of ρ
are the natural next steps.

## 6. Conclusion

Across replication, ablation, a survivorship-free regime, and a learned-parameter
test, we find no robust cross-sectional out-of-sample rank-predictive content for
the partial-moment metric as specified, and its headline autocorrelation innovation
cannot change the ranking. The honest verdict on the cross-sectional predictive
claim is negative — which, for a claim resting on a survivorship-prone universe and
a structurally inert adjustment term, is the more useful answer.

---

## Appendix — reproducibility

- Metric core and tests: `src/partial_moments.py`, `tests/test_partial_moments.py` (12/12).
- Equity replication (3 windows): `src/run_biased_arm.py`.
- RQ1 directional + delisting sensitivity: `src/run_partial_equity_rq1.py`.
- RQ2 ablation: `src/run_rq2_ablation.py`.
- RQ3 crypto (with committed bootstrap + economic panel): `src/run_crypto_rq3.py`.
- RQ4 learned params: `src/run_rq4_learned_params.py`.
- Robustness (total-return benchmark, FDR): `src/run_robustness_checks.py`.
- Data: yfinance (survivor prices, `^GSPC`, `^SP500TR`, `^IRX`); `fja05680/sp500`
  (point-in-time membership); Kaggle `borismarjanovic/price-volume-data-for-all-us-stocks-etfs`
  (equity) and `jessevent/all-crypto-currencies` (crypto).
- Decisions, gotchas, and dated findings: `context/`.

## Summary of verdicts

| RQ | Result | Confidence |
|----|--------|-----------|
| RQ2 | `(1−\|ρ\|)` term cannot change the ranking (≤0.045; 12/12 CIs include 0) | High |
| §4.2 | No consistent positive OOS correlation; sign-unstable; Sharpe no worse | Med–High |
| RQ1 | Survivorship correction shifts predictive OOS correlation down (lower bound) | Medium |
| RQ3 | No robust downside-forecast advantage over variance (+0.066, CI incl. 0) | Low–Med |
| RQ4 | Learned (q,n) do not reliably generalise (one OOS sign inversion of two) | Low–Med |
| §4.6 | Free survivorship-free equity prices unobtainable (structural) | High |

## References

Ang, J. S., & Chua, J. H. (1979). Composite measures for the evaluation of investment performance. *Journal of Financial and Quantitative Analysis, 14*(2), 361–384.

Bawa, V. S. (1975). Optimal rules for ordering uncertain prospects. *Journal of Financial Economics, 2*, 95–121.

Brown, S. J., Goetzmann, W. N., Ibbotson, R. G., & Ross, S. A. (1992). Survivorship bias in performance studies. *Review of Financial Studies, 5*(4), 553–580.

Brown, S. J., Goetzmann, W. N., & Ross, S. A. (1995). Survival. *Journal of Finance, 50*(3), 853–873.

Campbell, J. Y., & Thompson, S. B. (2008). Predicting excess stock returns out of sample. *Review of Financial Studies, 21*(4), 1509–1531.

Carhart, M. M., Carpenter, J. N., Lynch, A. W., & Musto, D. K. (2002). Mutual fund survivorship. *Review of Financial Studies, 15*(5), 1439–1463.

Elton, E. J., Gruber, M. J., & Blake, C. R. (1996). Survivor bias and mutual fund performance. *Review of Financial Studies, 9*(4), 1097–1120.

Fishburn, P. C. (1977). Mean-risk analysis with risk associated with below-target returns. *American Economic Review, 67*, 116–126.

Fishburn, P. C., & Kochenberger, G. A. (1979). Two-piece von Neumann–Morgenstern utility functions. *Decision Sciences, 10*, 503–518.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). …and the cross-section of expected returns. *Review of Financial Studies, 29*(1), 5–68.

Holthausen, D. M. (1981). A risk-return model with risk and return measured as deviations from a target return. *American Economic Review, 71*, 182–188.

Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision under risk. *Econometrica, 47*, 263–291.

Markowitz, H. (1959). *Portfolio selection: Efficient diversification of investments*. Yale University Press (Cowles Foundation Monograph No. 16; Ch. IX, semivariance).

Novy-Marx, R., & Velikov, M. (2016). A taxonomy of anomalies and their trading costs. *Review of Financial Studies, 29*(1), 104–147.

Sortino, F. A., & van der Meer, R. (1991). Downside risk. *Journal of Portfolio Management, 17*(4), 27–31.

Viole, F., & Nawrocki, D. (2016). Predicting risk/return performance using upper partial moment/lower partial moment metrics. *Journal of Mathematical Finance, 6*, 900–920.

Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies, 21*(4), 1455–1508.

White, H. (2000). A reality check for data snooping. *Econometrica, 68*(5), 1097–1126.
