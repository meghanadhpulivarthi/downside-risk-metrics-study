# Literature Review — Partial-Moment Metrics, Survivorship Bias, and Out-of-Sample Predictability

**Prepared 2026-07-12** as the evidence base for the audit of Viole & Nawrocki
(2016). Every entry was independently verified against publisher records
(Oxford Academic, JSTOR, Wiley, RePEc/IDEAS, PM Research) in a citation-verification
pass — **all 17 external works exist; corrections applied are noted inline.** This
review situates the audit in three literatures and doubles as the verified
bibliography for `writeup/partial_moment_audit.md`.

---

## 1. Downside risk and partial moments — the tradition the anchor builds on

The anchor's UPM/LPM metric is the latest in a fifty-year line of work replacing
variance with below-target (and above-target) partial moments.

- **Markowitz, H. (1959). *Portfolio Selection: Efficient Diversification of
  Investments*. Yale University Press (Cowles Foundation Monograph No. 16),
  Ch. IX.** *[corrected: publisher is Yale University Press, not Wiley; Wiley
  issued a later edition.]* The original proposal of **semivariance** as an
  alternative to variance — the conceptual ancestor of all lower-partial-moment
  downside measures. Establishes that the object Viole & Nawrocki generalise is
  seventy years old; their novelty is the ratio and the ρ term, not downside risk.

- **Bawa, V. S. (1975). Optimal rules for ordering uncertain prospects. *Journal
  of Financial Economics, 2*(1), 95–121.** Formalises **lower partial moments as a
  stochastic-dominance criterion**, giving LPM its axiomatic (utility-free)
  footing. Relevant because it means a UPM/LPM *ranking* claim inherits
  well-understood order conditions — against which the paper's empirical ranking
  can be judged.

- **Fishburn, P. C. (1977). Mean-risk analysis with risk associated with
  below-target returns. *American Economic Review, 67*(2), 116–126.** The canonical
  **utility-theoretic justification** for below-target (α-t) LPM risk. The anchor's
  "double non-stationary benchmark" (the target `y`) is a Fishburn target-return
  construct; auditing the metric means auditing the target choice.

- **Fishburn, P. C., & Kochenberger, G. A. (1979). Two-piece von Neumann–Morgenstern
  utility functions. *Decision Sciences, 10*(4), 503–518.** Establishes the
  **asymmetric (kinked) utility** that motivates treating upside and downside
  moments with different degree exponents — i.e. the paper's q≠n "investor type"
  parameters.

- **Holthausen, D. M. (1981). A risk-return model with risk and return measured as
  deviations from a target return. *American Economic Review, 71*(1), 182–188.**
  Extends Fishburn to measure **both return and risk relative to a target**,
  directly underpinning a UPM/LPM *ratio* of the kind the anchor uses.

- **Kahneman, D., & Tversky, A. (1979). Prospect theory: An analysis of decision
  under risk. *Econometrica, 47*(2), 263–291.** The behavioural foundation for
  loss aversion; the loss-aversion coefficient λ ≈ 2.25 is the source of the
  anchor's "Prospect" degree preset (q ≈ 1/2.25 ≈ 0.44). Grounds §2 of the audit.

- **Sortino, F. A., & van der Meer, R. (1991). Downside risk. *Journal of Portfolio
  Management, 17*(4), 27–31.** Coined "downside risk" in the practitioner
  literature and introduced the **Sortino ratio** (an LPM-based Sharpe analogue) —
  the closest established cousin of the anchor's metric and a natural baseline.

- **Ang, J. S., & Chua, J. H. (1979). Composite measures for the evaluation of
  investment performance. *Journal of Financial and Quantitative Analysis, 14*(2),
  361–384.** *[corrected: title is "Composite measure**s**" (plural).]* Proposes a
  performance measure substituting lower partial moments for standard deviation —
  an early, direct precedent for UPM/LPM performance evaluation.

**Takeaway for the audit:** the *downside-asymmetry* idea is well-founded and old;
what is new and untested in Viole & Nawrocki is (a) the specific UPM/LPM ratio
against a double non-stationary benchmark and (b) the `(1−|ρ|)` autocorrelation
discount. The audit correctly targets (b) (RQ2) and the *empirical* value of the
whole construct (RQ1, RQ3), not the settled theory of downside risk.

## 2. Survivorship bias in performance studies — the RQ1 foundation

This literature quantifies exactly the bias the anchor's ~300-surviving-firm
universe is exposed to, and grounds the direction/magnitude claims in RQ1 and §4.6.

- **Brown, S. J., Goetzmann, W. N., Ibbotson, R. G., & Ross, S. A. (1992).
  Survivorship bias in performance studies. *Review of Financial Studies, 5*(4),
  553–580.** *[corrected: "Goetzmann, W. **N.**"]* The foundational demonstration
  that **conditioning on survival inflates apparent performance and manufactures
  spurious persistence.** This is the theoretical warrant for RQ1's hypothesis and
  for treating a surviving-universe result as suspect by default.

- **Brown, S. J., Goetzmann, W. N., & Ross, S. A. (1995). Survival. *Journal of
  Finance, 50*(3), 853–873.** Shows that **even with no true skill, survival
  conditioning induces spurious persistence** — directly relevant to why the
  anchor's out-of-sample correlations require scrutiny, not just its in-sample ones.

- **Elton, E. J., Gruber, M. J., & Blake, C. R. (1996). Survivor bias and mutual
  fund performance. *Review of Financial Studies, 9*(4), 1097–1120.** *[corrected:
  title is "**Survivor** bias", not "Survivorship bias".]* Quantifies the upward
  return bias from excluding dead funds — an empirical benchmark for the magnitude
  of distortion RQ1 attempts to bound.

- **Carhart, M. M., Carpenter, J. N., Lynch, A. W., & Musto, D. K. (2002). Mutual
  fund survivorship. *Review of Financial Studies, 15*(5), 1439–1463.**
  Disaggregates survivorship into **look-ahead and back-fill** components and shows
  sample-construction choices materially move measured performance — the precise
  methodological hazard §4.6 documents for free equity data.

**Takeaway for the audit:** the survivorship literature says the anchor's design
is exposed to a first-order, well-quantified bias. It also frames the audit's own
constraint: our RQ1 is a *lower bound* precisely because the free data omits the
failure tail — the very effect this literature measures.

## 3. Out-of-sample predictability and data-mining — the audit's toolkit

This literature supplies the standards the audit applies: OOS evaluation, the
in-sample/OOS gap as an overfitting signal (RQ4), multiple-testing discipline
(§4.7 FDR), and the predictability-vs-tradability gap (§5 scope caveat).

- **Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical
  performance of equity premium prediction. *Review of Financial Studies, 21*(4),
  1455–1508.** The canonical demonstration that **most in-sample predictors fail
  out of sample**; establishes OOS evaluation as the default skeptical standard —
  the stance the entire audit adopts.

- **Campbell, J. Y., & Thompson, S. B. (2008). Predicting excess stock returns out
  of sample: Can anything beat the historical average? *Review of Financial
  Studies, 21*(4), 1509–1531.** The companion showing disciplined, economically
  restricted OOS forecasts can eke out modest gains — i.e. OOS protocol design
  matters, motivating the audit's walk-forward care.

- **White, H. (2000). A reality check for data snooping. *Econometrica, 68*(5),
  1097–1126.** The foundational test for whether a *best* model found by search
  beats a benchmark once **multiple testing** is accounted for — the formal warrant
  for RQ2's "best of 144 configurations still does nothing" framing and §4.7's FDR.

- **Harvey, C. R., Liu, Y., & Zhu, H. (2016). …and the cross-section of expected
  returns. *Review of Financial Studies, 29*(1), 5–68.** Argues the significance
  hurdle must be raised (t ≈ 3) to survive the factor-zoo multiple-testing problem —
  directly supports the FDR correction and the skeptical reading of "significant"
  survivor-universe correlations in §4.2/§4.7.

- **Novy-Marx, R., & Velikov, M. (2016). A taxonomy of anomalies and their trading
  costs. *Review of Financial Studies, 29*(1), 104–147.** Shows many statistically
  significant anomalies are eroded or eliminated by realistic transaction costs —
  the basis for the audit's §5 caveat that rank-predictive content ≠ tradable edge.

**Takeaway for the audit:** the negative-result methodology the audit uses
(OOS testing, adversarial search, multiplicity control, distinguishing statistical
from economic significance) is standard in this literature. The audit is applying
established finance-metascience tools to a claim that never faced them.

---

## Verification summary

| Cluster | Works | All real? | Corrections |
|---------|-------|-----------|-------------|
| Downside risk / partial moments | 8 | Yes | Markowitz publisher (Yale, not Wiley); Ang & Chua title "measures" (plural); minor issue numbers |
| Survivorship bias | 4 | Yes | Elton-Gruber-Blake title "Survivor" (not "Survivorship"); Brown 1992 "Goetzmann, W. N." |
| OOS predictability / data-mining | 5 | Yes | none material (author-name/typographic variants only) |

No fabricated references were found. The manuscript's reference list
(`writeup/partial_moment_audit.md`) has been updated with the corrections above.
