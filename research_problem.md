# Research Problem Report — Project B

## Is the partial-moment "predictive edge" real, or an artifact of survivorship bias and a curve-fit autocorrelation term?

**Type:** Replication + adversarial audit + methodological correction
**Anchor paper:** Viole & Nawrocki (2016), *Predicting Risk/Return Performance
Using Upper Partial Moment/Lower Partial Moment Metrics*, Journal of Mathematical
Finance 6, 900–920.
**Status:** Problem statement (pre-plan). Written 2026-07-12.

---

### 1. One-sentence problem

Viole & Nawrocki claim a partial-moment risk metric predicts out-of-sample
performance where classical (Sharpe/VaR) metrics cannot — but they test it on
*surviving* S&P 500 firms and bolt on a single-lag autocorrelation term with no
validation; this project asks whether the predictive edge survives once
survivorship bias is removed and the autocorrelation term is subjected to a
proper ablation.

### 2. Why this is a worthy problem (not a cookie-cutter project)

Re-deriving Sharpe or a textbook risk metric is exactly the "simple cookie
cutter" work Bianco dismisses. Auditing a *published predictive claim* for a
bias that could have manufactured the entire result is a different act — it is
the "find the problems not mentioned in the paper" move Halperin describes, and
survivorship bias is one of the most consequential, most cited, and most
under-checked biases in empirical finance. More than five people care about
whether downside-asymmetry risk metrics actually predict tail risk; that clears
Halperin's relevance bar.

The intellectual core is measurement integrity, which is where the reference set
repeatedly points (model validation, diagnostics, "the ill-conceived bias in how
we measure risk"). It plays directly to an empirical-experiment skill set and is
the more tractable of the two projects — lower infrastructure risk, cleaner data
story, sharper falsifiable claim.

### 3. What the anchor paper actually claims

The paper is fundamentally a **critique of how risk is measured**, plus a
proposed replacement.

- **The critique ("ill-conceived bias"):** classical benefit/cost metrics
  (Sharpe, Sortino, Treynor, Jensen's alpha, VaR) use a single observation as
  *both* the risk and the return term — but a return cannot be simultaneously a
  benefit and a cost. They also assume stationarity, which markets violate.
- **The replacement — partial moments:**
  - Lower Partial Moment: `LPM(n,h,i) = (1/T) Σ max(0, h − Rᵢₜ)ⁿ`
  - Upper Partial Moment: `UPM(q,l,i) = (1/T) Σ max(0, Rᵢₜ − l)^q`
  - **Explanatory metric:** `UPM(q,y,x) / LPM(n,y,x)`, evaluated against a
    *double non-stationary benchmark* (the asset-class benchmark and the
    risk-free rate, taking the greater of the two at each time t).
  - Degree exponents encode investor psychology: Risk-Averse (q:n = 0.25),
    Prospect-Theory (0.44, per Kahneman–Tversky), Risk-Neutral (1.0),
    Risk-Seeking (2.0).
- **The predictive twist:** multiply the explanatory ratio by a bubble/instability
  discount, `(UPM/LPM)·(1 − |ρ(x)|)`, where `ρ(x) = Cov(xₜ, xₜ₋₁)` is the
  **one-period autocorrelation**. Logic: high autocorrelation signals an unstable
  (bubble-like) run; the strategy is long-only, entering at ρ = 0 and fully
  exiting at |ρ| = 1.
- **Empirical design:** ~300 **surviving** S&P 500 companies over three
  overlapping 11-year windows (1978–89, 1988–99, 1998–2009); rank correlations
  between the metric and subsequent 1- and 2-year returns; CRSP market-cap index
  as the systemic benchmark.
- **Claim:** the explanatory metric shows explanatory-period correlation but no
  out-of-sample predictive power; the autocorrelation-adjusted *predictive*
  metric shows significant out-of-sample correlation; classical MPT metrics are
  sparse on both.

### 4. The soft spots — "problems not mentioned in the paper"

1. **Survivorship bias, front and center.** The universe is "≈300 *surviving*
   S&P 500 companies." Conditioning on survival over 11–31 years mechanically
   favors firms that avoided catastrophic downside — precisely the tail the LPM
   is meant to capture. This bias can *manufacture* a downside-risk predictive
   result. It is the central vulnerability.
2. **The autocorrelation term is unvalidated and arbitrary.** Why one-period lag?
   Why the linear `(1 − |ρ|)` form? There is no ablation showing the term adds
   predictive power beyond the base ratio, and no test that it is signal rather
   than curve-fit to these particular windows.
3. **Overlapping test windows.** The three periods overlap (1988–99 shares years
   with both neighbors), inflating apparent robustness.
4. **Rank-correlation-only, no economic test.** Predictive *rank correlation* is
   not the same as a tradable, cost-aware edge. No transaction costs, no turnover.
5. **Hand-set exponents and pre-2009 data.** The (q, n) degrees are fixed by
   assumption, and nothing after the GFC is tested.

### 5. Research questions and hypotheses

- **RQ1 (survivorship):** Does the predictive edge survive on a
  survivorship-bias-free universe (point-in-time constituents, or a delisting-
  inclusive dataset, or crypto — which sidesteps the bias differently)?
  - *H1:* A material part of the reported predictive correlation is attributable
    to survivorship bias and shrinks on a bias-free universe.
- **RQ2 (the autocorrelation term):** Under ablation, does the `(1 − |ρ|)`
  adjustment add out-of-sample predictive power over the base UPM/LPM ratio?
  - *H2:* The term adds little robust signal; its apparent value is period-
    specific.
- **RQ3 (does asymmetry beat variance at all):** Independent of the paper's exact
  construction, do partial-moment metrics predict future **downside/drawdown**
  better than variance/Sharpe out of sample?
  - *H3:* Partial-moment metrics predict *downside* risk modestly better than
    variance, but predict *returns* no better — i.e. the value is in risk
    forecasting, not return forecasting.
- **RQ4 (ML extension, optional):** Can the (q, n) degree parameters be *learned*
  from data rather than hand-set, and does a learned parameterization generalize
  out of sample?

Each hypothesis is falsifiable both ways. "The edge was survivorship bias" and
"the edge is real for downside forecasting only" are both strong, honest,
interview-ready findings.

### 6. Data

- **Bias-free equity option:** point-in-time S&P 500 (or broad-market)
  constituents *including delisted firms* — the correct fix for RQ1. If a
  survivorship-free source is not obtainable within scope, this constraint is a
  gating decision to resolve early.
- **Crypto option:** liquid crypto returns sidestep classical equity
  survivorship bias (different failure mode) and provide a clean second
  out-of-sample regime — useful as a robustness universe, not a replacement.
- **Frequency:** daily to monthly; the paper works at coarse horizons (1–2 year
  holding), so daily data with monthly rebalancing is ample and keeps scope small.

### 7. Methodology (sketch — full plan is a later artifact)

1. Re-implement `LPM`, `UPM`, the explanatory ratio, and the autocorrelation-
   adjusted predictive metric exactly as specified (equations 1–7), including the
   double non-stationary benchmark.
2. Reproduce the paper's rank-correlation result on a *surviving* universe first
   (sanity check that the implementation matches the claim).
3. Swap in the **survivorship-bias-free** universe; re-measure (RQ1).
4. **Ablate** the autocorrelation term and test alternative lags/forms (RQ2).
5. Reframe the target as future downside/drawdown and benchmark partial-moment
   metrics against variance/Sharpe out of sample (RQ3).
6. (Optional) Learn (q, n) via cross-validation and test generalization (RQ4).
7. Non-overlapping, walk-forward windows throughout; report with and without
   transaction costs where an economic test is implied.

### 8. Success criteria (Rule 4)

Succeeds if it delivers a **clean, honest verdict** on whether the partial-moment
predictive edge is real, with:
- a like-for-like comparison on surviving vs. survivorship-free universes;
- a proper ablation isolating the autocorrelation term's contribution;
- a separation of the metric's *risk-forecasting* value from its *return-
  forecasting* value.

Success is a defensible answer, not a positive one.

### 9. Risks and feasibility (4 months, self-built, part-time)

- **Main risk:** obtaining a genuinely survivorship-bias-free equity universe can
  be hard/costly. *Mitigation:* resolve the data source in week one; fall back to
  crypto + a delisting-inclusive academic dataset if needed; make the data
  limitation explicit rather than silent (Rule 12).
- **Low infrastructure risk:** this is econometrics/stats, not RL — the reason it
  is the more tractable of the two projects.
- **"Not exciting enough" risk:** framed as a metric re-implementation it is dull;
  framed as *auditing a published claim for a result-manufacturing bias* it is a
  sophisticated quant move. Framing is the difference.

### 10. What it signals for an MFE application

Demonstrates: understanding of survivorship bias and out-of-sample discipline
(the two things that separate real empirical finance from notebook exercises);
comfort with the utility-theoretic and downside-risk literature; and the
skeptical, validation-first instinct the reference set repeatedly rewards. The
survivorship-bias audit is a concrete, nameable skill an admissions reader (and
a future interviewer) will immediately recognize.

### 11. Open questions to resolve before the implementation plan

- Is a survivorship-bias-free equity universe obtainable within scope, or is
  crypto the primary universe?
- Which horizon and rebalancing frequency?
- Include the learned-(q,n) extension (RQ4) in scope, or hold it as a stretch goal?
