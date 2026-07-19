# Post-2016 Downside / Tail-Risk Metrics — Survey for Choosing a Build Target

**Prepared 2026-07-12.** Read to decide which modern metric to *improve* (the 2016
anchor is now just motivation). Focus: cross-sectional downside/tail-risk predictors
buildable from **free daily data** on a **survivorship-free testbed** (crypto with
dead coins + point-in-time equity). Sources read per candidate; details verified
against JFE/RFS/JFQA records where possible.

## The pattern across all six

Every modern downside/tail-risk metric shares two properties that matter for us:
1. **Estimated on survivors over multi-year windows** — the stocks that actually
   crashed to zero (the true tail) are dropped *before* they can enter a portfolio.
   So the measured "crash premium" / "tail premium" is structurally survivorship-
   exposed, and its robustness is *actively contested* (Levi-Welch 2020; RMIT 2025;
   Nevrla) — several premia shrink or flip under penny-stock filters or replication.
2. **Nearly all run on free daily data** — none of the *core* cross-sectional
   results require intraday (even BPQ's headline is monthly-semibetas-from-daily).

**Implication for "a better metric":** the honest, high-value constructive angle is
not "a new formula that beats variance" (the evidence says downside asymmetry ≈
variance) — it is a **survivorship-robust / death-aware estimator** of a tail-risk
metric, validated on the one testbed these papers never used (dead-coin crypto +
PIT equity). That is genuinely novel, tractable solo in ~3 weeks, and honest.

## Candidate comparison

| Metric | Paper | Est. from daily? | Estimation cost | Survivorship exposure | Robustness status | Fit for a 3-wk solo "better metric" |
|---|---|---|---|---|---|---|
| **Left-tail risk (VaR/ES)** | Atilgan-Bali-Demirtas-Gunaydin 2020, JFE | Yes (trailing-yr daily) | **Low** (percentile of daily returns) | **Acute** — the measure *is* recently-crashed stocks, which then delist | Sign contested (Huang+ find opposite) | **Best** — simplest; death-aware version is a clean contribution |
| **Kelly-Jiang tail risk λ** | Kelly & Jiang 2014, RFS | Yes (Hill est. on cross-section) | **Low** (one Hill formula) | Moderate (λ from contemporaneous crashes) but pricing test needs long histories | Canonical baseline; well-understood | Great **baseline**; improvable as death-aware λ |
| **Crash sensitivity / LTD** | Chabi-Yo-Ruenzi-Weigert 2018, JFQA | Yes (copula on daily) | **High** (copula, tail-data-hungry) | **High** — LTD needs survivors over multi-yr window | Premium contested | High upside, but heavy + data-hungry (risky for short crypto histories) |
| **Multivariate crash risk (MCRASH)** | Chabi-Yo-Huggenberger-Weigert 2022, JFE | Yes (nonparam co-exceedance) | High | **Severe** | **Actively contested** (RMIT 2025: "no robust premium") | Highest prestige/upside, highest execution risk |
| **Realized semibetas** | Bollerslev-Patton-Quaedvlieg 2022, JFE | Yes (monthly-from-daily is their *primary* spec) | Medium | Moderate (>$5 filter, S&P500 intraday) | βᴺ robust; four-way split contested (only βᴺ replicates) | Strong — mechanism (limits-to-arbitrage) is falsifiable in frictionless crypto |
| **GDA downside betas** | Farago-Tedongap 2018, JFE | Yes (daily + EGARCH) | Medium (EGARCH + 5 betas) | Moderate | Fragile internationally | Structural; sign-restriction falsification possible |
| **Down-beta (skeptic ref)** | Levi & Welch 2020, RFS | Yes | Low | Low (within-stock OOS) | The skeptic: "useful for neither hedging nor pricing" | Provides the *skeptical method*, not a build target |

## The three viable build targets (ranked for our constraints)

**1. Left-tail risk (Atilgan et al. 2020) — RECOMMENDED for a 3-week short paper.**
- Metric: trailing-year VaR₅ / ES₅ from daily returns. Trivial to compute.
- Why: the *most* survivorship-exposed metric (it literally measures recent crashes),
  so our dead-coin testbed is its natural habitat; the "left-tail momentum" claim
  (crashed stocks keep underperforming) is directly testable and its sign is contested.
- **Better metric = death-aware left-tail risk:** an ES/VaR estimator that assigns
  delisted assets their true terminal loss (−100%), tested against the naive
  survivor-only version and against variance for forecasting *future drawdown* OOS.
- Baselines: Kelly-Jiang λ, plain variance. Honest even if it only ties variance.

**2. Crash sensitivity / MCRASH (Chabi-Yo et al.) — higher prestige, higher risk.**
- Metric: lower-tail dependence via copula / co-exceedance.
- Why: severe survivorship exposure + *actively contested* robustness → a
  survivorship-free replication is publishable. But copula/LTD estimation is
  tail-data-hungry; short crypto histories strain it. Risky for 3 weeks solo.

**3. Realized semibetas (BPQ 2022) — cleanest mechanism test.**
- Metric: four signed semibetas (daily analogue is their primary spec).
- Why: their βᴺ premium is attributed to *short-sale/limits-to-arbitrage*; in
  frictionless, shortable crypto that premium should vanish — a falsifiable,
  novel prediction. Medium build cost.

## Recommendation

For a solo ~6-page paper in ~3 weeks, targeting MFE signaling + ICAIF/arXiv:
**improve the left-tail-risk measure (Atilgan et al. 2020) into a survivorship-robust
("death-aware") estimator, validated on the dead-coin crypto + PIT-equity testbed,
with Kelly-Jiang λ and variance as baselines.** Lowest execution risk, most direct
use of our existing testbed, and a genuine constructive contribution (a better
*estimator*, honestly evaluated) rather than a formula fishing for alpha.
