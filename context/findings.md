# Empirical Findings

Preliminary and confirmed results. Each flags its confidence and caveats. These
are hypotheses-in-progress until the full RQ pipeline verifies them — do not
quote as conclusions yet (Rule 12).

## 2026-07-12 — NEW PAPER pivot + KEY RESULT: death-aware left-tail risk

Pivoted from "audit of V&N 2016" to a constructive paper building on Atilgan et al.
(2020) left-tail risk. Target: ICAIF '26 short paper (deadline Aug 2) + arXiv; MFE
admissions signal. Testbed: dead-coin crypto (CMC dump) + PIT equity.

**KEY RESULT (outputs/*_death_aware_crypto), the paper's thesis:**
Left-tail ES5 forecasting future 90d drawdown, crypto, quarterly formation dates.
Partial Spearman (does each add beyond the other?):
  NAIVE survivor-selected design:  ES5 beyond vol = 0.10 (t=1.45, N.S.);
                                   vol beyond ES5 = 0.20 (t=3.5, sig).
  DEATH-AWARE point-in-time design: ES5 beyond vol = 0.148 (t=3.8, SIG);
                                    vol beyond ES5 = 0.11 (t=2.5).
=> In the survivor design, downside asymmetry looks redundant with variance (the
recurring "asymmetry ≈ variance" wall). Under death-aware (PIT) evaluation, left-
tail risk carries SIGNIFICANT incremental drawdown-forecasting power beyond vol —
and more than vol adds beyond it. **Survivorship bias MASKS the value of downside
asymmetry.** Also: death-aware mean realized drawdown 0.55 vs naive 0.47 (survivor
selection understates realized downside by ~7.6pp).
Confidence: MEDIUM (11-13 quarterly dates, naive t-stats, single asset class/dataset).
MUST solidify: monthly dates, test the DESIGN DIFFERENCE significance (not just each
leg), Newey-West/bootstrap, equity illustration, horizon/threshold robustness.

### 2026-07-12 SOLIDIFIED (monthly, 32 dates, Newey-West + block bootstrap) — thesis REVISED

Ran the hardened version (outputs/*_death_aware_crypto_solid). The "masking" flip
did NOT survive:
  naive  ES5-beyond-vol = 0.081 (t=3.03) — SIGNIFICANT (was t=1.45 at quarterly = underpowered)
  death  ES5-beyond-vol = 0.097 (t=6.17)
  DIFFERENCE death-naive = 0.016 (t=0.51, 95% CI [-0.047, +0.074]) — NOT significant.
Across the 12-config robustness grid the design difference is noise around 0
(-0.036 to +0.018). **RETRACT the "survivorship masks the value" thesis** — it was
a small-sample power artifact. (This is exactly the over-claim the solidification
test was designed to catch; caught it before writing.)

ROBUST results that DO survive:
1. **ES5 forecasts future drawdown incrementally BEYOND volatility, robustly** — all
   12 grid configs significant (t=2.9–9.7). Downside asymmetry is NOT redundant with
   variance for forecasting TAIL RISK (drawdown), in survivorship-free crypto.
   (Contrast: for forecasting RETURNS it was redundant — RQ3. The target matters.)
2. **Survivorship inflates the LEVEL of realized downside** — death-aware mean
   drawdown 0.557 vs naive 0.494 (+6.3pp, t=4.45), robust. Survivor selection
   understates how bad downside gets, but does NOT change the forecasting ranking.
New confidence: the two robust results are HIGH (robustness-gridded, HAC t). The
masking mechanism is REJECTED. Paper thesis must be rebuilt around #1 + #2.

## 2026-07-12 — AUTORESEARCH LOOP (paper spine): search overfits, holdout catches it

Built a Karpathy-style guarded metric-discovery loop (src/autoresearch/):
prepare_data.py (frozen panel + train/val/TEST splits), evaluate.py (frozen scorer,
never sees test except final), search_loop.py (greedy forward feature selection
maximizing VAL downside-forecast rank-corr), program.md. Testbed = survivorship-free
crypto; label = forward 90d max drawdown; score = mean per-date Spearman.

FIRST RESULT (outputs/*_autoresearch_loop): the loop "discovered" linear(var5,vol,skew):
  VAL 0.363 — beats best baseline (volatility 0.314) by +0.049. Looks like a win.
  LOCKED TEST 0.356 — WORSE than plain volatility (0.376). The gain evaporated.
=> Autonomous search manufactures a validation-beating metric whose edge is a
mirage; the locked holdout reveals it. This IS the paper: autonomous metric
discovery is a data-snooping engine; anti-snooping guardrails (locked holdout +
survivorship-free testbed) are necessary to tell signal from search noise.
(DES TEST 0.431 looks best on TEST but is WORST on VAL 0.232 — picking by TEST is
itself snooping; nice cautionary sub-point.)
Confidence: MEDIUM — single greedy path. MUST solidify: many searches (random/greedy
restarts) to show the VAL-gain -> TEST-decay is SYSTEMATIC; + equity cross-check.

## 2026-07-12 — Open-ended LLM autoresearch loop: first locked-test result

Built the faithful Karpathy structure (src/autoresearch/): program.md, agent-edited
metric.py, FROZEN run_metric.py (scores VAL, withholds val/test labels from the
metric) + evaluate.py, LOCKED final_eval.py (TEST once), analysis.ipynb, results.tsv.
An LLM researcher subagent ran 17 open-ended (nonlinear) experiments; VAL 0.314->0.338.

LOCKED-TEST (final_eval.py, touched once):
  discovered metric (downside-consensus rank-avg of var5/es10/lpm2/downside_dev x
    runup interaction): VAL 0.338 -> TEST 0.421 (t=20).
  baselines TEST: volatility 0.376, ES5 0.323, DES 0.431.
  => beats vol & ES5 on TEST, does NOT beat DES. "beats all baselines" = False.

NUANCE (vs the earlier greedy-linear run which gave TEST 0.356 < vol = clean mirage):
this LLM metric PARTIALLY transferred (beat vol OOS). And VAL-selection picked a
metric that is NOT the TEST-best (DES was best on TEST but WORST on VAL -> discarded).
=> single-search conclusions are UNRELIABLE (one search beats vol OOS, another
doesn't); "best on val" != "best on test". The real result requires a SYSTEMATIC
MULTI-RUN study: many searches, characterize the distribution of VAL gain vs
TEST transfer, and whether VAL-selection reliably beats simple baselines OOS.
Harness verified end-to-end. Next: the multi-run study (paper's core evidence).

## 2026-07-13 — Paper revised + RE-REVIEWED: Major -> Minor, fixes applied

Revised icaif_paper.md addressing all Round-1 criticals; re-review (verification mode)
independently confirmed all 4 CRITICAL + 3/4 MAJOR fully addressed, moved decision
Major -> MINOR REVISION. Re-review caught one real table error (crypto-2016 VaR5
mislabeled ns, should be †) + minor items. All fixed: corrected Table 2 crypto-2016
column with exact values (vol 0.623 highest; var5 0.611†; es5 0.623 ns; downside_dev
0.615 ns; pricing metrics all †), sourced the 43-54% figure, noted ES5 partial-corr
supersession. Paper now at submission-quality draft. Title: "Volatility Is Hard to
Beat: A Survivorship-Free Benchmark of Downside-Risk Metrics and the Limits of LLM
Metric Discovery." Remaining (future/non-blocking): cost-aware economic test,
faithful (non-adapted) metric implementations, more markets.

## 2026-07-13 — REVISION v2: significance tests + calm regimes (post peer-review)

src/autoresearch/run_multi_testbed_v2.py. Added block-bootstrap 95% CIs on
(rho_metric - rho_vol) per bed, and calm-regime beds. 5 test beds now:
crypto_2016_calm, crypto_2017_bull, crypto_2018_crash, equity_1994_99_calm (n=66),
equity_2005_09_gfc (n=54).
SIGNIFICANT wins over volatility (CI excludes 0): ONLY downside_dev (+0.025) and
var5 (+0.045) on crypto_2017_bull. Nothing else, anywhere.
On BOTH equity beds, volatility SIGNIFICANTLY BEATS downside_dev/es5/var5 (crypto
edge REVERSES on equity). All pricing metrics (VN, semibeta, down_beta, LTD, gda)
significantly WORSE than vol on ~every bed. Hill/Kelly-Jiang is NaN per-asset
(it's an aggregate measure — drop it, per domain reviewer).
=> HONEST REVISED CLAIM: no downside-risk metric CONSISTENTLY or GENERALIZABLY
beats volatility; the only significant wins are small, crypto-bull-specific, and
reverse on equity. Resolves the over-claim: report the crypto wins openly, show
they don't generalize. Construct-mismatch acknowledged: pricing metrics repurposed
as drawdown forecasters. Peer-review criticals (significance + calm regime) addressed.

## 2026-07-13 — MULTI-TEST-BED benchmark (3 beds x 10 metrics): headline robust

src/autoresearch/run_multi_testbed.py. Added 3 metrics (Levi-Welch down_beta,
Chabi-Yo LTD, Farago-Tedongap vol-down proxy) + TWO more test beds beyond crypto-2018.
TEST drawdown-forecast Spearman:
  metric        crypto2017  crypto2018  equityGFC(n=21707)
  volatility       0.355      0.376      0.564  <- baseline
  downside_dev     0.379      0.410      0.535
  var5             0.400      0.362      0.540
  es5              0.351      0.327      0.522
  down_semibeta    0.249      0.309      0.487
  down_beta(LW)    0.093      0.151      0.344
  ltd_crash(CY)   -0.154     -0.285     -0.046
  gda_voldown(FT) -0.105     -0.050     -0.321
  vn_ratio        -0.276     -0.346     -0.424
  hill_tail(KJ)   -0.078     -0.071       nan
Beats vol: crypto17 {downside_dev,var5}; crypto18 {downside_dev}; equityGFC NONE.
=> Across 3 independent test beds (2 crypto regimes + equity through 2008 crash),
NO novel published metric beats volatility; only downside_dev (semi-vol) beats it on
crypto and it FAILS to generalize to equity (0.535<0.564). On equity, volatility is
the single best of all 10. Addresses the "one test set isn't enough" objection.

## 2026-07-13 — ABLATIONS: headline robust to horizon and blow-up threshold

src/autoresearch/run_ablations.py.
HORIZON (TEST drawdown-forecast Spearman): 30d vol 0.384 / 90d vol 0.376; only
downside_dev (semi-vol) edges vol (0.406/0.409); all published metrics below vol
at both horizons (VN -0.34, Hill -0.08, semibeta ~0.31-0.35, ES5/VaR5 <vol).
(h=180 infeasible: test formation +180d exceeds 2018-11 data end -> NaN. Data limit.)
BLOW-UP THRESHOLD (TEST AUC): -50%/-70%/-90% -> vol 0.59/0.69/0.95, BEST at all;
nothing beats vol.
=> "published metrics don't beat volatility OOS" is robust to horizon + threshold.
downside_dev edges vol only on drawdown-rank (it's semi-vol) and loses on blow-up.
Remaining ablations for full paper (noted as future/limitations): trailing window,
universe size (TOP_N), survivorship on/off for the full benchmark (death_aware study
already showed survivorship shifts drawdown LEVEL not forecasting RANK), and
multiple test periods (currently one, 2018-dominated).

## 2026-07-13 — PUBLISHED-METRIC BENCHMARK: none beat volatility OOS (headline)

src/autoresearch/run_published_baselines.py. Published downside/tail metrics computed
per-coin (daily-crypto adaptations) and scored on the survivorship-free LOCKED TEST,
for drawdown-forecast (Spearman) and blow-up (AUC):
  metric            TEST rho   TEST blowup-AUC
  Viole-Nawrocki    -0.35      0.31   (anti-predictive / worse than random!)
  Kelly-Jiang Hill  -0.07      0.50   (coin flip)
  BPQ down-semibeta  0.31      0.60
  Atilgan ES5        0.32      0.58
  Atilgan VaR5       0.36      0.61
  downside_dev       0.41      0.64
  VOLATILITY         0.38      0.68   <- baseline, WINS overall
=> Every published metric underperforms plain volatility OOS on both tasks. Only
downside_dev edges vol on drawdown-rank (it's semi-vol) but loses on blow-up AUC.
V&N's own ratio is anti-predictive — closes the loop with the original audit.
Caveat: daily-crypto adaptations, some published metrics target returns not drawdown;
fair survivorship-free risk-forecasting benchmark, not exact replications.

THE PAPER: a survivorship-free benchmark showing published downside-risk metrics,
ML models, and automated metric discovery ALL fail to beat plain volatility out-of-
sample. Volatility is the hard-to-beat baseline; complexity overfits. Strong, honest,
novel, ICAIF/MFE-appropriate. Time to WRITE.

## 2026-07-13 — BLOW-UP PREDICTOR: volatility beats the ML model OOS too

src/autoresearch/run_blowup_predictor.py. Label: forward 90d return <= -80% (delist =
-100%); base rate ~1-2%. Fit on TRAIN, select by VAL AUC (grad_boost 0.733 > logit
0.689; but vol-alone VAL AUC 0.755 already beats both!). LOCKED TEST:
  grad_boost model AUC 0.590, lift 1.3x
  volatility ALONE AUC 0.680, lift 2.5x  <- simple baseline WINS again
  downside_dev 0.635; trailing_dd 0.459.
=> The multi-feature ML blow-up model OVERFITS and loses to plain volatility OOS.
BUT the honest useful positive: volatility is a genuinely useful blow-up flag
(AUC 0.68, 2.5x top-decile lift), survivorship-free.

ROBUST META-FINDING across the ENTIRE project (audit, DES, death-aware, LLM loop
18/50, 20k directed sweep, blow-up ML): simple volatility is a hard-to-beat downside
baseline; hand-crafted metrics, modern measures, automated search, and ML all overfit
and fail to reliably beat it out-of-sample. This is the paper. Continuing to hunt new
avenues for a "win" would itself be hypothesis-mining. Recommendation: WRITE the paper.

## 2026-07-12 — DIRECTED SWEEP (20k configs): harder search -> WORSE out-of-sample

src/autoresearch/sweep.py: 5 parameterized families x 4000 configs = 20000, scored on
VAL + locked TEST. Baseline vol: VAL 0.314, TEST 0.376.
  GLOBAL best VAL 0.3557 (gated family) -> locked TEST 0.3734 — BELOW vol (0.376)!
  best-of-N: VAL climbs 0.27->0.356 with intensity; TEST flat ~0.36-0.38.
  val-winners beat vol on TEST only 42.7% (worse than coin flip).
=> Cranking search intensity to 20k pushed VAL to its ceiling but the val-selected
metric UNDERPERFORMS volatility on the locked test. Decisive: "making the loop
succeed by searching harder" manufactures overfitting, not a useful metric.
Nuance: some FAMILIES modestly beat vol on test (consensus_boost 0.380, tilt_blend
0.384) — a small honest signal in the multi-downside-consensus DIRECTION, IF
validated pre-specified (not config-tuned).

## 2026-07-12 — LLM loop 18 vs 50 experiments: search intensity confirms best-of-N

Extended the LLM autoresearch loop from 17 to 50 experiments (45/50 beat vol baseline).
Locked-test comparison of the two intensities:
  18 exps (iter16): VAL 0.338 -> TEST 0.421
  50 exps (iter50): VAL 0.342 -> TEST 0.424
  => 3x more search: VAL +0.004 (inflated), TEST +0.003 (flat). Extra search bought
  ~nothing OOS. Confirms the multi-run best-of-N (VAL up with intensity, TEST plateau)
  using the authentic open-ended LLM agent.
Nuance: the discovered downside-consensus family robustly beats plain VOLATILITY on
test (~0.42 vs 0.376) across both runs — a small real gain, findable with minimal
search — but does NOT beat DES (0.431), and val-selection beats vol only ~54% of the
time (multi-run). So: the modest robust gain is cheap to find; all incremental search
beyond it is validation overfitting. (Agent caught/fixed a stray-line mishap in
metric.py at iter49, re-verified — fail-loud.)

## 2026-07-12 — MULTI-RUN STUDY (paper core): search inflates VAL, not TEST

src/autoresearch/multi_run_study.py: 4000 random nonlinear candidate metrics scored
on VAL + locked TEST; 250 simulated 15-experiment searches (pick best-VAL each).
Baseline volatility: VAL 0.314, TEST 0.376.

MONEY RESULT — best-of-N (search intensity vs transfer):
  N=1: VAL 0.154 / TEST 0.137;  N=10: 0.326/0.369;  N=100: 0.348/0.375;
  N=2000: VAL 0.351 / TEST 0.370.
  => VAL rises monotonically with search; TEST PLATEAUS at ~0.37 ≈ vol baseline.
  The validation gain from searching is pure overfitting — buys ~0 out-of-sample.

Val-selection barely beats vol OOS: 15-exp search winner beats vol on VAL by +0.018
but on TEST by only +0.003, and beats vol on TEST just 53.6% of the time (coin flip).
NOT catastrophic (val-test corr 0.948 — discovered metrics don't blow up) — just
USELESS: statistically indistinguishable from plain volatility OOS.

Triangulates all 3 runs (greedy winner<vol; single-LLM winner>vol; multi-run ~coin
flip) => THESIS: automated downside-metric discovery inflates validation scores that
are overfitting; a locked, survivorship-free holdout shows the discovered metrics do
not reliably beat volatility. The guardrails (locked holdout) are the contribution.
Confidence: HIGH for the best-of-N / no-reliable-improvement result. Caveats: single
asset class (crypto 2016-18), test period more forecastable (regime), random metric
grammar. Next: equity cross-check + write ICAIF short paper.

## 2026-07-12 — Literature review + citation verification

Ran a lit-review pass (writeup/literature_review.md): verified all 17 external
references against publisher records (parallel web lookups by theme). ALL are real
(no hallucinations). Corrections applied to the manuscript: Markowitz 1959
publisher = Yale University Press / Cowles Monograph 16 (was wrongly "Wiley");
Elton-Gruber-Blake 1996 title = "Survivor bias..." (was "Survivorship bias");
Ang & Chua title = "Composite measures" (plural); Brown 1992 = "Goetzmann, W. N.".
Review organizes the field into 3 clusters (downside-risk/partial moments;
survivorship bias; OOS-predictability/data-mining) tied to RQ1/RQ2/RQ3/§4.6.
Closes the peer-review "verify references at submission" residual.

## 2026-07-12 — Revision round 1 (post peer-review): robustness + fixes

Peer-review panel (ARS full mode) returned Major Revision; addressed:
- **RQ3 bootstrap CI now committed** in run_crypto_rq3.py (+0.066, CI [-0.065,
  +0.211], includes 0) + economic decile panel: ratio->return +0.0078/day,
  LPM2->drawdown +0.222. (was a real reproducibility gap — CI had been computed ad hoc)
- **Delisting sensitivity:** 0/104 universe-B firms have their LAST observation
  inside the 2009-01..2011-01 holding window (the delisted-tail names Kaggle
  retains all trade through end-2017, so none terminate in-window) => a -100%
  terminal-return convention cannot change RQ1; confirms free source has no
  in-window deaths to score.
- **Total-return benchmark:** ^GSPC->^SP500TR shifts every predictive-vs-holding
  rank corr by <=0.027 (max) across 1988-99 & 1998-2009 => price-vs-total-return
  mismatch is immaterial to conclusions.
- **Multiple testing:** BH-FDR over the 78-correlation family — 51 significant at
  raw p<.05, ALL 51 survive FDR at q<.05 => significance robust to multiplicity.
- **Appendix A engagement:** V&N explicitly concede survivorship pads EXPLANATORY
  ex-post correlations but argue it "does not offer an acceptable counterpoint to
  why these metrics completely fail out-of-sample." RQ1's downward shift in the
  PREDICTIVE OOS correlation speaks directly to (and against) that defense.
- **RQ2 reframed:** ρ median 0.076 => (1-|ρ|)≈0.92 near-constant multiplier =>
  structurally cannot re-rank cross-sectionally (a stronger, precise claim). The
  paper's separate DEFENSIVE-TIMING claim for ρ (its Fig 3) is not captured by
  rank correlation — flagged as untested / future work, not refuted.

## 2026-07-12 — RQ4: learned (q,n) do NOT reliably generalize

Output: outputs/*_rq4_learned_params. Grid-search (q,n) maximizing in-sample
Spearman(explanatory metric, 1y holding return) on a train window, applied OOS to
a later window. Walk-forward (equity survivors, monthly):
  train 1978-89 -> test 1988-99: learned (0.5, 2.0); in-sample +0.15, OOS -0.28
    (SIGN FLIP), vs best preset +0.27, ceiling +0.33; gap 0.43.
  train 1988-99 -> test 1998-09: learned (0.75, 0.1); in-sample +0.33, OOS +0.30,
    vs best preset +0.28, ceiling +0.34; gap 0.03.
Learned optimum UNSTABLE across eras (n=2.0 vs n=0.1) and inverted sign OOS in one
of two pairs, losing to a hand-set preset. No stable learnable structure =>
consistent with curve-fitting, not a real parameterizable signal. Confidence:
MEDIUM (2 walk-forward pairs, survivor universe).

## 2026-07-12 — RQ3 (crypto, survivorship-free): H3 NOT supported

Output: outputs/*_crypto_rq3. Genuinely survivorship-free universe: top-100 by
market cap AS OF 2017-06-30 (point-in-time; CMC retains crashed coins), n=69 with
≥200 daily obs. Explanatory 2016-07..2017-06; holding 2017-07..2018-06 (bull→crash).

- **Downside forecasting:** explanatory downside-LPM(2) vs holding max drawdown
  rho=0.40 (p=.001); variance foil rho=0.34 (p=.005). Difference +0.066, bootstrap
  95% CI [-0.065, +0.211] STRADDLES 0. => partial-moment downside metric does NOT
  robustly beat plain variance at forecasting downside. H3's core claim unsupported.
- **Return forecasting:** UPM/LPM ratio vs holding return rho=0.31 (p=.009);
  Sharpe rho=0.14 (ns). Contra H3, the ratio DOES correlate with returns here —
  likely momentum in the 2017-18 crypto regime, not a genuine risk-return split.
Confidence: LOW-MEDIUM (n=69, single extreme window, momentum-heavy regime).
Caveat: LPM/variance both predict drawdown partly via volatility persistence.

**Emerging overall verdict:** the partial-moment framework's specific claims do
not hold up — predictive term inert (RQ2, strong), edge survivorship-inflated
(RQ1, directional), and no robust downside-forecasting advantage over variance
(RQ3). Coherent honest-negative audit. Still to firm up: RQ3 across more
windows/universes; RQ4 (learned q,n); RQ1 with true deaths (crypto has them).

## 2026-07-12 — RQ2 CONFIRMED (H2): autocorrelation term adds no OOS power

Output: outputs/*_rq2_ablation. Formal ablation on survivor universe, 3 windows.
Grid = 3 lags {1,2,3} × 2 forms {linear 1-|ρ|, squared 1-ρ²} × 2 methods
{correlation, covariance} × 3 windows × 4 investor types = 144 cells.

**Result (HIGH confidence):**
- max |Δ rank-corr (adjusted − base)| over ALL 144 cells = 0.045; mean = 0.004.
- Paper's exact variant (lag1, linear, correlation): ALL 12 window×investor
  bootstrap 95% CIs (1000 resamples over stocks) straddle 0.
=> The (1-|ρ|) "predictive" adjustment (Eq 7) adds no robust/significant
out-of-sample predictive power over the base UPM/LPM ratio (Eq 4), regardless of
lag, functional form, covariance-vs-correlation reading, window, or investor
type. This falsifies the paper's central claim that the predictive metric beats
the explanatory metric OOS. Caveat: survivor universe (appropriate for H2 — the
question is whether the TERM carries signal, independent of survivorship).

## 2026-07-12 — Phase 2a: directional RQ1 (H1) lower-bound + Kaggle has NO deaths

Output: outputs/2026-07-12_09-26-14_partial_equity_rq1. PIT universe from
fja05680 (1998-2009): 840 members, 289 survivors, 551 left-index. Prices Kaggle.

**Critical data fact:** ALL 245 delisted-tail firms Kaggle retains END in 2017 —
Kaggle contains essentially ZERO firms that actually stopped trading. Its
"delisted" names are firms that FELL OUT of the S&P 500 but kept trading
(shrank/acquired-but-continued), not go-to-zero failures. A true bias-free equity
test is impossible on free data (confirmed twice over).

**FINDING (H1, directional, LOWER BOUND):** adding even those surviving
index-leavers to the survivor universe shifts the predictive metric's OOS rank
corr DOWN in every investor type:
  investor         A_survivors  B_+delisted  shift
  risk_neutral        +0.158      +0.024     -0.134   (edge essentially vanishes)
  prospect            -0.244      -0.362     -0.119
  risk_averse         -0.362      -0.454     -0.091
  risk_seeking        +0.298      +0.239     -0.060
Direction consistent with H1 (survivorship inflates the apparent edge). TRUE
effect is larger — the catastrophic-death tail is entirely absent. Confidence:
MEDIUM (mild correction only; a proper test needs in-window deaths, unobtainable
free for equity → deferred to the crypto arm with dead coins).

## 2026-07-12 — Phase 1 SOLIDIFIED across the paper's 3 windows (monthly)

Ran all three of the paper's overlapping windows on the survivor universe
(1978-89 n=157, 1988-99 n=239, 1998-2009 n=354; today's S&P 500, monthly).
Output: outputs/2026-07-12_09-15-01_biased_arm_multiwindow.

**FINDING 1 UPGRADED to HIGH confidence (H2 — autocorr term is inert).** The
predictive metric's OOS rank correlation differs from the plain explanatory
ratio by AT MOST 0.024, across every window × investor type. The (1−|ρ|)
adjustment does not change the cross-sectional ranking, ever, on this data.
Still to confirm on the bias-free universe + against a formal ablation (RQ2).

**FINDING 2 UPGRADED (no robust predictive edge; regime + parameter fragile).**
Predictive-vs-1y-holding rank corr by window (rows=investor type):
                 1978-1989  1988-1999  1998-2009
  risk_averse       +0.14     -0.11      -0.42
  prospect          +0.11     -0.01      -0.35
  risk_neutral      +0.01     +0.22      +0.13
  risk_seeking      -0.02     +0.27      +0.28
Sign is unstable across windows AND flips between risk-averse and risk-seeking.
Sharpe foil also unstable (+0.13 / -0.12 / -0.30). No consistent OOS predictive
power; the paper's "significant predictive, MPT sparse" pattern does NOT
replicate robustly on the survivor set. Confidence: MEDIUM-HIGH (biased universe,
^GSPC proxy, today's-survivors — this is the BIASED BASELINE for the RQ1 contrast).

## (superseded) 2026-07-12 — Phase 1 first monthly run, single window 1998–2009

**Config:** monthly returns; explanatory Jan-1998→Jan-2009 (dot-com + GFC);
holding 2009 (1yr) / 2009–2010 (2yr); universe = today's S&P 500 survivors
(431 with data, 354 with ≥100 monthly obs ≈ paper's ~300); double non-stationary
benchmark (^GSPC + ^IRX). Output: outputs/2026-07-12_09-09-43_biased_arm_replication.

**Objective (validate the implementation reproduces paper-scale behavior): MET.**
Significant out-of-sample rank correlations appear (|ρ_spearman| up to 0.42,
p<1e-11), unlike the earlier DAILY run which was null — confirming the null was a
frequency/regime artifact, not a code bug.

**PREVIEW FINDING 1 (H2 — the autocorrelation term is near-inert).** Even at
monthly frequency, median |ρ(x)| = 0.076, so (1−|ρ|) ≈ 0.92 for most stocks.
Explanatory-vs-predictive rank correlation across stocks = 0.94–0.99. The Eq-7
adjustment barely changes the cross-sectional ranking. Confidence: MEDIUM
(one window; must confirm across windows and vs an ablation in RQ2).

**PREVIEW FINDING 2 (parameter fragility).** The sign of the out-of-sample
correlation FLIPS with investor type: risk_averse (q=0.25) ρ≈−0.42,
risk_seeking (q=2) ρ≈+0.28. The hand-set exponents are not a monotone rescaling;
they reorder stocks and flip the verdict. Confidence: MEDIUM.

**PREVIEW FINDING 3 (does NOT reproduce the paper's specific pattern).** The
paper claimed predictive ≫ explanatory and "MPT metrics sparse." Here
explanatory ≈ predictive, and the Sharpe foil is ALSO significant (ρ≈−0.30).
Strong caveat: holding period 2009 is a violent GFC-rebound (mean-reversion)
regime, and this is a single window. NOT evidence against the paper yet — needs
the other windows + the bias-free arm. Confidence: LOW.

**Caveats binding all three:** single explanatory/holding window; 2009 holding
is idiosyncratic; universe is today's survivors over 1998–2009 (extreme
survivorship — intended for the biased arm, but means these numbers are the
biased baseline, not the paper's exact figures).

## 2026-07-13 — MECHANISM RESULT: drawdown predictability is volatility persistence

Reframing the paper from a mostly-negative benchmark to a positive/explanatory
claim: *why* does nothing reliably beat volatility? Script: `src/autoresearch/
run_mechanism.py`. Output: `outputs/2026-07-13_07-23-21_mechanism/mechanism.json`.
Same 5 beds as run_multi_testbed_v2 (crypto calm/bull/crash 2016/17/18; equity
1994-99 calm, 2005-09 GFC). Per-date cross-sectional Spearman, block-bootstrap 95% CI.

**M2 — the persistence chain (the positive mechanism). Confirmed on ALL 5 beds,
every CI excludes 0:**
  trailing_vol -> forward_vol : +0.68 / +0.42 / +0.42 / +0.84 / +0.80
  forward_vol  -> fwd_drawdown: +0.77 / +0.62 / +0.53 / +0.71 / +0.70
  (baseline trailing_vol -> fwd_drawdown: +0.62 / +0.36 / +0.38 / +0.61 / +0.56)
=> The reason trailing volatility forecasts forward drawdown is that volatility is
PERSISTENT (trailing->forward vol) and forward volatility mechanically drives
drawdown. Both links are stronger than the direct skill, so drawdown predictability
is, to first order, a volatility-persistence phenomenon.

**M1 — incremental signal beyond volatility (partial Spearman rho(M, fwd_dd | vol)).
A sharp MAGNITUDE-vs-SHAPE split:**
  Tail-MAGNITUDE (volatility family) carry a SMALL but real increment:
    var5:         +0.12* +0.19* +0.14* +0.04* +0.08*   (SIGNIFICANT on all 5 beds)
    downside_dev: +0.13* +0.14* +0.20*  +0.02   +0.01   (sig on 3 crypto beds)
    es5:          +0.14* +0.08*  +0.08  +0.03*  +0.00   (sig on 3 beds)
  Tail-SHAPE / asymmetry / co-crash carry NONE (~0 or significantly NEGATIVE):
    vn_ratio (V&N partial-moment):  -0.00  +0.01 -0.05 -0.05- -0.07-
    hill_tail:  -0.09- -0.13- -0.09-  nan   nan   (not estimable per-date on equity)
    ltd_crash:  -0.07  -0.04 -0.17-  -0.01  +0.00
    gda_voldown:+0.03  -0.05 +0.05   -0.01  -0.03
    down_beta:  -0.08-  +0.03 +0.02   +0.01  +0.04
  One mild exception: down_semibeta small positive increment on 3 beds
    (+0.08* crypto-bull, +0.13* crypto-crash, +0.05* equity-GFC).
  (* = 95% CI > 0; - = CI < 0.)

**Interpretation (the paper's new positive claim):** What little is forecastable
beyond volatility is the SIZE of tail losses (VaR/ES/downside-dev), NOT their
SHAPE/asymmetry. The asymmetry premise underlying partial-moment metrics — incl.
V&N's own UPM/LPM ratio — carries no incremental out-of-sample signal; it is flat
or negative on every bed. The magnitude increment is small (partial rho <=0.2,
mostly <=0.1) and largest in crypto.

**Not a contradiction with the head-to-head Table 2:** VaR5 has a significantly
POSITIVE partial rho on equity (+0.04, +0.08) yet is significantly WORSE than
volatility head-to-head on equity. Both hold: conditionally VaR carries a small
orthogonal signal, but as a STANDALONE ranker it is a noisier volatility proxy, so
it loses head-to-head. "Incremental" != "standalone" — this is the crux the mechanism
explains. Implication: a volatility + tail-magnitude composite should marginally beat
volatility; the increment is small (left as an implication, not chased, to avoid the
search-overfitting the paper warns about).

Confidence: MEDIUM-HIGH. 5 beds, 2 asset classes, calm+crash, block-bootstrap CIs.
Partial Spearman computed from the 3 pairwise rank correlations (standard). Caveat:
crypto beds have few formation dates (7-9); equity beds 54-66. hill_tail not
estimable per-date on equity (dropped there, as with Kelly-Jiang aggregate measure).

## 2026-07-18 — MECHANISM DEPTH: mediation, horizon dynamics, honest composite

Expansion of the mechanism (adds causal sharpening, dynamics, a pre-registered metric,
and 6 figures). Scripts: `run_mechanism_depth.py`, `run_composite_honest.py`,
`make_figures.py`. Outputs: `outputs/2026-07-18_08-29-36_mechanism_depth/`,
`outputs/2026-07-18_08-34-56_composite_honest/`, figures in `writeup/figures/`.

**MEDIATION (h=90): forward volatility is a (near-)sufficient statistic for trailing vol.**
partial rho(trailing_vol, dd | forward_vol) vs rho(forward_vol, dd | trailing_vol):
  crypto-16: +0.21 [0.15,0.30]  vs +0.60 [0.58,0.62]
  crypto-17: +0.15 [0.06,0.22]  vs +0.56 [0.45,0.64]
  crypto-18: +0.20 [0.15,0.25]  vs +0.45 [0.40,0.58]
  equity94-99:+0.02 [-0.02,0.05] vs +0.48 [0.45,0.51]
  equity05-09:+0.01 [-0.03,0.05] vs +0.50 [0.46,0.53]
=> On EQUITY, forward vol FULLY mediates (trailing|forward ≈ 0, CI covers 0): trailing
vol forecasts drawdown ONLY by forecasting forward vol. On CRYPTO, mediation is PARTIAL
(+0.15..0.21, CI excludes 0): a residual direct trailing-vol->drawdown link remains — the
BLOW-UP/DEATH channel that (pre-death) realized volatility misses. Ties the mechanism to
survivorship: fat-tailed, death-prone universes leave residual risk vol can't see.

**HORIZON (h=14..90): drawdown skill TRACKS volatility persistence (not "decay").**
Both move together: equity both RISE with h (persist 0.66->0.84, skill 0.47->0.61, roughly
constant gap); crypto both flat. Wherever persistence is higher, skill is higher. (Reframed
from my prior "decay" hypothesis — the data show co-movement, not decay.)

**PLACEBO:** permuting dd within date -> trailing-vol correlation ≈ 0 on all beds
(means +0.00..+0.03; the one marginal +0.028 in crypto-18 is negligible vs real ~0.38).
Confirms the block-bootstrap CIs are calibrated.

**HETEROGENEITY (n=5, descriptive):** blow-up rate / mean trailing excess kurtosis:
crypto 0.05-0.13 / 8-21;  equity ~0.00 / 3-4. The VaR incremental signal (partial rho|vol)
is systematically larger in the fat-tailed crypto beds (0.12-0.19) than equity (0.04-0.08).
Within crypto it does NOT track blowup-rate monotonically (n tiny) — it's a crypto-vs-equity
cluster separation.

**HONEST COMPOSITE (pre-registered, zero-tuning): comp = z(vol) + z(VaR5), equal weight.**
5-bed head-to-head diff (comp - vol) with block-bootstrap CI:
  crypto-16: -0.002 [-0.013,0.001] n.s.
  crypto-17: +0.047 [0.023,0.071]  ** SIG beats vol ** (fat-tailed bull)
  crypto-18: +0.014 [-0.034,0.067] n.s.
  equity94-99: -0.008 [-0.011,-0.004] SIG WORSE (noisy VaR hurts in thin tails)
  equity05-09: -0.002 [-0.009,0.004] n.s.
Locked panel test split: vol 0.376 vs comp 0.388 (diff +0.012, NW t=0.67, 7 dates; n.s.).
=> The mechanism's practical upshot, confirmed and bounded: a pre-registered vol+magnitude
composite gives a small SIGNIFICANT edge ONLY in the fat-tailed crypto-bull regime, and is
neutral-to-slightly-negative in thin-tailed equity. No tuning => not data-snooping. Honest
"better metric": marginal, regime-specific, exactly where M1 says magnitude carries signal.

FIGURES (writeup/figures/): fig1 partial-forest (magnitude vs shape), fig2 persistence
chain, fig3 mediation, fig4 horizon tracking, fig5 best-of-N (val inflates, test flat at
vol), fig6 increment-vs-tailfat.

Confidence: MEDIUM-HIGH. Same 5 beds/CIs as mechanism. Caveats: crypto few dates (7-9);
forward_vol->dd is contemporaneous/mechanical (predictive content is trailing->forward);
n=5 heterogeneity descriptive; composite locked-panel test underpowered (7 dates).

## 2026-07-18 (rev.) — PEER-REVIEW P0/P1: mediation REFUTED, magnitude-vs-shape FDR-ROBUST

ARS reviewer panel returned Major Revision (DA raised CRITICAL: the mediation "sufficient
statistic" claim is circular because forward vol and forward drawdown share the same window).
Two new experiments settle it.

**NON-OVERLAP MEDIATION (run_mediation_nonoverlap.py; outputs/2026-07-18_09-16-06_*).**
Mediator = vol over days 1-30; outcome = drawdown over days 31-120 (DISJOINT, genuinely
predictive). partial rho(trailing_vol, drawdown | forward_vol):
              OVERLAP (old, mechanical)   NON-OVERLAP (predictive)
  crypto-16:   +0.206                      +0.336
  crypto-17:   +0.152                      +0.171
  crypto-18:   +0.197                      +0.277
  equity94-99: +0.016 (CI covers 0)        +0.337  (CI [0.31,0.36])
  equity05-09: +0.013 (CI covers 0)        +0.321  (CI [0.28,0.35])
=> The "full mediation on equity" was ENTIRELY a same-window artifact. Under a predictive
design, controlling for forward vol does NOT remove trailing vol's content (partial +0.32-0.34
on equity ~ as large as the raw skill). **Forward volatility is NOT a sufficient statistic;
the strong "drawdown predictability IS volatility persistence / tested causally" claim FAILS
and is dropped.** (Caveat: non-overlap mediator is a shorter 30d/earlier window, also a noisier
vol proxy, so this is an upper bound on trailing vol's residual content — either way the strong
claim doesn't hold.) The DA and methodology reviewers were right.

What SURVIVES: (i) trailing vol is the dominant, hard-to-beat drawdown predictor (benchmark);
(ii) vol is persistent (Engle 1982; Bollerslev 1986) and realized vol is mechanically tied to
drawdown — so persistence is a leading but INCOMPLETE explanation; (iii) the magnitude-vs-shape
decomposition, which does not depend on mediation.

**BH-FDR + POWER (run_robustness_fdr.py; outputs/2026-07-18_09-20-34_*).** q=0.05.
Partial family (43 tests, survivor p<=0.0147): VaR partial SURVIVES on ALL 5 beds
(+0.12,+0.19,+0.13,+0.03,+0.08); downside_dev on 3 crypto; es5 on 2 crypto; **es5 equity-94-99
(+0.03) is nominal-only, does NOT survive FDR** (demote star). down_semibeta small positive
survives on 3 beds (documented shape exception). Shape metrics mostly significantly <=0.
Head-to-head family (survivor p<=0.04): only downside_dev/var5 on crypto-17 beat vol; all shape
metrics significantly worse. Confirms benchmark.
Small-sample power (crypto VaR partial): crypto-16 effect 0.124 > MDE@80% 0.094, LOO [0.106,0.139];
crypto-17 0.194 > MDE 0.091, LOO [0.172,0.207]; **crypto-18 0.135 < MDE 0.184 (n=7, UNDERPOWERED)**,
LOO [0.097,0.175] (stays positive). Report crypto-18 as suggestive.

**PLACEBO FACTUAL FIX (§4.8):** crypto-2018 placebo is +0.028 CI [0.003,0.059] — EXCLUDES 0
(negligible vs real ~0.38, but the manuscript's "CIs cover zero on every bed" is false; corrected).

Net: paper reframed AGAIN, honestly — title/spine move from "mechanism" to the FDR-robust
benchmark + magnitude-vs-shape, with volatility persistence as a caveated (not causal) explanation.

## 2026-07-18 (Appendix B) — economic backtest + conditional increment; appendix slimmed

Added two new experiments (user request: less code-description in appendix, more interesting exps).

**ECONOMIC BACKTEST (run_economic_backtest.py; outputs/2026-07-18_15-54-28_*).** Risk-sorted
terciles, hold low-risk / avoid high-risk; 20bps turnover cost; vol vs composite z(vol)+z(VaR).
Drawdown discrimination (fwd DD high− low tercile): vol +0.30/+0.12/+0.11/+0.11/+0.10 across
beds → volatility HAS real economic value (high-risk tercile draws down 10-30pp more). Composite
≈ vol everywhere (and trades more) EXCEPT crypto-2017 (net spread +0.84→+1.16, dd_disc
+0.12→+0.15). Crypto low-risk tercile also earns much higher fwd returns (low-vol anomaly);
equity spread ~flat/slightly negative but DD discrimination still positive. Resolves the §6
"not tradable" limitation: economics matches the rank-correlation story.

**CONDITIONAL INCREMENT (run_conditional_increment.py; outputs/2026-07-18_15-57-18_*).** Pool
dates by asset class, split at median aggregate vol. VaR partial increment: crypto low-stress
+0.127 [0.057,0.177] → high-stress +0.180 [0.09,0.241] (24 dates, 12/regime; suggestive,
CIs overlap); equity flat +0.056 → +0.050 (120 dates). => magnitude increment concentrates in
high-stress crypto dates; uniformly small on equity. "Fat tails" effect is cross-sectional AND
time-varying.

New figure fig7_economic.png (drawdown discrimination vol vs composite). Appendix restructured:
removed file-by-file code description → single repo link (PLACEHOLDER URL
github.com/meghanadhpulivarthi/partial-moments-audit — user to replace). New Appendix B holds
both experiments (Table B1 + Fig 7 + conditional test). PDF regenerated.
