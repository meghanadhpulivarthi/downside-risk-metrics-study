# Implementation Roadmap — Partial-Moment Predictive-Edge Audit

Concrete plan derived from `research_problem.md` (§7 methodology) + the kickoff
decisions in `context/decisions.md`. Anchor: Viole & Nawrocki (2016).

**Design in one line:** build the metric exactly as specified, then run it on two
universes — a deliberately *surviving* one (reproduce the claim) and a
*survivorship-bias-free* one (audit the claim). The gap between them answers RQ1.

Legend: ✅ unblocked now · ⛔ blocked on Kaggle token · ○ later

---

## Phase 0 — Metric core (foundation) ✅
Re-implement, from the paper's equations 1–7, verified against the PDF (not the
summary): `lpm(n,h,returns)`, `upm(q,l,returns)`, the explanatory ratio
`UPM(q,y,x)/LPM(n,y,x)`, the double non-stationary benchmark (max of asset-class
benchmark and risk-free at each t), and the predictive metric
`(UPM/LPM)·(1−|ρ|)` with `ρ = one-period autocorr`.
- Degree presets: Risk-Averse 0.25, Prospect 0.44, Risk-Neutral 1.0, Risk-Seeking 2.0.
- **Verify:** unit tests on hand-computable toy series; reproduce any worked
  number the paper prints. (Rule 9 — tests encode *why*.)
- Deliverable: `src/partial_moments.py` + `tests/`.

## Phase 1 — Biased-arm replication ✅
Reproduce the paper's rank-correlation result on a *surviving* universe (today's
S&P 500 back-filled via yfinance). Success = implementation matches the paper's
qualitative claim (explanatory-period correlation; weak base out-of-sample;
autocorr-adjusted metric predictive). This is the sanity check, not the finding.
- Non-overlapping walk-forward windows from the start (fixes §4.3).
- Deliverable: `notebooks/01_biased_arm_replication.ipynb`.

## Phase 2 — Bias-free universe assembly ⛔ (needs Kaggle token)
- Point-in-time S&P 500 membership from Wikipedia change log (✅ scraper can be
  built now; store snapshots to `data/`).
- Join Kaggle daily prices incl. delisted firms; produce a delisting-inclusive
  panel for the ~2009–2017 window. Log coverage gaps loudly (Rule 12).
- Deliverable: `src/build_universe.py`, `data/pit_membership.parquet`.

## Phase 3 — RQ1: survivorship test ⛔
Re-run Phase-1 metrics on the bias-free universe. **H1:** predictive correlation
shrinks materially vs the surviving universe. Like-for-like comparison table.

## Phase 4 — RQ2: autocorrelation-term ablation ✅ (runs on either universe)
Ablate `(1−|ρ|)`: base ratio vs adjusted; alternative lags; alternative forms.
**H2:** the term adds little robust out-of-sample signal; value is period-specific.

## Phase 5 — RQ3: risk- vs return-forecasting ✅
Reframe target as future downside/drawdown. Benchmark partial-moment metrics
against variance/Sharpe out of sample, for both returns and downside targets.
**H3:** modest edge for *downside* forecasting, none for *return* forecasting.

## Phase 6 — RQ4: learned (q,n) ○ (in scope)
Learn degree exponents via cross-validation; test out-of-sample generalization
vs hand-set presets. Guard hard against overfitting (walk-forward CV only).

## Phase 7 — Economic test + write-up ○
Transaction costs / turnover where an economic (not just rank-corr) claim is
implied. Then the paper via the ARS pipeline (`/ars-outline` → draft).

---

## Immediate next actions
1. ✅ **Phase 0 metric core** — read exact equations from the PDF, implement, unit-test.
2. ⛔ **You:** drop a Kaggle API token at `~/.kaggle/kaggle.json` to unblock Phase 2–3.
3. ✅ Wikipedia PIT-membership scraper can be built in parallel (no token needed).
