# Decisions

Choices made and why. Newest first.

## 2026-07-12 — Kickoff decisions (from problem-statement §11 gating questions)

- **Return frequency / rebalancing:** daily data, monthly rebalancing, 1–2yr
  holding horizon. Matches the paper's coarse horizon; keeps scope small.
- **RQ4 (learned (q,n) degree exponents) is IN SCOPE** from the start, not a
  stretch goal. Adds ML signal; accepts extra overfitting/infra risk.
- **Data access = free sources only** (no WRDS/CRSP/Bloomberg). This forces a
  *reconstructed* survivorship-bias-free universe rather than a bought one, and
  makes the data-limitation disclosure (Rule 12) a first-class part of the paper.

## 2026-07-12 — RQ1 data strategy REVISED (Kaggle inadequate; see gotchas.md)

Kaggle proved not survivorship-bias-free (misses the failure tail). New strategy:
- **RQ1 becomes a DIRECTIONAL / lower-bound equity test**, not a full bias-free
  replication. Universe from fja05680/sp500 point-in-time constituents (good,
  1996-2019). Prices from Kaggle for consistency. Compare:
    A = PIT members 1998-2009 still in index today (survivors)
    B = A + the delisted PIT members Kaggle DOES retain (~partial tail)
  Shift A→B lower-bounds the survivorship effect (H1). Coverage disclosed loudly.
  Handle firms that delisted mid-holding carefully (their loss must count).
- **Crypto = the genuinely-clean second regime** for the OOS asymmetry question
  (RQ3), NOT a fix for equity RQ1. Caveat: crypto has its own survivorship
  (dead coins) — include delisted coins where feasible, disclose otherwise.
- **Weight shifts toward RQ2/RQ3/RQ4**, which need no delisted-price data and
  already show strong results (RQ2: autocorr term inert).
- The coverage evidence itself is a reportable finding: free survivorship-free
  equity data is not obtainable, which is part of why published claims like this
  go under-audited.

## 2026-07-12 — Data feasibility spike results (see data_feasibility_probe.py)

- **Point-in-time S&P 500 membership: FREE + VIABLE.** Wikipedia
  `List_of_S&P_500_companies` returns 503 current constituents + a ~402-row
  "changes" table (Added/Removed/Effective Date) when fetched with a browser
  User-Agent (bare pandas.read_html gets HTTP 403). Change log realistically
  reaches back ~2 decades, NOT to the paper's 1978 start.
- **Consequence:** the bias-free arm targets a **post-~2005/post-GFC window**,
  which is a feature — the anchor paper tests nothing after 2009 (§4 soft spot).

## 2026-07-12 — Two-arm design + delisted-price source

- **Delisted-price source = Kaggle "Huge Stock Market Dataset"**
  (`borismarjanovic/price-volume-data-for-all-us-stocks-etfs`). Bias-free window
  therefore bounded ~2009–2017.
- **Two-arm design (the contrast is the RQ1 result):**
  - *Biased arm* — today's S&P 500 back-filled via yfinance (rebuilds the
    surviving universe on purpose) → reproduce the paper's claim.
  - *Bias-free arm* — point-in-time membership (Wikipedia) + Kaggle prices
    including delisted firms → the actual audit.
- **Action item (user):** provide Kaggle API token at `~/.kaggle/kaggle.json`
  (Kaggle → Settings → Create New API Token). `kaggle` pkg not yet installed;
  connectivity to kaggle.com confirmed. Blocks ONLY the bias-free arm; metric
  core + biased arm proceed without it.

## 2026-07-13 — Paper reframed to a mechanism; venue lowered to a workshop

- **Framing decision:** the mostly-negative "nothing beats volatility" benchmark
  is reframed around a POSITIVE mechanism (see findings 2026-07-13): drawdown
  predictability is a volatility-persistence phenomenon, and only tail-MAGNITUDE
  (not tail-SHAPE/asymmetry) carries incremental OOS signal. The 5-bed benchmark,
  the autoresearch/data-snooping result, and the volatility blow-up predictor
  become supporting evidence for the mechanism, not the headline.
  Rationale: we could not honestly produce a market-beating metric (every attempt
  failed OOS; fabricating one is the fraud the project audits). A mechanism is a
  genuine, defensible, positive contribution that the negative results support.
- **Venue decision:** retarget from ACM ICAIF MAIN short paper to a WORKSHOP /
  findings-style venue (user call). Honest negative+mechanism results clear a
  workshop bar and still serve MFE admissions as a citable arXiv paper. Note:
  "findings" is ACL/EMNLP terminology; ICAIF has no findings track — concrete
  options are an ICAIF-affiliated workshop, a NeurIPS/ICML "ML for finance/
  economics" workshop, or arXiv-only. Final venue TBD after seeing the mechanism
  draft; does not block the rewrite.

## 2026-07-18 (rev.4) — Peer-review P0/P1 revision; mechanism claim retracted

- ARS reviewer panel → Major Revision (DA CRITICAL: same-window mediation is circular).
- **P0 done:** non-overlapping mediation experiment REFUTED the "forward vol is a sufficient
  statistic / drawdown IS volatility persistence" claim (equity partial +0.32-0.34 under a
  predictive design vs ~0 under same-window). Dropped the causal/mechanism framing; retitled
  paper "Magnitude, Not Shape: A Survivorship-Free, Significance-Tested Benchmark of
  Downside-Risk Metrics." Fixed the placebo factual error (§4.8, crypto-18 CI excludes 0).
- **P1 done:** BH-FDR across the 45-cell families (VaR increment survives on all 5 beds;
  ES equity cell demoted); per-bed MDE + leave-one-date-out power checks (crypto-18 flagged
  underpowered); softened the composite "cannot be data-snooped" claim (lead with the n.s.
  locked-panel ceiling); added GARCH (Engle 1982, Bollerslev 1986), backtest-overfitting
  (Bailey & López de Prado 2014, 2015), LPM lineage + Rockafellar-Uryasev + Magdon-Ismail
  citations and positioning.
- New spine: FDR-robust benchmark + magnitude-vs-shape as the lead; volatility persistence
  demoted to a caveated (not causal) explanation. New scripts: run_mediation_nonoverlap.py,
  run_robustness_fdr.py. PDF = 10 pages.

## 2026-07-18 (re-review) — ACCEPT

ARS re-review (EIC verification mode) → **Accept**, conditional on Stage 4.5 citation check.
The lone CRITICAL (P0-1 mediation) verified RESOLVED: non-overlap experiment refuted the
sufficient-statistic claim and it was retracted across title/abstract/§4.1/§5/§7; decisive
number (equity predictive partial ρ +0.337/+0.321) reproduced from JSON. All P0/P1
FULLY_ADDRESSED + independently verified against outputs; all P2 addressed. One non-blocking
fix applied: normalized Table A1 / §4.2 rounding (crypto-18 VaR +0.14→+0.13, equity-94
+0.04→+0.03) to match robustness_fdr.json. Deep citation verification (Stage 4.5) still
recommended before finalization, though the added refs were spot-checked correct.

## 2026-07-18 (citation check) — Stage 4.5 done, 3 errors fixed

Verified all 20 references against publisher records (JSTOR/Oxford/ScienceDirect/RePEc/risk.net).
17 exact. Fixed 3: (1) Viole & Nawrocki (2016) TITLE was wrong ("The utility of partial
moments") → correct "Predicting Risk/Return Performance Using Upper Partial Moment/Lower
Partial Moment Metrics," J. Mathematical Finance 6(5):900–920. (2) Bailey et al. PBO: year
2015→2017, pages 39–69→39–70 (JCF 20(4)); in-text updated. (3) Andersen et al. (2003) was an
uncited reference → now cited in §4.1 (realized-vol forecastability). PDF regenerated.
