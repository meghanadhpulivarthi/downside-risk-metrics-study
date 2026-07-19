# Gotchas

Non-obvious failure modes discovered during the project. Newest first.

## 2026-07-12 — Free delisted-equity price data is now hard (data spike)

- **yfinance silently erases delisted firms.** `yf.download` for LEH (Lehman),
  ENE (Enron), WCOM (WorldCom) all return 0 rows with only a warning. Using
  yfinance as-is rebuilds the *exact survivorship-biased universe the paper is
  accused of* — the bias we are auditing. This is demonstrable evidence for the
  paper itself, not just a nuisance.
- **Stooq's free CSV endpoint is behind a JavaScript proof-of-work anti-bot wall
  now.** `https://stooq.com/q/d/l/?s=<ticker>&i=d` returns an HTML JS challenge,
  not CSV, for every ticker (delisted or live). The old "Stooq retains delisted
  tickers" trick no longer works for scripted access. Both the
  `pandas_datareader` stooq reader (NotImplementedError in v?) and direct CSV
  are dead paths.

## 2026-07-12 — DECISIVE: Kaggle dataset is NOT survivorship-bias-free (Phase 2)

Measured Kaggle "Huge Stock Market Dataset" coverage against fja05680/sp500
point-in-time constituents. Kaggle has ~88% of CURRENT members but only:
  - 44% of the 1998-2009 delisted/left-index tail (245/551)
  - 15-27% of index-REMOVED firms in EVERY era 1996-2017 (not just old ones)
Marquee bankruptcies ABSENT: Lehman, WaMu, Bear Stearns, Ambac, Enron, Fannie,
Freddie. Also absent (spot-checked, traded for years): Compaq, Merrill, Tyco,
Kodak, Sara Lee, Dow Jones. Present: Gillette, Wachovia, GM, Anheuser.
=> The 2017 Stooq-derived dump retained SOME acquired firms but dropped nearly
all bankruptcies/exits — i.e. it is still heavily survivorship-biased in exactly
the failure tail RQ1 is about. **Kaggle cannot serve as THE bias-free universe.**
This is §9's main risk materializing: free survivorship-free equity is not
obtainable. Decision escalated to user (see open-questions.md). fja05680/sp500
PIT constituents ARE good (1996-2019) — the gap is delisted PRICES, not membership.

## 2026-07-12 — FINDING: daily frequency makes the (1−|ρ|) term inert (Phase 1)

First Phase-1 run used DAILY returns (2010–2014 explanatory). Result: median
lag-1 autocorr |ρ| = 0.04, so the (1−|ρ|) discount ≈ 0.96 for nearly all stocks;
predictive metric is 0.99 rank-correlated with the plain explanatory ratio — the
Eq-7 adjustment does nothing. Daily equity returns are ~serially uncorrelated by
construction. The paper's autocorr term only bites at coarser (≈monthly)
frequency, where 11yr ≈ 132 obs (matches its "100+ obs to stabilize" text).
=> **Replication of the paper MUST use ~monthly returns.** The frequency choice
is itself RQ2 audit material: the marketed predictive term is frequency-fragile.
Also: 2010–2016 is a calm bull market; the paper used crisis-spanning "difficult
periods" — a downside metric may only show power across crashes (regime effect).
Neither confound lets us conclude anything about the paper from that first run.

## 2026-07-12 — Equation subtleties read from the PDF (pp.903–908)

These refine research_problem.md's summary and feed the audit directly:
- **h = l (footnote 3):** a single target is used for BOTH UPM and LPM. Not two
  independent gain/loss targets.
- **Double non-stationary benchmark (p.905):** the target at each time t is the
  ELEMENTWISE MAX of (asset-class benchmark return, risk-free rate). This
  time-varying series is the target `y` in Eq 4 / Eq 7. Systemic proxy = CRSP
  market-cap index; risk-free = 3-month T-bill.
- **ρ written as COVARIANCE, not correlation (Eq 5: ρ(x)=Cov(x_t,x_{t-1})).**
  But entry-at-0 / full-exit-at-|ρ|=1 and the (1−|ρ|) discount only make sense
  for a bounded [−1,1] correlation. Raw-return covariance is unbounded. => a
  real specification ambiguity. Implemented as a switch (rho_method); RQ2 must
  test both. THIS IS AUDIT MATERIAL, not just a coding note.
- **ρ computed over the ENTIRE explanatory period (footnote 10), not rolling.**
  The paper's "dynamic positioning" story (cut 50% when ρ hits 0.5, etc.) is
  narrative only — the tested metric uses one static ρ per asset per period.
- **Investor-type exponents ambiguous:** paper writes "q: n = 0.25 / 0.44 / 1 /
  2". Reads as a RATIO q/n, plausibly with n=1 (0.44 ≈ 1/2.25 = inverse of the
  Kahneman–Tversky loss-aversion λ). Implement (q,n) as separate params;
  default n=1, q ∈ {0.25,0.44,1.0,2.0}. Confirm against Tables 1–6 later.
- **Universe/periods:** ~300 SURVIVING S&P 500; three overlapping 11yr windows
  1978–89, 1988–99, 1998–2009, all described as "difficult market periods"
  (possible period cherry-picking, §4 soft spot). Four rank correlations per
  period; the headline "ex ante efficiency" is corr#4 = predictive metric vs
  holding-period mean return.

## 2026-07-12 — Wikipedia blocks default pandas.read_html

- `pd.read_html(url)` on Wikipedia -> HTTP 403 Forbidden (no User-Agent).
  Fix: `requests.get(url, headers={"User-Agent": "Mozilla/5.0 ..."})` then
  `pd.read_html(io.StringIO(resp.text))`.
