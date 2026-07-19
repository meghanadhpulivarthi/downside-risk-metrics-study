# Open Questions

Unresolved items. Resolve and move rationale to decisions.md when closed.

## RESOLVED 2026-07-12 — delisted-price source = Kaggle "Huge Stock Market Dataset"

Chose Boris Marjanovic's static dump (borismarjanovic/price-volume-data-for-all-
us-stocks-etfs): ~8k US stocks/ETFs incl. many delisted, daily ~1970s–2017.
Consequence: bias-free out-of-sample window bounded at 2017.
Kaggle access CONFIRMED working (user meghanadhpulivarthi, newer KGAT access_token
at ~/.kaggle/access_token; auth_method=ACCESS_TOKEN). Dataset = per-ticker OHLCV
.txt under Data/Stocks/ and Data/ETFs/, last updated 2017-11-16. Not yet
downloaded (~1GB) — pull as first step of Phase 2.

## OPEN — Exact out-of-sample window(s) for the bias-free arm

Depends on the delisted-price source's date coverage. Must be non-overlapping,
walk-forward (fixes the paper's overlapping-window soft spot, §4.3).
