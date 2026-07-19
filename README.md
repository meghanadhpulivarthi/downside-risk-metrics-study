# Magnitude, Not Shape: A Survivorship-Free Study of Downside-Risk Metrics

Code and results for the paper *"Magnitude, Not Shape: A Survivorship-Free,
Significance-Tested Evaluation of Downside-Risk Metrics."*

## What this is

Dozens of downside- and tail-risk metrics claim to improve on plain return
volatility, on the premise that the **shape** of the loss distribution (asymmetry,
tail heaviness, co-crash structure) carries information volatility misses. We test
that premise on the fairest field we can build — a **survivorship-free** universe
(a crypto panel retaining delisted coins, plus equity), a **locked** out-of-sample
holdout, multiple regimes, and **multiple-testing correction** — evaluating ten
metrics as forecasters of 90-day forward maximum drawdown across five test beds.

**Headline finding:** no metric — published or machine-discovered — reliably beats
plain volatility. What survives correction is a sharp law: **magnitude, not
shape** — only the *size* of tail losses (VaR/ES/downside deviation) adds
incremental signal beyond volatility (and only a little, mostly in fat-tailed
markets); asymmetry/shape metrics add nothing. We also show volatility persistence
is a leading but *not sufficient* explanation, that the edge is not economically
free, and that an autonomous LLM metric-discovery loop only inflates validation
while its locked-test score plateaus at volatility (a data-snooping caution).

## Repository layout

| Path | Contents |
|---|---|
| `src/` | metric implementations, data pipeline, left-tail measures, tests |
| `src/autoresearch/` | the evaluation harness: benchmark, partial-correlation decomposition, mediation, FDR + power, economic backtest, LLM/directed-search discovery, figures |
| `writeup/` | the paper (`volatility_persistence_paper.tex` / `.md`), compiled PDF, and `figures/` |
| `context/` | decision log, findings, gotchas (the project's reasoning trail) |

## Reproducing

```bash
uv sync                        # install dependencies (uses uv + pyproject.toml)
uv run python -u src/autoresearch/run_mechanism.py          # magnitude-vs-shape
uv run python -u src/autoresearch/run_robustness_fdr.py     # BH-FDR + power
uv run python -u src/autoresearch/run_mediation_nonoverlap.py
uv run python -u src/autoresearch/run_economic_backtest.py
uv run python -u src/autoresearch/make_figures.py           # regenerate figures
```

Each script prints a run header and writes a timestamped folder under `outputs/`.

### Data (not included)

The `data/` folder is git-ignored (large third-party datasets). To reproduce:
- **Crypto:** a CoinMarketCap daily OHLCV/market-cap dump (retaining delisted
  coins), e.g. via Kaggle.
- **Equity:** S&P 500 prices via `yfinance` and point-in-time index membership.

See `src/data_pipeline.py` for the exact loaders and expected file paths.

## Building the paper

The paper uses the ACM `acmart` (`sigconf`) class. Compile on Overleaf (select the
ACM template) or locally with a TeX distribution / `tectonic`:

```bash
tectonic -X compile writeup/volatility_persistence_paper.tex
```
