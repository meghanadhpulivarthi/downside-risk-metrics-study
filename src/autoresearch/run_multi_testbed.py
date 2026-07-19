"""
Multi-test-bed benchmark — the headline needs MORE THAN ONE test set.

A single test bed (crypto 2018) cannot support "nothing beats volatility OOS". Here
we evaluate the full metric set on THREE independent test beds:
  - crypto_2017 : crypto, 2017 formation dates (bull regime)
  - crypto_2018 : crypto, 2018 formation dates (crash regime)
  - equity_gfc  : S&P 500 survivors, 2005-2009 monthly (includes the 2008 crash)
and ask, per test bed, whether ANY downside/tail metric beats plain volatility at
forecasting forward 90d drawdown (per-date Spearman).

Metrics include the previously-benchmarked ones PLUS three added on request:
  down_beta (Levi-Welch / Ang-Chen-Xing), ltd_crash (Chabi-Yo lower-tail dependence),
  gda_voldown (Farago-Tedongap volatility-downside proxy).

Run: uv run --project <repo> python -u src/autoresearch/run_multi_testbed.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import evaluate as ev
import data_pipeline as dp
import partial_moments as pm

TRAILING_DAYS = 180
FORWARD_DAYS = 90
CRYPTO_CSV = ev.PROJECT_ROOT / "data" / "kaggle_crypto" / "crypto-markets.csv"
OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"


# ---------------- metric functions (all: higher = more predicted downside) ----------------

def compute_metrics(r, m):
    """r=asset trailing daily returns, m=market trailing daily returns (aligned)."""
    if len(r) < 60:
        return None
    downside = r[r < 0]
    out = {
        "volatility": float(np.std(r)),
        "downside_dev": float(np.sqrt(np.mean(downside ** 2))) if len(downside) else 0.0,
        "es5": float(-np.mean(r[r <= np.quantile(r, 0.05)])),
        "var5": float(-np.quantile(r, 0.05)),
    }
    # Viole-Nawrocki UPM/LPM (higher = riskier => negate ratio)
    try:
        out["vn_ratio"] = -pm.explanatory_metric(r, np.maximum(m, 0.0), q=1, n=1)
    except ZeroDivisionError:
        out["vn_ratio"] = np.nan
    # Kelly-Jiang Hill loss-tail exponent
    losses = -r[r < 0]
    if len(losses) >= 15:
        u = np.quantile(losses, 0.95); ex = losses[losses >= u]
        out["hill_tail"] = float(np.mean(np.log(ex / u))) if len(ex) >= 5 and u > 0 else np.nan
    else:
        out["hill_tail"] = np.nan
    # BPQ negative-negative semibeta
    dd = (r < 0) & (m < 0); denom = np.sum(m[m < 0] ** 2)
    out["down_semibeta"] = float(np.sum(r[dd] * m[dd]) / denom) if denom > 0 and dd.sum() >= 5 else np.nan
    # Levi-Welch / Ang-Chen-Xing downside beta
    down = m < 0
    out["down_beta"] = float(np.cov(r[down], m[down])[0, 1] / np.var(m[down])) if down.sum() >= 10 and np.var(m[down]) > 0 else np.nan
    # Chabi-Yo lower-tail dependence (co-exceedance in bottom quintile)
    if len(r) >= 30:
        rq, mq = np.quantile(r, 0.2), np.quantile(m, 0.2)
        m_ex = m <= mq
        out["ltd_crash"] = float(np.mean((r <= rq)[m_ex])) if m_ex.sum() >= 3 else np.nan
    else:
        out["ltd_crash"] = np.nan
    # Farago-Tedongap volatility-downside proxy: exposure to market vol in down states
    if down.sum() >= 10:
        mv = (m[down] ** 2); out["gda_voldown"] = float(np.cov(r[down], mv)[0, 1]) if np.var(mv) > 0 else np.nan
    else:
        out["gda_voldown"] = np.nan
    return out


METRICS = ["volatility", "downside_dev", "es5", "var5", "vn_ratio", "hill_tail",
           "down_semibeta", "down_beta", "ltd_crash", "gda_voldown"]


def score_testbed(close, market_ret, formation_dates, universe_fn):
    """Per-date Spearman(metric, forward drawdown), averaged; for each metric."""
    rows = []
    for fdate in formation_dates:
        universe = universe_fn(fdate)
        for asset in universe:
            if asset not in close.columns:
                continue
            r = close.loc[fdate - pd.Timedelta(days=TRAILING_DAYS):fdate, asset].dropna().pct_change().dropna()
            if len(r) < 60:
                continue
            m = market_ret.reindex(r.index).fillna(0.0).values
            feats = compute_metrics(r.values, m)
            if feats is None:
                continue
            fwd = close.loc[fdate:fdate + pd.Timedelta(days=FORWARD_DAYS), asset]
            import left_tail as lt
            dd = lt.forward_drawdown(fwd, fdate + pd.Timedelta(days=FORWARD_DAYS))
            if np.isnan(dd):
                continue
            feats.update({"formation_date": fdate, "fwd_dd": dd})
            rows.append(feats)
    df = pd.DataFrame(rows)
    scores = {}
    for metric in METRICS:
        rhos = []
        for _, g in df.groupby("formation_date"):
            gg = g[[metric, "fwd_dd"]].dropna()
            if len(gg) >= ev.MIN_COINS_PER_DATE:
                rhos.append(spearmanr(gg[metric], gg["fwd_dd"])[0])
        scores[metric] = float(np.mean(rhos)) if rhos else np.nan
    return scores, len(df)


def main():
    import datetime
    now = datetime.datetime.now()
    print("=" * 70)
    print(f"Multi-test-bed benchmark — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_multi_testbed"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ---- Crypto test beds (two regimes) ----
    close, mcap = dp.load_crypto_panel(CRYPTO_CSV)
    market = mcap.sum(axis=1, min_count=1).pct_change()
    def crypto_universe(fdate):
        try:
            return dp.point_in_time_top_n(mcap.loc[:fdate], fdate, 100)
        except ValueError:
            return []
    for name, dates in [("crypto_2017", pd.date_range("2017-01-31", "2017-09-30", freq="ME")),
                        ("crypto_2018", pd.date_range("2018-02-28", "2018-08-31", freq="ME"))]:
        scores, n = score_testbed(close, market, dates, crypto_universe)
        results[name] = scores
        print(f"\n[{name}] n={n}")

    # ---- Equity test bed: S&P 500 survivors incl. the 2008 crash ----
    tickers = dp.get_current_sp500_tickers(ev.PROJECT_ROOT / "data" / "sp500_current.csv")
    eq = dp.download_adjusted_close(tickers, "2004-01-01", "2010-01-01",
                                    ev.PROJECT_ROOT / "data" / "yf_sp500_prices_1978_2011.parquet")
    gspc = dp.download_adjusted_close(["^GSPC"], "2004-01-01", "2010-01-01",
                                      ev.PROJECT_ROOT / "data" / "yf_benchmark_1978_2011.parquet")
    eq_market = gspc["^GSPC"].pct_change()
    eq_cols = [t for t in eq.columns]
    scores, n = score_testbed(eq, eq_market, pd.date_range("2005-01-31", "2009-06-30", freq="ME"),
                              lambda d: eq_cols)
    results["equity_gfc"] = scores
    print(f"\n[equity_gfc] n={n}")

    # ---- Report matrix ----
    print("\n" + "=" * 70)
    print("TEST drawdown-forecast Spearman by test bed (higher = better; vol = baseline):")
    header = f"{'metric':14s}" + "".join(f"{tb:>13s}" for tb in results)
    print(header)
    for metric in METRICS:
        line = f"{metric:14s}" + "".join(f"{results[tb].get(metric, float('nan')):13.3f}" for tb in results)
        star = "  <- baseline" if metric == "volatility" else ""
        print(line + star)
    print("\nMetrics beating volatility per test bed:")
    for tb in results:
        vol = results[tb]["volatility"]
        beats = [mm for mm in METRICS if mm != "volatility" and results[tb].get(mm, np.nan) > vol]
        print(f"  {tb:14s}: {beats if beats else 'NONE'}")

    with open(output_dir / "multi_testbed.json", "w") as h:
        json.dump(results, h, indent=2)
    print(f"\nSaved -> {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
