"""
Multi-test-bed benchmark v2 — with SIGNIFICANCE and CALM regimes (revision).

Addresses the two critical review fixes:
  (1) every metric-vs-volatility comparison now gets a paired block-bootstrap
      95% CI on the per-date rho difference (not a bare point-estimate ranking);
  (2) adds CALM-regime test beds (crypto-2016 pre-bubble; equity 1994-1999 bull)
      so the evaluation is not all crash regimes.

A metric only "beats volatility" if the CI on (rho_metric - rho_vol) excludes 0.

Run: uv run --project <repo> python -u src/autoresearch/run_multi_testbed_v2.py
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
import left_tail as lt
from run_multi_testbed import compute_metrics, METRICS, TRAILING_DAYS, FORWARD_DAYS

OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"
CRYPTO_CSV = ev.PROJECT_ROOT / "data" / "kaggle_crypto" / "crypto-markets.csv"


def perdate_rhos(close, market_ret, formation_dates, universe_fn):
    """Return DataFrame index=formation_date, cols=METRICS of per-date cross-sectional rho."""
    per_date = {}
    for fdate in formation_dates:
        rows = []
        for asset in universe_fn(fdate):
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
            dd = lt.forward_drawdown(fwd, fdate + pd.Timedelta(days=FORWARD_DAYS))
            if np.isnan(dd):
                continue
            feats["fwd_dd"] = dd
            rows.append(feats)
        g = pd.DataFrame(rows)
        if len(g) < ev.MIN_COINS_PER_DATE:
            continue
        per_date[fdate] = {mm: (spearmanr(g[mm], g["fwd_dd"])[0] if g[mm].notna().sum() >= ev.MIN_COINS_PER_DATE else np.nan)
                           for mm in METRICS}
    return pd.DataFrame.from_dict(per_date, orient="index")


def block_bootstrap_diff(d, block=3, n_boot=3000):
    """Paired block-bootstrap mean + 95% CI for a per-date difference series."""
    d = np.asarray(d, dtype=float); d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(0)
    nb = int(np.ceil(n / block)); starts_max = max(1, n - block + 1)
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max, nb)
        means[i] = np.concatenate([d[s:s + block] for s in starts])[:n].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi)


def summarize_bed(name, rhos):
    """Per-metric mean rho + block-bootstrap CI on (metric - volatility) diff."""
    out = {"n_dates": int(len(rhos)), "metrics": {}}
    vol = rhos["volatility"]
    for mm in METRICS:
        mean_rho = float(rhos[mm].mean())
        if mm == "volatility":
            out["metrics"][mm] = {"rho": round(mean_rho, 3), "diff_vs_vol": 0.0, "ci": [0.0, 0.0], "sig_beats_vol": False}
            continue
        d, lo, hi = block_bootstrap_diff((rhos[mm] - vol).values)
        out["metrics"][mm] = {"rho": round(mean_rho, 3), "diff_vs_vol": round(d, 3),
                              "ci": [round(lo, 3), round(hi, 3)],
                              "sig_beats_vol": bool(lo > 0), "sig_worse": bool(hi < 0)}
    return out


def main():
    import datetime
    now = datetime.datetime.now()
    print("=" * 70)
    print(f"Multi-test-bed v2 (significance + calm regimes) — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_multi_testbed_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    beds = {}

    # --- Crypto beds: calm (2016), bull (2017), crash (2018) ---
    close, mcap = dp.load_crypto_panel(CRYPTO_CSV)
    cmkt = mcap.sum(axis=1, min_count=1).pct_change()
    def cuniv(fdate):
        try:
            return dp.point_in_time_top_n(mcap.loc[:fdate], fdate, 100)
        except ValueError:
            return []
    for name, dates in [("crypto_2016_calm", pd.date_range("2016-02-29", "2016-09-30", freq="ME")),
                        ("crypto_2017_bull", pd.date_range("2017-01-31", "2017-09-30", freq="ME")),
                        ("crypto_2018_crash", pd.date_range("2018-02-28", "2018-08-31", freq="ME"))]:
        beds[name] = perdate_rhos(close, cmkt, dates, cuniv)
        print(f"  {name}: {len(beds[name])} dates")

    # --- Equity beds from the 1978-2011 cache: calm bull (1994-99) and GFC (2005-09) ---
    tickers = dp.get_current_sp500_tickers(ev.PROJECT_ROOT / "data" / "sp500_current.csv")
    eq = dp.download_adjusted_close(tickers, "1993-01-01", "2010-01-01",
                                    ev.PROJECT_ROOT / "data" / "yf_sp500_prices_1978_2011.parquet")
    gspc = dp.download_adjusted_close(["^GSPC"], "1993-01-01", "2010-01-01",
                                      ev.PROJECT_ROOT / "data" / "yf_benchmark_1978_2011.parquet")
    emkt = gspc["^GSPC"].pct_change(); ecols = list(eq.columns)
    for name, dates in [("equity_1994_99_calm", pd.date_range("1994-01-31", "1999-06-30", freq="ME")),
                        ("equity_2005_09_gfc", pd.date_range("2005-01-31", "2009-06-30", freq="ME"))]:
        beds[name] = perdate_rhos(eq, emkt, dates, lambda d: ecols)
        print(f"  {name}: {len(beds[name])} dates")

    results = {name: summarize_bed(name, rhos) for name, rhos in beds.items() if len(rhos) >= 3}

    print("\n" + "=" * 70)
    print("Per bed: does any metric SIGNIFICANTLY beat volatility (95% CI on diff excludes 0)?")
    any_sig = []
    for name, res in results.items():
        vol_rho = res["metrics"]["volatility"]["rho"]
        sig_better = [mm for mm in METRICS if res["metrics"][mm].get("sig_beats_vol")]
        print(f"\n[{name}]  n_dates={res['n_dates']}  vol rho={vol_rho}")
        for mm in METRICS:
            if mm == "volatility":
                continue
            r = res["metrics"][mm]
            flag = "  ** SIG BEATS VOL **" if r["sig_beats_vol"] else (" (sig worse)" if r.get("sig_worse") else "")
            print(f"    {mm:16s} rho={r['rho']:+.3f}  diff={r['diff_vs_vol']:+.3f} CI{r['ci']}{flag}")
        if sig_better:
            any_sig.append((name, sig_better))

    print("\n" + "=" * 70)
    print("VERDICT: metrics that SIGNIFICANTLY beat volatility, by bed:")
    print(f"  {any_sig if any_sig else 'NONE on any test bed'}")
    with open(output_dir / "multi_testbed_v2.json", "w") as h:
        json.dump(results, h, indent=2)
    print(f"Saved -> {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
