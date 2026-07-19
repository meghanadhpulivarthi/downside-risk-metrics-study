"""
Volatility-persistence MECHANISM test.

The 5-bed benchmark (run_multi_testbed_v2.py) shows no downside-risk metric
reliably beats plain volatility at forecasting forward drawdown. This script asks
*why*, and tests one explanation:

    Cross-sectional drawdown predictability is a VOLATILITY-PERSISTENCE phenomenon.
    Trailing volatility forecasts forward volatility, which drives forward drawdown;
    downside-SHAPE information (asymmetry, tail heaviness, co-crash structure) carries
    ~no INCREMENTAL out-of-sample signal once volatility is controlled for.

Two evidence blocks, per test bed, per formation date (cross-sectional Spearman),
aggregated with a block-bootstrap 95% CI:

  M1 (mechanism): partial Spearman rho(M, fwd_dd | volatility) for every competing
                  metric M. If ~0 => shape adds nothing beyond volatility. A robust
                  nonzero partial rho would instead be a genuine incremental signal.
  M2 (chain):     rho(trailing_vol, forward_vol)  [vol is persistent]  and
                  rho(forward_vol, fwd_dd)        [forward vol drives drawdown].

We reuse compute_metrics / METRICS / horizons from run_multi_testbed and the paired
block-bootstrap from run_multi_testbed_v2, and the same 5 beds and universes.

Run: uv run --project <repo> python -u src/autoresearch/run_mechanism.py
"""

import datetime
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
from run_multi_testbed_v2 import block_bootstrap_diff

# Config — edit these directly
CRYPTO_CSV = ev.PROJECT_ROOT / "data" / "kaggle_crypto" / "crypto-markets.csv"
OUTPUT_DIR = ev.PROJECT_ROOT / "outputs"
MIN_PAIRS = ev.MIN_COINS_PER_DATE  # min cross-sectional assets to trust a per-date rho


def _spearman(a, b):
    """Cross-sectional Spearman over aligned finite pairs; NaN if too few pairs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < MIN_PAIRS:
        return np.nan
    return spearmanr(a[mask], b[mask])[0]


def _partial_spearman(r_m_dd, r_m_vol, r_vol_dd):
    """Partial Spearman of (M, fwd_dd) controlling for volatility, from the three
    pairwise rank correlations. NaN-safe."""
    if not all(np.isfinite([r_m_dd, r_m_vol, r_vol_dd])):
        return np.nan
    denom = np.sqrt((1.0 - r_m_vol ** 2) * (1.0 - r_vol_dd ** 2))
    if denom <= 0:
        return np.nan  # a pairwise rho hit +-1; partial is undefined here
    return (r_m_dd - r_m_vol * r_vol_dd) / denom


def per_date_series(close, market_ret, formation_dates, universe_fn):
    """
    For each formation date, build the cross-sectional frame (one row per asset) of
    all metrics + forward drawdown + forward realized volatility, then compute the
    per-date correlations for M1 (partial rhos) and M2 (persistence chain).

    Returns a dict of lists (one value per usable date) keyed by:
      f"partial::{metric}"  -> partial rho(metric, fwd_dd | vol)
      f"raw::{metric}"      -> raw rho(metric, fwd_dd)
      "chain_tvol_fvol"     -> rho(trailing_vol, forward_vol)
      "chain_fvol_dd"       -> rho(forward_vol, fwd_dd)
    """
    series = {f"partial::{m}": [] for m in METRICS if m != "volatility"}
    series.update({f"raw::{m}": [] for m in METRICS if m != "volatility"})
    series["raw::volatility"] = []
    series["chain_tvol_fvol"] = []
    series["chain_fvol_dd"] = []

    n_dates_used = 0
    for fdate in formation_dates:
        rows = []
        for asset in universe_fn(fdate):
            if asset not in close.columns:
                continue
            trailing = close.loc[fdate - pd.Timedelta(days=TRAILING_DAYS):fdate, asset].dropna()
            r = trailing.pct_change().dropna()
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
            # Forward realized volatility over the SAME window used for drawdown.
            fwd_r = fwd.dropna().pct_change().dropna()
            fwd_vol = float(np.std(fwd_r.values)) if len(fwd_r) >= 5 else np.nan
            feats["fwd_dd"] = dd
            feats["fwd_vol"] = fwd_vol
            rows.append(feats)

        g = pd.DataFrame(rows)
        if len(g) < MIN_PAIRS:
            # Too few assets on this date to trust a cross-sectional rho — skip, loudly.
            print(f"    skip {fdate.date()}: only {len(g)} assets (<{MIN_PAIRS})")
            continue
        n_dates_used += 1

        vol = g["volatility"].values
        dd = g["fwd_dd"].values
        r_vol_dd = _spearman(vol, dd)
        series["raw::volatility"].append(r_vol_dd)

        for metric in METRICS:
            if metric == "volatility":
                continue
            mv = g[metric].values
            r_m_dd = _spearman(mv, dd)
            r_m_vol = _spearman(mv, vol)
            series[f"raw::{metric}"].append(r_m_dd)
            series[f"partial::{metric}"].append(_partial_spearman(r_m_dd, r_m_vol, r_vol_dd))

        # M2 persistence chain
        series["chain_tvol_fvol"].append(_spearman(vol, g["fwd_vol"].values))
        series["chain_fvol_dd"].append(_spearman(g["fwd_vol"].values, dd))

    return series, n_dates_used


def summarize(series, n_dates):
    """Mean + block-bootstrap 95% CI for every per-date series; flag CI-excludes-0."""
    out = {"n_dates": n_dates, "partial": {}, "raw": {}, "chain": {}}
    for key, vals in series.items():
        mean, lo, hi = block_bootstrap_diff(vals)
        record = {"mean": round(mean, 3), "ci": [round(lo, 3), round(hi, 3)],
                  "sig_pos": bool(np.isfinite(lo) and lo > 0),
                  "sig_neg": bool(np.isfinite(hi) and hi < 0)}
        if key.startswith("partial::"):
            out["partial"][key.split("::", 1)[1]] = record
        elif key.startswith("raw::"):
            out["raw"][key.split("::", 1)[1]] = record
        else:
            out["chain"][key] = record
    return out


def build_beds():
    """The same 5 test beds as run_multi_testbed_v2: crypto calm/bull/crash + equity calm/GFC."""
    beds = {}

    close, mcap = dp.load_crypto_panel(CRYPTO_CSV)
    cmkt = mcap.sum(axis=1, min_count=1).pct_change()

    def cuniv(fdate):
        try:
            return dp.point_in_time_top_n(mcap.loc[:fdate], fdate, 100)
        except ValueError:
            return []

    beds["crypto_2016_calm"] = (close, cmkt,
                                pd.date_range("2016-02-29", "2016-09-30", freq="ME"), cuniv)
    beds["crypto_2017_bull"] = (close, cmkt,
                                pd.date_range("2017-01-31", "2017-09-30", freq="ME"), cuniv)
    beds["crypto_2018_crash"] = (close, cmkt,
                                 pd.date_range("2018-02-28", "2018-08-31", freq="ME"), cuniv)

    tickers = dp.get_current_sp500_tickers(ev.PROJECT_ROOT / "data" / "sp500_current.csv")
    eq = dp.download_adjusted_close(tickers, "1993-01-01", "2010-01-01",
                                    ev.PROJECT_ROOT / "data" / "yf_sp500_prices_1978_2011.parquet")
    gspc = dp.download_adjusted_close(["^GSPC"], "1993-01-01", "2010-01-01",
                                      ev.PROJECT_ROOT / "data" / "yf_benchmark_1978_2011.parquet")
    emkt = gspc["^GSPC"].pct_change()
    ecols = list(eq.columns)
    beds["equity_1994_99_calm"] = (eq, emkt,
                                   pd.date_range("1994-01-31", "1999-06-30", freq="ME"),
                                   lambda d: ecols)
    beds["equity_2005_09_gfc"] = (eq, emkt,
                                  pd.date_range("2005-01-31", "2009-06-30", freq="ME"),
                                  lambda d: ecols)
    return beds


def main():
    now = datetime.datetime.now()
    print("=" * 70)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : TRAILING={TRAILING_DAYS}d FORWARD={FORWARD_DAYS}d MIN_PAIRS={MIN_PAIRS}")
    print("=" * 70)

    output_dir = OUTPUT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_mechanism"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    beds = build_beds()
    results = {}
    for name, (close, mkt, dates, univ) in beds.items():
        print(f"\n[{name}] scoring {len(dates)} formation dates ...")
        series, n_dates = per_date_series(close, mkt, dates, univ)
        if n_dates < 3:
            print(f"  SKIP bed {name}: only {n_dates} usable dates")
            continue
        results[name] = summarize(series, n_dates)
        print(f"  used {n_dates} dates")

    # --- M1: partial rho (metric | vol) — the mechanism ---
    print("\n" + "=" * 70)
    print("M1  partial Spearman rho(metric, fwd_dd | volatility)  [~0 => no incremental signal]")
    print("=" * 70)
    competing = [m for m in METRICS if m != "volatility"]
    header = f"{'metric':16s}" + "".join(f"{n.split('_')[0][:7]:>9s}" for n in results)
    print(header)
    for metric in competing:
        cells = []
        for name in results:
            rec = results[name]["partial"][metric]
            flag = "*" if rec["sig_pos"] else ("-" if rec["sig_neg"] else " ")
            cells.append(f"{rec['mean']:+.2f}{flag}")
        print(f"{metric:16s}" + "".join(f"{c:>9s}" for c in cells))
    print("  * = partial rho CI>0 (adds signal beyond vol);  - = CI<0;  blank = indistinguishable from 0")

    # --- M2: persistence chain ---
    print("\n" + "=" * 70)
    print("M2  persistence chain  (mean rho [95% CI])")
    print("=" * 70)
    for name in results:
        c = results[name]["chain"]
        vdd = results[name]["raw"]["volatility"]
        print(f"[{name}]  n_dates={results[name]['n_dates']}")
        print(f"    trailing_vol -> forward_vol : {c['chain_tvol_fvol']['mean']:+.3f} "
              f"CI{c['chain_tvol_fvol']['ci']}")
        print(f"    forward_vol  -> fwd_drawdown: {c['chain_fvol_dd']['mean']:+.3f} "
              f"CI{c['chain_fvol_dd']['ci']}")
        print(f"    (trailing_vol-> fwd_drawdown: {vdd['mean']:+.3f} CI{vdd['ci']}  = baseline skill)")

    out_path = output_dir / "mechanism.json"
    with open(out_path, "w") as h:
        json.dump(results, h, indent=2)
    print(f"\nResults saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
