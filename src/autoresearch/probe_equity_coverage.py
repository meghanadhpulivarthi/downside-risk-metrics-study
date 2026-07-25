"""
Feasibility probe --- can we build a SURVIVORSHIP-FREE equity bed from the Kaggle
Huge Stock Market dump for the paper's two equity windows?

Gate before spending effort on the full bed + battery. For each window we:
  - get point-in-time S&P 500 members active in the window (historical constituents)
  - match them to Kaggle price files (which include delisted names)
  - for each monthly formation date, count members with >=60 trailing daily obs
    (a usable cross-section needs >= MIN_COINS_PER_DATE assets)
  - count how many matched members "die" inside the window (series ends early) --- the
    survivorship signal that free (yfinance-survivor) data hides
  - confirm a few famous delisted names are present (LEH, WCOM, ENE, ...)

Run: uv run --project <repo> python -u src/autoresearch/probe_equity_coverage.py
"""

import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import data_pipeline as dp
import evaluate as ev

# Config --- edit these directly
ROOT = ev.PROJECT_ROOT
HIST_CSV = ROOT / "data" / "sp500_historical_constituents.csv"
STOCKS_DIR = ROOT / "data" / "kaggle_huge_stock" / "Stocks"
TRAILING_DAYS = 180
FORWARD_DAYS = 90
MIN_COINS = ev.MIN_COINS_PER_DATE
WINDOWS = {
    "equity_1994_99_calm": ("1994-01-31", "1999-06-30"),
    "equity_2005_09_gfc":  ("2005-01-31", "2009-06-30"),
}
FAMOUS_DELISTED = ["LEH", "WCOM", "ENE", "BSC", "WM", "CIT", "ABK", "GM", "AIG"]


def probe_window(name, start, end, hist):
    print(f"\n{'='*70}\n[{name}]  {start} .. {end}\n{'='*70}")
    members = dp.pit_members_in_window(hist, pd.Timestamp(start) - pd.Timedelta(days=TRAILING_DAYS),
                                       pd.Timestamp(end) + pd.Timedelta(days=FORWARD_DAYS))
    members = sorted(members)
    prices, matched, missing = dp.load_kaggle_prices(members, STOCKS_DIR)
    print(f"  PIT members in window: {len(members)};  matched to Kaggle: {len(matched)};  missing: {len(missing)}")
    if prices.empty:
        print("  NO prices matched --- window not buildable from Kaggle.")
        return {"name": name, "buildable": False}

    prices = prices.sort_index()
    fdates = pd.date_range(start, end, freq="ME")
    per_date_counts, deaths = [], 0
    for fdate in fdates:
        n_ok = 0
        for t in prices.columns:
            trailing = prices.loc[fdate - pd.Timedelta(days=TRAILING_DAYS):fdate, t].dropna()
            if len(trailing) >= 60:
                n_ok += 1
        per_date_counts.append(n_ok)
    # deaths inside the window: last price before window end (proxy for delist/bankruptcy)
    win_end = pd.Timestamp(end) + pd.Timedelta(days=FORWARD_DAYS)
    for t in prices.columns:
        s = prices[t].dropna()
        if len(s) and pd.Timestamp(start) <= s.index.max() < win_end - pd.Timedelta(days=14):
            deaths += 1
    counts = np.array(per_date_counts)
    ok_dates = int((counts >= MIN_COINS).sum())
    print(f"  per-date usable cross-section (>=60 trailing obs): "
          f"min={counts.min()} median={int(np.median(counts))} max={counts.max()} over {len(counts)} dates")
    print(f"  formation dates clearing MIN_COINS={MIN_COINS}: {ok_dates}/{len(counts)}")
    print(f"  matched members that DIE inside the window (series ends early): {deaths}")
    return {"name": name, "buildable": ok_dates >= 3, "matched": len(matched),
            "median_cross_section": int(np.median(counts)), "ok_dates": ok_dates,
            "n_dates": len(counts), "deaths_in_window": deaths}


def main():
    now = datetime.datetime.now()
    print("=" * 70)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Stocks dir  : {STOCKS_DIR}  (exists={STOCKS_DIR.exists()})")
    print("=" * 70)

    hist = pd.read_csv(HIST_CSV)
    print(f"Loaded historical constituents: {len(hist)} rows")

    results = [probe_window(name, s, e, hist) for name, (s, e) in WINDOWS.items()]

    print("\n" + "=" * 70)
    print("FAMOUS DELISTED NAMES present in Kaggle?")
    _, matched, missing = dp.load_kaggle_prices(FAMOUS_DELISTED, STOCKS_DIR)
    print(f"  present: {matched}")
    print(f"  absent : {missing}")

    print("\n" + "=" * 70)
    print("VERDICT")
    for r in results:
        verdict = "BUILDABLE" if r.get("buildable") else "NOT buildable"
        print(f"  {r['name']:22s} {verdict}"
              + (f"  (median cross-section {r.get('median_cross_section')}, "
                 f"{r.get('ok_dates')}/{r.get('n_dates')} dates ok, {r.get('deaths_in_window')} deaths)"
                 if r.get("buildable") is not None else ""))
    print("=" * 70)


if __name__ == "__main__":
    main()
