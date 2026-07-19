"""
Data feasibility probe for the partial-moments audit.

Question it answers: can a *free* survivorship-bias-free-ish US equity path be
built here? Two capabilities are required and tested independently:
  (1) point-in-time S&P 500 membership  -> Wikipedia change log
  (2) prices for firms that later DELISTED -> Stooq (yfinance usually drops them)

This is a throwaway spike, not the pipeline. It only prints what worked.
"""

import sys
import datetime

import pandas as pd

now = datetime.datetime.now()
print("=" * 60)
print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Script      : {__file__}")
print("=" * 60)

# Known delisted US tickers to test survivorship coverage.
# LEH = Lehman Brothers (bankrupt 2008), ENE = Enron (2001), WCOM = WorldCom (2002)
DELISTED_TEST_TICKERS = ["LEH", "ENE", "WCOM"]


def probe_wikipedia_pit_membership():
    print("\n[1] Wikipedia point-in-time S&P 500 membership")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
    except Exception as exc:
        print(f"    FAILED to read Wikipedia tables: {exc!r}")
        return False
    print(f"    got {len(tables)} tables from the page")
    current = tables[0]
    print(f"    current constituents: {len(current)} rows, cols={list(current.columns)[:4]}...")
    if len(tables) > 1:
        changes = tables[1]
        print(f"    changes/history table: {len(changes)} rows -> point-in-time IS reconstructable")
        # earliest change date tells us how far back membership can be rebuilt
        print(f"    change-table columns: {list(changes.columns)}")
        return True
    print("    NO changes table found -> point-in-time NOT reconstructable from this page")
    return False


def probe_stooq_delisted():
    print("\n[2] Stooq prices for delisted tickers (survivorship coverage)")
    from pandas_datareader import data as pdr

    any_ok = False
    for ticker in DELISTED_TEST_TICKERS:
        symbol = f"{ticker}.US"
        try:
            frame = pdr.DataReader(symbol, "stooq")
        except Exception as exc:
            print(f"    {symbol}: FAILED {exc!r}")
            continue
        if frame is None or frame.empty:
            print(f"    {symbol}: empty (not retained)")
            continue
        first = frame.index.min().date()
        last = frame.index.max().date()
        print(f"    {symbol}: {len(frame)} rows, {first} -> {last}  OK (delisted data retained)")
        any_ok = True
    return any_ok


def probe_yfinance_delisted():
    print("\n[3] yfinance for the same delisted tickers (expected to be sparse)")
    import yfinance as yf

    for ticker in DELISTED_TEST_TICKERS:
        try:
            frame = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        except Exception as exc:
            print(f"    {ticker}: FAILED {exc!r}")
            continue
        print(f"    {ticker}: {len(frame)} rows")


wiki_ok = probe_wikipedia_pit_membership()
stooq_ok = probe_stooq_delisted()
probe_yfinance_delisted()

print("\n" + "=" * 60)
print("VERDICT")
print(f"  point-in-time membership (Wikipedia): {'YES' if wiki_ok else 'NO'}")
print(f"  delisted-firm prices (Stooq)        : {'YES' if stooq_ok else 'NO'}")
free_path_ok = wiki_ok and stooq_ok
print(f"  => free survivorship-free-ish path viable: {'YES' if free_path_ok else 'NO'}")
print("=" * 60)
sys.exit(0 if free_path_ok else 1)
