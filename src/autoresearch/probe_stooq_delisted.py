"""
Feasibility probe --- does Stooq carry daily history for the DELISTED S&P names that
the Kaggle dump lacked? Gate before any survivorship-free EQUITY bed effort.

For a real survivorship-free GFC bed we need the 2008 casualties (Lehman, Bear Stearns,
WaMu, Fannie/Freddie, ...) with daily prices spanning ~2005-2008. We test controls
(survivors) first to confirm the reader works, then the delisted names.

Run: uv run --project <repo> python -u src/autoresearch/probe_stooq_delisted.py
"""

import datetime

import pandas as pd
import pandas_datareader.data as pdr

# Config --- edit these directly
CONTROLS = ["AAPL.US", "IBM.US"]                       # survivors: confirm the reader works at all
DELISTED = ["LEH.US", "BSC.US", "WM.US", "WCOM.US",    # 2008 / early-2000s S&P casualties
            "ENE.US", "ABK.US", "FNM.US", "FRE.US", "AIG.US", "CIT.US"]
NEED_START, NEED_END = "2005-01-01", "2009-01-01"      # coverage a GFC bed would require


def probe(symbol):
    try:
        df = pdr.DataReader(symbol, "stooq")
    except Exception as exc:
        # loud: report the exact failure, never swallow silently
        return {"symbol": symbol, "ok": False, "error": repr(exc)[:120]}
    if df is None or df.empty:
        return {"symbol": symbol, "ok": False, "error": "empty frame"}
    df = df.sort_index()
    lo, hi = df.index.min(), df.index.max()
    covers_gfc = (lo <= pd.Timestamp(NEED_START)) and (hi >= pd.Timestamp("2008-06-01"))
    return {"symbol": symbol, "ok": True, "rows": len(df),
            "start": str(lo.date()), "end": str(hi.date()), "covers_gfc": bool(covers_gfc)}


def main():
    now = datetime.datetime.now()
    print("=" * 66)
    print(f"Stooq delisted-coverage probe --- {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GFC bed needs daily history spanning {NEED_START} .. {NEED_END}")
    print("=" * 66)

    print("\n[controls: survivors --- confirm the reader works]")
    for s in CONTROLS:
        print(f"  {s:10s} {probe(s)}")

    print("\n[delisted S&P casualties --- the real test]")
    hits = 0
    for s in DELISTED:
        r = probe(s)
        print(f"  {s:10s} {r}")
        if r.get("ok") and r.get("covers_gfc"):
            hits += 1

    print("\n" + "=" * 66)
    print(f"VERDICT: {hits}/{len(DELISTED)} delisted names have GFC-spanning daily history on Stooq.")
    print("  A survivorship-free equity GFC bed is viable only if this count is high.")
    print("=" * 66)


if __name__ == "__main__":
    main()
