"""
Reusable data helpers shared by both audit arms.

Biased arm (Phase 1) uses yfinance for the *surviving* universe on purpose.
Bias-free arm (Phase 2) will reuse the return/benchmark helpers on the Kaggle
panel. Kept deliberately small — no wrapper classes (code-style rule).
"""

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (research; partial-moments-audit)"}


def get_current_sp500_tickers(cache_path):
    """
    Today's S&P 500 constituents from Wikipedia (survivorship-BIASED on purpose —
    this is the biased arm's universe). Cached to CSV. yfinance uses '-' where
    tickers have '.', so BRK.B -> BRK-B.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        frame = pd.read_csv(cache_path)
        print(f"Loaded {len(frame)} S&P 500 tickers from cache {cache_path}")
        return frame["yf_ticker"].tolist()

    response = requests.get(WIKI_SP500_URL, headers=BROWSER_HEADERS, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    constituents = tables[0]
    tickers = constituents["Symbol"].astype(str).str.strip()
    frame = pd.DataFrame({
        "wiki_symbol": tickers,
        "yf_ticker": tickers.str.replace(".", "-", regex=False),
        "security": constituents["Security"].astype(str).values,
    })
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    print(f"Fetched {len(frame)} S&P 500 tickers from Wikipedia -> {cache_path}")
    return frame["yf_ticker"].tolist()


def download_adjusted_close(tickers, start, end, cache_path):
    """
    Daily auto-adjusted close for `tickers` over [start, end], one column each.
    Cached to parquet and restartable: only missing tickers are fetched.
    Tickers that return no data are dropped WITH a printed count (never silent).
    """
    import yfinance as yf

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = pd.DataFrame()
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        print(f"Price cache {cache_path}: {cached.shape[1]} tickers already present")

    missing = [ticker for ticker in tickers if ticker not in cached.columns]
    if not missing:
        print("All requested tickers already cached; no download needed.")
        return cached[[t for t in tickers if t in cached.columns]]

    print(f"Downloading {len(missing)} missing tickers via yfinance ...")
    raw = yf.download(
        missing, start=start, end=end,
        auto_adjust=True, progress=True, group_by="column", threads=True,
    )
    # With multiple tickers yfinance returns MultiIndex columns (field, ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        # single ticker -> flat columns; wrap into one named column
        close = raw[["Close"]].rename(columns={"Close": missing[0]})

    empties = [t for t in close.columns if close[t].dropna().empty]
    if empties:
        # Loud: these are almost always delisted/renamed — the survivorship signal.
        print(f"WARNING: {len(empties)} tickers returned no data (dropped): {empties[:10]}...")
        close = close.drop(columns=empties)

    combined = close if cached.empty else cached.join(close, how="outer")
    combined.to_parquet(cache_path)
    print(f"Saved price cache: {combined.shape[1]} tickers -> {cache_path}")
    return combined[[t for t in tickers if t in combined.columns]]


def to_daily_returns(prices):
    """Simple daily returns from an adjusted-close frame (rows=dates, cols=tickers)."""
    return prices.sort_index().pct_change().iloc[1:]


def to_monthly_returns(prices):
    """
    Simple MONTHLY returns from a daily adjusted-close frame: resample to
    month-end last price, then pct_change. The paper builds its metric at ~monthly
    frequency (11yr ≈ 132 obs), where the lag-1 autocorrelation term is not inert
    (see context/gotchas.md).
    """
    monthly = prices.sort_index().resample("ME").last()
    return monthly.pct_change().iloc[1:]


def annualized_pct_to_periodic_rate(rate_pct, periods_per_year):
    """
    Convert an annualized percent rate (e.g. ^IRX 13-week T-bill, where 0.05 means
    0.05%) to a per-period fraction. periods_per_year = 252 (daily) or 12 (monthly).
    Small but nonzero; matters for the double benchmark on down periods.
    """
    return (rate_pct / 100.0) / periods_per_year


# ---------------------------------------------------------------- Kaggle / PIT

import re


def _normalize_variants(ticker):
    """Symbol variants to try when matching a constituent ticker to a Kaggle file."""
    t = ticker.strip().upper()
    t = re.sub(r"-\d{6}$", "", t)  # strip fja05680 -YYYYMM removal-date suffix
    return {t, t.replace(".", "-"), t.replace(".", ""), t.replace("-", ""), t.rstrip("Q")}


def load_kaggle_prices(tickers, kaggle_stocks_dir):
    """
    Load daily close for `tickers` from the Kaggle Huge Stock Market dump
    (Data/Stocks/<sym>.us.txt, columns Date,Open,High,Low,Close,Volume,OpenInt;
    prices are already split/dividend adjusted per the dataset docs).

    Returns (prices_df[dates x matched tickers under their INPUT name], matched, missing).
    Missing/empty files are reported, never silently skipped.
    """
    kaggle_stocks_dir = Path(kaggle_stocks_dir)
    file_index = {p.name.split(".us.txt")[0].upper(): p for p in kaggle_stocks_dir.glob("*.txt")}

    series_by_ticker = {}
    matched, missing = [], []
    for ticker in tickers:
        path = None
        for variant in _normalize_variants(ticker):
            if variant in file_index:
                path = file_index[variant]
                break
        if path is None:
            missing.append(ticker)
            continue
        frame = pd.read_csv(path, usecols=["Date", "Close"])
        if frame.empty:
            print(f"    Kaggle file for {ticker} ({path.name}) is empty — skipping")
            missing.append(ticker)
            continue
        series = frame.set_index(pd.to_datetime(frame["Date"]))["Close"].sort_index()
        series_by_ticker[ticker] = series
        matched.append(ticker)

    print(f"    Kaggle price load: matched {len(matched)}/{len(tickers)}, missing {len(missing)}")
    prices = pd.DataFrame(series_by_ticker) if series_by_ticker else pd.DataFrame()
    return prices, matched, missing


def load_crypto_panel(csv_path):
    """
    Load the CoinMarketCap 'crypto-markets' dump into wide close + market-cap
    panels (rows=date, cols=coin slug). Crashed-but-listed coins are retained
    (the failure tail), so point-in-time selection here is genuinely
    survivorship-free in a way free equity data is not.
    """
    frame = pd.read_csv(csv_path, usecols=["slug", "date", "close", "market"])
    frame["date"] = pd.to_datetime(frame["date"])
    close = frame.pivot_table(index="date", columns="slug", values="close")
    market_cap = frame.pivot_table(index="date", columns="slug", values="market")
    return close.sort_index(), market_cap.sort_index()


def point_in_time_top_n(market_cap, as_of_date, n):
    """
    Top-n coin slugs by market cap AS OF a past date (nearest available row on or
    before as_of_date). Selection at T0 — not by survival to the dataset end — is
    what makes the universe survivorship-free.
    """
    available = market_cap.loc[:as_of_date]
    if available.empty:
        raise ValueError(f"no market-cap rows on or before {as_of_date}")
    snapshot = available.iloc[-1].dropna()
    return snapshot.sort_values(ascending=False).head(n).index.tolist()


def max_drawdown(price_series):
    """Max peak-to-trough drawdown magnitude (positive) over a price series."""
    clean = price_series.dropna()
    if len(clean) < 2:
        return np.nan
    running_max = clean.cummax()
    drawdown = clean / running_max - 1.0
    return float(-drawdown.min())


def pit_members_in_window(historical_constituents, start, end):
    """
    Union of all point-in-time S&P 500 tickers that were members at ANY snapshot in
    [start, end], from an fja05680/sp500 'date,tickers' frame. Returns normalized
    base tickers (removal-date suffix stripped). No membership = no bias-free test,
    so this is the spine of the universe.
    """
    frame = historical_constituents.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    window = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    members = set()
    for tickers_str in window["tickers"]:
        for raw in str(tickers_str).split(","):
            if raw.strip():
                base = re.sub(r"-\d{6}$", "", raw.strip().upper())
                members.add(base)
    return members
