"""Market data loader.

Pulls OHLCV from yfinance for a universe of tickers and index/sector ETFs.
All analysis uses daily close unless otherwise noted.
"""
from __future__ import annotations

import logging
from typing import Iterable, Dict

import pandas as pd

log = logging.getLogger(__name__)

INDEXES = ["^GSPC", "^NDX", "^DJI", "^VIX"]
SECTOR_ETFS = {
    "Tech": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "ConsDisc": "XLY",
    "ConsStaples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "RealEstate": "XLRE",
    "CommServ": "XLC",
}
MEGA_CAPS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "ASML", "TSM"]


def load_prices(
    tickers: Iterable[str],
    period: str = "6mo",
    interval: str = "1d",
    retries: int = 3,
    backoff_s: float = 8.0,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices indexed by date.

    Retries with backoff on rate-limit errors and falls back to a shorter
    period if the long one keeps failing.
    """
    import time
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance is required: pip install yfinance") from e
    tickers = list(dict.fromkeys(tickers))
    period_ladder = [period, "2y", "1y", "6mo"]
    # de-dupe while preserving order
    period_ladder = list(dict.fromkeys(period_ladder))

    last_err = None
    for p in period_ladder:
        for attempt in range(retries):
            try:
                data = yf.download(
                    tickers, period=p, interval=interval,
                    auto_adjust=True, progress=False, threads=False,
                )
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data["Close"]
                else:
                    prices = data[["Close"]].rename(columns={"Close": tickers[0]})
                prices = prices.dropna(how="all")
                if prices.empty or prices.shape[1] == 0:
                    raise RuntimeError("empty frame")
                if p != period:
                    log.warning("yfinance: fell back to period=%s", p)
                return prices
            except Exception as e:
                last_err = e
                log.warning("yfinance %s attempt %d/%d failed: %s", p, attempt + 1, retries, e)
                time.sleep(backoff_s * (attempt + 1))
    raise RuntimeError(f"yfinance failed for all periods: {last_err}")


def default_universe() -> list[str]:
    return INDEXES + list(SECTOR_ETFS.values()) + MEGA_CAPS


def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")
