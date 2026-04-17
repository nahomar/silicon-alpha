"""Sentiment ↔ price pattern detection.

Aggregates scraped items by day & ticker, then tests whether daily social
sentiment leads/lags that ticker's returns.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from .correlation import lead_lag


def _to_utc_day(ts) -> pd.Timestamp:
    """Coerce anything date-like to a tz-naive UTC-normalized Timestamp."""
    if ts is None or ts == "":
        return pd.NaT
    dt = pd.NaT
    # numeric epoch seconds
    try:
        f = float(ts)
        dt = pd.to_datetime(f, unit="s", utc=True, errors="coerce")
    except (TypeError, ValueError):
        pass
    if pd.isna(dt):
        dt = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(dt):
        return pd.NaT
    return dt.tz_convert("UTC").tz_localize(None).normalize()


def items_to_daily_sentiment(items: Iterable[dict]) -> pd.DataFrame:
    """Return a DataFrame of (date, ticker, mean_sentiment, volume)."""
    rows = []
    for it in items:
        sent = it.get("sentiment")
        if sent is None:
            continue
        ts = it.get("created_utc") or it.get("published") or it.get("created_at")
        day = _to_utc_day(ts)
        if pd.isna(day):
            continue
        for t in it.get("tickers", []) or ["_MARKET_"]:
            rows.append({"date": day, "ticker": t, "sentiment": float(sent)})
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "sentiment", "mentions"])
    df = pd.DataFrame(rows)
    agg = df.groupby(["date", "ticker"]).agg(
        sentiment=("sentiment", "mean"),
        mentions=("sentiment", "count"),
    ).reset_index()
    return agg


def align_sentiment_and_returns(
    sentiment_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily sentiment with per-ticker daily returns.

    Returns a long DataFrame: date, ticker, sentiment, mentions, ret.
    """
    if sentiment_df.empty:
        return pd.DataFrame()
    ret = prices.pct_change()
    long_ret = ret.stack().rename("ret").reset_index()
    long_ret.columns = ["date", "ticker", "ret"]
    # Always tz-naive day precision.
    long_ret["date"] = pd.to_datetime(long_ret["date"]).dt.tz_localize(None).dt.normalize()
    sentiment_df = sentiment_df.copy()
    sd = pd.to_datetime(sentiment_df["date"])
    if getattr(sd.dt, "tz", None) is not None:
        sd = sd.dt.tz_convert("UTC").dt.tz_localize(None)
    sentiment_df["date"] = sd.dt.normalize()
    merged = pd.merge(long_ret, sentiment_df, on=["date", "ticker"], how="inner")
    return merged


def lead_lag_per_ticker(merged: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """For each ticker, compute lead-lag between sentiment and returns."""
    results = []
    for ticker, g in merged.groupby("ticker"):
        g = g.sort_values("date")
        if len(g) < 10:
            continue
        lag, corr = lead_lag(g["sentiment"].reset_index(drop=True),
                             g["ret"].reset_index(drop=True), max_lag=max_lag)
        results.append({
            "ticker": ticker,
            "best_lag_days": lag,  # +k: sentiment leads returns by k days
            "corr": corr,
            "n_obs": len(g),
        })
    out = pd.DataFrame(results)
    if not out.empty:
        out = out.reindex(out["corr"].abs().sort_values(ascending=False).index)
    return out


def mention_spikes(sentiment_df: pd.DataFrame, window: int = 14, z: float = 2.0) -> pd.DataFrame:
    """Tickers whose mention-count today is z std-devs above trailing mean."""
    if sentiment_df.empty:
        return sentiment_df
    pivot = sentiment_df.pivot_table(index="date", columns="ticker",
                                     values="mentions", aggfunc="sum").fillna(0.0)
    mu = pivot.rolling(window).mean()
    sd = pivot.rolling(window).std().replace(0, np.nan)
    z_scores = (pivot - mu) / sd
    latest = z_scores.iloc[-1].dropna()
    spikes = latest[latest >= z].sort_values(ascending=False)
    return spikes.to_frame(name="mention_z")
