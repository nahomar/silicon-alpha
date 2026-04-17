"""Unified dataset — the 'monolithic world state'.

Every timestep produces a single vector combining:
  - per-ticker log returns
  - per-ticker 21-day realized vol
  - per-ticker cross-sectional rank of return (relative strength)
  - market-wide sentiment (from scraped history, aligned by day)

This feeds the tokenizer, transformer, diffusion model, and RL env.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from data.market_data import load_prices
from data.storage import load_history
from .config import Config

log = logging.getLogger(__name__)


def _daily_market_sentiment() -> pd.Series:
    """Mean VADER compound per day from our scrape history."""
    h = load_history()
    if h.empty or "sentiment" not in h.columns:
        return pd.Series(dtype=float, name="sentiment")
    ts = pd.to_datetime(h.get("created_utc", h.get("published", h.get("created_at"))),
                        errors="coerce", utc=True)
    h = h.assign(_dt=ts).dropna(subset=["_dt"])
    h["_day"] = h["_dt"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    s = h.groupby("_day")["sentiment"].mean()
    s.name = "sentiment"
    return s


def build_panel(cfg: Config | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (prices, features, meta) — features is the state vector."""
    cfg = cfg or Config()
    prices = load_prices(cfg.tickers, period=cfg.history_period, interval=cfg.interval)
    prices = prices.dropna(axis=1, how="all").ffill().dropna()

    log_ret = np.log(prices / prices.shift(1)).dropna()
    vol21 = log_ret.rolling(21).std().dropna()
    rank = log_ret.rank(axis=1, pct=True)
    sent = _daily_market_sentiment().reindex(log_ret.index).fillna(0.0)

    common = log_ret.index.intersection(vol21.index).intersection(rank.index)
    ret = log_ret.loc[common]
    vol = vol21.loc[common]
    rnk = rank.loc[common]
    sent = sent.loc[common]

    ret.columns = [f"ret_{c}" for c in ret.columns]
    vol.columns = [f"vol_{c}" for c in vol.columns]
    rnk.columns = [f"rank_{c}" for c in rnk.columns]

    features = pd.concat([ret, vol, rnk], axis=1)
    features["sentiment"] = sent.values
    meta = pd.DataFrame(index=features.index, data={"date": features.index})

    log.info("Unified panel: %s rows × %s features", *features.shape)
    return prices.loc[common], features, meta


def save_panel(prefix: str = "unified") -> Path:
    cfg = Config()
    cfg.ensure_dirs()
    prices, features, _ = build_panel(cfg)
    out = cfg.checkpoints_dir / f"{prefix}_panel.parquet"
    features.to_parquet(out)
    prices.to_parquet(cfg.checkpoints_dir / f"{prefix}_prices.parquet")
    return out
