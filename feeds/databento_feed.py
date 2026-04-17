"""Databento adapter — paid, but the best available equities L2/MBO.

https://databento.com
Requires: pip install databento ; DATABENTO_API_KEY in env.

Covers: NYSE, Nasdaq (ITCH), CME futures (MDP 3.0), OPRA options.
Schema used: mbp-10 (top 10 levels of book, every update).
Cost: ~$25/mo starter for a single dataset+symbol; historical bulk is
separate. Delivered over HTTPS (historical) or Live Gateway (real-time).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import pandas as pd


@dataclass
class DatabentoFeed:
    dataset: str = "XNAS.ITCH"     # Nasdaq TotalView
    schema: str = "mbp-10"
    api_key: Optional[str] = None

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("DATABENTO_API_KEY")

    def historical(self, symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
        try:
            import databento as db
        except ImportError as e:
            raise RuntimeError("pip install databento") from e
        if not self.api_key:
            raise RuntimeError("Set DATABENTO_API_KEY")
        client = db.Historical(self.api_key)
        data = client.timeseries.get_range(
            dataset=self.dataset, schema=self.schema,
            symbols=list(symbols), start=start, end=end,
        )
        df = data.to_df()
        # databento columns: bid_px_00, ask_px_00, bid_sz_00, ask_sz_00, ...
        return df.rename(columns={
            "bid_px_00": "bid_px", "bid_sz_00": "bid_sz",
            "ask_px_00": "ask_px", "ask_sz_00": "ask_sz",
            "price": "last_px", "size": "last_sz", "side": "last_side",
        })

    def live_iter(self, symbols: Iterable[str]) -> Iterator[dict]:
        """Yield TOB dicts from Databento Live Gateway (real-time)."""
        try:
            import databento as db
        except ImportError as e:
            raise RuntimeError("pip install databento") from e
        if not self.api_key:
            raise RuntimeError("Set DATABENTO_API_KEY")
        client = db.Live(key=self.api_key)
        client.subscribe(
            dataset=self.dataset, schema=self.schema, symbols=list(symbols),
        )
        for rec in client:
            # normalize to our schema
            yield {
                "ts": rec.ts_event,
                "bid_px": rec.levels[0].bid_px / 1e9,
                "bid_sz": rec.levels[0].bid_sz,
                "ask_px": rec.levels[0].ask_px / 1e9,
                "ask_sz": rec.levels[0].ask_sz,
            }
