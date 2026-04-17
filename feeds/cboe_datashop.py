"""CBOE DataShop replay feed.

Reads CSV / CSV.zst files downloaded from CBOE DataShop and replays them
as an async stream with the same dict schema as feeds/coinbase_feed.py and
feeds/databento_opra.py.

Typical DataShop columns: underlying_symbol, quote_datetime, root, expiration,
strike, option_type, open, high, low, close, trade_volume, bid_size, bid,
ask_size, ask, underlying_bid, underlying_ask, implied_volatility, delta,
gamma, theta, vega, rho, open_interest.

Usage:
    feed = CBOEDataShopReplay("data/cboe/spx_0dte_2024_12.csv.zst")
    async for ev in feed.live_iter(): ...
"""
from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class CBOEDataShopReplay:
    path: str
    speed: float = 1.0               # 1.0 = real-time replay; 0.0 = as fast as possible
    chunksize: int = 100_000

    def _open(self):
        p = Path(self.path)
        if p.suffix.endswith("zst"):
            try:
                import zstandard as zstd
            except ImportError as e:
                raise RuntimeError("pip install zstandard") from e
            dctx = zstd.ZstdDecompressor()
            return io.TextIOWrapper(dctx.stream_reader(p.open("rb")), encoding="utf-8")
        return p.open("r")

    @staticmethod
    def _to_event(row) -> dict:
        ts = pd.to_datetime(row.get("quote_datetime")).value // 1_000_000  # ns → ms
        cp = str(row.get("option_type", "")).upper()[:1]
        return {
            "kind": "tob",
            "ts": int(ts),
            "underlying": row.get("underlying_symbol") or row.get("root"),
            "strike": float(row.get("strike")) if row.get("strike") is not None else None,
            "expiry": str(row.get("expiration")),
            "cp_flag": cp,
            "bid_px": float(row.get("bid")) if row.get("bid") is not None else None,
            "bid_sz": float(row.get("bid_size")) if row.get("bid_size") is not None else None,
            "ask_px": float(row.get("ask")) if row.get("ask") is not None else None,
            "ask_sz": float(row.get("ask_size")) if row.get("ask_size") is not None else None,
            "iv": float(row.get("implied_volatility")) if row.get("implied_volatility") is not None else None,
            "delta": float(row.get("delta")) if row.get("delta") is not None else None,
            "gamma": float(row.get("gamma")) if row.get("gamma") is not None else None,
            "vega": float(row.get("vega")) if row.get("vega") is not None else None,
        }

    async def live_iter(self) -> AsyncIterator[dict]:
        fh = self._open()
        prev_ts: Optional[int] = None
        for chunk in pd.read_csv(fh, chunksize=self.chunksize):
            for row in chunk.to_dict(orient="records"):
                ev = self._to_event(row)
                if self.speed > 0 and prev_ts is not None:
                    dt_s = max(0.0, (ev["ts"] - prev_ts) / 1000 / self.speed)
                    if dt_s > 0:
                        await asyncio.sleep(min(dt_s, 1.0))
                prev_ts = ev["ts"]
                yield ev
