"""Financial Modeling Prep (FMP) feed adapter — BUDGET tier.

FMP Ultimate ($19-$29/mo) offers unlimited real-time US equity + options
quote pulls. NOT a true tick stream; you poll. For 0DTE mid-frequency
this is usually fine (1-5 Hz is plenty for signal generation).

Endpoints used:
  /api/v4/options/chain/SPX          full option chain
  /api/v3/quote-short/SPX            underlying NBBO
  /api/v3/historical-chart/1min/SPX  intraday bars for backtest

Docs: https://site.financialmodelingprep.com/developer/docs
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterable, List, Optional

import httpx

log = logging.getLogger(__name__)

BASE = "https://financialmodelingprep.com/api"


@dataclass
class FMPFeed:
    api_key: Optional[str] = field(default=None)
    poll_interval_s: float = 1.0        # how fast to re-query
    _client: httpx.AsyncClient = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set FMP_API_KEY or pass api_key=...")
        self._client = httpx.AsyncClient(timeout=10)

    async def underlying_quote(self, symbol: str) -> dict:
        r = await self._client.get(
            f"{BASE}/v3/quote-short/{symbol}",
            params={"apikey": self.api_key})
        r.raise_for_status()
        data = r.json()
        if not data:
            return {}
        q = data[0]
        return {
            "ts": int(time.time() * 1000),
            "symbol": q.get("symbol"),
            "price": q.get("price"),
            "volume": q.get("volume"),
        }

    async def options_chain(self, underlying: str, expiry: Optional[str] = None
                             ) -> List[dict]:
        """Full option chain for underlying. expiry filter optional (YYYY-MM-DD)."""
        params = {"apikey": self.api_key}
        if expiry:
            params["expiration"] = expiry
        r = await self._client.get(
            f"{BASE}/v4/options/chain/{underlying}", params=params)
        r.raise_for_status()
        out = []
        now = int(time.time() * 1000)
        for row in r.json():
            out.append({
                "kind": "tob",
                "ts": now,
                "underlying": underlying,
                "strike": row.get("strike"),
                "expiry": row.get("expiration"),
                "cp_flag": (row.get("type") or "").upper()[:1],
                "bid_px": row.get("bid"),
                "bid_sz": row.get("bidSize"),
                "ask_px": row.get("ask"),
                "ask_sz": row.get("askSize"),
                "iv": row.get("impliedVolatility"),
                "delta": row.get("delta"),
                "gamma": row.get("gamma"),
                "theta": row.get("theta"),
                "vega": row.get("vega"),
            })
        return out

    async def live_iter(self, underlying: str, expiry: Optional[str] = None
                         ) -> AsyncIterator[dict]:
        """Polling-loop analog to websocket live_iter."""
        while True:
            try:
                chain = await self.options_chain(underlying, expiry)
                for row in chain:
                    yield row
            except Exception as e:
                log.warning("FMP chain fetch failed: %s", e)
            await asyncio.sleep(self.poll_interval_s)

    async def close(self):
        if self._client:
            await self._client.aclose()
