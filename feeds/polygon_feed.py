"""Polygon.io adapter — US equities NBBO + L2 snapshots.

https://polygon.io
Requires: POLYGON_API_KEY. Stocks Advanced plan for real-time; Starter plan
returns 15-min delayed.

REST used for snapshots; WebSocket for live quotes/trades.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Optional

import httpx

log = logging.getLogger(__name__)


@dataclass
class PolygonFeed:
    api_key: Optional[str] = None
    base_url: str = "https://api.polygon.io"
    ws_url: str = "wss://socket.polygon.io/stocks"

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set POLYGON_API_KEY")

    def nbbo_snapshot(self, ticker: str) -> dict:
        r = httpx.get(
            f"{self.base_url}/v3/quotes/{ticker}",
            params={"apiKey": self.api_key, "limit": 1, "order": "desc"},
            timeout=15,
        )
        r.raise_for_status()
        q = r.json().get("results", [{}])[0]
        return {
            "ts": q.get("sip_timestamp"),
            "bid_px": q.get("bid_price"), "bid_sz": q.get("bid_size"),
            "ask_px": q.get("ask_price"), "ask_sz": q.get("ask_size"),
        }

    async def live_iter(self, symbols: Iterable[str]) -> AsyncIterator[dict]:
        try:
            import websockets
        except ImportError as e:
            raise RuntimeError("pip install websockets") from e
        async with websockets.connect(self.ws_url) as ws:
            await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
            params = ",".join(f"Q.{s}" for s in symbols)  # Q = quotes
            await ws.send(json.dumps({"action": "subscribe", "params": params}))
            async for msg in ws:
                for ev in json.loads(msg):
                    if ev.get("ev") != "Q":
                        continue
                    yield {
                        "ts": ev.get("t"),
                        "symbol": ev.get("sym"),
                        "bid_px": ev.get("bp"), "bid_sz": ev.get("bs"),
                        "ask_px": ev.get("ap"), "ask_sz": ev.get("as"),
                    }

    # ------------------------------------------------------------------
    # Options endpoints (budget tier)
    # ------------------------------------------------------------------

    def options_chain_snapshot(self, underlying: str,
                               expiry: str | None = None) -> list[dict]:
        """Options NBBO + Greeks via REST.

        underlying: 'SPX', 'SPY', 'NDX', 'QQQ', ...
        expiry: optional 'YYYY-MM-DD' filter; omit for all expirations.
        """
        params = {"apiKey": self.api_key, "limit": 250, "order": "asc"}
        if expiry:
            params["expiration_date"] = expiry
        r = httpx.get(
            f"{self.base_url}/v3/snapshot/options/{underlying}",
            params=params, timeout=20)
        r.raise_for_status()
        out = []
        for res in r.json().get("results", []):
            d = res.get("details", {})
            q = res.get("last_quote", {})
            g = res.get("greeks", {})
            out.append({
                "kind": "tob",
                "ts": q.get("sip_timestamp"),
                "underlying": underlying,
                "strike": d.get("strike_price"),
                "expiry": d.get("expiration_date"),
                "cp_flag": (d.get("contract_type") or "").upper()[:1],
                "bid_px": q.get("bid"), "bid_sz": q.get("bid_size"),
                "ask_px": q.get("ask"), "ask_sz": q.get("ask_size"),
                "iv": res.get("implied_volatility"),
                "delta": g.get("delta"), "gamma": g.get("gamma"),
                "theta": g.get("theta"), "vega": g.get("vega"),
            })
        return out

    async def options_live_iter(self, underlying: str,
                                 expiry: str | None = None,
                                 poll_s: float = 1.0
                                 ) -> AsyncIterator[dict]:
        """Polygon's options WS requires Options Advanced ($$$).
        Budget tier polls the snapshot endpoint at `poll_s`."""
        while True:
            try:
                for row in self.options_chain_snapshot(underlying, expiry):
                    yield row
            except Exception as e:
                log.warning("polygon options snapshot failed: %s", e)
            await asyncio.sleep(poll_s)
