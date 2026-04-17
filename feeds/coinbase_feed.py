"""Coinbase Advanced Trade WebSocket feed — FREE real-time L2.

Endpoint: wss://advanced-trade-ws.coinbase.com
Docs:     https://docs.cloud.coinbase.com/advanced-trade-api/docs/ws-channels

This works with NO API key for market data. Perfect for live-testing the
entire MM stack end-to-end before you pay for equities data.

Usage:
    import asyncio
    from feeds.coinbase_feed import coinbase_l2_stream
    async def main():
        async for tob in coinbase_l2_stream(["BTC-USD"]):
            print(tob)
    asyncio.run(main())
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Iterable

import httpx

log = logging.getLogger(__name__)

WS_URL = "wss://advanced-trade-ws.coinbase.com"
REST_URL = "https://api.exchange.coinbase.com"


def _now_ms() -> int:
    return int(time.time() * 1000)


async def coinbase_l2_stream(products: Iterable[str]) -> AsyncIterator[dict]:
    """Yield TOB dicts {'ts', 'product_id', 'bid_px', 'bid_sz', 'ask_px', 'ask_sz'}.

    Subscribes to the `level2` channel; maintains a local book for each product
    and emits a new TOB snapshot every update.
    """
    try:
        import websockets
    except ImportError as e:
        raise RuntimeError("pip install websockets httpx") from e
    products = list(products)
    books: dict[str, dict[str, dict[float, float]]] = {
        p: {"bids": {}, "asks": {}} for p in products
    }
    async with websockets.connect(WS_URL, ping_interval=20, max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "type": "subscribe",
            "channel": "level2",
            "product_ids": products,
        }))
        async for msg in ws:
            data = json.loads(msg)
            if data.get("channel") != "l2_data":
                continue
            for event in data.get("events", []):
                p = event.get("product_id")
                if p not in books:
                    continue
                for u in event.get("updates", []):
                    px = float(u["price_level"])
                    sz = float(u["new_quantity"])
                    side = "bids" if u["side"].lower().startswith("b") else "asks"
                    if sz == 0:
                        books[p][side].pop(px, None)
                    else:
                        books[p][side][px] = sz
                b = books[p]
                if not b["bids"] or not b["asks"]:
                    continue
                bid_px = max(b["bids"])
                ask_px = min(b["asks"])
                yield {
                    "ts": _now_ms(),
                    "product_id": p,
                    "bid_px": bid_px,
                    "bid_sz": b["bids"][bid_px],
                    "ask_px": ask_px,
                    "ask_sz": b["asks"][ask_px],
                }


async def coinbase_multi_stream(products: Iterable[str]) -> AsyncIterator[dict]:
    """Unified stream: yields dicts tagged with channel type.

    Types:
      {'kind': 'tob',   'ts', 'product_id', 'bid_px', 'bid_sz', 'ask_px', 'ask_sz'}
      {'kind': 'trade', 'ts', 'product_id', 'last_px', 'last_sz', 'last_side'}

    'last_side' semantics: +1 = taker bought (aggressor at ask), -1 = taker sold.

    Reconnects with exponential backoff on WS errors so long-running runners
    survive transient drops.
    """
    try:
        import websockets
    except ImportError as e:
        raise RuntimeError("pip install websockets httpx") from e
    products = list(products)
    backoff = 1.0
    while True:
        books: dict[str, dict[str, dict[float, float]]] = {
            p: {"bids": {}, "asks": {}} for p in products
        }
        try:
            async for ev in _coinbase_multi_once(products, books):
                backoff = 1.0
                yield ev
        except Exception as e:
            log.warning("coinbase WS disconnected: %s — backoff %.1fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


async def _coinbase_multi_once(products, books) -> AsyncIterator[dict]:
    import websockets
    async with websockets.connect(
        WS_URL, ping_interval=20, max_size=16 * 1024 * 1024,
        open_timeout=30, close_timeout=10,
    ) as ws:
        await ws.send(json.dumps({
            "type": "subscribe", "channel": "level2", "product_ids": products,
        }))
        await ws.send(json.dumps({
            "type": "subscribe", "channel": "market_trades", "product_ids": products,
        }))
        async for msg in ws:
            data = json.loads(msg)
            ch = data.get("channel")
            if ch == "l2_data":
                for event in data.get("events", []):
                    p = event.get("product_id")
                    if p not in books:
                        continue
                    for u in event.get("updates", []):
                        px = float(u["price_level"])
                        sz = float(u["new_quantity"])
                        side = "bids" if u["side"].lower().startswith("b") else "asks"
                        if sz == 0:
                            books[p][side].pop(px, None)
                        else:
                            books[p][side][px] = sz
                    b = books[p]
                    if not b["bids"] or not b["asks"]:
                        continue
                    bid_px = max(b["bids"])
                    ask_px = min(b["asks"])
                    yield {
                        "kind": "tob",
                        "ts": _now_ms(),
                        "product_id": p,
                        "bid_px": bid_px, "bid_sz": b["bids"][bid_px],
                        "ask_px": ask_px, "ask_sz": b["asks"][ask_px],
                    }
            elif ch == "market_trades":
                for event in data.get("events", []):
                    for tr in event.get("trades", []):
                        side_raw = str(tr.get("side", "")).lower()
                        # Coinbase reports the MAKER side; aggressor is opposite.
                        # "buy" means maker bought → aggressor sold → -1.
                        aggressor = -1 if side_raw.startswith("b") else +1
                        yield {
                            "kind": "trade",
                            "ts": _now_ms(),
                            "product_id": tr.get("product_id"),
                            "last_px": float(tr.get("price", 0.0)),
                            "last_sz": float(tr.get("size", 0.0)),
                            "last_side": aggressor,
                        }


async def coinbase_rest_snapshot(product: str, level: int = 2) -> dict:
    """Level-2 snapshot via REST, as a warmup before subscribing WS."""
    url = f"{REST_URL}/products/{product}/book?level={level}"
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.json()
