"""Live news recorder (IBKR's free Dow Jones / Briefing.com feeds). Record only.

Headlines are NOT used in trading decisions: headline data failed its backtest
(docs/expiry_edge_plan.md, step 2 / headlines). Recording them every day builds the
live dataset to test that again honestly later. A few high-severity words raise an
alert during market hours so the owner knows something big is happening.

The broad tape needs a subscription the paper account does not have, so the recorder
asks for the latest headlines on SPY, QQQ and the largest S&P companies every few
minutes, without blocking the trading loop (async requests).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

SYMBOLS = ("SPY", "QQQ", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "JPM")
PROVIDERS = "DJ-RT+DJ-N+BRFG+BRFUPDN"
HOT = ("trading halt", "halted", "circuit breaker", "attack", "invasion", "missile", "war ", "emergency",
       "terror", "default", "explosion", "nuclear", "assassinat", "martial law")
MAX_ALERTS = 5


def is_hot(headline: str) -> bool:
    h = (headline or "").lower()
    return any(w in h for w in HOT)


class NewsRecorder:
    def __init__(self, ib, day, path: Path, alert=lambda t, x: None, every_s: float = 180.0):
        from ib_async import Stock
        self.ib, self.path, self.alert, self.every = ib, Path(path), alert, every_s
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seen: set = set()
        self.last = None
        self.alerts = 0
        self.seeded: set = set()          # first answer per symbol = old headlines: recorded, never alerted
        cons = [Stock(s, "SMART", "USD") for s in SYMBOLS]
        self.cons = [c for c in ib.qualifyContracts(*cons) if c and c.conId]

    def poll(self, now) -> None:
        """Start one async request per symbol every `every` seconds; never blocks."""
        if self.last is not None and (now - self.last).total_seconds() < self.every:
            return
        self.last = now
        for c in self.cons:
            fut = asyncio.ensure_future(self.ib.reqHistoricalNewsAsync(c.conId, PROVIDERS, "", "", 10))
            fut.add_done_callback(lambda f, sym=c.symbol: self._done(f, sym))

    def _done(self, fut, sym: str) -> None:
        try:
            items = fut.result() or []
        except Exception:
            return
        self.record(items, sym)

    def record(self, items, sym: str) -> int:
        new = 0
        fresh = sym in self.seeded
        self.seeded.add(sym)
        with open(self.path, "a") as fh:
            for x in items:
                key = getattr(x, "articleId", None) or (getattr(x, "time", ""), getattr(x, "headline", ""))
                if key in self.seen:
                    continue
                self.seen.add(key)
                head = str(getattr(x, "headline", ""))
                hot = is_hot(head)
                fh.write(json.dumps({"seen_at": str(pd.Timestamp.now(tz="America/New_York").tz_localize(None)),
                                     "time": str(getattr(x, "time", "")), "provider": getattr(x, "providerCode", ""),
                                     "symbol": sym, "headline": head, "hot": hot}) + "\n")
                new += 1
                if hot and fresh and self.alerts < MAX_ALERTS:
                    self.alerts += 1
                    self.alert("big news", head[:200])
        return new
