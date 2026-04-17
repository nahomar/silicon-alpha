"""Databento OPRA MBO feed adapter — US equity / index options real-time.

Emits the same async-iterator-of-dicts contract as feeds/coinbase_feed.py,
extended with option-specific fields (underlying, strike, expiry, cp_flag).

Cost: Databento OPRA Live starts in the low thousands $/mo plus per-byte
fees at full firehose rates. For algorithm dev, use historical batches
first (feeds/databento_feed.py::DatabentoFeed.historical).

Requires:
  pip install databento
  env: DATABENTO_API_KEY
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Optional

log = logging.getLogger(__name__)


def _ts_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class DatabentoOPRAFeed:
    dataset: str = "OPRA.PILLAR"    # Options Price Reporting Authority via Pillar
    schema: str = "mbp-10"          # top-10 levels of book
    api_key: Optional[str] = None

    def __post_init__(self):
        self.api_key = self.api_key or os.getenv("DATABENTO_API_KEY")
        if not self.api_key:
            log.warning("DATABENTO_API_KEY not set; live_iter will raise")

    async def live_iter(self, symbols: Iterable[str]) -> AsyncIterator[dict]:
        """Yield normalized option events.

        Symbol examples: 'SPX   260417C05500000' (OCC 21-char) or 'SPX.OPT'.
        """
        try:
            import databento as db
        except ImportError as e:
            raise RuntimeError("pip install databento") from e
        if not self.api_key:
            raise RuntimeError("Set DATABENTO_API_KEY")
        client = db.Live(key=self.api_key)
        client.subscribe(dataset=self.dataset, schema=self.schema,
                         symbols=list(symbols))
        loop = asyncio.get_event_loop()
        for rec in await loop.run_in_executor(None, lambda: list(client)):
            try:
                yield self._normalize(rec)
            except Exception as e:  # pragma: no cover
                log.warning("record normalize failed: %s", e)

    @staticmethod
    def _normalize(rec) -> dict:
        """Map Databento MBP-10 → our common schema."""
        # OCC symbol like 'SPX   260417C05500000':
        #   root(6)+expiry(YYMMDD)+cp(1)+strike(8, 1e-3)
        raw_sym = getattr(rec, "symbol", "") or ""
        root = raw_sym[:6].strip()
        yymmdd = raw_sym[6:12]
        cp = raw_sym[12:13]
        strike_raw = raw_sym[13:21]
        try:
            strike = int(strike_raw) / 1e3 if strike_raw.strip() else None
            expiry = f"20{yymmdd[0:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}" if yymmdd.strip() else None
        except Exception:
            strike = None; expiry = None

        lvl0 = rec.levels[0] if getattr(rec, "levels", None) else None
        return {
            "kind": "tob" if lvl0 else "trade",
            "ts": _ts_ms(),
            "underlying": root,
            "strike": strike,
            "expiry": expiry,
            "cp_flag": cp,
            "bid_px": (lvl0.bid_px / 1e9) if lvl0 else None,
            "bid_sz": lvl0.bid_sz if lvl0 else None,
            "ask_px": (lvl0.ask_px / 1e9) if lvl0 else None,
            "ask_sz": lvl0.ask_sz if lvl0 else None,
            "last_px": getattr(rec, "price", None) and rec.price / 1e9,
            "last_sz": getattr(rec, "size", None),
            "last_side": getattr(rec, "side", None),
        }
