"""Spread-aware paper broker (Phase 5 upgrade).

Closes the simulation-to-real gap that blows up naive backtests:

  • Market orders fill at the OPPOSITE side of the book (buys at ask, sells
    at bid). NEVER at mid. That bakes the 3.6 %-average SPX 0DTE spread
    friction into every paper P&L.
  • Partial fills: an order can take at most `fill_rate_per_tick` × the
    displayed volume-at-strike (default 10 %). Remaining qty stays
    resting; the next tick's volume can top it up.
  • Realized-spread cost is tagged on every fill so post-trade analysis
    can separate "alpha" from "execution shortfall".

Orders NEVER reach a real exchange. Everything logged through
mm/persistence.py::RollingParquetWriter for the post-trade analyzer.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np

from mm.persistence import RollingParquetWriter

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order / fill model
# ---------------------------------------------------------------------------

@dataclass
class PaperOrder:
    order_id: str
    ts: int
    symbol: str
    side: str                        # "buy" | "sell"
    qty: float                       # original total qty
    order_type: str                  # "market" | "limit"
    limit_px: Optional[float] = None
    remaining: float = 0.0
    status: str = "resting"          # resting | filled | partial | canceled
    # Running aggregates across partials:
    filled_qty: float = 0.0
    avg_fill_px: float = 0.0
    total_spread_cost: float = 0.0   # $ spent crossing the spread
    last_fill_ts: Optional[int] = None

    def on_partial(self, qty: float, px: float, mid: float, ts: int) -> None:
        new_qty = self.filled_qty + qty
        if new_qty <= 0:
            return
        self.avg_fill_px = (self.avg_fill_px * self.filled_qty + px * qty) / new_qty
        self.filled_qty = new_qty
        self.remaining = max(0.0, self.qty - self.filled_qty)
        sign = 1.0 if self.side == "buy" else -1.0
        self.total_spread_cost += sign * (px - mid) * qty
        self.last_fill_ts = ts
        if self.remaining <= 1e-9:
            self.status = "filled"
        else:
            self.status = "partial"


# ---------------------------------------------------------------------------
# PaperBroker
# ---------------------------------------------------------------------------

@dataclass
class PaperBroker:
    max_age_s: float = 30.0            # cancel stale resting orders
    stale_ticks: int = 3
    fill_rate_per_tick: float = 0.10   # fraction of displayed volume we can take
    penny_threshold: float = 0.05      # used by close_for_pennies
    tick_size: float = 0.05

    _orders: Dict[str, PaperOrder] = field(default_factory=dict, init=False)
    _order_writer: RollingParquetWriter = field(
        default_factory=lambda: RollingParquetWriter("odte_orders",
                                                      flush_every_n=100,
                                                      flush_every_s=5.0),
        init=False,
    )
    _fill_writer: RollingParquetWriter = field(
        default_factory=lambda: RollingParquetWriter("odte_fills",
                                                      flush_every_n=100,
                                                      flush_every_s=5.0),
        init=False,
    )

    # -------- limit order (legacy path) -------------------------------
    def submit(self, symbol: str, side: str, qty: float, limit_px: float,
               current_bid: float, current_ask: float,
               volume_at_strike: float = 0.0) -> PaperOrder:
        now_ms = int(time.time() * 1000)
        oid = uuid.uuid4().hex[:12]
        order = PaperOrder(order_id=oid, ts=now_ms, symbol=symbol,
                           side=side, qty=float(qty), remaining=float(qty),
                           order_type="limit", limit_px=float(limit_px))
        self._log_order(order)
        mid = 0.5 * (current_bid + current_ask) if current_bid and current_ask else limit_px
        crossed = ((side == "buy" and limit_px >= current_ask and current_ask > 0)
                   or (side == "sell" and limit_px <= current_bid and current_bid > 0))
        if crossed:
            fill_px = current_ask if side == "buy" else current_bid
            take = self._partial_take(qty, volume_at_strike)
            order.on_partial(take, fill_px, mid, now_ms)
            self._log_fill(order, take, fill_px, mid)
            if order.status != "filled":
                self._orders[oid] = order
        else:
            self._orders[oid] = order
        return order

    # -------- market order (spec'd path) ------------------------------
    def submit_market(self, symbol: str, side: str, qty: float,
                      current_bid: float, current_ask: float,
                      volume_at_strike: float = 0.0,
                      reason: str = "") -> PaperOrder:
        now_ms = int(time.time() * 1000)
        oid = uuid.uuid4().hex[:12]
        mid = 0.5 * (current_bid + current_ask) if current_bid and current_ask else 0.0
        fill_px = current_ask if side == "buy" else current_bid
        order = PaperOrder(order_id=oid, ts=now_ms, symbol=symbol,
                           side=side, qty=float(qty), remaining=float(qty),
                           order_type="market")
        self._log_order(order, extra={"reason": reason})

        if fill_px <= 0:
            order.status = "canceled"
            self._log_order(order, extra={"reason": "no_quote"})
            return order

        take = self._partial_take(qty, volume_at_strike)
        order.on_partial(take, fill_px, mid, now_ms)
        self._log_fill(order, take, fill_px, mid, extra={"reason": reason})
        if order.status != "filled":
            self._orders[oid] = order
        return order

    # -------- market-data tick ---------------------------------------
    def on_tob(self, symbol: str, bid_px: float, ask_px: float,
               ts_ms: int, volume_at_strike: float = 0.0) -> List[PaperOrder]:
        """May fill resting limits and may top up partial fills."""
        filled: List[PaperOrder] = []
        mid = 0.5 * (bid_px + ask_px) if bid_px and ask_px else 0.0
        for oid, o in list(self._orders.items()):
            if o.symbol != symbol:
                continue
            # cancel stale
            age_s = (ts_ms - o.ts) / 1000.0
            if age_s > self.max_age_s:
                o.status = "canceled"
                self._orders.pop(oid, None)
                self._log_order(o, extra={"reason": "stale"})
                continue
            # partial-top-up for a market order whose first tick was thin
            if o.order_type == "market" and o.remaining > 0 and mid > 0:
                fill_px = ask_px if o.side == "buy" else bid_px
                take = self._partial_take(o.remaining, volume_at_strike)
                if take > 0:
                    o.on_partial(take, fill_px, mid, ts_ms)
                    self._log_fill(o, take, fill_px, mid,
                                    extra={"reason": "topup"})
            # price-cross fill for a resting limit
            if o.order_type == "limit" and o.remaining > 0:
                crossed = ((o.side == "buy" and o.limit_px is not None
                            and ask_px > 0 and o.limit_px >= ask_px)
                           or (o.side == "sell" and o.limit_px is not None
                               and bid_px > 0 and o.limit_px <= bid_px))
                if crossed:
                    fill_px = ask_px if o.side == "buy" else bid_px
                    take = self._partial_take(o.remaining, volume_at_strike)
                    if take > 0:
                        o.on_partial(take, fill_px, mid, ts_ms)
                        self._log_fill(o, take, fill_px, mid,
                                        extra={"reason": "cross"})
            if o.status == "filled":
                filled.append(o)
                self._orders.pop(oid, None)
        return filled

    def cancel(self, order_id: str) -> bool:
        o = self._orders.pop(order_id, None)
        if o is not None:
            o.status = "canceled"
            self._log_order(o, extra={"reason": "user_cancel"})
            return True
        return False

    def open_orders(self) -> List[PaperOrder]:
        return list(self._orders.values())

    # -------- helpers -------------------------------------------------
    def _partial_take(self, requested: float, volume_at_strike: float) -> float:
        """Cap per-tick fill size at `fill_rate_per_tick` × displayed volume.

        If volume_at_strike is 0 (no displayed-size info) we assume full-take
        so the smoke test doesn't deadlock.
        """
        if volume_at_strike <= 0:
            return float(requested)
        cap = max(1.0, volume_at_strike * self.fill_rate_per_tick)
        return float(min(requested, cap))

    # -------- persistence ---------------------------------------------
    def _log_order(self, o: PaperOrder, extra: Optional[Dict] = None) -> None:
        row = {
            "ts": o.ts, "order_id": o.order_id, "symbol": o.symbol,
            "side": o.side, "qty": o.qty, "order_type": o.order_type,
            "limit_px": o.limit_px, "status": o.status,
        }
        if extra:
            row.update(extra)
        self._order_writer.write(row)

    def _log_fill(self, o: PaperOrder, qty: float, px: float, mid: float,
                  extra: Optional[Dict] = None) -> None:
        row = {
            "ts": o.last_fill_ts, "order_id": o.order_id, "symbol": o.symbol,
            "side": o.side, "fill_qty": qty, "fill_px": px, "mid_at_fill": mid,
            "spread_cost_per_unit": (px - mid) if o.side == "buy" else (mid - px),
            "cum_filled": o.filled_qty, "remaining": o.remaining,
            "avg_fill_px": o.avg_fill_px, "total_spread_cost": o.total_spread_cost,
        }
        if extra:
            row.update(extra)
        self._fill_writer.write(row)

    def close(self) -> None:
        self._order_writer.close()
        self._fill_writer.close()
