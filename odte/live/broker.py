"""Broker connections: Interactive Brokers (ib_async) and a historical replay.

Both expose the same small interface the runner uses:
  now()                         current time, US/Eastern, naive
  sleep(seconds)
  spot()                        index level (signal symbol)
  subscribe(day, strikes)       start quotes for today's 0DTE calls and puts
  snapshot()                    current quotes, in the historical chain schema
  place_fly(order, price)       SELL the iron fly for a credit -> handle
  modify(handle, price), cancel(handle)
  filled(handle)                (filled quantity, average credit)
  settlement(day)               the index close the options settle on
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TZ = "America/New_York"
CHAIN_COLS = ["quote_datetime", "root", "expiration", "strike", "option_type",
              "bid", "ask", "bid_size", "ask_size"]


def strikes_for(spot: float, inner_band: float, band: float, outer_step: float,
                step: float = 5.0, max_lines: int = 95) -> list:
    """Every `step` strike within inner_band, then every outer_step out to band;
    trimmed (outermost first) so calls + puts fit in max_lines."""
    lo_i, hi_i = spot * (1 - inner_band), spot * (1 + inner_band)
    inner = np.arange(math.floor(lo_i / step) * step, math.ceil(hi_i / step) * step + step / 2, step)
    outer = np.arange(math.floor(spot * (1 - band) / outer_step) * outer_step,
                      math.ceil(spot * (1 + band) / outer_step) * outer_step + outer_step / 2, outer_step)
    ks = sorted(set(np.round(np.concatenate([inner, outer]), 2)))
    while 2 * len(ks) > max_lines:
        ks.pop(0 if abs(ks[0] - spot) >= abs(ks[-1] - spot) else -1)
    return [float(k) for k in ks]


# ---------------------------------------------------------------------------
# Replay: a historical day, minute quotes, virtual clock
# ---------------------------------------------------------------------------

class ReplayBroker:
    """Plays back data/odte_0dte/<day>.parquet with a virtual clock.

    A resting sell fills only when the fly's touch credit (sell both bodies
    at the bid, buy both wings at the ask) reaches the limit: a real buyer
    paid our price. XSP orders are priced from SPX quotes / 10."""

    def __init__(self, day, data_dir="data/odte_0dte", settle=None, start="09:25"):
        self.day = pd.Timestamp(day).normalize()
        self.hist = pd.read_parquet(f"{data_dir}/{self.day:%Y-%m-%d}.parquet")
        self.hist["quote_datetime"] = pd.to_datetime(self.hist["quote_datetime"])
        self.t = self.day + pd.Timedelta(hours=int(start[:2]), minutes=int(start[3:]))
        self._settle = settle
        self.orders = {}

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += pd.Timedelta(seconds=seconds)
        self._check_fills()

    def _visible(self):
        return self.hist[self.hist["quote_datetime"] <= self.t]

    def spot(self):
        v = self._visible()
        if v.empty:
            v = self.hist[self.hist["quote_datetime"] <= self.hist["quote_datetime"].min()]
        from odte.eval.expiry_edge import implied_forward, minute_grid
        F = implied_forward(minute_grid(v)).dropna()
        return float(F.iloc[-1]) if len(F) else float(v["strike"].median())

    def subscribe(self, day, strikes, xsp_strikes=()):
        self.strikes = set(strikes)

    def snapshot_xsp(self):
        return pd.DataFrame(columns=CHAIN_COLS)        # no historical XSP quotes

    def snapshot(self):
        v = self._visible()
        v = v[v["strike"].isin(self.strikes)] if getattr(self, "strikes", None) else v
        last = v.sort_values("quote_datetime").groupby(["option_type", "strike"]).tail(1).copy()
        last["quote_datetime"] = self.t
        return last[CHAIN_COLS]

    def _touch_credit(self, o):
        s = 10.0 if o.symbol == "XSP" else 1.0
        q = self._visible().sort_values("quote_datetime").groupby(["option_type", "strike"]).tail(1)
        q = q.set_index(["option_type", "strike"])
        try:
            K, W = o.K * s, o.W * s
            c = (q.at[("C", K), "bid"] + q.at[("P", K), "bid"]
                 - q.at[("C", K + W), "ask"] - q.at[("P", K - W), "ask"])
        except KeyError:
            return float("nan")
        return c / s

    def place_fly(self, order, price):
        h = len(self.orders)
        self.orders[h] = {"order": order, "price": price, "filled": 0, "avg": float("nan"), "live": True}
        self._check_fills()
        return h

    def modify(self, h, price):
        if self.orders[h]["live"]:
            self.orders[h]["price"] = price
            self._check_fills()

    def cancel(self, h):
        self.orders[h]["live"] = False

    def filled(self, h):
        o = self.orders[h]
        return o["filled"], o["avg"]

    def _check_fills(self):
        for o in self.orders.values():
            if o["live"] and not o["filled"]:
                tc = self._touch_credit(o["order"])
                if np.isfinite(tc) and tc >= o["price"] - 1e-9:
                    o.update(filled=o["order"].qty, avg=o["price"], live=False)

    def settlement(self, day):
        if self._settle is not None:
            return self._settle
        st = pd.read_csv("data/odte_0dte/settlements.csv", parse_dates=["date"]).set_index("date")
        return float(st.loc[pd.Timestamp(day).normalize(), "spx_close"])


# ---------------------------------------------------------------------------
# Interactive Brokers
# ---------------------------------------------------------------------------

class IBKRBroker:
    """IB Gateway / TWS through ib_async. Paper port by default (Config)."""

    def __init__(self, cfg):
        from ib_async import IB, Index
        cfg.check()
        self.cfg = cfg
        self.ib = IB()
        self.ib.connect(cfg.host, cfg.resolved_port(), clientId=cfg.client_id, timeout=20)
        self.ib.reqMarketDataType(1)          # live data (paper shares the live subscriptions)
        self.index = Index(cfg.signal_symbol, "CBOE", "USD")
        self.ib.qualifyContracts(self.index)
        self.idx_ticker = self.ib.reqMktData(self.index, "", False, False)
        self.tickers = {}

    def now(self):
        return pd.Timestamp.now(tz=TZ).tz_localize(None)

    def sleep(self, seconds):
        self.ib.sleep(seconds)

    def spot(self):
        for _ in range(20):
            p = self.idx_ticker.marketPrice()
            if p and math.isfinite(p) and p > 0:
                return float(p)
            self.ib.sleep(0.5)
        raise RuntimeError(f"no {self.cfg.signal_symbol} price: check market data permissions")

    def _option(self, symbol, day, k, right):
        from ib_async import Option
        tc = "SPXW" if symbol == "SPX" else "XSP"
        return Option(symbol, f"{pd.Timestamp(day):%Y%m%d}", k, right, "SMART",
                      multiplier="100", currency="USD", tradingClass=tc)

    def subscribe(self, day, strikes, xsp_strikes=()):
        """Quotes for the signal symbol's strikes and, if trading XSP, XSP's own."""
        self.day = pd.Timestamp(day).normalize()
        for sym, ks in ((self.cfg.signal_symbol, strikes), ("XSP", xsp_strikes)):
            cons = [self._option(sym, day, k, r) for k in ks for r in ("C", "P")]
            cons = [c for c in (self.ib.qualifyContracts(*cons) if cons else []) if c and c.conId]
            for c in cons:
                self.tickers[(sym, c.right, float(c.strike))] = self.ib.reqMktData(c, "", False, False)

    def _rows(self, sym):
        now = self.now()
        num = lambda x: float(x) if x is not None and math.isfinite(x) and x >= 0 else np.nan
        root = {"SPX": "SPXW", "XSP": "XSP"}[sym]
        rows = [{"quote_datetime": now, "root": root, "expiration": self.day, "strike": k,
                 "option_type": r, "bid": num(t.bid), "ask": num(t.ask),
                 "bid_size": num(t.bidSize), "ask_size": num(t.askSize)}
                for (s, r, k), t in self.tickers.items() if s == sym]
        return pd.DataFrame(rows, columns=CHAIN_COLS).dropna(subset=["bid", "ask"])

    def snapshot(self):
        return self._rows(self.cfg.signal_symbol)

    def snapshot_xsp(self):
        return self._rows("XSP")

    def place_fly(self, order, price):
        from ib_async import ComboLeg, Contract, LimitOrder
        day = self.day
        legs = [("C", order.K, "BUY"), ("P", order.K, "BUY"),
                ("C", order.K + order.W, "SELL"), ("P", order.K - order.W, "SELL")]
        cons = self.ib.qualifyContracts(*[self._option(order.symbol, day, k, r) for r, k, _ in legs])
        if len(cons) != 4 or not all(c.conId for c in cons):
            raise RuntimeError("could not qualify all four legs")
        combo = Contract(symbol=order.symbol, secType="BAG", currency="USD", exchange="SMART",
                         comboLegs=[ComboLeg(conId=c.conId, ratio=1, action=a, exchange="SMART")
                                    for c, (_, _, a) in zip(cons, legs)])
        # SELLing the long fly = selling the iron fly; a positive limit is the credit.
        o = LimitOrder("SELL", order.qty, round(price, 2), tif="DAY")
        return self.ib.placeOrder(combo, o)

    def modify(self, trade, price):
        if not trade.isDone():
            trade.order.lmtPrice = round(price, 2)
            self.ib.placeOrder(trade.contract, trade.order)

    def cancel(self, trade):
        if not trade.isDone():
            self.ib.cancelOrder(trade.order)

    def filled(self, trade):
        self.ib.sleep(0)
        s = trade.orderStatus
        return int(s.filled or 0), float(s.avgFillPrice or float("nan"))

    def settlement(self, day):
        bars = self.ib.reqHistoricalData(self.index, endDateTime="", durationStr="2 D",
                                         barSizeSetting="1 day", whatToShow="TRADES", useRTH=True)
        day = pd.Timestamp(day).normalize()
        for b in bars:
            if pd.Timestamp(b.date).normalize() == day:
                return float(b.close)
        raise RuntimeError(f"no {self.cfg.signal_symbol} close for {day:%Y-%m-%d} yet")
