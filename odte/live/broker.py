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
ROOTS = {"SPX": "SPXW", "XSP": "XSP", "NDX": "NDXP", "RUT": "RUTW"}     # PM-settled 0DTE classes
REPLAY_DIRS = {"NDX": "data/odte_0dte_ndxp", "RUT": "data/odte_0dte_rutw"}
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

    def subscribe_extra(self, sym, day, strikes):
        """Another index's own 0DTE quotes (shadow only), from its history folder."""
        if not hasattr(self, "extra"):
            self.extra = {}
        h = pd.read_parquet(f"{REPLAY_DIRS[sym]}/{self.day:%Y-%m-%d}.parquet")
        h["quote_datetime"] = pd.to_datetime(h["quote_datetime"])
        self.extra[sym] = (h[h["strike"].isin(set(strikes))], set(strikes))

    def snapshot_extra(self, sym):
        h, _ = self.extra[sym]
        v = h[h["quote_datetime"] <= self.t]
        last = v.sort_values("quote_datetime").groupby(["option_type", "strike"]).tail(1).copy()
        last["quote_datetime"] = self.t
        return last[CHAIN_COLS]

    def settlement_extra(self, sym, day):
        st = pd.read_csv(f"{REPLAY_DIRS[sym]}/settlements.csv", parse_dates=["date"]).set_index("date")
        return float(st.loc[pd.Timestamp(day).normalize(), "spx_close"])

    def prev_close_extra(self, sym, day):
        st = pd.read_csv(f"{REPLAY_DIRS[sym]}/settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
        st = st[st.index < pd.Timestamp(day).normalize()]
        return float(st.iloc[-1]) if len(st) else float("nan")

    def subscribe_next(self, day, expiration, contracts):
        """Next-expiry SPXW quotes for (right, strike) pairs, from data/odte_next (replaces any earlier set)."""
        if getattr(self, "_next_hist", None) is None:
            h = pd.read_parquet(f"data/odte_next/{self.day:%Y-%m-%d}.parquet")
            h["quote_datetime"] = pd.to_datetime(h["quote_datetime"])
            self._next_hist = h
        self._next_set = {(r, float(k)) for r, k in contracts}

    def snapshot_next(self):
        h = getattr(self, "_next_hist", None)
        if h is None or not getattr(self, "_next_set", None):
            return pd.DataFrame(columns=CHAIN_COLS)
        v = h[h["quote_datetime"] <= self.t]
        v = v[[(r, float(k)) in self._next_set for r, k in zip(v["option_type"], v["strike"])]]
        last = v.sort_values("quote_datetime").groupby(["option_type", "strike"]).tail(1).copy()
        last["quote_datetime"] = self.t
        return last[CHAIN_COLS]

    def reconcile(self, day) -> dict:
        """End-of-day broker view (replay: from the simulated order book)."""
        return {"open_orders": sum(1 for o in self.orders.values() if o["live"]),
                "orders_with_fills": sum(1 for o in self.orders.values() if o["filled"] > 0),
                "later_expiry_positions": 0}

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

    live = True

    def now(self):
        return pd.Timestamp.now(tz=TZ).tz_localize(None)

    def start_news(self, day, path, alert):
        from odte.live.news import NewsRecorder
        self.ib.RequestTimeout = 15
        try:
            self.news = NewsRecorder(self.ib, day, path, alert)
        finally:
            self.ib.RequestTimeout = 0

    def poll_news(self, now):
        if getattr(self, "news", None) is not None:
            self.news.poll(now)

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
        tc = ROOTS[symbol]
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
        root = ROOTS[sym]
        rows = [{"quote_datetime": now, "root": root, "expiration": self.day, "strike": k,
                 "option_type": r, "bid": num(t.bid), "ask": num(t.ask),
                 "bid_size": num(t.bidSize), "ask_size": num(t.askSize)}
                for (s, r, k), t in self.tickers.items() if s == sym]
        return pd.DataFrame(rows, columns=CHAIN_COLS).dropna(subset=["bid", "ask"])

    def snapshot(self):
        return self._rows(self.cfg.signal_symbol)

    def subscribe_extra(self, sym, day, strikes):
        """Another index's 0DTE quotes for the shadow book (uses extra quote lines)."""
        cons = [self._option(sym, day, k, r) for k in strikes for r in ("C", "P")]
        cons = [c for c in (self.ib.qualifyContracts(*cons) if cons else []) if c and c.conId]
        if not cons:
            raise RuntimeError(f"no {sym} 0DTE contracts resolved")
        for c in cons:
            self.tickers[(sym, c.right, float(c.strike))] = self.ib.reqMktData(c, "", False, False)

    def snapshot_extra(self, sym):
        return self._rows(sym)

    def settlement_extra(self, sym, day):
        """Official close from Yahoo (no index data subscription needed)."""
        import yfinance as yf
        day = pd.Timestamp(day).normalize()
        h = yf.Ticker({"NDX": "^NDX", "RUT": "^RUT"}[sym]).history(start=day, end=day + pd.Timedelta(days=1),
                                                                 auto_adjust=False)
        if h.empty:
            raise RuntimeError(f"no {sym} close for {day:%Y-%m-%d} yet")
        return float(h["Close"].iloc[-1])

    def prev_close_extra(self, sym, day):
        import yfinance as yf
        day = pd.Timestamp(day).normalize()
        h = yf.Ticker({"NDX": "^NDX", "RUT": "^RUT"}[sym]).history(start=day - pd.Timedelta(days=10), end=day,
                                                                 auto_adjust=False)
        h = h[pd.DatetimeIndex(h.index).tz_localize(None).normalize() < day]
        return float(h["Close"].iloc[-1]) if len(h) else float("nan")

    def snapshot_xsp(self):
        return self._rows("XSP")

    def subscribe_next(self, day, expiration, contracts):
        """Next-expiry SPXW quotes for a few (right, strike) pairs (lead shadow's term inputs).
        Cancels the previous set first, so it holds at most len(contracts) quote lines."""
        for t in getattr(self, "next_tickers", {}).values():
            try:
                self.ib.cancelMktData(t.contract)
            except Exception:
                pass
        self.next_tickers, self.next_exp = {}, pd.Timestamp(expiration).normalize()
        cons = [self._option("SPX", expiration, float(k), r) for r, k in contracts]
        cons = [c for c in (self.ib.qualifyContracts(*cons) if cons else []) if c and c.conId]
        for c in cons:
            self.next_tickers[(c.right, float(c.strike))] = self.ib.reqMktData(c, "", False, False)
        return len(cons)

    def snapshot_next(self):
        now = self.now()
        num = lambda x: float(x) if x is not None and math.isfinite(x) and x >= 0 else np.nan
        rows = [{"quote_datetime": now, "root": "SPXW", "expiration": getattr(self, "next_exp", pd.NaT),
                 "strike": k, "option_type": r, "bid": num(t.bid), "ask": num(t.ask),
                 "bid_size": num(t.bidSize), "ask_size": num(t.askSize)}
                for (r, k), t in getattr(self, "next_tickers", {}).items()]
        return pd.DataFrame(rows, columns=CHAIN_COLS).dropna(subset=["bid", "ask"])

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

    def reconcile(self, day) -> dict:
        """End of day, read-only: what IBKR itself shows, to compare with the journal.
        open orders left; distinct orders with executions today; option positions
        expiring after today (the bot only trades same-day options, so expect 0)."""
        day = pd.Timestamp(day).normalize()
        self.ib.sleep(1)
        open_orders = len(self.ib.openTrades())
        ids = set()
        for f in self.ib.reqExecutions():
            t = pd.Timestamp(f.execution.time)
            t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(TZ).tz_localize(None)
            if t.normalize() == day and f.contract.secType in ("BAG", "OPT"):
                ids.add(f.execution.orderId)
        later = [p for p in self.ib.positions()
                 if p.contract.secType == "OPT" and p.position != 0
                 and str(p.contract.lastTradeDateOrContractMonth)[:8] > f"{day:%Y%m%d}"]
        return {"open_orders": open_orders, "orders_with_fills": len(ids),
                "later_expiry_positions": len(later)}

    def settlement(self, day):
        bars = self.ib.reqHistoricalData(self.index, endDateTime="", durationStr="2 D",
                                         barSizeSetting="1 day", whatToShow="TRADES", useRTH=True)
        day = pd.Timestamp(day).normalize()
        for b in bars:
            if pd.Timestamp(b.date).normalize() == day:
                return float(b.close)
        raise RuntimeError(f"no {self.cfg.signal_symbol} close for {day:%Y-%m-%d} yet")
