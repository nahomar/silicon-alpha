"""Live MM stack runner on Coinbase L2 + trades (FREE real-time crypto).

Streams:
  - level2        → top-of-book updates
  - market_trades → real trade prints with aggressor side

Persists three rotating parquet files per hour:
  reports/mm_data/YYYY-MM-DD/HH/tob.parquet
                                trades.parquet
                                quotes.parquet

The hourly LaunchAgent tick (see mm_fill_trainer.py) reads these, labels
fills, and retrains the FillProbabilityModel.

Run manually:
    WARMUP_MIN=2 python mm_live_coinbase.py
Or via LaunchAgent (com.nahom.mmlive.plist) with KeepAlive.

LIVE_TRADING is intentionally NOT wired to any broker.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from feeds import coinbase_multi_stream
from mm import (
    microprice_features, vpin, realized_variance, variance_ratio,
    ShortHorizonPredictor, PRISM,
)
from mm.avellaneda_stoikov import ASParams
from mm.prism import PRISMConfig
from mm.persistence import RollingParquetWriter
from mm.calibration import calibrate_as_params

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("mm_live")

PRODUCT = os.getenv("CB_PRODUCT", "BTC-USD")
WARMUP_MIN = int(os.getenv("WARMUP_MIN", "2"))
PREDICT_EVERY_N_TOB = int(os.getenv("PREDICT_EVERY_N_TOB", "5"))


class State:
    tob_buf: deque = deque(maxlen=2048)
    trade_buf: deque = deque(maxlen=4096)
    tob_count: int = 0
    tob_w = RollingParquetWriter("tob", flush_every_n=500, flush_every_s=5.0)
    trade_w = RollingParquetWriter("trades", flush_every_n=200, flush_every_s=5.0)
    quote_w = RollingParquetWriter("quotes", flush_every_n=200, flush_every_s=5.0)
    prism: PRISM | None = None
    running: bool = True


def _install_signal_handlers(st: State):
    def _stop(*_):
        st.running = False
        log.info("shutdown requested; flushing writers…")
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)


async def _warmup(duration_s: int):
    """Returns (tob_df with DatetimeIndex, raw tob list with ts ms, trades list)."""
    log.info("Warmup: %ds of L2/trades for %s", duration_s, PRODUCT)
    tob_rows = []
    trade_rows = []
    t_end = time.time() + duration_s
    async for ev in coinbase_multi_stream([PRODUCT]):
        if ev["kind"] == "tob":
            tob_rows.append(ev)
        elif ev["kind"] == "trade":
            trade_rows.append(ev)
        if time.time() >= t_end:
            break
    tob_raw = pd.DataFrame(tob_rows)
    trade_raw = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(
        columns=["ts", "last_px", "last_sz", "last_side"]
    )
    if tob_raw.empty:
        return pd.DataFrame(), tob_raw, trade_raw
    tob_idxed = tob_raw.drop(columns=["kind", "product_id"], errors="ignore").copy()
    tob_idxed["ts_dt"] = pd.to_datetime(tob_idxed["ts"], unit="ms")
    return tob_idxed.set_index("ts_dt"), tob_raw, trade_raw


def _build_predictor(warmup: pd.DataFrame) -> ShortHorizonPredictor:
    feats = microprice_features(warmup)
    tox = pd.DataFrame({
        "vpin": 0.0,
        "rv": realized_variance(feats["mid"]).fillna(0.0),
        "vr": variance_ratio(feats["mid"]).fillna(1.0),
    }, index=feats.index)
    pred = ShortHorizonPredictor(horizon_steps=5)
    X = pred.build_features(feats, tox)
    y = pred.build_target(feats["mid"])
    m = y.notna()
    if m.sum() < 200:
        raise RuntimeError("Not enough warmup rows to fit predictor")
    pred.fit(X[m], y[m])
    return pred


async def main():
    st = State()
    _install_signal_handlers(st)

    warmup, warmup_tob_raw, warmup_trades = await _warmup(WARMUP_MIN * 60)
    log.info("warmup rows: tob=%d  trades=%d", len(warmup), len(warmup_trades))
    if len(warmup) < 300:
        log.warning("insufficient warmup; raise WARMUP_MIN")
        return
    pred = _build_predictor(warmup)

    # Calibrate A-S params to the actual regime instead of guessing.
    as_params = calibrate_as_params(
        tob=warmup_tob_raw[["ts", "bid_px", "bid_sz", "ask_px", "ask_sz"]],
        trades=warmup_trades[["ts", "last_px", "last_sz", "last_side"]]
              if not warmup_trades.empty else warmup_trades,
        gamma=float(os.getenv("MM_GAMMA", "0.02")),
        horizon=1.0,
        inv_limit=int(os.getenv("MM_INV_LIMIT", "5")),
    )

    st.prism = PRISM(
        predictor=pred,
        cfg=PRISMConfig(
            vpin_widen=0.35, vpin_pull=0.65, sigma_halt=1.0,
            as_params=as_params,
        ),
    )

    log.info("Entering live loop")
    async for ev in coinbase_multi_stream([PRODUCT]):
        if not st.running:
            break
        if ev["kind"] == "tob":
            st.tob_w.write({k: ev[k] for k in
                            ("ts", "product_id", "bid_px", "bid_sz", "ask_px", "ask_sz")})
            st.tob_buf.append({"bid_px": ev["bid_px"], "bid_sz": ev["bid_sz"],
                               "ask_px": ev["ask_px"], "ask_sz": ev["ask_sz"]})
            st.tob_count += 1
            if len(st.tob_buf) >= 64 and (st.tob_count % PREDICT_EVERY_N_TOB == 0):
                tob_df = pd.DataFrame(list(st.tob_buf))
                tob_df.index = pd.RangeIndex(len(tob_df))
                trade_df = pd.DataFrame(list(st.trade_buf)) if st.trade_buf else pd.DataFrame(
                    {"last_px": [], "last_sz": [], "last_side": []}
                )
                q = st.prism.decide(tob_df, trade_df)
                if "pulled" not in q:
                    st.quote_w.write({
                        "ts": ev["ts"], "product_id": ev["product_id"],
                        "bid": q["bid"], "ask": q["ask"], "micro": q["micro"],
                        "mu": q["mu"], "sigma": q["sigma"], "vpin": q["vpin"],
                        "inv": q["inv"], "reservation": q["reservation"],
                        "half_spread": q["half_spread"],
                        "bid_queue_ahead": q.get("bid_queue_ahead", 0.0),
                        "ask_queue_ahead": q.get("ask_queue_ahead", 0.0),
                    })
        elif ev["kind"] == "trade":
            st.trade_w.write({k: ev[k] for k in
                              ("ts", "product_id", "last_px", "last_sz", "last_side")})
            st.trade_buf.append({"last_px": ev["last_px"], "last_sz": ev["last_sz"],
                                 "last_side": ev["last_side"]})

    for w in (st.tob_w, st.trade_w, st.quote_w):
        w.close()
    log.info("shutdown clean")


if __name__ == "__main__":
    asyncio.run(main())
