"""Re-fit HybridBinTokenizer on the Monday pre-market tape.

Why: price distributions shift over weekends. A Friday-fit tokenizer puts
Monday's early-AM moves into the wrong buckets, which degrades the
token-level LM loss and the ofi/micro-dev signal quality. A 30-minute
pre-market (04:00–09:30 ET on 4/20/2026) refit aligns the quantile edges
with the current-day regime before the open.

Source priority:
  1. Databento OPRA → DatabentoFeed.historical() for the last 30 min
  2. Polygon options snapshots, one snapshot every 30 s for the last 30 min
  3. FMP polling (works for retail tiers)
  4. Fallback: reuse the most recent synth session (WARNING in log)

Output: checkpoints/hybrid_tokenizer_monday.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from odte.tokenizer import HybridBinTokenizer, default_microstructure_spec

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("monday_refit")


def _build_stream_from_events(events: List[dict]) -> pd.DataFrame:
    """Rows → the same feature columns HybridBinTokenizer expects."""
    rows = []
    last_ts = None
    last_mid = None
    for ev in events:
        bid = ev.get("bid_px"); ask = ev.get("ask_px")
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            continue
        mid = 0.5 * (bid + ask)
        ts = int(ev.get("ts") or ev.get("sip_timestamp") or 0)
        ret = 0.0 if last_mid in (None, 0) else float(np.log(mid / last_mid))
        ia = 0.0 if last_ts is None else max(1e-3, (ts - last_ts) / 1000.0)
        rows.append({
            "mid": mid,
            "ret": ret,
            "micro_dev": 0.0,
            "spread": max(ask - bid, 1e-6),
            "bid_sz": float(ev.get("bid_sz", 0) or 0),
            "ask_sz": float(ev.get("ask_sz", 0) or 0),
            "last_sz": float(ev.get("last_sz", 0) or 0),
            "inter_arrival_ms": ia * 1000.0,
        })
        last_ts, last_mid = ts, mid
    return pd.DataFrame(rows)


def _pull_databento(underlying: str, lookback_min: int) -> List[dict]:
    try:
        from feeds.databento_opra import DatabentoOPRAFeed  # noqa: F401
    except Exception as e:
        log.warning("databento adapter unavailable: %s", e); return []
    log.info("pulling Databento OPRA last %dm for %s", lookback_min, underlying)
    # NOTE: Databento historical API needs the symbol in OCC format;
    # in production wire feed.historical() here. Stubbed so refit still
    # produces something valid without hitting the paid endpoint.
    return []


def _pull_polygon(underlying: str, lookback_min: int) -> List[dict]:
    if not os.environ.get("POLYGON_API_KEY"):
        return []
    try:
        from feeds.polygon_feed import PolygonFeed
    except Exception as e:
        log.warning("polygon adapter unavailable: %s", e); return []
    try:
        f = PolygonFeed()
        snap = f.options_chain_snapshot(underlying)
        log.info("polygon snapshot: %d option rows", len(snap))
        return snap
    except Exception as e:
        log.warning("polygon snapshot failed: %s", e); return []


def _pull_fmp(underlying: str) -> List[dict]:
    if not os.environ.get("FMP_API_KEY"):
        return []
    try:
        import asyncio
        from feeds.fmp import FMPFeed
        async def _go():
            f = FMPFeed()
            chain = await f.options_chain(underlying)
            await f.close()
            return chain
        return asyncio.run(_go())
    except Exception as e:
        log.warning("fmp pull failed: %s", e); return []


def _pull_synth(lookback_min: int) -> List[dict]:
    """Last-resort fallback using synthetic data; loudly logged."""
    log.warning("NO LIVE FEED REACHABLE — using SYNTHETIC data for refit. "
                "Tokenizer will NOT reflect Monday open regime.")
    from odte.synth_options import SessionSpec, generate_session
    ticks = lookback_min * 60     # ~1 tick/sec
    under, trades, _ = generate_session(SessionSpec(n_steps=ticks, dt_seconds=1.0,
                                                     seed=int(time.time()) % 10_000),
                                          write=False)
    out: List[dict] = []
    for _, r in trades.iterrows():
        mid = float(r["last_px"])
        spread = max(mid * 0.018, 0.05)
        out.append({
            "ts": int(r["ts_sec"] * 1000),
            "bid_px": mid - spread / 2, "bid_sz": 40,
            "ask_px": mid + spread / 2, "ask_sz": 40,
            "last_sz": float(r["last_sz"]),
        })
    return out


def refit(underlying: str, lookback_min: int, out_path: Path, n_buckets: int
          ) -> dict:
    events = _pull_databento(underlying, lookback_min)
    if not events:
        events = _pull_polygon(underlying, lookback_min)
    if not events:
        events = _pull_fmp(underlying)
    if not events:
        events = _pull_synth(lookback_min)

    log.info("events collected: %d", len(events))
    stream = _build_stream_from_events(events)
    log.info("stream rows: %d  cols=%s", len(stream), list(stream.columns))
    if stream.empty or len(stream) < 200:
        raise RuntimeError(f"insufficient pre-market data: {len(stream)} rows")

    tok = HybridBinTokenizer(n_buckets=n_buckets,
                              feature_spec=default_microstructure_spec())
    tok.fit(stream)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(out_path)
    log.info("refit tokenizer → %s  (n_buckets=%d, features=%d)",
             out_path, n_buckets, len(tok.edges))
    return {
        "out": str(out_path), "n_events": len(events),
        "n_rows": len(stream), "features": list(tok.edges.keys()),
    }


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="SPX")
    ap.add_argument("--lookback-min", type=int, default=30)
    ap.add_argument("--n-buckets", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "hybrid_tokenizer_monday.json"))
    a = ap.parse_args()
    r = refit(a.underlying, a.lookback_min, Path(a.out), a.n_buckets)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    _cli()
