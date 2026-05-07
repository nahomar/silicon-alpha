"""Binance crypto L3-ish recorder + raw-parquet packer.

Records top-of-book updates (bookTicker) and aggregated trades (aggTrade)
from Binance public WebSocket streams. Free, no API key required. Each
WebSocket message is shaped into the same column schema that
databento_to_datashop_schema produces, so the existing prepare_features +
HybridBinTokenizer pipeline can consume crypto data without modification.

Modality ID convention:
  0 = OPRA, 1 = ES (CME futures), 2 = SPY (US equity), 3 = crypto.

Two phases:
  A. record_loop()   — async WS → raw parquet shards (no tokenization).
                       Long-running. Writes 1 shard every shard_rows events.
  B. pack_crypto_raw_dir() — same two-pass packer pattern as cme_es_pack:
                       fit HybridBinTokenizer streaming, then tokenize +
                       write parquet shards with modality_id=3.

Phase A runs continuously to accumulate corpus. Phase B runs as a batch
job after enough data lands. They're decoupled so the recorder is robust
to packer changes.

Required packages (all pip-installable, no special infra):
    pip install websockets pandas pyarrow numpy

Usage:
  # Long-running recorder (run in tmux / nohup / Modal Function):
  python -m odte.data.crypto_recorder record \\
      --symbols btcusdt,ethusdt,solusdt \\
      --out-dir /scratch/$USER/data/crypto_raw \\
      --shard-rows 200000 --max-hours 24

  # Pack raw parquets into tokenized shards (modality 3):
  python -m odte.data.crypto_recorder pack \\
      --raw-dir /scratch/$USER/data/crypto_raw \\
      --out-dir /scratch/$USER/data/packed/crypto

After packing, merge with other modalities:
  python -m odte.data.multimodal_interleave \\
      --inputs /scratch/.../packed/{opra,es,spy_nbbo,spy_l3,crypto} \\
      --output /scratch/.../packed/multimodal_v2 \\
      --fallback-modalities 0,1,2,2,3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BINANCE_WS = "wss://stream.binance.com:9443/stream?streams="
DEFAULT_SYMBOLS = ["btcusdt", "ethusdt", "solusdt"]
MODALITY_CRYPTO = 3


# ---------------------------------------------------------------------------
# Phase A — async recorder
# ---------------------------------------------------------------------------

async def _record_symbol(symbol: str, queue: asyncio.Queue,
                         out_dir: Path, shard_rows: int,
                         max_hours: float) -> None:
    """Drain `queue` for one symbol, accumulate feature rows, flush every
    `shard_rows`. Stops after `max_hours` (0 = forever)."""
    out_dir = out_dir / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + max_hours * 3600 if max_hours > 0 else float("inf")

    state = {"bid": None, "ask": None, "bid_size": None, "ask_size": None}
    rows: list[dict] = []
    shard_idx = 0

    while time.time() < deadline:
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=10.0)
        except asyncio.TimeoutError:
            continue
        stream = msg.get("stream", "")
        data = msg.get("data", msg)
        ts_ms = int(data.get("E", time.time() * 1000))

        if "bookTicker" in stream:
            state["bid"] = float(data["b"])
            state["ask"] = float(data["a"])
            state["bid_size"] = float(data["B"])
            state["ask_size"] = float(data["A"])
            trade_volume = 0.0
        elif "aggTrade" in stream:
            trade_volume = float(data.get("q", 0.0))
        else:
            continue

        if state["bid"] is None:
            continue

        rows.append({
            "quote_datetime": pd.Timestamp(ts_ms, unit="ms"),
            "bid": state["bid"], "ask": state["ask"],
            "bid_size": state["bid_size"], "ask_size": state["ask_size"],
            "trade_volume": trade_volume,
        })

        if len(rows) >= shard_rows:
            df = pd.DataFrame(rows)
            path = out_dir / f"{symbol}_raw_{shard_idx:06d}.parquet"
            df.to_parquet(path, compression="zstd")
            log.info("[%s] shard %d  rows=%d  bid=%.2f", symbol, shard_idx,
                     len(rows), state["bid"])
            rows.clear()
            shard_idx += 1

    if rows:
        df = pd.DataFrame(rows)
        path = out_dir / f"{symbol}_raw_{shard_idx:06d}.parquet"
        df.to_parquet(path, compression="zstd")
        log.info("[%s] final shard %d  rows=%d", symbol, shard_idx, len(rows))


async def _multiplex(symbols: List[str], queues: dict, reconnect: int = 5
                     ) -> None:
    """Single Binance WebSocket multi-stream connection. Routes each
    incoming message to the per-symbol queue. Auto-reconnects."""
    import websockets
    streams = "/".join(f"{s}@bookTicker/{s}@aggTrade" for s in symbols)
    url = BINANCE_WS + streams
    while True:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                log.info("connected to Binance: %d symbols", len(symbols))
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    stream = msg.get("stream", "")
                    sym = stream.split("@")[0] if "@" in stream else None
                    if sym in queues:
                        try:
                            queues[sym].put_nowait(msg)
                        except asyncio.QueueFull:
                            pass  # drop on backpressure (very rare)
        except Exception as e:
            log.warning("ws error: %s; reconnect in %ds", e, reconnect)
            await asyncio.sleep(reconnect)


async def record_loop(symbols: List[str], out_dir: Path,
                      shard_rows: int, max_hours: float) -> None:
    """Run multiplexer + per-symbol writers concurrently."""
    queues = {s: asyncio.Queue(maxsize=50000) for s in symbols}
    tasks = [asyncio.create_task(_multiplex(symbols, queues))]
    for s in symbols:
        tasks.append(asyncio.create_task(
            _record_symbol(s, queues[s], out_dir, shard_rows, max_hours)))
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        log.info("interrupted; flushing")


# ---------------------------------------------------------------------------
# Phase B — pack raw parquets into tokenized shards (modality_id=3)
# ---------------------------------------------------------------------------

def _iter_raw_parquet_chunks(raw_dir: Path, chunk_rows: int = 200_000
                             ) -> Iterable[pd.DataFrame]:
    """Stream raw recorder parquets. Each yielded chunk has the columns
    prepare_features() expects (quote_datetime, bid, ask, bid_size,
    ask_size, trade_volume).
    """
    raw_dir = Path(raw_dir).expanduser()
    files = sorted(raw_dir.rglob("*_raw_*.parquet"))
    if not files:
        raise RuntimeError(f"no raw crypto parquets under {raw_dir}")
    log.info("found %d raw shards under %s", len(files), raw_dir)
    for f in files:
        df = pd.read_parquet(f)
        # Read in chunks if huge
        for start in range(0, len(df), chunk_rows):
            yield df.iloc[start:start + chunk_rows].copy()


def pack_crypto_raw_dir(raw_dir: Path, out_dir: Path,
                        n_buckets: int = 64,
                        shard_rows: int = 1_000_000,
                        feature_spec: Optional[dict] = None
                        ) -> dict:
    """Pack raw recorder parquets into tokenized shards with modality_id=3.
    Two-pass: fit tokenizer streaming, then tokenize + write."""
    from .datashop_pack import (
        DataShopPacker, default_feature_spec_v1, prepare_features,
    )
    from .streaming_quantiles import fit_hybrid_from_chunks
    from ..tokenizer import HybridBinTokenizer

    raw_dir = Path(raw_dir).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = feature_spec or default_feature_spec_v1()

    packer = DataShopPacker(out_dir=out_dir, n_buckets=n_buckets,
                            shard_rows=shard_rows, feature_spec=spec)

    # Fit pass
    n_chunks = [0]; n_rows = [0]
    def _fit_chunks():
        t0 = time.time()
        for ch in _iter_raw_parquet_chunks(raw_dir):
            feats = prepare_features(ch)
            n_chunks[0] += 1
            n_rows[0] += len(feats)
            if n_chunks[0] % 50 == 0:
                rate = n_rows[0] / max(1e-3, time.time() - t0)
                log.info("fit chunk %d  rows=%d  (%.0fk/s)",
                         n_chunks[0], n_rows[0], rate / 1000)
            yield feats

    log.info("fitting tokenizer (modality=%d, %d features)…",
             MODALITY_CRYPTO, len(spec))
    t0 = time.time()
    edges = fit_hybrid_from_chunks(_fit_chunks(), spec, n_buckets=n_buckets)
    tok = HybridBinTokenizer(n_buckets=n_buckets, feature_spec=spec)
    tok.edges = edges
    log.info("tokenizer fit in %.1fs (%d rows)", time.time() - t0, n_rows[0])

    # Write pass — vectorized buffers, same fix as the cme_es_pack speedup.
    log.info("writing tokenized shards…")
    t0 = time.time(); shard_idx = 0; n_written = 0
    buf_ts: list[np.ndarray] = []
    buf_tok: list[list] = []
    buf_mod: list[np.ndarray] = []
    buf_count = 0

    def _flush(n: int):
        nonlocal shard_idx, n_written, buf_ts, buf_tok, buf_mod, buf_count
        ts_concat = np.concatenate(buf_ts)
        mod_concat = np.concatenate(buf_mod)
        path = out_dir / f"shard_{shard_idx:06d}.parquet"
        pd.DataFrame({
            "ts": ts_concat[:n], "tokens": buf_tok[:n],
            "modality_id": mod_concat[:n],
        }).to_parquet(path, compression="zstd")
        carry_ts = ts_concat[n:]; carry_mod = mod_concat[n:]
        carry_tok = buf_tok[n:]
        buf_ts = [carry_ts] if carry_ts.size else []
        buf_mod = [carry_mod] if carry_mod.size else []
        buf_tok = carry_tok
        buf_count -= n; n_written += n; shard_idx += 1

    for ch in _iter_raw_parquet_chunks(raw_dir):
        feats = prepare_features(ch)
        toks = tok.tokenize_batch(feats, feature_order=list(spec))
        n_chunk = toks.shape[0]
        buf_ts.append(feats["ts_ms"].values.astype(np.int64))
        buf_tok.extend(toks.tolist())
        buf_mod.append(np.full(n_chunk, MODALITY_CRYPTO, dtype=np.int8))
        buf_count += n_chunk
        while buf_count >= shard_rows:
            _flush(shard_rows)
            if shard_idx % 5 == 0:
                rate = n_written / max(1e-3, time.time() - t0)
                log.info("wrote %d shards  rows=%d  (%.0fk/s)",
                         shard_idx, n_written, rate / 1000)
    if buf_count > 0:
        _flush(buf_count)

    log.info("done: %d shards  %d rows  in %.1fs",
             shard_idx, n_written, time.time() - t0)
    return {"shards": shard_idx, "rows": n_written,
            "modality_id": MODALITY_CRYPTO, "n_features": len(spec)}


# ---------------------------------------------------------------------------
# CLI dispatch: record | pack
# ---------------------------------------------------------------------------

def _cli():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="long-running WS recorder")
    r.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    r.add_argument("--out-dir", type=Path, default=Path("data/crypto_raw"))
    r.add_argument("--shard-rows", type=int, default=200_000)
    r.add_argument("--max-hours", type=float, default=24.0,
                   help="0 = run forever")

    p = sub.add_parser("pack", help="pack raw recorder parquets into tokenized shards")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--shard-rows", type=int, default=1_000_000)

    args = ap.parse_args()
    if args.cmd == "record":
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        asyncio.run(record_loop(symbols, args.out_dir,
                                args.shard_rows, args.max_hours))
    elif args.cmd == "pack":
        res = pack_crypto_raw_dir(args.raw_dir, args.out_dir,
                                  shard_rows=args.shard_rows)
        print(res)


if __name__ == "__main__":
    _cli()
