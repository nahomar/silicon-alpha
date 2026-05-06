"""K-way time-ordered merge of packed token shards from multiple modalities
into a single interleaved corpus.

Input: N directories of `shard_*.parquet` files (or `opra_*.parquet`),
each with columns:
  - ts            int64 ms
  - tokens        list[int16], length n_features
  - modality_id   int8 (0=OPRA, 1=ES, 2=SPY by convention)

Output: directory of merged `shard_*.parquet` files, time-ordered across
all input modalities, same column schema. Each row's modality_id is
preserved so downstream `modality_emb` can attribute attention to its
source venue.

The merger uses a streaming heap-based k-way merge so memory stays bounded
by ~(N modalities × shard_buffer × row_size). 1M rows × 7 int16 tokens +
metadata ≈ 60 MB per shard buffer × 4 modalities = ~240 MB peak. Safe on
any modern node.

Usage:
    python -m odte.data.multimodal_interleave \\
        --inputs /scratch/.../packed/opra,/scratch/.../packed/es \\
        --output /scratch/.../packed/multimodal
"""
from __future__ import annotations

import argparse
import heapq
import logging
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _iter_shard_rows(shard_paths: List[Path]
                     ) -> Iterator[Tuple[int, np.ndarray, int]]:
    """Yield (ts, tokens, modality_id) tuples in shard-file order.

    Reads parquet shards lazily — each file is read fully into memory then
    streamed row-by-row, then released. For 1M-row shards this peaks at
    ~50 MB per modality stream during merge.
    """
    for shard in shard_paths:
        df = pd.read_parquet(shard, columns=["ts", "tokens", "modality_id"])
        for ts, toks, mod in zip(
            df["ts"].values, df["tokens"].values, df["modality_id"].values
        ):
            yield int(ts), np.asarray(toks, dtype=np.int16), int(mod)


def _shards_for(d: Path) -> List[Path]:
    """All packed shards in `d`, sorted by filename. Accepts both
    `shard_*.parquet` (new packer) and `opra_*.parquet` (legacy)."""
    paths = sorted(list(d.glob("shard_*.parquet")) + list(d.glob("opra_*.parquet")))
    if not paths:
        raise RuntimeError(f"no parquet shards under {d}")
    return paths


def merge_modalities(
    input_dirs: List[Path],
    out_dir: Path,
    shard_rows: int = 1_000_000,
) -> dict:
    """K-way time-ordered merge.

    Args:
        input_dirs: list of directories, each containing shard_*.parquet
        out_dir: destination for merged shards
        shard_rows: rows per output shard

    Returns dict with shard count + total rows + per-modality counts.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    iters: List[Iterator] = []
    for d in input_dirs:
        d = Path(d).expanduser()
        shards = _shards_for(d)
        log.info("source %s: %d shards", d, len(shards))
        iters.append(_iter_shard_rows(shards))

    # Heap entry: (ts, idx, tokens, modality_id) — idx breaks ties
    # deterministically and keeps the heap totally-ordered even when
    # multiple modalities share a timestamp.
    heap: list = []
    for idx, it in enumerate(iters):
        try:
            ts, toks, mod = next(it)
            heapq.heappush(heap, (ts, idx, toks, mod))
        except StopIteration:
            pass

    buf_ts: list[int] = []
    buf_tok: list[list[int]] = []
    buf_mod: list[int] = []
    shard_idx = 0
    n_total = 0
    per_modality: dict[int, int] = {}
    t0 = time.time()

    while heap:
        ts, idx, toks, mod = heapq.heappop(heap)
        buf_ts.append(ts)
        buf_tok.append(toks.tolist())
        buf_mod.append(mod)
        per_modality[mod] = per_modality.get(mod, 0) + 1
        # Pull the next row from that modality's iterator.
        try:
            nts, ntoks, nmod = next(iters[idx])
            heapq.heappush(heap, (nts, idx, ntoks, nmod))
        except StopIteration:
            pass

        if len(buf_ts) >= shard_rows:
            out = out_dir / f"shard_{shard_idx:06d}.parquet"
            pd.DataFrame({
                "ts": buf_ts, "tokens": buf_tok, "modality_id": buf_mod,
            }).to_parquet(out, compression="zstd")
            n_total += len(buf_ts)
            shard_idx += 1
            buf_ts.clear(); buf_tok.clear(); buf_mod.clear()
            if shard_idx % 5 == 0:
                rate = n_total / max(1e-3, time.time() - t0)
                log.info("wrote %d shards  rows=%d  (%.0fk rows/s)",
                         shard_idx, n_total, rate / 1000)

    if buf_ts:
        out = out_dir / f"shard_{shard_idx:06d}.parquet"
        pd.DataFrame({
            "ts": buf_ts, "tokens": buf_tok, "modality_id": buf_mod,
        }).to_parquet(out, compression="zstd")
        n_total += len(buf_ts)
        shard_idx += 1

    log.info("merge done: %d shards  %d rows  per-modality=%s  in %.1fs",
             shard_idx, n_total, per_modality, time.time() - t0)
    return {"shards": shard_idx, "rows": n_total,
            "per_modality": per_modality}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True,
                    help="comma-separated list of packed-shard directories")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--shard-rows", type=int, default=1_000_000)
    args = ap.parse_args()
    dirs = [Path(p) for p in args.inputs.split(",") if p.strip()]
    res = merge_modalities(dirs, args.output, shard_rows=args.shard_rows)
    print(res)
