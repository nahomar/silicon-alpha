"""CME E-mini S&P 500 (ES) futures -> sharded tokenized parquet.

Cross-asset fusion Phase-2.5: adds ES MBP-1 as a lead signal for the SPX
0DTE options forecast. ES typically leads OPRA by 1-10 ms during fast
moves — exactly the lead-lag edge cross-asset transformers exploit.

Implementation note: GLBX MBP-1 and OPRA MBP-1 share the same Databento
column schema (`ts_event`, `bid_px_00`, `ask_px_00`, `bid_sz_00`,
`ask_sz_00`, `action`, `size`, ...). So we can reuse `databento_pack.
databento_to_datashop_schema` and `prepare_features` directly. The only
ES-specific bits are: (1) the symbology argument when fetching, (2) the
modality_id stamped on output shards so the trainer's ShardedTokenDataset
can interleave OPRA + ES streams without confusing them.

Output shards are written with the SAME columns as OPRA shards plus a
`modality_id` int8 column. This keeps the existing tokenizer + dataset
code unchanged; only the multi-modal interleaver needs to know about
modality_id.

Budget reality check (verified 2026-05-05 against actual Databento
estimate API):
  - ES.c.0 mbp-1 for 5 days: ~$6.10 actual.
  - Single-day exploration window: <$1.50.

Usage (one-day pack from already-fetched DBN files):
    pack_es_dbn_dir(
        dbn_dir=Path("~/sol_xfer/ES").expanduser(),
        out_dir=Path("/scratch/.../databento_es_packed/2026-04-20"),
    )
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .databento_pack import iter_dbn_chunks
from .datashop_pack import (
    DataShopPacker, default_feature_spec_v1, prepare_features,
)
from .streaming_quantiles import fit_hybrid_from_chunks
from ..tokenizer import HybridBinTokenizer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ES_DATASET = "GLBX.MDP3"
ES_SCHEMA = "mbp-1"
ES_STYPE_IN_CONTINUOUS = "continuous"
ES_DEFAULT_SYMBOL = "ES.c.0"

# Modality ID stamped on every ES shard so the multi-modal interleaver can
# distinguish ES vs OPRA tokens at training time.
MODALITY_OPRA = 0
MODALITY_ES = 1
MODALITY_SPY = 2


# ---------------------------------------------------------------------------
# Pack from already-downloaded DBN files
# ---------------------------------------------------------------------------

def pack_dbn_dir(
    dbn_dir: Path,
    out_dir: Path,
    modality_id: int,
    feature_spec: Optional[dict] = None,
    n_buckets: int = 64,
    shard_rows: int = 1_000_000,
    file_glob: str = "*.dbn.zst",
) -> dict:
    """Generic DBN-directory packer. Works for any single-instrument
    Databento MBP-1 feed (ES, SPY, etc.) because they share the same
    DBN column schema.

    Two-pass:
      1. Fit pass — stream every DBN file once to fit the streaming
         quantile tokenizer.
      2. Write pass — stream a second time, tokenize per chunk, write
         1M-row parquet shards.

    Both passes use constant memory via `iter_dbn_chunks` (yields
    ~200k-row pandas DataFrames at a time).

    Returns a dict with shard count + row count + token vocab size.
    """
    dbn_dir = Path(dbn_dir).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = feature_spec or default_feature_spec_v1()
    dbn_files = sorted(dbn_dir.glob(file_glob))
    if not dbn_files:
        raise RuntimeError(f"no {file_glob} files under {dbn_dir}")
    log.info("packing %d DBN files from %s -> %s (modality=%d, %d features)",
             len(dbn_files), dbn_dir, out_dir, modality_id, len(spec))

    packer = DataShopPacker(
        out_dir=out_dir, n_buckets=n_buckets, shard_rows=shard_rows,
        feature_spec=spec,
    )

    # ---- Pass 1: fit tokenizer ----
    n_chunks = [0]; n_rows = [0]
    def _fit_chunks():
        t0 = time.time()
        for p in dbn_files:
            log.info("fit pass: %s", p.name)
            for ch in iter_dbn_chunks(p):
                feats = prepare_features(ch)
                n_chunks[0] += 1
                n_rows[0] += len(feats)
                if n_chunks[0] % 50 == 0:
                    rate = n_rows[0] / max(1e-3, time.time() - t0)
                    log.info("  fit chunk %d  rows=%d  (%.0fk rows/s)",
                             n_chunks[0], n_rows[0], rate / 1000)
                yield feats

    log.info("fitting tokenizer...")
    t0 = time.time()
    edges = fit_hybrid_from_chunks(_fit_chunks(), spec, n_buckets=n_buckets)
    tok = HybridBinTokenizer(n_buckets=n_buckets, feature_spec=spec)
    tok.edges = edges
    packer.tokenizer = tok
    log.info("tokenizer fit in %.1fs (%d rows scanned)",
             time.time() - t0, n_rows[0])

    # ---- Pass 2: tokenize + write ----
    log.info("writing shards...")
    t0 = time.time()
    shard_idx = 0
    n_written = 0
    buf_tokens: list[list[int]] = []
    buf_meta: list[dict] = []

    for p in dbn_files:
        log.info("write pass: %s", p.name)
        for ch in iter_dbn_chunks(p):
            feats = prepare_features(ch)
            toks = tok.tokenize_batch(feats, feature_order=list(spec))
            for i, tok_row in enumerate(toks):
                buf_tokens.append(tok_row.tolist())
                buf_meta.append({
                    "ts": int(feats.iloc[i]["ts_ms"]),
                    "modality_id": int(modality_id),
                })
            while len(buf_tokens) >= shard_rows:
                shard_path = out_dir / f"shard_{shard_idx:06d}.parquet"
                pd.DataFrame({
                    "ts": [m["ts"] for m in buf_meta[:shard_rows]],
                    "tokens": buf_tokens[:shard_rows],
                    "modality_id": [m["modality_id"] for m in buf_meta[:shard_rows]],
                }).to_parquet(shard_path, compression="zstd")
                n_written += shard_rows
                shard_idx += 1
                buf_tokens = buf_tokens[shard_rows:]
                buf_meta = buf_meta[shard_rows:]
                if shard_idx % 5 == 0:
                    rate = n_written / max(1e-3, time.time() - t0)
                    log.info("  wrote %d shards  rows=%d  (%.0fk rows/s)",
                             shard_idx, n_written, rate / 1000)

    if buf_tokens:
        shard_path = out_dir / f"shard_{shard_idx:06d}.parquet"
        pd.DataFrame({
            "ts": [m["ts"] for m in buf_meta],
            "tokens": buf_tokens,
            "modality_id": [m["modality_id"] for m in buf_meta],
        }).to_parquet(shard_path, compression="zstd")
        n_written += len(buf_tokens)
        shard_idx += 1

    log.info("wrote %d shards (%d rows) in %.1fs",
             shard_idx, n_written, time.time() - t0)

    return {
        "shards": shard_idx, "rows": n_written,
        "n_features": len(spec), "modality_id": modality_id,
        "n_buckets": n_buckets,
    }


def pack_es_dbn_dir(dbn_dir: Path, out_dir: Path, **kwargs) -> dict:
    """Convenience wrapper for ES with modality_id=1."""
    return pack_dbn_dir(
        dbn_dir=dbn_dir, out_dir=out_dir,
        modality_id=MODALITY_ES, **kwargs,
    )


def pack_spy_dbn_dir(dbn_dir: Path, out_dir: Path, **kwargs) -> dict:
    """Convenience wrapper for SPY with modality_id=2.

    Works for both EQUS.MINI mbp-1 and XNAS.ITCH mbp-1/mbp-10 shards —
    DBN column schema is shared. For mbp-10, only level-0 columns are
    consumed by the v1 feature spec; deeper levels are dropped by
    `prepare_features` (which only references bid_px_00 / ask_px_00).
    To use multi-level depth, build a v2-or-deeper feature_spec.
    """
    return pack_dbn_dir(
        dbn_dir=dbn_dir, out_dir=out_dir,
        modality_id=MODALITY_SPY, **kwargs,
    )


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbn-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--modality", type=int, choices=(0, 1, 2),
                    required=True, help="0=OPRA, 1=ES, 2=SPY")
    args = ap.parse_args()
    res = pack_dbn_dir(args.dbn_dir, args.out_dir, modality_id=args.modality)
    print(res)
