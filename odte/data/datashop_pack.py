"""CBOE DataShop → sharded tokenized parquet.

Reads raw DataShop daily CSVs (plain or zstd-compressed), computes the
per-event features HybridBinTokenizer expects, and writes ~1 GB shards of
int16 token sequences to disk.

Shard format (per file):
    reports/odte_shards/opra_NNNNNN.parquet
Columns:
    ts            int64  ms
    tokens        list<int16>   flat sequence
    underlying    str
    expiry        str
    day           str  YYYY-MM-DD

Designed to run on a laptop for a few days of DataShop, and on many
parallel workers for the full multi-year corpus (cloud phase).
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from odte.tokenizer import HybridBinTokenizer
from .streaming_quantiles import fit_hybrid_from_chunks

log = logging.getLogger(__name__)

# Column schema observed on CBOE DataShop SPX option trades:
DATASHOP_COLS = [
    "underlying_symbol", "quote_datetime", "root", "expiration", "strike",
    "option_type", "open", "high", "low", "close",
    "trade_volume", "bid_size", "bid", "ask_size", "ask",
    "underlying_bid", "underlying_ask",
    "implied_volatility", "delta", "gamma", "theta", "vega", "rho",
    "open_interest",
]


def _open(path: Path) -> io.TextIOBase:
    if path.suffix.endswith("zst"):
        try:
            import zstandard as zstd
        except ImportError as e:
            raise RuntimeError("pip install zstandard") from e
        return io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(path.open("rb")),
                                encoding="utf-8")
    return path.open("r")


def iter_csv_chunks(path: Path, chunksize: int = 100_000,
                    usecols: Sequence[str] | None = None) -> Iterator[pd.DataFrame]:
    """Stream a (possibly .zst) CSV in chunks."""
    with _open(path) as fh:
        for chunk in pd.read_csv(fh, chunksize=chunksize, usecols=usecols):
            yield chunk


def prepare_features(chunk: pd.DataFrame) -> pd.DataFrame:
    """Derive the feature columns HybridBinTokenizer expects.

    Always-emitted columns:
        ret, mid, micro_dev, spread, bid_sz, ask_sz, last_sz, inter_arrival_ms

    Phase-6 v1 microstructure factor columns (always computed; opt-in via
    `default_feature_spec_v2()`):
        ofi          single-level Cont (2014) order-flow imbalance — net of
                     bid-side and ask-side queue pressure events. Empirically
                     explains a large fraction of short-horizon mid-price
                     change in equity LOB; first principled factor to add
                     when the existing 7-feature set caps at ~AUC 0.64
                     across multi-day h=10 directional eval.
        mid_accel    2nd-difference of log-mid (i.e. diff of `ret`).
                     Captures momentum acceleration / regime shifts the
                     1st-order return token can't express.
        spread_vel   1st-difference of spread. Liquidity-regime signal —
                     widening spread often presages directional moves.
    """
    df = chunk.copy()
    df["quote_datetime"] = pd.to_datetime(df["quote_datetime"], errors="coerce")
    df = df.dropna(subset=["quote_datetime"])
    df["ts_ms"] = (df["quote_datetime"].astype("int64") // 1_000_000)
    df["mid"] = (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0
    df["spread"] = (df["ask"].astype(float) - df["bid"].astype(float)).clip(lower=1e-6)
    df["bid_sz"] = df["bid_size"].astype(float).fillna(0.0)
    df["ask_sz"] = df["ask_size"].astype(float).fillna(0.0)
    df["last_sz"] = df["trade_volume"].astype(float).fillna(0.0)
    df["micro_dev"] = 0.0                          # placeholder until full-depth feed
    df = df.sort_values("ts_ms")
    df["inter_arrival_ms"] = df["ts_ms"].diff().fillna(0.0).clip(lower=1e-3)
    df["ret"] = np.log(df["mid"]).diff().fillna(0.0)

    # ---- Phase-6 v1 factors ----
    bid = df["bid"].astype(float).values
    ask = df["ask"].astype(float).values
    bid_sz = df["bid_sz"].values
    ask_sz = df["ask_sz"].values
    bid_prev = np.concatenate([bid[:1], bid[:-1]])
    ask_prev = np.concatenate([ask[:1], ask[:-1]])
    bid_sz_prev = np.concatenate([bid_sz[:1], bid_sz[:-1]])
    ask_sz_prev = np.concatenate([ask_sz[:1], ask_sz[:-1]])
    # Cont (2014) per-event bid/ask contributions:
    e_bid = np.where(bid > bid_prev, bid_sz,
            np.where(bid < bid_prev, -bid_sz_prev, bid_sz - bid_sz_prev))
    e_ask = np.where(ask < ask_prev, ask_sz,
            np.where(ask > ask_prev, -ask_sz_prev, ask_sz - ask_sz_prev))
    df["ofi"] = (e_bid - e_ask).astype(np.float64)
    df["mid_accel"] = df["ret"].diff().fillna(0.0)
    df["spread_vel"] = df["spread"].diff().fillna(0.0)
    return df


def default_feature_spec_v1() -> Dict[str, str]:
    """7-feature spec used by all packs prior to Phase 6 (back-compat)."""
    return {
        "ret": "quantile", "mid": "quantile",
        "spread": "log", "bid_sz": "log", "ask_sz": "log",
        "last_sz": "log", "inter_arrival_ms": "log",
    }


def default_feature_spec_v2() -> Dict[str, str]:
    """10-feature spec adding Phase-6 v1 microstructure factors.

    The new factors are signed and can have heavy tails on either side, so
    "quantile" (empirical-bucket) is the right tokenizer kind — "log"
    would clip the negatives. Order is: original 7 first, then new 3, so
    feature index 0 (`ret`) is preserved for downstream code that expects
    "feature 0 = return-token" (e.g. dir_baseline_*.py, eval_loop.py)."""
    base = default_feature_spec_v1()
    base.update({
        "ofi": "quantile",
        "mid_accel": "quantile",
        "spread_vel": "quantile",
    })
    return base


@dataclass
class DataShopPacker:
    out_dir: Path
    # Default = v1 spec (7 features, pre-Phase-6) for back-compat. Pass
    # `feature_spec=default_feature_spec_v2()` to opt into the +3 microstructure
    # factor columns.
    # NOTE: "micro_dev" intentionally absent from both v1 and v2 specs until
    # the full-depth feed lands. prepare_features() still writes the column
    # as 0.0 (placeholder) but feeding a dead constant into the tokenizer
    # produces degenerate quantile edges and a wasted feature dimension in
    # the model.
    # TODO(phase-2-backfill): restore "micro_dev": "quantile" once the
    # full-depth feed is wired and prepare_features computes it.
    feature_spec: Dict[str, str] = field(
        default_factory=default_feature_spec_v1
    )
    n_buckets: int = 64
    shard_rows: int = 1_000_000
    tokenizer: Optional[HybridBinTokenizer] = None

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # fit streaming tokenizer
    # ------------------------------------------------------------------
    def fit_tokenizer(self, csv_paths: Iterable[Path],
                      chunksize: int = 100_000) -> HybridBinTokenizer:
        def chunks():
            for p in csv_paths:
                log.info("fit tokenizer: reading %s", p)
                for ch in iter_csv_chunks(Path(p), chunksize=chunksize):
                    yield prepare_features(ch)
        edges = fit_hybrid_from_chunks(chunks(), self.feature_spec,
                                       n_buckets=self.n_buckets,
                                       checkpoint=self.out_dir / "_fit_ckpt")
        tok = HybridBinTokenizer(n_buckets=self.n_buckets,
                                 feature_spec=self.feature_spec)
        tok.edges = edges
        tok.save(self.out_dir / "tokenizer.json")
        self.tokenizer = tok
        return tok

    # ------------------------------------------------------------------
    # pack to shards
    # ------------------------------------------------------------------
    def pack(self, csv_paths: Iterable[Path], chunksize: int = 100_000
             ) -> list[Path]:
        if self.tokenizer is None:
            raise RuntimeError("call fit_tokenizer() first")
        tok = self.tokenizer
        buffer: list[dict] = []
        shard_idx = 0
        shard_paths: list[Path] = []
        for p in csv_paths:
            log.info("pack: %s", p)
            for ch in iter_csv_chunks(Path(p), chunksize=chunksize):
                feats = prepare_features(ch)
                toks = tok.tokenize_batch(feats, feature_order=list(self.feature_spec))
                # flatten per-row into one int16 sequence
                for i, row in feats.reset_index(drop=True).iterrows():
                    buffer.append({
                        "ts": int(row["ts_ms"]),
                        "underlying": str(row.get("underlying_symbol") or row.get("root") or ""),
                        "expiry": str(row.get("expiration") or ""),
                        "day": pd.Timestamp(row["quote_datetime"]).strftime("%Y-%m-%d"),
                        "tokens": toks[i].tolist(),
                    })
                    if len(buffer) >= self.shard_rows:
                        shard_paths.append(self._flush_shard(buffer, shard_idx))
                        shard_idx += 1
                        buffer = []
        if buffer:
            shard_paths.append(self._flush_shard(buffer, shard_idx))
        log.info("packed %d shards → %s", len(shard_paths), self.out_dir)
        return shard_paths

    def _flush_shard(self, buffer: list[dict], idx: int) -> Path:
        df = pd.DataFrame(buffer)
        path = self.out_dir / f"opra_{idx:06d}.parquet"
        df.to_parquet(path, index=False)
        log.info("wrote shard %s (%d rows)", path, len(df))
        return path


def pack_folder(folder: Path, out_dir: Path,
                pattern: str = "*.csv*", n_buckets: int = 64) -> list[Path]:
    """Convenience: fit tokenizer then pack every matching file under folder."""
    folder = Path(folder)
    paths = sorted(folder.rglob(pattern))
    if not paths:
        raise RuntimeError(f"no files matched {pattern!r} under {folder}")
    packer = DataShopPacker(out_dir=out_dir, n_buckets=n_buckets)
    packer.fit_tokenizer(paths)
    return packer.pack(paths)
