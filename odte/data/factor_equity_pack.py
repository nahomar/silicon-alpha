"""Longer-horizon factor equity ingest + packer (modality_id=4).

Daily/weekly cross-sectional equity factor data to complement the
microstructure modalities (OPRA, ES, SPY-microstructure, crypto). This
adds *macro-flow* context: regime indicators, sector rotation, factor
return spreads — the slow-moving structural information that pure
sub-second tape can't see.

Source: yfinance (free, no API key, decent quality for liquid US names
back ~25 years). Daily OHLCV per ticker → cross-sectional features per
day → tokenized parquet shards in the same row schema as the rest of
the corpus.

Universe (default, conservative): SPY, QQQ, IWM (broad benchmarks) +
SPDR sector ETFs (XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC)
+ ^VIX (vol regime). 14 instruments. Cheap to pull, captures most of
the factor structure traded at daily horizon.

Per-day row built with the same 7-feature spec the microstructure path
uses, so prepare_features() consumes it unchanged. The features are
recomputed differently though — they're cross-sectional aggregates:

  ret              = SPY log return that day
  mid              = SPY close
  spread           = (highest sector return - lowest sector return) — sector dispersion
  bid_sz           = SPY dollar volume (proxy for participation)
  ask_sz           = QQQ dollar volume
  last_sz          = ^VIX close (vol regime)
  inter_arrival_ms = trading days since last entry (essentially 1)

This is *not* the most theoretically clean factor representation — it's
a practical compression of the daily cross-section into the existing
schema so the multimodal model can attend across timescales without
schema conflicts. A v2 could expand the feature_spec for richer factors.

Usage:
    python -m odte.data.factor_equity_pack \\
        --start 2020-01-01 --end 2026-04-25 \\
        --out-dir /scratch/$USER/data/packed/factor_equity
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODALITY_FACTOR_EQUITY = 4

DEFAULT_UNIVERSE = [
    # Broad benchmarks
    "SPY", "QQQ", "IWM",
    # Sector ETFs
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
    # Vol regime
    "^VIX",
]


def fetch_daily_panel(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Pull daily OHLCV for `symbols` between [start, end). Returns a
    multi-index DataFrame (date × symbol) with columns Open/High/Low/Close/Volume.
    Skips symbols that fail to download.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("pip install yfinance") from e

    log.info("yfinance: pulling %d symbols [%s..%s]", len(symbols), start, end)
    # group_by="ticker" gives a clean multi-index
    raw = yf.download(symbols, start=start, end=end,
                      group_by="ticker", auto_adjust=True,
                      progress=False, threads=True)
    if raw.empty:
        raise RuntimeError("yfinance returned empty frame; check date range / symbols")

    # Long-format conversion
    rows: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            sub = raw[sym].copy()
        except KeyError:
            log.warning("symbol %s not in yfinance response; skipping", sym)
            continue
        sub["symbol"] = sym
        sub.index.name = "date"
        rows.append(sub.reset_index())
    if not rows:
        raise RuntimeError("no symbols loaded")
    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["Close", "Volume"])
    log.info("loaded %d rows across %d symbols",
             len(df), df["symbol"].nunique())
    return df


def build_cross_section(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long-format panel into one row per date with the
    cross-sectional features we'll feed prepare_features().

    Returns a DataFrame with columns:
        quote_datetime  — pd.Timestamp at midnight UTC
        bid             — SPY close (kept as 'bid' to match prepare_features schema)
        ask             — SPY close × (1 + sector_dispersion)
        bid_size        — SPY dollar volume
        ask_size        — QQQ dollar volume
        trade_volume    — ^VIX close (vol regime as 'trade_volume' proxy)

    The schema names are kept identical to the microstructure path so
    `prepare_features` runs unchanged. Semantics differ (daily, not
    microstructure) but that's intentional — the modality_id=4 stamp is
    what tells the model these tokens come from a different timescale.
    """
    pivot_close = panel.pivot(index="date", columns="symbol", values="Close")
    pivot_vol = panel.pivot(index="date", columns="symbol", values="Volume")
    pivot_close = pivot_close.dropna(how="all")

    # SPY / QQQ are required for the mid/spread proxies; bail early if missing
    for req in ("SPY", "QQQ"):
        if req not in pivot_close.columns:
            raise RuntimeError(f"required ticker {req} missing from panel")

    spy_close = pivot_close["SPY"]
    qqq_close = pivot_close["QQQ"]
    vix_close = pivot_close.get("^VIX", pd.Series(20.0, index=spy_close.index))

    # Sector dispersion — daily return spread across the 11 sector ETFs.
    sector_cols = [c for c in pivot_close.columns
                   if c in ("XLK", "XLF", "XLE", "XLV", "XLY",
                            "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC")]
    sector_close = pivot_close[sector_cols]
    sector_ret = np.log(sector_close).diff()
    sector_disp = (sector_ret.max(axis=1) - sector_ret.min(axis=1)).fillna(0.0)

    spy_dollar_vol = (pivot_close["SPY"] * pivot_vol["SPY"]).fillna(0.0)
    qqq_dollar_vol = (pivot_close["QQQ"] * pivot_vol["QQQ"]).fillna(0.0)

    out = pd.DataFrame({
        "quote_datetime": pd.to_datetime(spy_close.index),
        "bid": spy_close.values,
        # ask = SPY × (1 + sector_disp) — synthetic 'spread' from dispersion
        "ask": (spy_close * (1.0 + sector_disp)).values,
        "bid_size": spy_dollar_vol.values,
        "ask_size": qqq_dollar_vol.values,
        "trade_volume": vix_close.reindex(spy_close.index).fillna(20.0).values,
    })
    out = out.dropna()
    log.info("cross-section: %d daily rows", len(out))
    return out


def pack_factor_equity(start: str, end: str, out_dir: Path,
                       symbols: Optional[List[str]] = None,
                       n_buckets: int = 64,
                       shard_rows: int = 1_000_000,
                       feature_spec: Optional[dict] = None) -> dict:
    """End-to-end: fetch yfinance → cross-section → tokenize → write
    parquet shards stamped with modality_id=4.
    """
    from .datashop_pack import (
        DataShopPacker, default_feature_spec_v1, prepare_features,
    )
    from .streaming_quantiles import fit_hybrid_from_chunks
    from ..tokenizer import HybridBinTokenizer

    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    syms = symbols or DEFAULT_UNIVERSE
    spec = feature_spec or default_feature_spec_v1()

    panel = fetch_daily_panel(syms, start, end)
    xs = build_cross_section(panel)
    if xs.empty:
        raise RuntimeError("empty cross-section; nothing to pack")

    feats = prepare_features(xs)

    # Single-pass tokenizer fit (small data — 25 years × 252 days ≈ 6300 rows).
    log.info("fitting tokenizer on %d rows…", len(feats))
    edges = fit_hybrid_from_chunks([feats], spec, n_buckets=n_buckets)
    tok = HybridBinTokenizer(n_buckets=n_buckets, feature_spec=spec)
    tok.edges = edges

    toks = tok.tokenize_batch(feats, feature_order=list(spec))
    n = toks.shape[0]
    ts_arr = feats["ts_ms"].values.astype(np.int64)
    mod_arr = np.full(n, MODALITY_FACTOR_EQUITY, dtype=np.int8)

    # Write 1M-row shards (likely just one file given the small scale).
    shard_idx = 0; n_written = 0
    while n_written < n:
        end_idx = min(n_written + shard_rows, n)
        path = out_dir / f"shard_{shard_idx:06d}.parquet"
        pd.DataFrame({
            "ts": ts_arr[n_written:end_idx],
            "tokens": toks[n_written:end_idx].tolist(),
            "modality_id": mod_arr[n_written:end_idx],
        }).to_parquet(path, compression="zstd")
        log.info("wrote %s  rows=%d", path, end_idx - n_written)
        n_written = end_idx
        shard_idx += 1

    return {"shards": shard_idx, "rows": n, "n_features": len(spec),
            "modality_id": MODALITY_FACTOR_EQUITY,
            "universe": syms, "start": start, "end": end}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-04-25")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--symbols", default=None,
                    help="comma-separated; defaults to SPY/QQQ/IWM/sectors/^VIX")
    args = ap.parse_args()
    syms = args.symbols.split(",") if args.symbols else None
    res = pack_factor_equity(args.start, args.end, args.out_dir, symbols=syms)
    print(res)
