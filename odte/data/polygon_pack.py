"""Polygon.io flat-files OPRA -> sharded tokenized parquet.

Adapter that pulls SPX/SPXW options tick data from Polygon's S3-compatible
flat-files endpoint, merges the separate trades + quotes streams into one
event stream with forward-filled BBO, translates columns to the schema
odte.data.datashop_pack.prepare_features() expects, and hands off to the
existing DataShopPacker pipeline.

Why Polygon flat files (not the REST API):
  - REST has rate limits / pagination that make multi-year pulls painful.
  - Flat files are daily .csv.gz per underlying at a fixed S3 prefix.
  - Bulk download via boto3 with custom endpoint, ~100x faster.

Why merge trades + quotes ourselves:
  - Polygon delivers them as TWO separate files (trades_v1/* and
    quotes_v1/*). datashop_pack.prepare_features() expects one unified
    stream with both trade size and BBO on every row.
  - We concat + sort by timestamp + forward-fill BBO on trade rows.
  - Non-trade (quote) rows get trade_volume = 0.

Schema assumptions (verify on first real pull; TODOs flag fragile spots):
  Polygon options trades CSV columns (approximate):
      ticker, conditions, correction, exchange, price, sip_timestamp,
      size, sequence_number, tape
  Polygon options quotes CSV columns (approximate):
      ticker, ask_exchange, ask_price, ask_size, bid_exchange, bid_price,
      bid_size, sequence_number, sip_timestamp, tape

  sip_timestamp is nanoseconds since epoch.
  Prices are already in dollars (NOT fixed-point, unlike Databento).

Cost shape:
  - Flat-files bandwidth is free within your Polygon plan.
  - Only recurring cost is the monthly plan ($199/mo Options Advanced
    or equivalent). 3 years SPX+SPXW typically fits in 1-3 months of
    a paid plan — total cost $199-800.

Usage:
    # smoke: 1 trading day of SPX, pack to shards
    python -m odte.data.polygon_pack --smoke

    # multi-day:
    python -m odte.data.polygon_pack \
        --start 2024-01-03 --end 2024-01-31 \
        --symbols SPX SPXW \
        --out-dir reports/odte_shards_real

Environment variables:
    POLYGON_S3_ACCESS_KEY  - flat-files S3 access key
    POLYGON_S3_SECRET_KEY  - flat-files S3 secret key
    (REST POLYGON_API_KEY is not used here; flat files use S3 creds.)
"""
from __future__ import annotations

import argparse
import gzip
import io
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import numpy as np
import pandas as pd

from .datashop_pack import DataShopPacker, prepare_features

log = logging.getLogger(__name__)

# Polygon flat-files endpoint + bucket. Documented at
# https://polygon.io/docs/stocks/getting-started (flat files section).
POLYGON_ENDPOINT = "https://files.polygon.io"
POLYGON_BUCKET = "flatfiles"

# Prefix layout for options flat files. Polygon reshuffles this
# occasionally; verify current structure at dashboard/flat-files.
# TODO(verify-on-first-pull): confirm exact prefix is
#   us_options_opra/trades_v1/YYYY/MM/DD/ or similar.
TRADES_PREFIX_FMT = "us_options_opra/trades_v1/{yyyy}/{mm}/{dd}.csv.gz"
QUOTES_PREFIX_FMT = "us_options_opra/quotes_v1/{yyyy}/{mm}/{dd}.csv.gz"


# ---------------------------------------------------------------------------
# Polygon flat-files S3 client
# ---------------------------------------------------------------------------

@dataclass
class PolygonFlatFiles:
    access_key: Optional[str] = None   # None -> POLYGON_S3_ACCESS_KEY
    secret_key: Optional[str] = None   # None -> POLYGON_S3_SECRET_KEY
    cache_dir: Path = Path("data/polygon_raw")

    def __post_init__(self):
        self.access_key = self.access_key or os.getenv("POLYGON_S3_ACCESS_KEY")
        self.secret_key = self.secret_key or os.getenv("POLYGON_S3_SECRET_KEY")
        if not (self.access_key and self.secret_key):
            raise RuntimeError(
                "Polygon flat-files credentials missing. Set "
                "POLYGON_S3_ACCESS_KEY and POLYGON_S3_SECRET_KEY env vars. "
                "Generate at https://polygon.io/dashboard/flat-files ."
            )
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _client(self):
        try:
            import boto3  # noqa: F401
        except ImportError as e:
            raise RuntimeError("pip install boto3") from e
        import boto3
        return boto3.client(
            "s3",
            endpoint_url=POLYGON_ENDPOINT,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name="us-east-1",
        )

    def download(self, key: str) -> Path:
        """Download a single flat-file key with on-disk caching."""
        local = self.cache_dir / key.replace("/", "_")
        if local.exists() and local.stat().st_size > 0:
            log.info("cache hit: %s", local)
            return local
        log.info("polygon s3 get: %s", key)
        self._client().download_file(POLYGON_BUCKET, key, str(local))
        return local

    def day_keys(self, d: date) -> tuple[str, str]:
        """Return (trades_key, quotes_key) for a given date."""
        fmt = dict(yyyy=d.strftime("%Y"), mm=d.strftime("%m"), dd=d.strftime("%d"))
        return TRADES_PREFIX_FMT.format(**fmt), QUOTES_PREFIX_FMT.format(**fmt)


# ---------------------------------------------------------------------------
# CSV parsing + schema translation
# ---------------------------------------------------------------------------

def _read_csv_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rb") as f:
        return pd.read_csv(io.BytesIO(f.read()))


def _filter_symbols(df: pd.DataFrame, roots: List[str]) -> pd.DataFrame:
    """Keep only rows whose option ticker starts with O:SPX or O:SPXW etc.

    Polygon option ticker format: "O:ROOTYYMMDDCPPPPPPPPP" — e.g.
    "O:SPX240419C05000000". We match by root prefix after the "O:".
    """
    if "ticker" not in df.columns:
        # Some flat-file versions name it "T" or "underlying".
        alt = next((c for c in ("T", "underlying_ticker", "sym") if c in df.columns), None)
        if alt is None:
            raise RuntimeError(f"no ticker column found. columns={list(df.columns)}")
        df = df.rename(columns={alt: "ticker"})
    # Build regex ^O:(ROOT1|ROOT2)\d{6}[CP]
    import re
    root_re = "|".join(re.escape(r) for r in roots)
    pattern = re.compile(rf"^O:({root_re})\d{{6}}[CP]\d+$")
    mask = df["ticker"].astype(str).str.match(pattern)
    return df[mask].copy()


def _trades_to_events(df_t: pd.DataFrame) -> pd.DataFrame:
    """Convert a trades-file DataFrame to the unified event schema."""
    return pd.DataFrame({
        "ts_ns":         df_t["sip_timestamp"].astype("int64"),
        "is_trade":      True,
        "trade_size":    df_t["size"].astype(float),
        "trade_price":   df_t["price"].astype(float),
        "ticker":        df_t["ticker"].astype(str),
        # BBO cols left NaN here — forward-filled from quote events.
        "bid":           np.nan,
        "ask":           np.nan,
        "bid_size_raw":  np.nan,
        "ask_size_raw":  np.nan,
    })


def _quotes_to_events(df_q: pd.DataFrame) -> pd.DataFrame:
    """Convert a quotes-file DataFrame to the unified event schema."""
    return pd.DataFrame({
        "ts_ns":         df_q["sip_timestamp"].astype("int64"),
        "is_trade":      False,
        "trade_size":    0.0,
        "trade_price":   np.nan,
        "ticker":        df_q["ticker"].astype(str),
        "bid":           df_q["bid_price"].astype(float),
        "ask":           df_q["ask_price"].astype(float),
        "bid_size_raw":  df_q["bid_size"].astype(float),
        "ask_size_raw":  df_q["ask_size"].astype(float),
    })


def merge_trades_quotes(df_trades: pd.DataFrame,
                        df_quotes: pd.DataFrame) -> pd.DataFrame:
    """Interleave trades + quotes, sort, forward-fill BBO onto trade rows.

    Produces a DataFrame in the schema datashop_pack.prepare_features()
    expects. Drops the initial trade rows that occur before any quote
    (no BBO to forward-fill from).
    """
    ev_t = _trades_to_events(df_trades) if len(df_trades) else pd.DataFrame()
    ev_q = _quotes_to_events(df_quotes) if len(df_quotes) else pd.DataFrame()
    df = pd.concat([ev_t, ev_q], ignore_index=True)
    if df.empty:
        return df
    # Per-ticker forward-fill of BBO. GroupBy-ffill preserves chronology.
    df = df.sort_values(["ticker", "ts_ns"], kind="mergesort").reset_index(drop=True)
    df[["bid", "ask", "bid_size_raw", "ask_size_raw"]] = (
        df.groupby("ticker", sort=False)
          [["bid", "ask", "bid_size_raw", "ask_size_raw"]]
          .ffill()
    )
    # Rows with NaN BBO (trades before the first quote of the day) are
    # unusable. Drop them; usually <1% of rows.
    df = df.dropna(subset=["bid", "ask"])
    # Translate to datashop_pack schema.
    return pd.DataFrame({
        "quote_datetime":  pd.to_datetime(df["ts_ns"], unit="ns"),
        "bid":             df["bid"],
        "ask":             df["ask"],
        "bid_size":        df["bid_size_raw"].fillna(0.0),
        "ask_size":        df["ask_size_raw"].fillna(0.0),
        "trade_volume":    df["trade_size"].fillna(0.0),
        "underlying_symbol": df["ticker"],   # full option ticker; parent root
                                             # extracted downstream if needed
    })


# ---------------------------------------------------------------------------
# Day-by-day chunk iterator (drop-in for DataShopPacker)
# ---------------------------------------------------------------------------

def iter_day_chunks(ff: PolygonFlatFiles, d: date, symbols: List[str],
                    chunk_rows: int = 500_000) -> Iterator[pd.DataFrame]:
    trades_key, quotes_key = ff.day_keys(d)
    tpath = ff.download(trades_key)
    qpath = ff.download(quotes_key)
    df_t = _filter_symbols(_read_csv_gz(tpath), symbols)
    df_q = _filter_symbols(_read_csv_gz(qpath), symbols)
    merged = merge_trades_quotes(df_t, df_q)
    if merged.empty:
        log.warning("no rows after merge for %s %s", d, symbols)
        return
    for i in range(0, len(merged), chunk_rows):
        yield merged.iloc[i:i + chunk_rows].copy()


def _date_range(start: str, end: str) -> list[date]:
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    days: list[date] = []
    cur = d0
    while cur < d1:
        # Skip weekends; holidays are filtered at download time (404s).
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# End-to-end: fetch + pack
# ---------------------------------------------------------------------------

def pack_polygon(start: str, end: str, symbols: List[str],
                 cache_dir: Path, out_dir: Path,
                 n_buckets: int = 64,
                 shard_rows: int = 1_000_000) -> list[Path]:
    ff = PolygonFlatFiles(cache_dir=Path(cache_dir))
    days = _date_range(start, end)
    if not days:
        raise RuntimeError(f"empty date range {start}..{end}")
    log.info("polygon pack: %d trading days %s..%s symbols=%s",
             len(days), start, end, symbols)

    packer = DataShopPacker(out_dir=Path(out_dir),
                            n_buckets=n_buckets, shard_rows=shard_rows)

    # Fit tokenizer streaming over all days.
    from .streaming_quantiles import fit_hybrid_from_chunks
    from odte.tokenizer import HybridBinTokenizer

    def _all_chunks():
        for d in days:
            try:
                for ch in iter_day_chunks(ff, d, symbols):
                    yield prepare_features(ch)
            except Exception as e:
                log.warning("skip %s: %s", d, e)

    edges = fit_hybrid_from_chunks(_all_chunks(), packer.feature_spec,
                                   n_buckets=n_buckets,
                                   checkpoint=Path(out_dir) / "_fit_ckpt")
    tok = HybridBinTokenizer(n_buckets=n_buckets,
                             feature_spec=packer.feature_spec)
    tok.edges = edges
    tok.save(Path(out_dir) / "tokenizer.json")
    packer.tokenizer = tok

    # Pack shards with a second pass through the data. First pass
    # populated the local cache so this pass is ~free.
    buffer: list[dict] = []
    shard_idx = 0
    shard_paths: list[Path] = []
    for d in days:
        try:
            for ch in iter_day_chunks(ff, d, symbols):
                feats = prepare_features(ch)
                toks = tok.tokenize_batch(
                    feats, feature_order=list(packer.feature_spec))
                for i, row in feats.reset_index(drop=True).iterrows():
                    buffer.append({
                        "ts": int(row["ts_ms"]),
                        "underlying": str(row.get("underlying_symbol") or ""),
                        "expiry": "",  # option-ticker-parsable if needed
                        "day": d.strftime("%Y-%m-%d"),
                        "tokens": toks[i].tolist(),
                    })
                    if len(buffer) >= packer.shard_rows:
                        shard_paths.append(packer._flush_shard(buffer, shard_idx))
                        shard_idx += 1
                        buffer = []
        except Exception as e:
            log.warning("skip pack %s: %s", d, e)
    if buffer:
        shard_paths.append(packer._flush_shard(buffer, shard_idx))
    log.info("polygon pack done: %d shards in %s", len(shard_paths), out_dir)
    return shard_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="ISO date, inclusive")
    ap.add_argument("--end", default=None, help="ISO date, exclusive")
    ap.add_argument("--symbols", nargs="+", default=["SPX", "SPXW"])
    ap.add_argument("--cache-dir", default="data/polygon_raw")
    ap.add_argument("--out-dir", default="reports/odte_shards_real")
    ap.add_argument("--n-buckets", type=int, default=64)
    ap.add_argument("--smoke", action="store_true",
                    help="fetch 1 trading day of SPX only; validates the path")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if a.smoke:
        a.start = a.start or "2024-01-03"
        a.end = a.end or "2024-01-04"
        a.symbols = ["SPX"]
        log.info("SMOKE MODE: start=%s end=%s symbols=%s",
                 a.start, a.end, a.symbols)

    if not (a.start and a.end):
        ap.error("--start and --end are required (unless --smoke)")

    pack_polygon(start=a.start, end=a.end, symbols=a.symbols,
                 cache_dir=Path(a.cache_dir), out_dir=Path(a.out_dir),
                 n_buckets=a.n_buckets)


if __name__ == "__main__":
    _cli()
