"""Databento OPRA -> sharded tokenized parquet.

Thin adapter that pulls Databento MBP-1 (market-by-price top-of-book)
tick data for SPX/SPXW options, translates the column schema to match
odte.data.datashop_pack.prepare_features(), and hands off to the existing
DataShopPacker pipeline (tokenizer fit + shard writer).

Why MBP-1 specifically:
  - Single stream contains both BBO updates AND trades (distinguished by
    `action == 'T'`), so we get the quote + trade feature set
    prepare_features() expects with one subscription.
  - Bandwidth / cost lands in the sweet spot for Phase 2 pretrain vs full
    MBO (market-by-order / L3). Upgrade to MBO only at Phase 4 if the
    transformer proves it needs book-depth state.

Why SPX/SPXW parent symbology:
  - SPX = traditional monthly-expiry index options; SPXW = weekly/daily
    (including 0DTE). "Parent" symbology expands to every child contract
    (all strikes + expirations) under the root without forcing us to
    enumerate.
  - Databento bills per instrument-day of coverage, so parent symbology
    is the same price as listing every child — no penalty.

Cost shape (verify current pricing at databento.com/pricing):
  - Historical MBP-1 is billed per GB of compressed DBN output.
  - Typical: ~1-5 GB/day across SPX+SPXW at current volumes.
  - 3 years SPX+SPXW MBP-1 tick is usually $400-1500 depending on
    vol regime (2022-2023 was noisy, 2024+ is 0DTE-heavy).

Usage:
    # one-shot, fetches + packs
    python -m odte.data.databento_pack \
        --start 2024-01-01 --end 2024-02-01 \
        --raw-dir data/databento_raw \
        --out-dir reports/odte_shards_real \
        --symbols SPX SPXW

    # smoke test first — 1 day, tiny spend, validates the full path
    python -m odte.data.databento_pack --smoke

Environment:
    DATABENTO_API_KEY must be set. The client reads it automatically.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .datashop_pack import DataShopPacker, prepare_features

# Auto-load .env from project root if present. Quietly no-ops if
# python-dotenv isn't installed or no .env exists.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger(__name__)

# Databento dataset + schema constants. Keep names explicit so the
# symbolic dependency on the databento client stays visible.
DATASET_OPRA = "OPRA.PILLAR"
SCHEMA_MBP1 = "cmbp-1"   # OPRA.PILLAR is SIP-consolidated; use cmbp-1 not mbp-1
STYPE_PARENT = "parent"   # SPX.OPT resolves to every SPX option contract


# ---------------------------------------------------------------------------
# Schema translation: Databento MBP-1 -> datashop_pack prepare_features()
# ---------------------------------------------------------------------------

_RENAME_MAP = {
    # Databento MBP-1 field -> CBOE-style name prepare_features() expects
    "ts_event":    "quote_datetime",
    "bid_px_00":   "bid",
    "ask_px_00":   "ask",
    "bid_sz_00":   "bid_size",
    "ask_sz_00":   "ask_size",
}


def databento_to_datashop_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Translate a Databento MBP-1 DataFrame to datashop_pack schema.

    MBP-1 rows carry top-of-book BBO at `ts_event`. Rows where `action`
    is 'T' are trade executions; their `size` field populates
    `trade_volume`. Non-trade rows get `trade_volume = 0`.

    Prices are ints scaled by 1e-9 in DBN fixed-point; databento's
    .to_df() handles the scaling if passed the right flag, but we
    defensively rescale if the values look unscaled.
    """
    # Project/rename columns that survive the adapter. Some Databento
    # versions emit ts_event as an int64 ns timestamp; others already
    # pandas datetime64. Both are handled by prepare_features().
    df = df.rename(columns={k: v for k, v in _RENAME_MAP.items() if k in df.columns})

    # Defensive: if prices look like DBN fixed-point (very large ints),
    # rescale by 1e-9. Databento docs: fixed-point scale factor = 1e-9.
    for col in ("bid", "ask"):
        if col in df.columns and df[col].dtype.kind in ("i", "u"):
            max_val = df[col].max()
            if max_val > 1e6:  # no real option BBO is > $1M; this is fixed-point
                df[col] = df[col].astype("float64") * 1e-9

    # trade_volume: present only on action='T' rows. prepare_features()
    # reads this as the `trade_volume` column.
    if "action" in df.columns and "size" in df.columns:
        is_trade = df["action"].astype(str).str.upper() == "T"
        df["trade_volume"] = np.where(is_trade, df["size"].astype(float), 0.0)
    else:
        df["trade_volume"] = 0.0

    # Required shape check before returning — if any of these is missing,
    # prepare_features() will blow up with a less-clear KeyError.
    required = {"quote_datetime", "bid", "ask", "bid_size", "ask_size", "trade_volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"databento->datashop schema translation missing columns: {missing}. "
            f"Got columns: {sorted(df.columns)}. Check Databento schema "
            f"version (this adapter assumes MBP-1)."
        )
    return df


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

@dataclass
class DatabentoFetcher:
    api_key: Optional[str] = None  # None -> reads DATABENTO_API_KEY env
    raw_dir: Path = Path("data/databento_raw")
    dataset: str = DATASET_OPRA
    schema: str = SCHEMA_MBP1
    stype_in: str = STYPE_PARENT

    def __post_init__(self):
        self.raw_dir = Path(self.raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = self.api_key or os.getenv("DATABENTO_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No Databento API key. Set DATABENTO_API_KEY env var or "
                "pass api_key=... explicitly."
            )

    def _client(self):
        # Lazy import: keeps the CBOE/DataShop code path importable
        # even without databento installed.
        try:
            import databento as db
        except ImportError as e:
            raise RuntimeError("pip install databento>=0.40") from e
        return db.Historical(key=self.api_key)

    def cost_estimate(self, start: str, end: str,
                      symbols: List[str]) -> dict:
        """Dry-run metadata call to get size + cost before downloading."""
        client = self._client()
        # get_cost() returns billable units (USD) under current pricing.
        cost = client.metadata.get_cost(
            dataset=self.dataset, schema=self.schema,
            start=start, end=end,
            symbols=[f"{s}.OPT" for s in symbols],
            stype_in=self.stype_in,
        )
        size = client.metadata.get_billable_size(
            dataset=self.dataset, schema=self.schema,
            start=start, end=end,
            symbols=[f"{s}.OPT" for s in symbols],
            stype_in=self.stype_in,
        )
        return {"cost_usd": float(cost), "bytes": int(size),
                "gb": round(int(size) / 1e9, 2)}

    def fetch_range(self, start: str, end: str,
                    symbols: List[str]) -> Path:
        """Download one time range via streaming get_range (only for <5 GB).

        For larger requests use fetch_range_batch() — Databento's own client
        warns that streaming get_range over 5 GB tends to time out mid-pull.
        Use this only for schema probes / tiny windows.
        """
        client = self._client()
        safe = f"{symbols[0]}_{start}_{end}".replace(":", "").replace("-", "")
        out = self.raw_dir / f"{safe}.dbn.zst"
        if out.exists():
            log.info("cache hit: %s", out)
            return out
        log.info("fetching databento %s %s [%s..%s] symbols=%s -> %s (streaming)",
                 self.dataset, self.schema, start, end, symbols, out)
        client.timeseries.get_range(
            dataset=self.dataset, schema=self.schema,
            start=start, end=end,
            symbols=[f"{s}.OPT" for s in symbols],
            stype_in=self.stype_in,
            path=str(out),
        )
        return out

    # ------------------------------------------------------------------
    # Batch API path — required for >5 GB requests
    # ------------------------------------------------------------------

    def _find_existing_job(self, client, params: dict) -> Optional[dict]:
        """Idempotency: search existing jobs for one matching the params.

        Saves money (no double-billing) and time (resume a job already
        running from a prior run). Matches on dataset/schema/symbols/
        start/end — the billable parameters.
        """
        try:
            jobs = client.batch.list_jobs(states="received,queued,processing,done")
        except Exception as e:
            log.warning("list_jobs failed; can't dedupe (%s)", e)
            return None
        # Normalize symbol list for comparison.
        want_syms = set(params.get("symbols") or [])
        if isinstance(want_syms, str):
            want_syms = {want_syms}
        want_syms = {str(s).strip() for s in want_syms}
        for j in jobs:
            if (str(j.get("dataset")) != str(params.get("dataset"))
                    or str(j.get("schema")) != str(params.get("schema"))
                    or str(j.get("stype_in")) != str(params.get("stype_in"))):
                continue
            # Symbols may be str or list in the response.
            jsym = j.get("symbols")
            if isinstance(jsym, str):
                jsym = {s.strip() for s in jsym.split(",")}
            elif isinstance(jsym, list):
                jsym = {str(s).strip() for s in jsym}
            else:
                jsym = set()
            if jsym != want_syms:
                continue
            # Date boundaries can come back as various formats; compare
            # the prefix YYYY-MM-DD which is enough for day-granularity pulls.
            def _d(x):
                return str(x)[:10] if x is not None else ""
            if (_d(j.get("start")) != _d(params.get("start"))
                    or _d(j.get("end")) != _d(params.get("end"))):
                continue
            return j
        return None

    def fetch_range_batch(self, start: str, end: str, symbols: List[str],
                          poll_interval_s: int = 30,
                          max_wait_s: int = 7200) -> list[Path]:
        """Submit a batch job, poll until done, download files.

        Returns list of .dbn.zst paths (one per split_duration=day split).
        Idempotent: if a matching job exists and is done we just download
        it; if still running we poll the existing one.
        """
        client = self._client()
        bento_symbols = [f"{s}.OPT" for s in symbols]
        params = dict(
            dataset=self.dataset, schema=self.schema,
            start=start, end=end,
            symbols=bento_symbols, stype_in=self.stype_in,
        )
        existing = self._find_existing_job(client, params)
        if existing:
            job_id = existing["id"]
            log.info("batch: found existing job %s (state=%s) — reusing",
                     job_id, existing.get("state"))
            job = existing
        else:
            log.info("batch: submitting new job %s %s [%s..%s] symbols=%s",
                     self.dataset, self.schema, start, end, symbols)
            job = client.batch.submit_job(
                dataset=self.dataset, schema=self.schema,
                start=start, end=end,
                symbols=bento_symbols, stype_in=self.stype_in,
                encoding="dbn", compression="zstd",
                split_duration="day",
                delivery="download",
            )
            job_id = job["id"]
            log.info("batch: submitted job %s", job_id)

        # Poll until done or errored.
        t0 = time.time()
        while job.get("state") not in ("done",):
            if job.get("state") in ("expired", "errored", "canceled"):
                raise RuntimeError(
                    f"batch job {job_id} ended in state={job['state']}"
                )
            if time.time() - t0 > max_wait_s:
                raise TimeoutError(
                    f"batch job {job_id} not done in {max_wait_s}s"
                )
            time.sleep(poll_interval_s)
            # Refresh state by listing jobs and finding ours.
            try:
                updated = [j for j in client.batch.list_jobs(
                    states="received,queued,processing,done,expired,errored,canceled"
                ) if j["id"] == job_id]
                if updated:
                    job = updated[0]
                    log.info("batch %s state=%s  elapsed=%ds",
                             job_id, job["state"], int(time.time() - t0))
            except Exception as e:
                log.warning("batch poll failed (retrying): %s", e)
        # Done — download.
        log.info("batch: downloading files for job %s", job_id)
        paths = client.batch.download(
            job_id=job_id, output_dir=str(self.raw_dir),
        )
        log.info("batch: downloaded %d file(s): %s",
                 len(paths), [str(p.name) for p in paths])
        # Filter to the actual data files (skip manifest / metadata).
        data_files = [p for p in paths if str(p).endswith(".dbn.zst")]
        return data_files


# ---------------------------------------------------------------------------
# .dbn.zst -> pandas (chunked)
# ---------------------------------------------------------------------------

def iter_dbn_chunks(path: Path, chunk_rows: int = 200_000
                    ) -> Iterable[pd.DataFrame]:
    """Stream a .dbn.zst file into pandas chunks without ever loading
    the whole file into RAM.

    Critical for 1-day+ OPRA cmbp-1 pulls: a single day of SPX+SPXW is
    ~1B rows / 100+ GB decompressed, which does NOT fit in Mac memory.
    We use DBNStore.to_df(count=N) which returns a DataFrameIterator
    that yields batches of N rows at ~52 MB each, constant memory.
    """
    try:
        import databento as db  # noqa: F401
    except ImportError as e:
        raise RuntimeError("pip install databento>=0.40") from e
    import databento as db
    store = db.DBNStore.from_file(str(path))
    for batch in store.to_df(count=chunk_rows):
        # Schema translation happens per-batch so we never materialize
        # the full-file DataFrame in memory.
        yield databento_to_datashop_schema(batch)


# ---------------------------------------------------------------------------
# High-level: fetch + pack
# ---------------------------------------------------------------------------

def pack_databento(start: str, end: str, symbols: List[str],
                   raw_dir: Path, out_dir: Path,
                   n_buckets: int = 64,
                   shard_rows: int = 1_000_000,
                   api_key: Optional[str] = None,
                   skip_cost_check: bool = False,
                   max_spend_usd: float = 50.0,
                   force_streaming: bool = False) -> list[Path]:
    """End-to-end: cost-check, fetch via batch API, fit tokenizer, pack.

    Uses Databento's batch API for all pulls by default because the
    streaming get_range() API times out for requests over ~5 GB (and
    every cmbp-1 day is ~50 GB billable). Set force_streaming=True
    only for tiny schema probes under 5 GB.

    Refuses to download if the estimated cost exceeds `max_spend_usd`
    unless skip_cost_check=True. Prevents a typo'd date range from
    billing $5k accidentally.
    """
    fetcher = DatabentoFetcher(api_key=api_key, raw_dir=raw_dir)
    est = fetcher.cost_estimate(start, end, symbols)
    log.info("databento cost estimate: $%.2f for %.2f GB",
             est["cost_usd"], est["gb"])
    if not skip_cost_check and est["cost_usd"] > max_spend_usd:
        raise RuntimeError(
            f"estimated cost ${est['cost_usd']:.2f} exceeds max_spend "
            f"${max_spend_usd:.2f}. Raise max_spend_usd or pass "
            f"skip_cost_check=True to override."
        )
    # Fetch — default to batch API (reliable for large pulls); only
    # use streaming get_range() for explicit small-sample paths.
    if force_streaming or est["gb"] < 3.0:
        dbn_paths = [fetcher.fetch_range(start, end, symbols)]
    else:
        dbn_paths = fetcher.fetch_range_batch(start, end, symbols)
    log.info("databento fetch done: %d file(s)", len(dbn_paths))

    # Pack via existing DataShopPacker — it doesn't care whether the
    # chunks came from CSV or DBN as long as schema matches.
    packer = DataShopPacker(out_dir=Path(out_dir),
                            n_buckets=n_buckets, shard_rows=shard_rows)

    def _all_chunks():
        for p in dbn_paths:
            for ch in iter_dbn_chunks(p):
                yield prepare_features(ch)

    # Pre-populate the streaming fit via the shared helper.
    from .streaming_quantiles import fit_hybrid_from_chunks
    from odte.tokenizer import HybridBinTokenizer
    edges = fit_hybrid_from_chunks(_all_chunks(), packer.feature_spec,
                                   n_buckets=n_buckets,
                                   checkpoint=Path(out_dir) / "_fit_ckpt")
    tok = HybridBinTokenizer(n_buckets=n_buckets,
                             feature_spec=packer.feature_spec)
    tok.edges = edges
    tok.save(Path(out_dir) / "tokenizer.json")
    packer.tokenizer = tok

    # Second pass: tokenize and write shards. Re-reads the DBN files
    # from disk — the first-pass iterator is consumed by fit.
    buffer: list[dict] = []
    shard_idx = 0
    shard_paths: list[Path] = []
    for p in dbn_paths:
        for ch in iter_dbn_chunks(p):
            feats = prepare_features(ch)
            toks = tok.tokenize_batch(feats, feature_order=list(packer.feature_spec))
            for i, row in feats.reset_index(drop=True).iterrows():
                buffer.append({
                    "ts": int(row["ts_ms"]),
                    "underlying": str(symbols[0]),  # parent sym
                    "expiry": "",  # populate via instrument_id->definition later
                    "day": pd.Timestamp(row["quote_datetime"]).strftime("%Y-%m-%d"),
                    "tokens": toks[i].tolist(),
                })
                if len(buffer) >= packer.shard_rows:
                    shard_paths.append(packer._flush_shard(buffer, shard_idx))
                    shard_idx += 1
                    buffer = []
    if buffer:
        shard_paths.append(packer._flush_shard(buffer, shard_idx))
    log.info("databento pack done: %d shards in %s", len(shard_paths), out_dir)
    return shard_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="ISO date e.g. 2024-01-01")
    ap.add_argument("--end", default=None, help="ISO date, exclusive")
    ap.add_argument("--symbols", nargs="+", default=["SPX", "SPXW"])
    ap.add_argument("--raw-dir", default="data/databento_raw")
    ap.add_argument("--out-dir", default="reports/odte_shards_real")
    ap.add_argument("--n-buckets", type=int, default=64)
    ap.add_argument("--max-spend-usd", type=float, default=50.0)
    ap.add_argument("--skip-cost-check", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="fetch 1 day of SPX only, validates end-to-end path")
    ap.add_argument("--cost-only", action="store_true",
                    help="print cost estimate and exit (no download)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if a.smoke:
        a.start = a.start or "2024-01-03"   # 1 trading day of 2024
        a.end = a.end or "2024-01-04"
        a.symbols = ["SPX"]
        a.max_spend_usd = 10.0
        log.info("SMOKE MODE: start=%s end=%s symbols=%s spend_cap=$%.2f",
                 a.start, a.end, a.symbols, a.max_spend_usd)

    if not (a.start and a.end):
        ap.error("--start and --end are required (unless --smoke)")

    fetcher = DatabentoFetcher(raw_dir=Path(a.raw_dir))
    est = fetcher.cost_estimate(a.start, a.end, a.symbols)
    print(f"cost_estimate: ${est['cost_usd']:.2f}  size: {est['gb']:.2f} GB")

    if a.cost_only:
        return

    pack_databento(start=a.start, end=a.end, symbols=a.symbols,
                   raw_dir=Path(a.raw_dir), out_dir=Path(a.out_dir),
                   n_buckets=a.n_buckets,
                   max_spend_usd=a.max_spend_usd,
                   skip_cost_check=a.skip_cost_check)


if __name__ == "__main__":
    _cli()
