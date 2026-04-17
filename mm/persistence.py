"""Append-only hourly-rotating parquet writers.

Layout:
    reports/mm_data/YYYY-MM-DD/HH/tob.parquet
                                  trades.parquet
                                  quotes.parquet

Why hourly rotation:
  - bounded file size (BTC L2 ≈ 40 MB/day),
  - easy parallel reads for training,
  - if a file is corrupted on crash, you lose at most one hour.

Each writer flushes every `flush_every_n` rows OR every `flush_every_s`
seconds, whichever comes first.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "reports" / "mm_data"


def _hour_dir(ts_ms: int) -> Path:
    t = time.gmtime(ts_ms / 1000)
    return DATA_ROOT / time.strftime("%Y-%m-%d", t) / time.strftime("%H", t)


@dataclass
class RollingParquetWriter:
    name: str
    flush_every_n: int = 2000
    flush_every_s: float = 10.0
    _buf: list = field(default_factory=list)
    _last_flush: float = field(default_factory=time.time)
    _current_dir: Path | None = None

    def write(self, row: dict) -> None:
        self._buf.append(row)
        if (len(self._buf) >= self.flush_every_n
                or time.time() - self._last_flush >= self.flush_every_s):
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        ts = self._buf[0].get("ts", int(time.time() * 1000))
        d = _hour_dir(int(ts))
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{self.name}.parquet"
        df = pd.DataFrame(self._buf)
        # append by reading + concat (simplest; parquet has no native append)
        if path.exists():
            try:
                prev = pd.read_parquet(path)
                df = pd.concat([prev, df], ignore_index=True)
            except Exception as e:
                log.warning("read of %s failed (%s); overwriting", path, e)
        df.to_parquet(path, index=False)
        log.debug("flushed %d rows to %s", len(self._buf), path)
        self._buf.clear()
        self._last_flush = time.time()
        self._current_dir = d

    def close(self) -> None:
        self.flush()


def read_recent(name: str, hours: int = 24) -> pd.DataFrame:
    """Load the last `hours` hours of a given stream, concatenated."""
    if not DATA_ROOT.exists():
        return pd.DataFrame()
    now = time.time()
    cutoff = now - hours * 3600
    frames = []
    for day_dir in sorted(DATA_ROOT.iterdir()):
        if not day_dir.is_dir():
            continue
        for hour_dir in sorted(day_dir.iterdir()):
            if not hour_dir.is_dir():
                continue
            p = hour_dir / f"{name}.parquet"
            if not p.exists():
                continue
            try:
                df = pd.read_parquet(p)
                if "ts" in df.columns:
                    df = df[df["ts"] >= cutoff * 1000]
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                log.warning("read %s failed: %s", p, e)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "ts" in out.columns:
        out = out.sort_values("ts").reset_index(drop=True)
    return out
