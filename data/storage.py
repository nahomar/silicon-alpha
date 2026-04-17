"""Simple file-based storage for scraped items.

Writes each run to reports/raw/{source}_{timestamp}.json and keeps a rolling
parquet file at reports/raw/history.parquet (if pandas is available).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "reports" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def save_items(source: str, items: Iterable[dict]) -> Path:
    items = list(items)
    ts = time.strftime("%Y%m%dT%H%M%S")
    path = RAW_DIR / f"{source}_{ts}.json"
    path.write_text(json.dumps(items, indent=2, default=str))
    _append_history(source, items)
    return path


def _append_history(source: str, items: list[dict]) -> None:
    if not items:
        return
    df = pd.DataFrame(items)
    df["_source"] = source
    df["_fetched_at"] = time.time()
    history = RAW_DIR / "history.parquet"
    try:
        if history.exists():
            prev = pd.read_parquet(history)
            df = pd.concat([prev, df], ignore_index=True)
        df.to_parquet(history, index=False)
    except Exception:
        # parquet engine might be missing; fall back to CSV append
        csv = RAW_DIR / "history.csv"
        header = not csv.exists()
        df.to_csv(csv, mode="a", header=header, index=False)


def load_history() -> pd.DataFrame:
    p = RAW_DIR / "history.parquet"
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    csv = RAW_DIR / "history.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()
