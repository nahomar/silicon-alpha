"""Coinbase Exchange public daily candles for every USD pair (listed or delisted), for sweep 5 family K.

    python3 -m odte.research.coinbase_daily fetch    # -> data/xasset/coinbase_daily.parquet
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "xasset" / "coinbase_daily.parquet"
API = "https://api.exchange.coinbase.com"
HDR = {"User-Agent": "python-research-client"}
START, END = pd.Timestamp("2019-01-01"), pd.Timestamp("2026-11-07")


def get(url: str):
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            time.sleep(2 ** k)
        except Exception:                                        # noqa: BLE001
            time.sleep(2 ** k)
    return None


def fetch() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prods = [p for p in get(f"{API}/products") if p.get("quote_currency") == "USD"]
    print(f"{len(prods)} USD products", flush=True)
    rows = []
    for i, p in enumerate(prods):
        t0 = START
        while t0 < END:
            t1 = min(t0 + pd.Timedelta(days=299), END)
            c = get(f"{API}/products/{p['id']}/candles?granularity=86400&start={t0:%Y-%m-%dT00:00:00Z}"
                    f"&end={t1:%Y-%m-%dT00:00:00Z}")
            time.sleep(0.15)
            for ts, lo, hi, op, cl, vol in (c or []):
                rows.append((p["id"], p["base_currency"], pd.Timestamp(ts, unit="s"), op, hi, lo, cl, vol))
            t0 = t1 + pd.Timedelta(days=1)
        if i % 25 == 0:
            print(f"{i}/{len(prods)} {p['id']} rows so far {len(rows)}", flush=True)
    d = pd.DataFrame(rows, columns=["product", "base", "date", "open", "high", "low", "close", "volume"])
    d = d.drop_duplicates(["product", "date"]).sort_values(["product", "date"])
    d.to_parquet(OUT)
    print(f"done: {len(d)} rows, {d['product'].nunique()} products", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(fetch() if sys.argv[1:] == ["fetch"] else 0)
