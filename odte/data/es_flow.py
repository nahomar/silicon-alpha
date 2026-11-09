"""ES futures order flow: who was aggressive, buyers or sellers, minute by minute.

Databento GLBX.MDP3 `trades` for the continuous front month (ES.c.0). Every
trade carries its aggressor side (Databento `side`: 'B' = the buyer crossed
the spread, 'A' = the seller did, 'N' = unknown). Raw trades are never
stored: each UTC day is fetched, reduced to 1-minute sums, and dropped.

Output: data/futures/es_flow/YYYY-MM.parquet, one row per minute with
  ts          minute start, UTC epoch seconds (the minute is known at ts + 60)
  buy, sell   aggressive buy / sell volume (contracts)
  big_buy, big_sell   the same for trades of >= 20 contracts
  n           number of trades

Usage:
  python3 -m odte.data.es_flow fetch --start 2025-08-31 --end 2026-09-05 --max-usd 155
  python3 -m odte.data.es_flow check     # buy-sell imbalance should rise with price
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "futures" / "es_flow"
BIG = 20
KW = dict(dataset="GLBX.MDP3", schema="trades", symbols=["ES.c.0"], stype_in="continuous")


def reduce_day(a) -> pd.DataFrame:
    """Structured trades array -> per-minute aggressor volume."""
    if len(a) == 0:
        return pd.DataFrame(columns=["ts", "buy", "sell", "big_buy", "big_sell", "n"])
    ts = (a["ts_recv"].astype(np.int64) // 10**9) // 60 * 60
    side = a["side"]
    side = side.astype("U1") if side.dtype.kind in ("S", "U") else np.array([chr(x) for x in side])
    size = a["size"].astype(float)
    df = pd.DataFrame({"ts": ts, "buy": np.where(side == "B", size, 0.0),
                       "sell": np.where(side == "A", size, 0.0)})
    df["big_buy"] = np.where(size >= BIG, df["buy"], 0.0)
    df["big_sell"] = np.where(size >= BIG, df["sell"], 0.0)
    df["n"] = 1
    return df.groupby("ts", as_index=False).sum()


def fetch(start, end, max_usd: float, workers: int = 4) -> None:
    import databento as db
    from odte.data.fetch_0dte import _retry
    c = db.Historical(os.environ["DATABENTO_API_KEY"])
    cost = float(c.metadata.get_cost(start=start, end=end, **KW))
    print(f"ES.c.0 trades {start}..{end}: ${cost:.2f} (cap ${max_usd:.2f})", flush=True)
    if cost > max_usd:
        raise SystemExit("ABORT: estimate exceeds the cap; nothing was downloaded.")
    OUT.mkdir(parents=True, exist_ok=True)
    done_file = OUT / ".done_days"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    days = [d for d in pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
            if f"{d:%Y-%m-%d}" not in done]
    lock, parts = threading.Lock(), {}

    def one(d):
        s, e = d.isoformat(), (d + pd.Timedelta(days=1)).isoformat()
        store = _retry(lambda: c.timeseries.get_range(start=s, end=e, **KW), f"{d:%Y-%m-%d} es trades")
        arrs = list(store.to_ndarray(count=2_000_000)) if store.nbytes else []
        m = reduce_day(np.concatenate(arrs)) if arrs else reduce_day([])
        with lock:
            parts.setdefault(f"{d:%Y-%m}", []).append(m)
            done.add(f"{d:%Y-%m-%d}")
            if len(done) % 20 == 0:
                print(f"  {len(done)} days", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, days))
    for month, ms in parts.items():
        p = OUT / f"{month}.parquet"
        old = [pd.read_parquet(p)] if p.exists() else []
        m = pd.concat(old + ms).groupby("ts", as_index=False).sum().sort_values("ts")
        m.to_parquet(p.with_suffix(".tmp"), index=False)
        p.with_suffix(".tmp").replace(p)
    done_file.write_text("\n".join(sorted(done)))
    print(f"ES flow: {len(done)} days -> {OUT}")


def load() -> pd.DataFrame:
    fs = sorted(OUT.glob("20??-??.parquet"))
    return (pd.concat(pd.read_parquet(f) for f in fs).groupby("ts", as_index=False).sum()
            .sort_values("ts").reset_index(drop=True)) if fs else pd.DataFrame()


def check() -> None:
    """Sanity: aggressive buying should coincide with rising prices."""
    from odte.data.futures import OUT as BARS
    f = load()
    bars = pd.read_parquet(BARS / "es_1m.parquet", columns=["ts", "close", "instrument_id"])
    m = f.merge(bars, on="ts")
    same = m["instrument_id"].eq(m["instrument_id"].shift())
    m["ret"] = np.log(m["close"]).diff().where(same)
    m["imb"] = (m["buy"] - m["sell"]) / (m["buy"] + m["sell"]).clip(lower=1)
    print(f"{len(f):,} minutes, {f['n'].sum():,.0f} trades; corr(imbalance, same-minute return) "
          f"= {m[['imb', 'ret']].corr().iloc[0, 1]:+.3f} (should be clearly positive)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--max-usd", type=float, required=True)
    sp.add_parser("check")
    a = ap.parse_args()
    fetch(a.start, a.end, a.max_usd) if a.cmd == "fetch" else check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
