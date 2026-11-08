"""Pre-open SPX open interest for dealer-gamma features (Databento OPRA statistics).

For each session: the next 5 SPXW expirations, strikes within +/-5% of the
prior close every 5 points, and the open-interest statistic published before
09:35 ET. That figure reflects positions as of the previous close, so it is
known before the session opens. About $0.001 per session.

Output: data/oi/YYYY-MM-DD.parquet with expiration, strike, option_type, oi.

Usage:
  python3 -m odte.data.fetch_oi --start 2022-05-16 --end 2026-11-06 --max-usd 5
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

from .fetch_0dte import DATASET, TZ, _retry

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "oi"
OPEN_INTEREST = 9          # databento StatType.OPEN_INTEREST
N_EXPIRIES, BAND, STEP = 5, 0.05, 5.0


def _symbols(day, sessions, center) -> list:
    exps = list(sessions[sessions >= day][:N_EXPIRIES])
    lo = np.floor(center * (1 - BAND) / STEP) * STEP
    hi = np.ceil(center * (1 + BAND) / STEP) * STEP
    return [f"SPXW  {e:%y%m%d}{cp}{int(round(k * 1000)):08d}"
            for e in exps for k in np.arange(lo, hi + STEP / 2, STEP) for cp in "CP"]


def _window(day):
    d = pd.Timestamp(day).normalize()
    s = d.tz_localize(TZ).tz_convert("UTC").isoformat()
    e = (d + pd.Timedelta(hours=9, minutes=35)).tz_localize(TZ).tz_convert("UTC").isoformat()
    return s, e


def fetch_day(client, day, symbols) -> pd.DataFrame:
    s, e = _window(day)
    store = _retry(lambda: client.timeseries.get_range(
        dataset=DATASET, schema="statistics", symbols=symbols, stype_in="raw_symbol",
        start=s, end=e), f"{day:%Y-%m-%d} oi")
    df = store.to_df(map_symbols=True)
    if df.empty:
        return pd.DataFrame(columns=["expiration", "strike", "option_type", "oi"])
    df = df[df["stat_type"] == OPEN_INTEREST].reset_index()
    df = df.sort_values("ts_recv").groupby("symbol").tail(1)       # latest OI per contract
    sym = df["symbol"].astype(str)
    return pd.DataFrame({
        "expiration": pd.to_datetime(sym.str.slice(6, 12), format="%y%m%d"),
        "strike": sym.str.slice(13, 21).astype(np.int64) / 1000.0,
        "option_type": sym.str.slice(12, 13),
        "oi": df["quantity"].astype(float),
    }).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    import databento as db
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    st = pd.read_csv(ROOT / "data" / "odte_0dte" / "settlements.csv", parse_dates=["date"])
    st = st.set_index("date")["spx_close"].sort_index()
    sessions = st.index
    days = [d for d in sessions if pd.Timestamp(a.start) <= d <= pd.Timestamp(a.end)]
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days if not (OUT / f"{d:%Y-%m-%d}.parquet").exists()
            and len(st[st.index < d])]
    syms = {d: _symbols(d, sessions, float(st[st.index < d].iloc[-1])) for d in todo}

    def price(d):
        s, e = _window(d)
        return d, float(_retry(lambda: client.metadata.get_cost(
            dataset=DATASET, schema="statistics", symbols=syms[d], stype_in="raw_symbol",
            start=s, end=e), f"{d:%Y-%m-%d} price"))
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        costs = dict(ex.map(price, todo))
    total = sum(costs.values())
    print(f"{len(todo)} sessions, estimated ${total:.2f} (cap ${a.max_usd:.2f})", flush=True)
    if total > a.max_usd:
        print("ABORT: estimate exceeds the cap; nothing was downloaded.")
        return 2
    lock, done = threading.Lock(), [0]

    def one(d):
        oi = fetch_day(client, d, syms[d])
        f = OUT / f"{d:%Y-%m-%d}.parquet"
        oi.to_parquet(f.with_suffix(".tmp"), index=False)
        f.with_suffix(".tmp").replace(f)
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  {done[0]}/{len(todo)} sessions", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, todo))
    print(f"open interest: {len(list(OUT.glob('20??-??-??.parquet')))} sessions -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
