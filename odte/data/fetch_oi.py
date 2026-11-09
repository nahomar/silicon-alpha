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
    # Before 2022 SPXW expired Mon/Wed/Fri only, so look further ahead: dates
    # without an expiry just do not resolve.
    n = N_EXPIRIES if day >= pd.Timestamp("2022-05-16") else 8
    exps = list(sessions[sessions >= day][:n])
    lo = np.floor(center * (1 - BAND) / STEP) * STEP
    hi = np.ceil(center * (1 + BAND) / STEP) * STEP
    return [f"SPXW  {e:%y%m%d}{cp}{int(round(k * 1000)):08d}"
            for e in exps for k in np.arange(lo, hi + STEP / 2, STEP) for cp in "CP"]


def _window(day, end_hhmm: str = "09:35"):
    d = pd.Timestamp(day).normalize()
    s = d.tz_localize(TZ).tz_convert("UTC").isoformat()
    h, m = int(end_hhmm[:2]), int(end_hhmm[3:])
    e = (d + pd.Timedelta(hours=h, minutes=m)).tz_localize(TZ).tz_convert("UTC").isoformat()
    return s, e


def fetch_day(client, day, symbols, end_hhmm: str = "09:35") -> pd.DataFrame:
    s, e = _window(day, end_hhmm)
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


def fetch_today(day, center: float, max_usd: float = 0.05) -> Path:
    """Live use: today's pre-open open interest around `center` (index level).

    Expiries come from the NYSE session calendar (future days included), so
    this works before today's close exists anywhere. Run after 09:35 ET."""
    import databento as db
    from odte.data.calendar import OUT as CAL
    day = pd.Timestamp(day).normalize()
    f = OUT / f"{day:%Y-%m-%d}.parquet"
    if f.exists():
        return f
    sessions = pd.DatetimeIndex(pd.read_parquet(CAL / "sessions.parquet")["session"]).sort_values()
    syms = _symbols(day, sessions, center)
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    # Same-day OPRA data after 09:30 ET needs a live licence; the OI statistic
    # is published before the open, so stop the window at 09:29.
    s, e = _window(day, "09:29")
    cost = float(client.metadata.get_cost(dataset=DATASET, schema="statistics", symbols=syms,
                                          stype_in="raw_symbol", start=s, end=e))
    if cost > max_usd:
        raise SystemExit(f"ABORT: open interest for {day:%Y-%m-%d} quoted ${cost:.3f} > cap ${max_usd}")
    oi = fetch_day(client, day, syms, "09:29")
    OUT.mkdir(parents=True, exist_ok=True)
    oi.to_parquet(f.with_suffix(".tmp"), index=False)
    f.with_suffix(".tmp").replace(f)
    print(f"open interest {day:%Y-%m-%d}: {len(oi)} contracts, ${cost:.4f}", flush=True)
    return f


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
    if st.index.min() > pd.Timestamp(a.start) - pd.Timedelta(days=10):
        # Settlement file does not reach back far enough: use Yahoo closes.
        import yfinance as yf
        h = yf.Ticker("^GSPC").history(start=pd.Timestamp(a.start) - pd.Timedelta(days=20),
                                       end=pd.Timestamp(a.end) + pd.Timedelta(days=15),
                                       auto_adjust=False)["Close"]
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        st = pd.concat([h, st]).groupby(level=0).last().sort_index()
    sessions = st.index
    days = [d for d in sessions if pd.Timestamp(a.start) <= d <= pd.Timestamp(a.end)]
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days if not (OUT / f"{d:%Y-%m-%d}.parquet").exists()
            and len(st[st.index < d])]
    syms = {d: _symbols(d, sessions, float(st[st.index < d].iloc[-1])) for d in todo}

    def price(d):
        s, e = _window(d)
        try:
            return d, float(_retry(lambda: client.metadata.get_cost(
                dataset=DATASET, schema="statistics", symbols=syms[d], stype_in="raw_symbol",
                start=s, end=e), f"{d:%Y-%m-%d} price"))
        except db.BentoClientError as ex:
            if "symbology_invalid_request" in str(ex):   # nothing listed in range that day
                return d, 0.0
            raise
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        costs = dict(ex.map(price, todo))
    skipped = [d for d in todo if costs[d] <= 0]
    todo = [d for d in todo if costs[d] > 0]
    total = sum(costs.values())
    print(f"{len(todo)} sessions, estimated ${total:.2f} (cap ${a.max_usd:.2f}); "
          f"{len(skipped)} days with no resolvable contracts skipped", flush=True)
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
