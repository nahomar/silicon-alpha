"""Execution-research data (owner approved $31.08 on 2026-11-11, quoted per part):

  A      XSP 0DTE near-the-money 1-minute quotes, 2024-11-11 .. 2026-11-10   -> data/odte_xsp/        cap $2.275
  B_spx  SPX 0DTE near-the-money 1-SECOND quotes, 2026-08-17 .. 2026-11-10   -> data/odte_1s/spx/     cap $20.415
  B_xsp  XSP 0DTE near-the-money 1-SECOND quotes, same sessions              -> data/odte_1s/xsp/     cap $8.405
  M      (owner approved up to $8.00) same-day options of SPY QQQ IWM TLT GLD USO XLF EEM, 1-minute,
         +/-3% of the previous close, 2024-11-11 .. 2026-11-10 -> data/odte_etf/<ROOT>/   total cap $8.00

Contracts are exactly the ones quoted: zero_dte_symbols(root, day, previous SPX close (/10 for XSP),
band, step). Every session is priced first (free); a part aborts before downloading if its total
exceeds its cap, and re-checks the running total before each session. Existing files are skipped,
so a rerun resumes. Files use the research chain schema (odte/data/fetch_0dte.reduce_frame).

  python3 -m odte.data.fetch_exec A [--dry-run]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from odte.data.fetch_0dte import _retry, reduce_frame, session_window, zero_dte_symbols

ROOT = Path(__file__).resolve().parents[2]
DATASET = "OPRA.PILLAR"
JOBS = {
    "A": ("XSP", "cbbo-1m", 0.03, 1.0, "2024-11-11", "2026-11-10", ROOT / "data" / "odte_xsp", 2.275),
    "B_spx": ("SPXW", "cbbo-1s", 0.02, 5.0, "2026-08-17", "2026-11-10", ROOT / "data" / "odte_1s" / "spx", 20.415),
    "B_xsp": ("XSP", "cbbo-1s", 0.03, 1.0, "2026-08-17", "2026-11-10", ROOT / "data" / "odte_1s" / "xsp", 8.405),
}
LOG = ROOT / "data" / "odte_1s" / "spend_log.jsonl"


def spent_so_far(prefix: str) -> float:
    """Dollars already spent on parts whose name starts with `prefix` (resumed runs count them)."""
    if not LOG.exists():
        return 0.0
    tot = 0.0
    for line in LOG.read_text().splitlines():
        r = json.loads(line)
        if r["part"] == prefix or (prefix == "M" and r["part"].startswith("M_")):
            tot += r["usd"]
    return tot


def sessions(a, b) -> list:
    st = pd.read_csv(ROOT / "data" / "odte_0dte" / "settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
    cal = pd.DatetimeIndex(pd.read_parquet(ROOT / "data" / "calendar" / "sessions.parquet")["session"])
    return [(d, float(st[st.index < d].iloc[-1])) for d in cal if pd.Timestamp(a) <= d <= pd.Timestamp(b) and d in st.index]


def run(name: str, dry: bool = False) -> int:
    import databento as db
    root, schema, band, step, a, b, out, cap = JOBS[name]
    out.mkdir(parents=True, exist_ok=True)
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    todo = [(d, prev / (10 if root == "XSP" else 1)) for d, prev in sessions(a, b)
            if not (out / f"{d:%Y-%m-%d}.parquet").exists()]
    syms = {d: zero_dte_symbols(root, d, c, band, step) for d, c in todo}

    def price(item):
        d, _ = item
        s, e = session_window(d)
        try:
            return d, float(_retry(lambda: client.metadata.get_cost(dataset=DATASET, schema=schema, symbols=syms[d],
                                                                    stype_in="raw_symbol", start=s, end=e), f"{d:%Y-%m-%d} price"))
        except db.BentoClientError as ex:
            if "symbology_invalid_request" in str(ex):
                return d, 0.0
            raise
    with ThreadPoolExecutor(8) as ex:
        costs = dict(ex.map(price, todo))
    todo = [(d, c) for d, c in todo if costs[d] > 0]
    total = sum(costs[d] for d, _ in todo)
    prev = spent_so_far(name)
    cap = cap - prev                                  # the approval covers the whole part, resumed or not
    print(f"{name}: {len(todo)} sessions to fetch, estimated ${total:.2f}; already spent ${prev:.2f}; "
          f"remaining cap ${cap:.3f}", flush=True)
    if total > cap:
        print("ABORT: estimate exceeds the cap; nothing downloaded.")
        return 2
    if dry:
        return 0
    spent = 0.0
    with tempfile.TemporaryDirectory(dir=out) as td:
        for d, _ in todo:
            if spent + costs[d] > cap + 1e-9:
                print(f"{d:%Y-%m-%d} STOP: would pass the cap (spent ${spent:.2f})")
                break
            s, e = session_window(d)
            tmp = Path(td) / f"{d:%Y%m%d}.dbn.zst"
            def _get():
                tmp.unlink(missing_ok=True)                 # a failed attempt can leave a partial file
                client.timeseries.get_range(dataset=DATASET, schema=schema, symbols=syms[d],
                                            stype_in="raw_symbol", start=s, end=e, path=str(tmp))
            _retry(_get, f"{d:%Y-%m-%d}")
            spent += costs[d]
            parts = [reduce_frame(ch.reset_index(), d) for ch in db.DBNStore.from_file(str(tmp)).to_df(count=500_000, map_symbols=True)]
            tmp.unlink(missing_ok=True)
            chain = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            f = out / f"{d:%Y-%m-%d}.parquet"
            chain.to_parquet(f.with_suffix(".tmp"), index=False)
            f.with_suffix(".tmp").replace(f)
            with open(LOG, "a") as fh:
                fh.write(json.dumps({"part": name, "day": f"{d:%Y-%m-%d}", "usd": round(costs[d], 6), "rows": len(chain)}) + "\n")
            print(f"{d:%Y-%m-%d}  {len(chain):>10,} rows  ${costs[d]:.3f}  (spent ${spent:.2f})", flush=True)
    print(f"{name}: done, spent ${spent:.2f}")
    return 0


MULTI = ("SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "XLF", "EEM")
MULTI_CAP = 8.00
ETF = ROOT / "data" / "odte_etf"


def closes(root: str) -> pd.Series:
    """Daily closes (Yahoo), kept append-only in data/odte_etf/<ROOT>/settlements.csv."""
    import yfinance as yf
    f = ETF / root / "settlements.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    h = yf.Ticker(root).history(start="2024-10-20", end="2026-11-11", auto_adjust=False)["Close"]
    new = pd.DataFrame({"date": pd.to_datetime(h.index).tz_localize(None).normalize(), "close": h.to_numpy()})
    if f.exists():
        old = pd.read_csv(f, parse_dates=["date"])
        new = pd.concat([old, new[~new["date"].isin(set(old["date"]))]], ignore_index=True).sort_values("date")
    new.to_csv(f, index=False)
    return new.set_index("date")["close"]


def run_multi(dry: bool = False, workers: int = 4) -> int:
    import threading
    import databento as db
    client = db.Historical(os.environ["DATABENTO_API_KEY"])
    cal = pd.DatetimeIndex(pd.read_parquet(ROOT / "data" / "calendar" / "sessions.parquet")["session"])
    days = [d for d in cal if pd.Timestamp("2024-11-11") <= d <= pd.Timestamp("2026-11-10")]
    jobs = []
    for r in MULTI:
        c = closes(r)
        step = 0.5 if float(c.iloc[-1]) < 100 else 1.0
        for d in days:
            prev = c[c.index < d]
            if len(prev) and not (ETF / r / f"{d:%Y-%m-%d}.parquet").exists():
                jobs.append((r, d, zero_dte_symbols(r, d, float(prev.iloc[-1]), 0.03, step)))

    def price(j):
        r, d, syms = j
        s, e = session_window(d)
        try:
            return float(_retry(lambda: client.metadata.get_cost(dataset=DATASET, schema="cbbo-1m", symbols=syms,
                                                                 stype_in="raw_symbol", start=s, end=e), f"{r} {d:%Y-%m-%d} price"))
        except db.BentoClientError:
            return 0.0                                   # no same-day expiry listed that session
    with ThreadPoolExecutor(8) as ex:
        costs = list(ex.map(price, jobs))
    todo = [(j, c) for j, c in zip(jobs, costs) if c > 0]
    total = sum(c for _, c in todo)
    by = pd.Series({r: sum(c for (j, c) in todo if j[0] == r) for r in MULTI}).round(3).to_dict()
    prev = spent_so_far("M")
    cap = MULTI_CAP - prev
    print(f"M: {len(todo)} market-sessions to fetch, estimated ${total:.2f}; already spent ${prev:.2f}; remaining cap "
          f"${cap:.2f}; by market {by}", flush=True)
    if total > cap:
        print("ABORT: estimate exceeds the cap; nothing downloaded.")
        return 2
    if dry:
        return 0
    lock, spent = threading.Lock(), [0.0]

    def fetch(item):
        (r, d, syms), cost = item
        with lock:
            if spent[0] + cost > cap + 1e-9:
                return
            spent[0] += cost
        s, e = session_window(d)
        out = ETF / r
        with tempfile.TemporaryDirectory(dir=out) as td:
            tmp = Path(td) / f"{d:%Y%m%d}.dbn.zst"
            def _get():
                tmp.unlink(missing_ok=True)
                client.timeseries.get_range(dataset=DATASET, schema="cbbo-1m", symbols=syms,
                                            stype_in="raw_symbol", start=s, end=e, path=str(tmp))
            _retry(_get, f"{r} {d:%Y-%m-%d}")
            parts = [reduce_frame(ch.reset_index(), d) for ch in db.DBNStore.from_file(str(tmp)).to_df(count=500_000, map_symbols=True)]
        chain = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        f = out / f"{d:%Y-%m-%d}.parquet"
        chain.to_parquet(f.with_suffix(".tmp"), index=False)
        f.with_suffix(".tmp").replace(f)
        with lock:
            with open(LOG, "a") as fh:
                fh.write(json.dumps({"part": f"M_{r}", "day": f"{d:%Y-%m-%d}", "usd": round(cost, 6), "rows": len(chain)}) + "\n")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, todo))
    print(f"M: done, spent ${spent[0]:.2f}")
    return 0


if __name__ == "__main__":
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "M":
        sys.exit(run_multi("--dry-run" in sys.argv))
    sys.exit(run(sys.argv[1], "--dry-run" in sys.argv))
