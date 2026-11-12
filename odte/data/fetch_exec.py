"""Execution-research data (owner approved $31.08 on 2026-11-11, quoted per part):

  A      XSP 0DTE near-the-money 1-minute quotes, 2024-11-11 .. 2026-11-10   -> data/odte_xsp/        cap $2.275
  B_spx  SPX 0DTE near-the-money 1-SECOND quotes, 2026-08-17 .. 2026-11-10   -> data/odte_1s/spx/     cap $20.415
  B_xsp  XSP 0DTE near-the-money 1-SECOND quotes, same sessions              -> data/odte_1s/xsp/     cap $8.405

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
    print(f"{name}: {len(todo)} sessions to fetch, estimated ${total:.2f} (cap ${cap:.3f})", flush=True)
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
            _retry(lambda: client.timeseries.get_range(dataset=DATASET, schema=schema, symbols=syms[d],
                                                       stype_in="raw_symbol", start=s, end=e, path=str(tmp)), f"{d:%Y-%m-%d}")
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


if __name__ == "__main__":
    LOG.parent.mkdir(parents=True, exist_ok=True)
    sys.exit(run(sys.argv[1], "--dry-run" in sys.argv))
