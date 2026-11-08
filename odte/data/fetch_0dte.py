"""Download SPXW 0DTE one-minute quotes from Databento, one session at a time.

Built for a nearly full disk. Each session is fetched to a temporary
.dbn.zst, reduced to same-day-expiry contracts within +/- `band` of the
implied forward during regular hours, saved as
data/odte_0dte/YYYY-MM-DD.parquet, and the temporary file is deleted before
the next session starts. Sessions already on disk are skipped, so a rerun
resumes where it stopped.

Spend guard: every session is priced first (Databento's metadata calls are
free). The run aborts before downloading anything if the total exceeds
--max-usd, and checks the running total again before each session.

Settlement: SPXW 0DTE cash-settles at the SPX official close. Daily closes
come from Yahoo (^GSPC) and are written to settlements.csv; the study falls
back to the last implied forward for any day that is missing.

Usage:
  python3 -m odte.data.fetch_0dte --start 2026-10-01 --end 2026-11-06 --max-usd 35 --dry-run
  python3 -m odte.data.fetch_0dte --start 2026-10-01 --end 2026-11-06 --max-usd 35
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "odte_0dte"
DATASET = "OPRA.PILLAR"
SCHEMA = "cbbo-1m"
PARENT = "SPXW.OPT"
# Cash-settled 0DTE roots: options root -> (Yahoo ticker of the settlement index, data folder)
ROOTS = {"SPXW": ("^GSPC", OUT_DIR),
         "NDXP": ("^NDX", ROOT / "data" / "odte_0dte_ndxp"),
         "RUTW": ("^RUT", ROOT / "data" / "odte_0dte_rutw")}
TZ = "America/New_York"


def _retry(fn, label: str, attempts: int = 5):
    """Call fn(), retrying dropped connections and 5xx errors with backoff.

    4xx errors (billing, auth, bad request) are raised immediately: retrying
    them cannot help and could hide an account problem.
    """
    import databento as db
    import requests
    for i in range(attempts):
        try:
            return fn()
        except db.BentoClientError:
            raise
        except (db.BentoError, requests.exceptions.RequestException,
                ConnectionError, TimeoutError) as e:
            if i == attempts - 1:
                raise
            wait = 20 * (i + 1)
            msg = getattr(e, "http_status", None) or str(e)[:80]
            print(f"{label}  transient error ({msg}); retry in {wait}s", flush=True)
            time.sleep(wait)


def session_window(day: pd.Timestamp) -> tuple:
    """09:25 to 16:01 New York time, as UTC ISO strings (end exclusive)."""
    d = pd.Timestamp(day).normalize()
    start = (d + pd.Timedelta(hours=9, minutes=25)).tz_localize(TZ).tz_convert("UTC")
    end = (d + pd.Timedelta(hours=16, minutes=1)).tz_localize(TZ).tz_convert("UTC")
    return start.isoformat(), end.isoformat()


def reduce_frame(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Databento cbbo DataFrame -> canonical chain rows for `day`'s expiry.

    Symbols are OSI: 'SPXW  261106C05800000' = root(6) yymmdd C/P strike*1000(8).
    The quote time is ts_recv (when the consolidated quote was recorded),
    converted to naive New York time.
    """
    if df.empty:
        return df.iloc[0:0]
    sym = df["symbol"].astype(str)
    keep = sym.str.slice(6, 12) == pd.Timestamp(day).strftime("%y%m%d")
    if not keep.any():
        return df.iloc[0:0]
    d = df.loc[keep]
    s = sym[keep]
    ts = d["ts_recv"] if "ts_recv" in d.columns else d.index.to_series()
    ts = pd.to_datetime(ts, utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    out = pd.DataFrame({
        "quote_datetime": ts.to_numpy(),
        "root": s.str.slice(0, 6).str.strip().to_numpy(),
        "expiration": pd.Timestamp(day).normalize(),
        "strike": s.str.slice(13, 21).astype(np.int64).to_numpy() / 1000.0,
        "option_type": s.str.slice(12, 13).to_numpy(),
        "bid": d["bid_px_00"].astype("float64").to_numpy(),
        "ask": d["ask_px_00"].astype("float64").to_numpy(),
        "bid_size": d["bid_sz_00"].astype("float32").to_numpy(),
        "ask_size": d["ask_sz_00"].astype("float32").to_numpy(),
    })
    t = out["quote_datetime"]
    rth = (t.dt.time >= pd.Timestamp("09:25").time()) & (t.dt.time <= pd.Timestamp("16:00").time())
    return out[rth].reset_index(drop=True)


def band_filter(chain: pd.DataFrame, band: float) -> pd.DataFrame:
    """Keep strikes within +/- band of the session's median implied forward."""
    from odte.eval.expiry_edge import implied_forward, minute_grid
    F = implied_forward(minute_grid(chain)).median()
    if not np.isfinite(F):
        return chain
    lo, hi = F * (1 - band), F * (1 + band)
    return chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)].reset_index(drop=True)


def zero_dte_symbols(root: str, day: pd.Timestamp, center: float, band: float,
                     step: float = 5.0) -> list:
    """OSI symbols for `day`'s expiry, every `step` points within +/- band.

    Asking only for these, instead of the parent (every listed expiry), cuts
    the bytes, cost and time several-fold. Strikes that are not listed simply
    do not resolve and are skipped by the API.
    """
    lo = np.floor(center * (1 - band) / step) * step
    hi = np.ceil(center * (1 + band) / step) * step
    ymd = pd.Timestamp(day).strftime("%y%m%d")
    return [f"{root:<6s}{ymd}{cp}{int(round(k * 1000)):08d}"
            for k in np.arange(lo, hi + step / 2, step) for cp in ("C", "P")]


def fetch_session(client, day: pd.Timestamp, tmp_dir: Path, band: float,
                  parent: str = PARENT, symbols=None) -> pd.DataFrame:
    import databento as db
    start, end = session_window(day)
    tmp = tmp_dir / f"{day:%Y%m%d}.dbn.zst"
    try:
        def _get():
            tmp.unlink(missing_ok=True)
            client.timeseries.get_range(dataset=DATASET, schema=SCHEMA,
                                        symbols=symbols or [parent],
                                        stype_in="raw_symbol" if symbols else "parent",
                                        start=start, end=end, path=str(tmp))
        _retry(_get, f"{day:%Y-%m-%d}")
        store = db.DBNStore.from_file(str(tmp))
        parts = []
        for chunk in store.to_df(count=500_000, map_symbols=True):
            chunk = chunk.reset_index()
            parts.append(reduce_frame(chunk, day))
        chain = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    finally:
        tmp.unlink(missing_ok=True)
    if chain.empty:
        return chain
    return band_filter(chain, band)


def fetch_settlements(days: list, out: Path, index: str = "^GSPC") -> pd.DataFrame:
    import yfinance as yf
    lo = min(days) - pd.Timedelta(days=7)
    hi = max(days) + pd.Timedelta(days=1)
    h = yf.Ticker(index).history(start=f"{lo:%Y-%m-%d}", end=f"{hi:%Y-%m-%d}",
                                   auto_adjust=False)
    s = pd.DataFrame({"date": pd.to_datetime(h.index).tz_localize(None).normalize(),
                      "spx_close": h["Close"].to_numpy()})
    s.to_csv(out, index=False)
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True, help="last session, inclusive")
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--band", type=float, default=0.06)
    ap.add_argument("--root", default="SPXW", choices=list(ROOTS),
                    help="cash-settled 0DTE options root")
    ap.add_argument("--out", default=None, help="default: the root's own data folder")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="request only the session's own expiry near the money, "
                         "several sessions in parallel")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    import databento as db
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("DATABENTO_API_KEY is not set")
        return 1
    client = db.Historical(key)
    parent = f"{a.root}.OPT"
    index, default_out = ROOTS[a.root]
    out = Path(a.out) if a.out else default_out
    out.mkdir(parents=True, exist_ok=True)

    days = list(pd.bdate_range(a.start, a.end))
    syms = {}
    if a.fast:
        # Band centre = previous session's index close (known before the open).
        import yfinance as yf
        h = yf.Ticker(index).history(start=pd.Timestamp(a.start) - pd.Timedelta(days=10),
                                     end=pd.Timestamp(a.end) + pd.Timedelta(days=1),
                                     auto_adjust=False)["Close"]
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        for d in days:
            prev = h[h.index < d]
            if len(prev):
                syms[d] = zero_dte_symbols(a.root, d, float(prev.iloc[-1]), a.band + 0.01)
    todo, costs = [], {}
    for d in days:
        if (out / f"{d:%Y-%m-%d}.parquet").exists():
            continue
        s, e = session_window(d)
        sym, st = (syms[d], "raw_symbol") if d in syms else ([parent], "parent")
        try:
            c = float(_retry(lambda: client.metadata.get_cost(
                dataset=DATASET, schema=SCHEMA, symbols=sym,
                stype_in=st, start=s, end=e), f"{d:%Y-%m-%d} price"))
        except db.BentoClientError as ex:
            # On a market holiday no SPXW contract is listed for the window,
            # so the parent symbol does not resolve: 422, not a $0 quote.
            if "symbology_invalid_request" in str(ex):
                print(f"{d:%Y-%m-%d}  no listed contracts (holiday) -- skipped")
                continue
            raise
        if c <= 0:
            print(f"{d:%Y-%m-%d}  no data (holiday?) -- skipped")
            continue
        costs[d] = c
        todo.append(d)
    total = sum(costs.values())
    print(f"{len(todo)} sessions to fetch, estimated ${total:.2f} (cap ${a.max_usd:.2f})")
    if total > a.max_usd:
        print("ABORT: estimate exceeds the cap; nothing was downloaded.")
        return 2
    if a.dry_run:
        for d in todo:
            print(f"  {d:%Y-%m-%d}  ${costs[d]:.2f}")
        return 0

    import threading
    from concurrent.futures import ThreadPoolExecutor
    spent, lock = [0.0], threading.Lock()

    def one(d, td):
        with lock:
            if spent[0] + costs[d] > a.max_usd:
                print(f"{d:%Y-%m-%d}  STOP: would pass the cap (spent ${spent[0]:.2f})")
                return
            spent[0] += costs[d]              # reserve before downloading
        chain = fetch_session(client, d, Path(td), a.band, parent=parent,
                              symbols=syms.get(d) if a.fast else None)
        if chain.empty:
            print(f"{d:%Y-%m-%d}  empty after filtering")
            return
        f = out / f"{d:%Y-%m-%d}.parquet"
        tmp = f.with_suffix(".tmp")
        chain.to_parquet(tmp, index=False)
        tmp.replace(f)                        # atomic: a killed run leaves no half file
        print(f"{d:%Y-%m-%d}  {len(chain):>8,} rows  "
              f"{chain['strike'].nunique()} strikes  ${costs[d]:.2f}  "
              f"(spent ${spent[0]:.2f})", flush=True)

    with tempfile.TemporaryDirectory(dir=out) as td:
        with ThreadPoolExecutor(max_workers=a.workers if a.fast else 1) as ex:
            list(ex.map(lambda d: one(d, td), todo))

    have = sorted(pd.Timestamp(p.stem) for p in out.glob("20??-??-??.parquet"))
    if have:
        try:
            s = fetch_settlements(have, out / "settlements.csv", index=index)
            print(f"settlements: {len(s)} SPX closes -> {out/'settlements.csv'}")
        except Exception as e:
            print(f"settlements fetch failed ({e}); study will use last forward")
    return 0


if __name__ == "__main__":
    sys.exit(main())
