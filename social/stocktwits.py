"""Stocktwits recorder: timestamped finance chatter for SPX, SPY, ES and VIX.

Source: Stocktwits' public symbol streams (api.stocktwits.com/api/2), no key
needed (tested 2026-11-08). Each call returns the 30 newest messages before
an optional `max` id, newest first, so history is read by paging *backward*.
Measured: 60 pages of $SPX went back 16 days in 80 s with no rate limiting.

Why it is worth recording: it is X-style chatter but finance-only, and each
message can carry a Bullish/Bearish tag set by the poster at posting time, a
ready-made sentiment label.

Paging state, per symbol, in _state.json:
  newest_id  highest id saved. Updated only after a catch-up has paged all
             the way down to the previous newest_id, so an interrupted
             catch-up re-reads the gap instead of skipping it.
  oldest_id  lowest id saved. The backfill continues below it until
             messages are older than --start, so an interrupted backfill
             resumes where it stopped.
Rows are deduplicated on id, so re-reading is safe.

What is stored, and what is deliberately not
--------------------------------------------
  stored   id, stream, created_utc, retrieved_utc, user_hash, body,
           sentiment (poster's own tag), symbols mentioned
  dropped  likes, reshares, follower counts. These are as of when we fetch,
           which for backfilled rows is the future relative to the post.
  hashed   usernames.
`retrieved_utc` lets a feature builder require retrieved_utc <= decision
time on live data.

Storage: data/social/stocktwits/{SYMBOL}/{YYYY-MM-DD}.parquet (UTC dates).

Usage:
  python3 -m social.stocktwits sync --start 2026-10-01   # backfill + catch up
  python3 -m social.stocktwits sync                      # catch up (launchd)
  python3 -m social.stocktwits status
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "social" / "stocktwits"
API = "https://api.stocktwits.com/api/2/streams/symbol"
SYMBOLS = ("SPX", "ES_F", "VIX", "SPY")   # SPY last: busiest, slowest to backfill
UA = "Mozilla/5.0 (silicon-alpha research recorder)"
PAUSE_S = 0.8
WRITE_EVERY = 20          # pages between writes during long runs
COLUMNS = ["id", "stream", "created_utc", "retrieved_utc", "user_hash",
           "body", "sentiment", "symbols"]


def _get(symbol: str, max_id=None, attempts: int = 30) -> list:
    url = f"{API}/{symbol}.json" + (f"?max={max_id}" if max_id else "")
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8")).get("messages", [])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            # 429 = rate limit (unauthenticated access is ~200 calls/hour).
            # It is expected during a backfill, so wait it out rather than
            # fail: up to ~2.5 h of 5-minute waits before giving up.
            wait = 300 if e.code == 429 else 10 * (i + 1)
            if i == attempts - 1 or (e.code != 429 and i >= 5):
                raise
            print(f"  {symbol}: HTTP {e.code}, retry in {wait}s", flush=True)
            time.sleep(wait)
        except Exception as e:
            if i == attempts - 1:
                raise
            print(f"  {symbol}: {str(e)[:80]}, retry in {10 * (i + 1)}s", flush=True)
            time.sleep(10 * (i + 1))
    return []


def _hash(name) -> str:
    return hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:16] if name else ""


def _row(m: dict, stream: str, now: int) -> dict:
    sent = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
    syms = [s.get("symbol") for s in (m.get("symbols") or []) if s.get("symbol")]
    return {
        "id": int(m["id"]),
        "stream": stream,
        "created_utc": int(pd.Timestamp(m["created_at"]).timestamp()),
        "retrieved_utc": now,
        "user_hash": _hash((m.get("user") or {}).get("username")),
        "body": m.get("body"),
        "sentiment": sent,
        "symbols": ",".join(syms),
    }


def _dir(symbol: str) -> Path:
    return OUT / symbol


def _load_state(symbol: str) -> dict:
    p = _dir(symbol) / "_state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_state(symbol: str, st: dict) -> None:
    d = _dir(symbol)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "_state.tmp"
    tmp.write_text(json.dumps(st))
    tmp.replace(d / "_state.json")


def _write(symbol: str, rows: list) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["day"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    n_new = 0
    for day, part in df.groupby("day"):
        p = _dir(symbol) / f"{day}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        part = part.drop(columns="day")
        if p.exists():
            old = pd.read_parquet(p)
            before = len(old)
            part = pd.concat([old, part]).drop_duplicates("id", keep="first")
            n_new += len(part) - before
        else:
            n_new += len(part)
        tmp = p.with_suffix(".tmp")
        part.sort_values("created_utc").to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(p)
    return n_new


def _page_down(symbol: str, start_max, stop_id=None, stop_ts=None, st=None,
               track_oldest: bool = False) -> int:
    """Page backward from `start_max` until id <= stop_id or ts < stop_ts.

    Writes every WRITE_EVERY pages. With track_oldest, oldest_id in the state
    advances as pages are written, which makes a backfill resumable.
    """
    max_id, buf, total, pages = start_max, [], 0, 0
    while True:
        msgs = _get(symbol, max_id)
        time.sleep(PAUSE_S)
        pages += 1
        if not msgs:
            break
        now = int(time.time())
        done = False
        for m in msgs:
            mid = int(m["id"])
            ts = int(pd.Timestamp(m["created_at"]).timestamp())
            if (stop_id is not None and mid <= stop_id) or \
               (stop_ts is not None and ts < stop_ts):
                done = True
                break
            buf.append(_row(m, symbol, now))
        max_id = min(int(m["id"]) for m in msgs) - 1
        if done or pages % WRITE_EVERY == 0:
            total += _write(symbol, buf)
            if track_oldest and buf:
                st["oldest_id"] = min(r["id"] for r in buf)
                st["oldest_utc"] = min(r["created_utc"] for r in buf)
                _save_state(symbol, st)
            if pages % (WRITE_EVERY * 5) == 0 and buf:
                print(f"  {symbol}: back to "
                      f"{pd.Timestamp(min(r['created_utc'] for r in buf), unit='s', tz='UTC'):%Y-%m-%d %H:%M}Z "
                      f"(+{total})", flush=True)
            buf = []
        if done:
            break
    total += _write(symbol, buf)
    if track_oldest and buf:
        st["oldest_id"] = min(r["id"] for r in buf)
        st["oldest_utc"] = min(r["created_utc"] for r in buf)
        _save_state(symbol, st)
    return total


def sync_symbol(symbol: str, start=None) -> int:
    st = _load_state(symbol)
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp()) if start else None
    total = 0
    if "newest_id" not in st:
        # First run: take the top page, fix newest_id, then backfill below it.
        top = _get(symbol)
        time.sleep(PAUSE_S)
        if not top:
            return 0
        now = int(time.time())
        rows = [_row(m, symbol, now) for m in top]
        total += _write(symbol, rows)
        st.update(newest_id=max(r["id"] for r in rows),
                  oldest_id=min(r["id"] for r in rows),
                  oldest_utc=min(r["created_utc"] for r in rows))
        _save_state(symbol, st)
    else:
        # Catch-up: page down to the previous newest_id. Only then move it,
        # so an interrupted catch-up re-reads the gap next time.
        top = _get(symbol)
        time.sleep(PAUSE_S)
        if top:
            new_top = max(int(m["id"]) for m in top)
            if new_top > st["newest_id"]:
                total += _page_down(symbol, None, stop_id=st["newest_id"])
                st["newest_id"] = new_top
                _save_state(symbol, st)
    if start_ts is not None and st.get("oldest_utc", 0) > start_ts:
        total += _page_down(symbol, st["oldest_id"] - 1, stop_ts=start_ts,
                            st=st, track_oldest=True)
    return total


def sync(start=None, symbols=SYMBOLS, min_free_gb: float = 1.0) -> dict:
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < min_free_gb:
        print(f"SKIP: only {free_gb:.1f} GB free (< {min_free_gb} GB)")
        return {}
    OUT.mkdir(parents=True, exist_ok=True)
    lock = open(OUT / ".sync.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("SKIP: another stocktwits sync is running")
        return {}
    out = {}
    for s in symbols:
        out[s] = sync_symbol(s, start)
        print(f"{s}: +{out[s]} new", flush=True)
    return out


def status() -> None:
    for s in SYMBOLS:
        files = sorted(_dir(s).glob("20??-??-??.parquet"))
        if not files:
            print(f"{s}: empty")
            continue
        df = pd.concat(pd.read_parquet(f, columns=["created_utc", "sentiment"]) for f in files)
        tagged = df["sentiment"].notna().mean()
        print(f"{s}: {len(df):,} msgs, {files[0].stem} .. {files[-1].stem}, "
              f"{tagged:.0%} sentiment-tagged, "
              f"{sum(f.stat().st_size for f in files)/1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sync")
    s.add_argument("--start", default=None, help="backfill down to this UTC date")
    s.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    s.add_argument("--min-free-gb", type=float, default=1.0)
    sp.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "sync":
        sync(start=a.start, symbols=tuple(a.symbols), min_free_gb=a.min_free_gb)
    else:
        status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
