"""Reddit recorder: timestamped posts and comments, backfilled and kept current.

Source: the Arctic Shift archive (arctic-shift.photon-reddit.com), which
ingests Reddit about 20 seconds after posting (median retrieved_on minus
created_utc on WSB, measured 2026-11-07). That makes it usable both for
history and as a near-live feed. reddit.com's own JSON returns 403 without
OAuth, and its listings cannot page back further than about 1,000 items.

One command does both jobs. `sync` reads the newest timestamp already on
disk and fetches everything after it, up to now. The first run, given
--start, is the backfill; every later run (launchd, every 5 minutes) just
catches up. It is idempotent: rows are deduplicated on id, so overlapping or
repeated runs are safe.

What is stored, and what is deliberately not
--------------------------------------------
  stored   id, kind, subreddit, created_utc, retrieved_on, author_hash,
           title, text, link_id, parent_id, flair, removed
  dropped  score, upvote ratio, num_comments. The archive refreshes these on
           a second pass (~36 h later, `_meta.retrieved_2nd_on`), so they
           describe the future relative to the post. A backtest that used
           them would be reading tomorrow's votes. Leaving them out makes
           that leak impossible rather than merely discouraged.
  hashed   author names: only unique-user counts are needed, so the raw
           username is never written to disk.

`retrieved_on` is kept so a feature builder can require
retrieved_on <= decision time, the strict point-in-time rule, rather than
just created_utc <= decision time.

Storage: data/social/reddit/{subreddit}/{posts|comments}/{YYYY-MM-DD}.parquet
(UTC dates). Writes are skipped if the disk has under --min-free-gb left.

Usage:
  python3 -m social.reddit sync --start 2026-10-01      # backfill + catch up
  python3 -m social.reddit sync                         # catch up (launchd)
  python3 -m social.reddit status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import fcntl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "social" / "reddit"
API = "https://arctic-shift.photon-reddit.com/api"
SUBS = ("wallstreetbets", "options", "thetagang")
KINDS = {"posts": "posts/search", "comments": "comments/search"}
UA = "silicon-alpha-research-recorder/0.1"
PAGE = 100
PAUSE_S = 0.4            # between requests; the archive is a free service
OVERLAP_S = 120          # re-read this much on catch-up; dedup handles it
COLUMNS = ["id", "kind", "subreddit", "created_utc", "retrieved_on",
           "author_hash", "title", "text", "link_id", "parent_id", "flair",
           "removed"]


def _get(path: str, params: dict, attempts: int = 5) -> list:
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8")).get("data")
            if data is None:
                raise ValueError("response had no data field")
            return data
        except Exception as e:                       # network or bad payload
            if i == attempts - 1:
                raise
            wait = 10 * (i + 1)
            detail = str(e)[:80]
            if isinstance(e, urllib.error.HTTPError):
                try:
                    detail += " " + e.read().decode("utf-8", "replace")[:160]
                except Exception:
                    pass
            print(f"  retry in {wait}s ({detail})", flush=True)
            time.sleep(wait)
    return []


def _author_hash(name) -> str:
    if not name or name in ("[deleted]", "AutoModerator"):
        return name or ""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def _row(x: dict, kind: str) -> dict:
    text = x.get("body") if kind == "comments" else x.get("selftext")
    removed = (text in ("[removed]", "[deleted]")
               or bool(x.get("removed_by_category")))
    return {
        "id": x["id"],
        "kind": kind,
        "subreddit": x.get("subreddit", ""),
        "created_utc": int(x["created_utc"]),
        "retrieved_on": int(x.get("retrieved_on") or 0),
        "author_hash": _author_hash(x.get("author")),
        "title": x.get("title") if kind == "posts" else None,
        "text": text,
        "link_id": x.get("link_id") if kind == "comments" else None,
        "parent_id": x.get("parent_id") if kind == "comments" else None,
        "flair": x.get("link_flair_text") if kind == "posts" else None,
        "removed": removed,
    }


def fetch_range(sub: str, kind: str, after: int, before: int):
    """Yield rows with after <= created_utc < before, oldest first.

    The archive's `after` is EXCLUSIVE (tested 2026-11-08: an item at second
    t is not returned for after=t). So each page re-reads from one second
    before its last timestamp, and `seen` drops the repeats. That way a page
    that ends partway through a second cannot lose the rest of that second.
    Completeness needs every page to be larger than the busiest single
    second, which holds on the real archive (>= 100 per page); a page that
    falls entirely inside one second is logged as a possible gap.
    """
    cursor = after - 1                     # make the caller's bound inclusive
    seen = set()
    while cursor < before:
        # limit="auto" lets the archive size the page (measured: ~350 vs 100
        # per request, ~5x the throughput), so page length can't signal the end.
        batch = _get(KINDS[kind], {"subreddit": sub, "after": cursor,
                                   "before": before, "limit": "auto",
                                   "sort": "asc"})
        time.sleep(PAUSE_S)
        if not batch:
            break
        fresh = [x for x in batch if x["id"] not in seen]
        for x in fresh:
            seen.add(x["id"])
            yield _row(x, kind)
        last = int(max(x["created_utc"] for x in batch))
        first = int(min(x["created_utc"] for x in batch))
        if fresh and first == last and last == cursor + 1:
            # A whole page inside one second: the time-only cursor cannot
            # page within a second, so anything past this page is skipped.
            # Pages are >= 100 and WSB peaks at a few per second, so this
            # should never fire; if it does, the gap is logged, not hidden.
            print(f"  WARN {sub}/{kind}: page filled second {last}; "
                  f"items beyond {len(batch)} in that second may be missed",
                  flush=True)
        if not fresh:
            # Only repeats: the tail second was already read. Step past it,
            # or stop if that reaches the end of the range.
            if last + 1 >= before:
                break
            cursor = max(cursor + 1, last)
            continue
        # Back up one second so a page ending mid-second loses nothing.
        cursor = max(cursor + 1, last - 1)


def _day_path(sub: str, kind: str, day: str) -> Path:
    return OUT / sub / kind / f"{day}.parquet"


def _write(sub: str, kind: str, rows: list) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["day"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    n_new = 0
    for day, part in df.groupby("day"):
        p = _day_path(sub, kind, day)
        p.parent.mkdir(parents=True, exist_ok=True)
        part = part.drop(columns="day")
        if p.exists():
            old = pd.read_parquet(p)
            before = len(old)
            part = pd.concat([old, part]).drop_duplicates("id", keep="first")
            n_new += len(part) - before
        else:
            n_new += len(part)
        part = part.sort_values("created_utc")
        tmp = p.with_suffix(".tmp")
        part.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(p)                                # atomic swap
    return n_new


def _latest_ts(sub: str, kind: str):
    d = OUT / sub / kind
    files = sorted(d.glob("20??-??-??.parquet")) if d.exists() else []
    if not files:
        return None
    return int(pd.read_parquet(files[-1], columns=["created_utc"])["created_utc"].max())


def sync(start=None, subs=SUBS, min_free_gb: float = 1.0,
         chunk_s: int = 6 * 3600) -> dict:
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < min_free_gb:
        print(f"SKIP: only {free_gb:.1f} GB free (< {min_free_gb} GB)")
        return {}
    OUT.mkdir(parents=True, exist_ok=True)
    lock = open(OUT / ".sync.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("SKIP: another sync is running")
        return {}
    now = int(time.time())
    summary = {}
    # Posts for every sub first (small, fast), then comments with the
    # biggest sub last, so a long WSB backfill never starves the others.
    order = ([(s, "posts") for s in subs]
             + [(s, "comments") for s in sorted(subs, key=lambda x: x == "wallstreetbets")])
    for sub, kind in order:
        last = _latest_ts(sub, kind)
        if last is not None:
            after = last - OVERLAP_S
        elif start is not None:
            after = int(pd.Timestamp(start, tz="UTC").timestamp())
        else:
            after = now - 3600                    # first run, no start
        total = 0
        # Write in chunks so a long backfill survives interruption.
        t = after
        while t < now:
            hi = min(t + chunk_s, now)
            rows = list(fetch_range(sub, kind, t, hi))
            total += _write(sub, kind, rows)
            t = hi
            if hi - after > chunk_s:              # progress on backfills
                print(f"  {sub}/{kind} through "
                      f"{pd.Timestamp(hi, unit='s', tz='UTC'):%Y-%m-%d %H:%M}Z "
                      f"(+{total})", flush=True)
        summary[f"{sub}/{kind}"] = total
        print(f"{sub}/{kind}: +{total} new", flush=True)
    return summary


def status() -> None:
    for sub in SUBS:
        for kind in KINDS:
            d = OUT / sub / kind
            files = sorted(d.glob("20??-??-??.parquet")) if d.exists() else []
            if not files:
                print(f"{sub}/{kind}: empty")
                continue
            n = sum(pd.read_parquet(f, columns=["id"]).shape[0] for f in files)
            last = _latest_ts(sub, kind)
            size = sum(f.stat().st_size for f in files) / 1e6
            print(f"{sub}/{kind}: {n:,} rows, {files[0].stem} .. {files[-1].stem}, "
                  f"newest {pd.Timestamp(last, unit='s', tz='UTC'):%Y-%m-%d %H:%M}Z, "
                  f"{size:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sync")
    s.add_argument("--start", default=None, help="backfill from this UTC date if empty")
    s.add_argument("--subs", nargs="+", default=list(SUBS))
    s.add_argument("--min-free-gb", type=float, default=1.0)
    sp.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "sync":
        sync(start=a.start, subs=tuple(a.subs), min_free_gb=a.min_free_gb)
    else:
        status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
