"""Evening check, scheduled Mon-Fri 13:30 Pacific (after every close, half days too).

  1. Did today's bot run finish? If not, alert: the bot cannot report its own absence.
  2. Is the disk getting full? Alert below MIN_FREE_GB.
  3. Back up the bot's records to the PRIVATE repo ~/projects/silicon-alpha-records
     (journals, recorded quotes, alerts, registry). The quotes are IBKR market data
     for personal use: the repo must stay private.

  python -m odte.live.watch
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from odte.live.config import Config

ROOT = Path(__file__).resolve().parents[2]
RECORDS = Path.home() / "projects" / "silicon-alpha-records"
MIN_FREE_GB = 10.0


def finished(cfg: Config, day) -> bool:
    p = cfg.journal_dir / f"{pd.Timestamp(day):%Y-%m-%d}.jsonl"
    if not p.exists():
        return False
    return any(json.loads(l).get("kind") == "day_summary" for l in p.read_text().splitlines() if l.strip())


def disk_free_gb(path: Path = ROOT) -> float:
    return shutil.disk_usage(path).free / 1e9


def _sources(cfg: Config) -> list:
    """(source, path inside the records repo); replay/test files are skipped."""
    live = cfg.journal_dir.parent
    out = [(p, Path("journal") / p.name) for p in sorted(cfg.journal_dir.glob("20??-??-??.jsonl"))]
    out += [(p, Path("chains") / p.name) for p in sorted(cfg.chain_dir.glob("20??-??-??.parquet"))]
    for sub in ("xsp", "ndx", "rut"):
        out += [(p, Path("chains") / sub / p.name) for p in sorted((cfg.chain_dir / sub).glob("20??-??-??.parquet"))]
    out += [(p, Path("news") / p.name) for p in sorted((live / "news").glob("20??-??-??.jsonl"))]
    for name in ("registry.json", "logs/alerts.log"):
        if (live / name).exists():
            out.append((live / name, Path(name)))
    return out


def backup(cfg: Config, dest: Path = RECORDS, push: bool = True) -> str:
    if not (dest / ".git").exists():
        raise RuntimeError(f"records repo missing at {dest}")
    n = 0
    for src, rel in _sources(cfg):
        tgt = dest / "live" / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if not tgt.exists() or tgt.stat().st_size != src.stat().st_size or tgt.read_bytes() != src.read_bytes():
            shutil.copy2(src, tgt)
            n += 1
    git = lambda *a: subprocess.run(["git", "-C", str(dest), *a], capture_output=True, text=True, timeout=120)
    git("add", "-A")
    if not git("status", "--porcelain").stdout.strip():
        return "backup: nothing new"
    r = git("commit", "-m", f"records {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    if r.returncode:
        raise RuntimeError(f"commit failed: {r.stderr.strip()[:200]}")
    if push:
        r = git("push", "-q", "origin", "HEAD")
        if r.returncode:
            raise RuntimeError(f"push failed: {r.stderr.strip()[:200]}")
    return f"backup: {n} file(s) updated" + (" and pushed" if push else "")


def main() -> int:
    from odte.live.notify import notify
    from odte.live.runner import is_trading_day
    cfg = Config(mode="paper")
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    if is_trading_day(today) and not finished(cfg, today):
        notify("bot did NOT finish today", f"{today:%Y-%m-%d}: no end-of-day summary. Check IB Gateway, "
                                           "the Mac, and data/live/logs/daily.err")
    free = disk_free_gb()
    if free < MIN_FREE_GB:
        notify("disk almost full", f"{free:.1f} GB free (want {MIN_FREE_GB:.0f}+). The bot may fail.")
    try:
        print(backup(cfg), flush=True)
    except Exception as e:
        notify("backup failed", f"{e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
