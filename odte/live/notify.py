"""Alerts for the scheduled bot: a macOS notification plus one line in
data/live/logs/alerts.log (the record to read later).

Title and text go to osascript as arguments, never inside the script text, so
nothing in a message can be run as AppleScript.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

LOG = Path(__file__).resolve().parents[2] / "data" / "live" / "logs" / "alerts.log"
_SCRIPT = ["-e", "on run argv", "-e", "display notification (item 2 of argv) with title (item 1 of argv)",
           "-e", "end run"]


def notify(title: str, text: str, log: Path = LOG, desktop: bool = True) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}  {title}: {text}\n")
    print(f"ALERT {title}: {text}", flush=True)
    if desktop and sys.platform == "darwin":
        try:
            subprocess.run(["/usr/bin/osascript", *_SCRIPT, title, text[:400]], timeout=10, check=False,
                           capture_output=True)
        except Exception:
            pass                                           # the log line is the record


def day_text(s: dict) -> str:
    """One line for the end-of-day alert."""
    sh = s.get("shadow", {})
    parts = [f"{k} ${v.get('half', 0):+,.0f}" for k, v in sh.items()
             if k in ("frozen_lead", "event_seller", "sell_only", "side_seller", "event_seller:NDX", "event_seller:RUT",
                      "xindex_spx", "calm_seller")]
    return (f"{s['day']}: {s['orders']} orders, {s['fills']} filled, P&L ${s['pnl']:+,.2f}"
            + (f". Shadow (half-gap): {', '.join(parts)}" if parts else ""))
