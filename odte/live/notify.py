"""Alerts for the scheduled bot: one line in data/live/logs/alerts.log (the record to read later),
a push to the owner's phone through ntfy (if a topic is configured), and a macOS notification
when running on a Mac.

ntfy topic: env NTFY_TOPIC, else the first line of ~/.config/silicon-alpha/ntfy_topic (chmod 600).
The topic is a long random name: anyone who knows it can read the alerts, so it is never committed.

Title and text go to osascript as arguments, never inside the script text, so
nothing in a message can be run as AppleScript.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

LOG = Path(__file__).resolve().parents[2] / "data" / "live" / "logs" / "alerts.log"
_SCRIPT = ["-e", "on run argv", "-e", "display notification (item 2 of argv) with title (item 1 of argv)",
           "-e", "end run"]


TOPIC_FILE = Path.home() / ".config" / "silicon-alpha" / "ntfy_topic"


def _topic() -> str | None:
    t = os.environ.get("NTFY_TOPIC", "").strip()
    if not t and TOPIC_FILE.exists():
        t = TOPIC_FILE.read_text().strip().splitlines()[0] if TOPIC_FILE.read_text().strip() else ""
    return t or None


def push(title: str, text: str) -> bool:
    """Phone push through ntfy.sh; never raises (the log line is the record)."""
    topic = _topic()
    if not topic:
        return False
    try:
        import requests
        high = any(w in title.lower() for w in ("fail", "did not", "waiting", "stopped", "disk", "missed", "reconcile"))
        r = requests.post(f"https://ntfy.sh/{topic}", data=text[:3000].encode("utf-8"), timeout=10,
                          headers={"Title": title[:200].encode("ascii", "ignore").decode(),
                                   "Priority": "high" if high else "default", "Tags": "chart_with_upwards_trend"})
        return r.ok
    except Exception:
        return False


def notify(title: str, text: str, log: Path = LOG, desktop: bool = True) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}  {title}: {text}\n")
    print(f"ALERT {title}: {text}", flush=True)
    if desktop:
        push(title, text)
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
