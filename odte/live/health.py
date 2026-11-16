"""Pre-market health check: is everything ready before the bot starts? Pushes an alert if not.

Checks: IB Gateway's API answers on 127.0.0.1:4002 and reports a paper account; disk space; the
models load and predict; the trading calendar knows today.
  python -m odte.live.health            # print the report; alert on any failure
"""
from __future__ import annotations

import shutil
import socket
import sys

import numpy as np
import pandas as pd

from odte.live.config import Config, LIVE, ROOT
from odte.live.notify import notify


def check(cfg: Config = Config()) -> list:
    problems = []
    port = cfg.resolved_port()
    try:
        with socket.create_connection((cfg.host, port), timeout=5):
            pass
    except OSError as e:
        problems.append(f"IB Gateway not listening on {cfg.host}:{port} ({e.__class__.__name__})")
    else:
        try:
            from ib_async import IB
            ib = IB()
            ib.connect(cfg.host, port, clientId=99, timeout=15, readonly=True)
            accts = ib.managedAccounts()
            ib.disconnect()
            if not accts:
                problems.append("IB Gateway answered but reports no account (not logged in?)")
            elif not any(a.startswith("D") for a in accts):
                problems.append(f"IB Gateway is logged into a NON-paper account {accts}: the bot is paper-only")
        except Exception as e:
            problems.append(f"IB Gateway API handshake failed: {e!r}"[:200])
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < 10:
        problems.append(f"disk nearly full: {free:.1f} GB free")
    try:
        import pickle
        from odte.live.linear import linear_predict
        b = pickle.loads((LIVE / "model_frozen.pkl").read_bytes())
        x = np.full((1, len(b["features"])), np.nan)
        if not all(np.isfinite(linear_predict(m, x)).all() for m in b["models"].values()):
            problems.append("frozen model predicts non-finite values")
    except Exception as e:
        problems.append(f"frozen model unusable: {e!r}"[:200])
    try:
        s = pd.DatetimeIndex(pd.read_parquet(ROOT / "data" / "calendar" / "sessions.parquet")["session"])
        if s.max() < pd.Timestamp.now() + pd.Timedelta(days=30):
            problems.append(f"trading calendar ends {s.max():%Y-%m-%d}: extend it")
    except Exception as e:
        problems.append(f"trading calendar unreadable: {e!r}"[:200])
    return problems


def main() -> int:
    p = check()
    if p:
        notify("bot NOT ready", "; ".join(p))
        return 1
    print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M} health OK: gateway logged in (paper), disk, models, calendar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
