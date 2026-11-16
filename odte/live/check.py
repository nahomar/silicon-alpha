"""Read-only check that the bot can work with your IBKR account. Places no orders.

Run with IB Gateway (paper, API port 4002) open and logged in:
  .venv-live/bin/python -m odte.live.check
"""
from __future__ import annotations

import math
import sys

import pandas as pd

from odte.live.broker import IBKRBroker, strikes_for
from odte.live.config import Config


def main() -> int:
    cfg = Config(mode="paper")
    ok = True
    try:
        b = IBKRBroker(cfg)
    except Exception as e:
        print(f"1. connect to IB Gateway on port {cfg.resolved_port()}: FAIL ({e})\n"
              "   Is IB Gateway running, logged into the PAPER account, with the API enabled?")
        return 1
    print(f"1. connected (server version {b.ib.client.serverVersion()})")
    accts = b.ib.managedAccounts()
    print(f"2. accounts: {accts}  ({'paper' if any(a.startswith('D') for a in accts) else 'CHECK: not a paper id'})")
    try:
        spot = b.spot()
        print(f"3. {cfg.signal_symbol} index price: {spot:,.2f}")
    except Exception as e:
        print(f"3. index price: FAIL ({e}); subscribe to CBOE index data and share it with paper")
        return 1
    day = b.now().normalize()
    ks = strikes_for(spot, cfg.inner_band, cfg.band, cfg.outer_step, max_lines=10)
    b.subscribe(day, ks)
    b.sleep(4)
    snap = b.snapshot()
    n = len(snap)
    print(f"4. 0DTE {cfg.signal_symbol} option quotes: {n} of {2 * len(ks)} contracts quoted"
          + ("" if n else "  FAIL: needs OPRA options data shared with paper, or the market is closed"))
    ok &= n > 0
    if n:
        atm = snap.iloc[(snap["strike"] - spot).abs().argsort()[:2]]
        print(atm[["strike", "option_type", "bid", "ask"]].to_string(index=False))
    try:
        legs = [b._option(cfg.symbol, day, round(spot / 10), r) for r in ("C", "P")]
        q = b.ib.qualifyContracts(*legs)
        print(f"5. {cfg.symbol} 0DTE contracts resolve: {'yes' if all(c.conId for c in q) else 'NO'}")
    except Exception as e:
        print(f"5. {cfg.symbol} contracts: FAIL ({e})")
        ok = False
    for sym in cfg.shadow_indexes:                   # shadow-only indexes (need extra quote lines)
        try:
            from odte.live.broker import ROOTS
            prev = b.prev_close_extra(sym, day)
            k = round(prev / 50) * 50 if sym == "NDX" else round(prev / 5) * 5
            q = b.ib.qualifyContracts(b._option(sym, day, k, "C"))
            print(f"6. {ROOTS[sym]} 0DTE contract resolves: {'yes' if q and q[0].conId else 'NO'} "
                  f"(prev close {prev:,.2f}; shadow only)")
        except Exception as e:
            print(f"6. {sym} shadow contracts: FAIL ({e}); the SPX bot is unaffected")
    b.ib.disconnect()
    print("\nREADY for paper trading." if ok else "\nNot ready yet; see the lines above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
