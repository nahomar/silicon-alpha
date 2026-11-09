"""One-shot sealed test of the improvement-round winner (C1) on SPX 2021H2 + 2022H1.

Gate and design: docs/expiry_edge_plan.md, 2026-11-09. Refuses if either block
is already in the lockbox ledger or strategy code is uncommitted; records both
blocks as opened.

  python3 -m odte.eval.sealed_c1
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.eval import improve as im
from odte.eval.lockbox import LEDGER, _git, load_ledger

ROOT = Path(__file__).resolve().parents[2]
LO, HI = pd.Timestamp("2021-07-01"), pd.Timestamp("2022-05-13")
TAG = "sealed-c1"


def main() -> int:
    led = load_ledger()
    if any(e["data"] == "odte_0dte" and e["block"] in ("2021H2", "2022H1") for e in led):
        raise SystemExit("REFUSED: 2021H2 / 2022H1 already opened")
    if _git("status", "--porcelain", "--", "odte/eval", "odte/data"):
        raise SystemExit("REFUSED: uncommitted strategy code")
    log = ROOT / "reports" / "expiry_edge" / f"run-{TAG}.log"
    cmd = [sys.executable, "-W", "ignore", "-m", "odte.eval.expiry_edge", "--data-dir", "data/odte_0dte",
           "--holdout-from", "2022-05-14", "--report-from", f"{LO:%Y-%m-%d}", "--report-to", f"{HI:%Y-%m-%d}",
           "--min-train-days", "40", "--refit-every", "5", "--no-importance", "--write", "--tag", TAG,
           "--sets", "+cal", "+cal+ctx"]
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                            env={**__import__("os").environ, "PYTHONPATH": str(ROOT)}).returncode
    if rc:
        raise SystemExit(f"dataset build failed ({rc}); see {log}. Nothing recorded.")
    from odte.eval.expiry_edge import paired_diff
    from odte.live.model import feature_list
    d = im.load([ROOT / "reports" / "expiry_edge" / f"decisions-{TAG}.parquet"], allow_sealed=True)
    d = d[d["day"] <= HI].reset_index(drop=True)
    full = feature_list()
    d["tested"] = d["day"].between(LO, HI)
    ok = ~d["post"].to_numpy()
    p0 = im.walk_forward(d, full, d["ratio"].clip(upper=4.0).to_numpy(), "ridge")
    p1 = im.walk_forward(d, full, d["ror_half"].to_numpy(), "ridge")
    sells = {"always_sell": ok.copy(), "C0_current": ok & (p0 < 0.85), "C1_profit": ok & (p1 > 0)}
    res = {}
    for cost in ("half", "touch"):
        ser = {k: im.daily(d, v, f"ror_{cost}") for k, v in sells.items()}
        res[cost] = {k: {**im.stats(x), "trades": int((sells[k] & d["tested"].to_numpy()).sum()),
                         "days": int(len(x)),
                         "vs_C0": paired_diff(x, ser["C0_current"]),
                         "vs_always": paired_diff(x, ser["always_sell"])} for k, x in ser.items()}
    h = res["half"]["C1_profit"]
    res["verdict"] = "PASS" if (h["total"] > 0 and h["vs_C0"]["mean"] > 0
                                and h["vs_always"]["mean"] > 0) else "FAIL"
    (ROOT / "reports" / "sealed_c1.json").write_text(json.dumps(res, indent=2, default=float))
    commit = _git("rev-parse", "--short", "HEAD")
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    for block, (a, b) in (("2021H2", ("2021-07-01", "2021-12-31")), ("2022H1", ("2022-01-01", "2022-05-13"))):
        led.append({"data": "odte_0dte", "block": block, "start": a, "end": b, "opened_at": now,
                    "commit": commit, "strategies": ["C1_profit (improvement round)"],
                    "primary": "today_cost_half_gap", "note": "opened together for the C1 sealed test",
                    "results_sealed_c1": res, "forced": None})
    LEDGER.write_text(json.dumps(led, indent=2, default=str))
    for cost in ("half", "touch"):
        print(f"\n== {cost} of today's gap ($1k worst case per trade), {LO:%Y-%m-%d} .. {HI:%Y-%m-%d}")
        for k, v in res[cost].items():
            print(f"  {k:12s} {v['days']} days {v['trades']:4d} trades  total ${v['total']:>8,.0f}  "
                  f"Sharpe {v['sharpe']:+.2f} [{v['sharpe_ci'][0]:+.2f}, {v['sharpe_ci'][1]:+.2f}]  "
                  f"maxDD ${v['max_dd']:,.0f}  vs C0 {v['vs_C0']['mean']:+.1f}/day  "
                  f"vs always {v['vs_always']['mean']:+.1f}/day")
    print(f"\nVERDICT (pre-registered gate): {res['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
