"""Sequential lockbox evaluation: each 6-month block can be opened once.

Walk-forward training stops the *model* from seeing the future. It does not
stop the *researcher*: once a period's results have been looked at and the
strategy changed because of them, that period can no longer test the changed
strategy. So the evaluation data is cut into blocks, and:

  1. A block is scored once, for the strategy as committed beforehand. The
     ledger records the git commit, so "the strategy" means exact code.
  2. Models train on everything before the block (expanding window), and
     sessions after the block are dropped entirely, a second guard against
     peeking through any feature.
  3. After a block's result is seen, the strategy may be revised, but the
     revision is scored only on a later, still-sealed block.
  4. The ledger (docs/lockbox_ledger.json) refuses to reopen a block.
     --force exists only for crashed runs and must give a reason that is
     written to the ledger.

Usage:
  python3 -m odte.eval.lockbox plan  --data-dir data/odte_0dte --first 2016-01-01 --last 2022-05-13
  python3 -m odte.eval.lockbox open  --data-dir data/odte_0dte --block 2016H2 \\
          --strategy "ml_ridge+cal+ctx -postevt"
  python3 -m odte.eval.lockbox ledger
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs" / "lockbox_ledger.json"


def blocks(first: str, last: str) -> list:
    """Half-year blocks (H1 = Jan-Jun, H2 = Jul-Dec) covering [first, last]."""
    out = []
    for y in range(pd.Timestamp(first).year, pd.Timestamp(last).year + 1):
        for h, (a, b) in (("H1", ("01-01", "06-30")), ("H2", ("07-01", "12-31"))):
            lo, hi = pd.Timestamp(f"{y}-{a}"), pd.Timestamp(f"{y}-{b}")
            lo, hi = max(lo, pd.Timestamp(first)), min(hi, pd.Timestamp(last))
            if lo <= hi:
                out.append({"block": f"{y}{h}", "start": f"{lo:%Y-%m-%d}", "end": f"{hi:%Y-%m-%d}"})
    return out


def load_ledger() -> list:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else []


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def open_block(data_dir: str, block: str, strategy, first: str, last: str,
               min_train_days: int = 40, refit_every: int = 5, sets=("+cal", "+cal+ctx"),
               force_reason: str | None = None) -> dict:
    data_key = Path(data_dir).name
    spec = {b["block"]: b for b in blocks(first, last)}
    if block not in spec:
        raise SystemExit(f"unknown block {block}; plan has {list(spec)}")
    led = load_ledger()
    prior = [e for e in led if e["data"] == data_key and e["block"] == block]
    if prior and not force_reason:
        raise SystemExit(f"REFUSED: {data_key} {block} was already opened on "
                         f"{prior[0]['opened_at']} (commit {prior[0]['commit']}).")
    if _git("status", "--porcelain", "--", "odte/eval", "odte/data"):
        raise SystemExit("REFUSED: uncommitted strategy code; commit it so the ledger "
                         "pins exactly what was tested.")
    b = spec[block]
    tag = f"lockbox-{data_key}-{block}"
    holdout = (pd.Timestamp(b["end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cmd = [sys.executable, "-W", "ignore", "-m", "odte.eval.expiry_edge",
           "--data-dir", data_dir, "--min-train-days", str(min_train_days),
           "--refit-every", str(refit_every), "--holdout-from", holdout,
           "--report-from", b["start"], "--report-to", b["end"],
           "--no-importance", "--write", "--tag", tag, "--sets", *sets]
    log = ROOT / "reports" / "expiry_edge" / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                            env={**__import__("os").environ, "PYTHONPATH": str(ROOT)}).returncode
    if rc != 0:
        raise SystemExit(f"study failed (exit {rc}); see {log}. Block NOT recorded.")
    rep = json.loads((ROOT / "reports" / "expiry_edge" / f"study-{tag}.json").read_text())
    md = rep["modes"]["fixed6"]
    st = md["strategies"]
    strategies = [strategy] if isinstance(strategy, str) else list(strategy)
    ref = st["always_short_fly"]
    results = {}
    for name in strategies:
        r = st.get(name)
        if r is None:
            raise SystemExit(f"strategy {name!r} not in report; have {list(st)}")
        cmp_ = md["comparisons"].get(name, {}).get("vs_always_short_fly", {})
        results[name] = {"n_test_days": r["n_test_days"], "total": r["total_pnl"],
                         "sharpe": r["sharpe"], "sharpe_ci": r["sharpe_ci"],
                         "max_drawdown": r["max_drawdown"], "worst_day": r["worst_day"],
                         "vs_always_short_per_day": cmp_.get("mean"),
                         "vs_always_short_ci": cmp_.get("ci")}
    entry = {
        "data": data_key, "block": block, "start": b["start"], "end": b["end"],
        "strategies": strategies, "commit": _git("rev-parse", "--short", "HEAD"),
        "opened_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "always_short_total": ref["total_pnl"], "results": results,
        "forced": force_reason,
    }
    led.append(entry)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=2, default=str))
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    o = sp.add_parser("open")
    for x in (p, o):
        x.add_argument("--data-dir", required=True)
        x.add_argument("--first", default="2016-01-01")
        x.add_argument("--last", default="2022-05-13")
    o.add_argument("--block", required=True)
    o.add_argument("--strategy", nargs="+", required=True,
                   help="one or more strategies, all fixed before opening")
    o.add_argument("--force-reason", default=None)
    o.add_argument("--sets", nargs="+", default=["+cal", "+cal+ctx", "+cal+ctx+gex"])
    sp.add_parser("ledger")
    a = ap.parse_args()
    if a.cmd == "plan":
        opened = {(e["data"], e["block"]) for e in load_ledger()}
        for b in blocks(a.first, a.last):
            state = "OPENED" if (Path(a.data_dir).name, b["block"]) in opened else "sealed"
            print(f"  {b['block']}  {b['start']} .. {b['end']}  {state}")
    elif a.cmd == "open":
        e = open_block(a.data_dir, a.block, a.strategy, a.first, a.last,
                       sets=tuple(a.sets), force_reason=a.force_reason)
        print(f"{e['data']} {e['block']} @ {e['commit']}  (always-short ${e['always_short_total']:,.0f})")
        for name, r in e["results"].items():
            ci = r["vs_always_short_ci"] or [float("nan")] * 2
            print(f"  {name:36s} {r['n_test_days']} days  total ${r['total']:,.0f}  Sharpe {r['sharpe']:+.2f} "
                  f"[{r['sharpe_ci'][0]:+.2f}, {r['sharpe_ci'][1]:+.2f}]  max DD ${r['max_drawdown']:,.0f}  "
                  f"vs always-short {r['vs_always_short_per_day']:+.1f}/day CI [{ci[0]:+.0f}, {ci[1]:+.0f}]")
    else:
        for e in load_ledger():
            for name, r in e.get("results", {}).items():
                print(f"  {e['opened_at']}  {e['data']:<14s} {e['block']}  {name} @ {e['commit']}  "
                      f"total ${r['total']:,.0f}  Sharpe {r['sharpe']:+.2f}"
                      + (f"  FORCED: {e['forced']}" if e.get("forced") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
