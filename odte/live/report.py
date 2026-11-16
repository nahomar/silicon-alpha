"""Study report from the bot's journals: did we get a signal, and what to do next.

  python -m odte.live.report                 # every live day so far + running totals
  python -m odte.live.report --day 2026-11-11

Per day: orders, fills, P&L, what unfilled orders would have made (adverse selection),
the shadow book at half-gap and at the touch, SPX vs XSP quoted gaps (the cost question),
guard events, news counts. Running totals: per strategy, days, trades, P&L, mean per day
with a rough 95% interval, which is what the go/no-go rules (docs/go_no_go.md) need.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from odte.live.config import Config


def read(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def day_report(recs: list, news_path: Path | None = None) -> dict:
    kinds = lambda k: [r for r in recs if r.get("kind") == k]
    summ = (kinds("day_summary") or [{}])[-1]
    dec = kinds("decision")
    spx_gap = [(d["mid"] - d["touch"]) / d["mid"] for d in dec if d.get("mid") and d.get("touch") and d["mid"] > 0]
    xsp_gap = [(d["xsp"]["fly_credit_mid"] - d["xsp"]["fly_credit"]) / d["xsp"]["fly_credit_mid"]
               for d in dec if isinstance(d.get("xsp"), dict) and d["xsp"].get("fly_credit_mid", 0) > 0
               and d["xsp"].get("fly_credit") is not None]
    book = {}
    for r in kinds("shadow_settle"):
        keys = [r["strategy"]] + ([f"{r['strategy']}:{r['index']}"] if r.get("index") else [])
        for key in keys:                                  # the lead shadow also per index
            b = book.setdefault(key, {"trades": 0, "half": 0.0, "touch": 0.0, "xsp_half": 0.0})
            b["trades"] += 1
            for k in ("half", "touch", "xsp_half"):
                v = r.get(k)
                if v is not None and math.isfinite(v):
                    b[k] += v
    fills = kinds("fill")
    news = []
    if news_path and news_path.exists():
        news = read(news_path)
    return {
        "day": summ.get("day"), "code": (kinds("start") or [{}])[0].get("code"),
        "decisions": summ.get("decisions"), "orders": summ.get("orders"), "fills": summ.get("fills"),
        "pnl": summ.get("pnl"), "unfilled": summ.get("unfilled"),
        "unfilled_would_have_pnl": summ.get("unfilled_would_have_pnl"),
        "share_of_gap_paid": [f.get("share_of_gap_paid") for f in fills],
        "spx_fly_gap_median": float(np.median(spx_gap)) if spx_gap else None,
        "xsp_fly_gap_median": float(np.median(xsp_gap)) if xsp_gap else None,
        "skip_reasons": pd.Series([d.get("reason") for d in dec if d.get("action") == "skip"]).value_counts().to_dict(),
        "shocks": len(kinds("shock")), "big_day": bool(kinds("big_day")), "crash": bool(kinds("crash")),
        "reconcile_ok": summ.get("reconcile_ok"),
        "notes": [r.get("text") for r in kinds("note")][:10],
        "news_headlines": len(news), "news_hot": sum(1 for n in news if n.get("hot")),
        "shadow": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()}
                   for k, v in sorted(book.items())},
    }


def running(days: list) -> pd.DataFrame:
    rows = []
    for d in days:
        for k, v in d["shadow"].items():
            rows.append({"day": d["day"], "strategy": k, "trades": v["trades"], "half": v["half"], "touch": v["touch"]})
    if not rows:
        return pd.DataFrame()
    x = pd.DataFrame(rows)
    all_days = sorted({d["day"] for d in days})
    out = []
    for k, g in x.groupby("strategy"):
        s = g.set_index("day")["half"].reindex(all_days, fill_value=0.0)
        n = len(s)
        se = s.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
        out.append({"strategy": k, "days": n, "trades": int(g["trades"].sum()), "half_total": s.sum(),
                    "touch_total": g["touch"].sum(), "mean_day": s.mean(),
                    "ci95": [s.mean() - 1.96 * se, s.mean() + 1.96 * se] if n > 1 else None})
    return pd.DataFrame(out).sort_values("half_total", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None)
    a = ap.parse_args()
    cfg = Config()
    paths = sorted(cfg.journal_dir.glob("20??-??-??.jsonl"))
    if a.day:
        paths = [p for p in paths if p.stem == a.day]
    days = [day_report(read(p), cfg.journal_dir.parent / "news" / p.name) for p in paths]
    days = [d for d in days if d["day"]]
    for d in days:
        print(json.dumps(d, indent=1, default=str))
    r = running(days)
    if len(r):
        print("\nRUNNING TOTALS (shadow book, $300 worst case per trade; go/no-go needs >= 60 days)")
        print(r.round(2).to_string(index=False))
    a_fill = sum(d.get("pnl") or 0 for d in days)
    a_unf = sum(d.get("unfilled_would_have_pnl") or 0 for d in days)
    print(f"\nadverse selection so far: filled orders ${a_fill:+,.2f} vs unfilled would-have ${a_unf:+,.2f} "
          f"over {len(days)} day(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
