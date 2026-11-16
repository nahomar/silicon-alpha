"""Never-seen sessions after 2026-11-06 (owner approved $10.61 for 2026-11-09..11): the locked
model B0 and the three frozen contenders (C7, NF, D4L), exactly as frozen, on research-quality data.

Runs under round 2's lock: ROUND2_UNLOCK must equal the challenger freeze commit in
docs/round2_plan.md. Nothing is trained here.

  ROUND2_UNLOCK=<commit> python3 -m odte.research.newdays 2026-11-09 2026-11-11
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import pandas as pd

from odte.research import exam, exam_guard as eg, library, menu3, round4, zoo

ROOT = exam.ROOT
OUT = ROOT / "reports" / "round2" / "newdays.json"
MENU = ("fly", "condor", "put_spread", "call_spread")
LATE = ("15:00", "15:30")
SCALE = 0.3                                            # $1k labels -> the shadow book's $300


def main(a: str, b: str) -> int:
    if not eg.round2_unlocked():
        raise eg.ExamLocked("set ROUND2_UNLOCK to the challenger freeze commit in docs/round2_plan.md")
    days = exam.sessions_between(a, b)
    rows = exam.build_rows(days)                       # same code path as the development matrix
    from odte.live import frozen
    from odte.live.config import Config
    b0 = pickle.loads(frozen.model_path(Config()).read_bytes())
    d4 = pickle.loads(round4.FINAL.read_bytes())
    recs = []
    for (sym, day), g in rows.groupby(["index", "day"]):
        folder = library.DIRS[sym]
        st = pd.read_csv(folder / "settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
        zd = zoo.Day(day, pd.read_parquet(folder / f"{day:%Y-%m-%d}.parquet"), float(st[day]), np.nan, {})
        r10 = zd.straddle_rel_spread(day + pd.Timedelta(hours=10))
        R = b0["R_today"][sym]
        k_day = float(np.clip(R / r10, 0.1, 2.0)) if np.isfinite(r10) and r10 > 0 else 1.0
        for _, r in g.iterrows():
            x = r.to_dict()
            post = bool(r["post"])
            t0, _ = frozen.choose(frozen.predict_row(b0, x), post)
            ev = round4.ev_row(d4, x, zd, r["hhmm"], k_day)
            best = max(ev, key=ev.get) if ev else None
            t4 = "none" if (post or r["hhmm"] in LATE or best is None or ev[best] <= 0) else best
            choice = {"B0": t0, "C7": t0 if r["hhmm"] not in LATE else "none",
                      "NF": "fly_narrow" if t0 == "fly" else t0, "D4L": t4}
            rec = {"day": day, "index": sym, "hhmm": r["hhmm"], **{f"{k}_trade": v for k, v in choice.items()}}
            for name, trade in choice.items():
                for share, tag in ((0.5, "half"), (1.0, "touch")):
                    if trade == "none":
                        rec[f"{name}_{tag}"] = 0.0
                        continue
                    rule = menu3.NEW["fly_narrow"] if trade == "fly_narrow" else library.TRADES[trade]
                    tr = zoo.trade(rule, zd, r["hhmm"], k_day, share)
                    rec[f"{name}_{tag}"] = float(tr["ret"]) * SCALE if tr else np.nan
            recs.append(rec)
    df = pd.DataFrame(recs)
    out = {}
    for scope, x in (("SPX", df[df["index"] == "SPX"]), ("all", df)):
        daily = x.groupby("day")[[c for c in df.columns if c.endswith(("_half", "_touch"))]].sum()
        out[scope] = {"daily_half": {str(k.date()): {m: round(v[f"{m}_half"], 1) for m in ("B0", "C7", "NF", "D4L")}
                                     for k, v in daily.iterrows()},
                      "total_half": {m: round(float(daily[f"{m}_half"].sum()), 1) for m in ("B0", "C7", "NF", "D4L")},
                      "total_touch": {m: round(float(daily[f"{m}_touch"].sum()), 1) for m in ("B0", "C7", "NF", "D4L")},
                      "trades": {m: int((x[f"{m}_trade"] != "none").sum()) for m in ("B0", "C7", "NF", "D4L")}}
    always = rows.assign(r=np.where(rows["post"], 0.0, rows["y_fly_half"].fillna(0.0)) * SCALE)
    out["always_sell_half"] = {"SPX": round(float(always[always["index"] == "SPX"]["r"].sum()), 1),
                               "all": round(float(always["r"].sum()), 1)}
    out["days"] = [str(d.date()) for d in days]
    OUT.write_text(json.dumps(out, indent=2, default=float))
    df.to_parquet(OUT.with_suffix(".parquet"), index=False)
    print(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
