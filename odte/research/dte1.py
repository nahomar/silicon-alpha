"""1DTE selling test (docs/dte1_plan.md)."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TIMES = tuple(f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30))
FLY = {"structure": "iron_fly", "wing": 1.5}


def one(day: str):
    from odte.research import zoo
    st = pd.read_csv(ROOT / "data/odte_0dte/settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
    cal = pd.DatetimeIndex(pd.read_parquet(ROOT / "data/calendar/sessions.parquet")["session"])
    d = pd.Timestamp(day)
    nxt = cal[cal > d]
    p0, p1 = ROOT / f"data/odte_0dte/{day}.parquet", ROOT / f"data/odte_next/{day}.parquet"
    if not len(nxt) or not p0.exists() or not p1.exists() or d not in st.index:
        return []
    ch1 = pd.read_parquet(p1)
    exp = pd.Timestamp(ch1["expiration"].iloc[0]).normalize()
    if exp != nxt[0] or exp not in st.index:
        return []
    z0 = zoo.Day(d, pd.read_parquet(p0), float(st[d]), np.nan, {})
    z1 = zoo.Day(d, ch1, float(st[exp]), np.nan, {})
    out = []
    for hh in TIMES:
        r = {"day": d, "hhmm": hh, "gap_days": int((exp - d).days)}
        for tag, z in (("d0", z0), ("d1", z1)):
            for share, s in ((0.5, "half"), (1.0, "touch")):
                tr = zoo.trade(FLY, z, hh, 1.0, share)
                r[f"{tag}_{s}"] = float(tr["ret"]) if tr else np.nan
        out.append(r)
    return out


def main() -> int:
    days = [p.stem for p in sorted((ROOT / "data/odte_next").glob("20??-??-??.parquet")) if p.stem <= "2026-11-06"]
    with ProcessPoolExecutor(10) as ex:
        rows = [r for part in ex.map(one, days, chunksize=8) for r in part]
    d = pd.DataFrame(rows)
    d.to_parquet(ROOT / "reports/round2/dte1.parquet", index=False)
    era = d[d["day"] >= "2022-05-16"]
    rng = np.random.default_rng(7)

    def ci(x):
        x = x.dropna().to_numpy()
        b = x[rng.integers(0, len(x), (5000, len(x)))].mean(axis=1)
        return float(x.mean()), [float(np.percentile(b, 5)), float(np.percentile(b, 95))], len(x)
    print("daily-expiry era 2022-05-16 .. 2026-11-06, mean $ per $1k risk per trade (half-gap), 90% CI:")
    res = {}
    for hh in TIMES:
        g = era[era["hhmm"] == hh]
        m1, c1, n1 = ci(g["d1_half"]); m0, c0, _ = ci(g["d0_half"])
        both = g.dropna(subset=["d0_half", "d1_half"]); md, cd, _ = ci(both["d1_half"] - both["d0_half"])
        res[hh] = {"1dte": m1, "1dte_ci": c1, "0dte": m0, "diff": md, "diff_ci": cd, "n": n1,
                   "1dte_touch": float(g["d1_touch"].mean()), "1dte_win": float((g["d1_half"] > 0).mean())}
        print(f"  {hh}: 1DTE {m1:+6.1f} [{c1[0]:+.1f}, {c1[1]:+.1f}]  0DTE {m0:+6.1f}  1DTE-0DTE {md:+6.1f} [{cd[0]:+.1f}, {cd[1]:+.1f}]"
              f"  touch {res[hh]['1dte_touch']:+.1f}  win {res[hh]['1dte_win']:.0%}  (n {n1})")
    p = era[era["hhmm"] == "15:30"]
    print("\n15:30 1DTE by year:", p.groupby(p["day"].dt.year)["d1_half"].mean().round(1).to_dict())
    print("15:30 1DTE worst 5 trades:", p.nsmallest(5, "d1_half")[["day", "d1_half"]].assign(day=lambda x: x["day"].dt.date).to_dict("records"))
    print("15:30 1DTE: Friday entries (over the weekend)", round(float(p[p["gap_days"] > 1]["d1_half"].mean()), 1),
          "| other days", round(float(p[p["gap_days"] == 1]["d1_half"].mean()), 1))
    r = res["15:30"]
    verdict = "PASS" if (r["1dte_ci"][0] > 0 and r["diff"] > 0) else "FAIL"
    print("\nprimary (15:30):", verdict)
    (ROOT / "reports/round2/dte1.json").write_text(json.dumps({"by_time": res, "verdict": verdict}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
