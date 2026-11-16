"""Round 3 (docs/round2_plan.md): B0 with a 14-shape menu, vs round 2's challenger C7.

  python3 -m odte.research.round3
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from odte.research import meta, menu3, round2

MENU14 = list(round2.MENU) + list(menu3.NEW)
PIN = ("pin_put_fly", "pin_call_fly")


def load() -> pd.DataFrame:
    d = meta.load()
    lab = pd.read_parquet(menu3.OUT)
    return d.merge(lab, on=["day", "index", "hhmm"], how="left", validate="one_to_one")


def walk_menu(d, X, menu, alpha=1000.0) -> dict:
    days = np.array(sorted(d["day"].unique()))
    dd = d["day"].to_numpy()
    ok = ~d["post"].to_numpy()
    P = {k: np.full(len(d), np.nan) for k in menu}
    for k0 in range(meta.MIN_DAYS, len(days), meta.REFIT):
        tr_days = dd < days[k0]
        te = (dd >= days[k0]) & ((dd < days[k0 + meta.REFIT]) if k0 + meta.REFIT < len(days) else True)
        for k in menu:
            y = d[f"y_{k}_half"].to_numpy(float)
            tr = tr_days & ok & np.isfinite(y)
            if tr.sum() < 1000:
                continue
            P[k][te] = round2.ridge(alpha).fit(X[tr], y[tr]).predict(X[te])
    return P


def decide_menu(d, P, menu, tag, drop_late=False) -> pd.DataFrame:
    M = np.column_stack([P[k] for k in menu])
    Y = np.column_stack([d[f"y_{k}_{tag}"].to_numpy(float) for k in menu])
    has = np.isfinite(M).any(axis=1)
    Mf = np.where(np.isfinite(M), M, -np.inf)
    best = Mf.argmax(axis=1)
    take = has & (Mf.max(axis=1) > 0) & ~d["post"].to_numpy()
    if drop_late:
        take &= ~d["hhmm"].isin(round2.LATE).to_numpy()
    ret = np.where(take, Y[np.arange(len(d)), best], 0.0)
    ret = np.where(np.isfinite(ret), ret, 0.0)
    return pd.DataFrame({"day": d["day"], "index": d["index"], "hhmm": d["hhmm"], "tested": has, "take": take,
                         "trade": np.where(take, np.array(menu)[best], "none"), "ret": ret})


def evaluate(d, P, menu, drop_late) -> tuple:
    all_days = pd.DatetimeIndex(sorted(d["day"].unique()))
    wd = all_days[(all_days >= round2.WIN[0]) & (all_days <= round2.WIN[1])]
    ser = {}
    for tag in ("half", "touch"):
        dec = decide_menu(d, P, menu, tag, drop_late)
        ser[tag] = dec[dec["tested"]].groupby("day")["ret"].sum().reindex(wd, fill_value=0.0)
        if tag == "half":
            spx = dec[dec["tested"] & (dec["index"] == "SPX")].groupby("day")["ret"].sum().reindex(wd, fill_value=0.0)
            t = dec[dec["take"] & dec["day"].between(*round2.WIN)]
            shapes = t.groupby("trade")["ret"].agg(["count", "sum", "mean"])
            shapes["share_picks"] = shapes["count"] / shapes["count"].sum()
            shapes["share_profit"] = shapes["sum"] / t["ret"].sum()
            take = float(dec.loc[dec["tested"] & dec["day"].between(*round2.WIN), "take"].mean())
    h = ser["half"]
    top = h.sort_values().tail(10).index
    r = {"primary_capped_sharpe": round2.capped(h), "sharpe": round2.sharpe(h), "mean_day": float(h.mean()),
         "touch_mean_day": float(ser["touch"].mean()), "no_top10_sharpe": round2.sharpe(h.drop(top)),
         "spx_sharpe": round2.sharpe(spx), "half1_capped": round2.capped(h[h.index <= round2.HALVES[0][1]]),
         "half2_capped": round2.capped(h[h.index >= round2.HALVES[1][0]]),
         "exam_window_mean": float(h[h.index >= pd.Timestamp("2026-09-08")].mean()), "take_rate": take}
    return r, h, shapes


def main() -> int:
    d = load()
    X = round2.features(d, round2.B0_GROUPS)
    P = walk_menu(d, X, MENU14)
    prev = json.loads((round2.OUT / "candidates.json").read_text())
    c7 = prev["C7_no_late"]
    out = {}
    for name, late in (("R3", False), ("R3L", True)):
        r, h, shapes = evaluate(d, P, MENU14, late)
        r["vs_C7"] = r["primary_capped_sharpe"] - c7["primary_capped_sharpe"]
        r["beats_C7_both_halves"] = bool(r["half1_capped"] > c7["half1_capped"] and r["half2_capped"] > c7["half2_capped"])
        out[name] = r
        round2.show(name, r)
        print(f"   vs C7 {r['vs_C7']:+.2f}, both halves: {r['beats_C7_both_halves']}")
        print("   shapes (2023-07..2026-11):")
        print(shapes.sort_values("count", ascending=False).round(3).to_string().replace("\n", "\n      "))
        out[name]["shapes"] = shapes.round(4).to_dict(orient="index")
    # the pin butterflies on their own, wherever the model picked them
    dec = decide_menu(d, P, MENU14, "half")
    t = dec[dec["take"] & dec["trade"].isin(PIN) & dec["day"].between(*round2.WIN)]
    if len(t):
        y = t["ret"]
        print(f"\npin butterflies picked {len(t)} times: hit rate {(y > 0).mean():.0%}, average win {y[y > 0].mean() / 1000:.1f}x risk,"
              f" wins > 2x risk: {(y > 2000).sum()}, total ${y.sum():+,.0f}")
    (round2.OUT / "round3.json").write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
