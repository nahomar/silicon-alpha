"""Phase 1 library (docs/master_plan.md): one row per (index, day, half-hour decision).

Features: the live bot's session features (session_rows, same code as the bot), calendar,
market context, SPX dealer gamma and term structure, backward-looking flow, strategy-rule
signals; cross-index and factory signals are added when the indexes are combined.

Labels: return per $1,000 worst case of five candidate trades at that minute (zoo engine),
at half of today's gap and at the touch: short iron fly, short iron condor, put credit
spread, call credit spread, long straddle.

Development days only: exam days are refused by odte/research/exam_guard.py.

  python3 -m odte.research.library build SPX      # or NDX / RUT
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import exam_guard as eg
from odte.research import zoo

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "research"
DIRS = {"SPX": ROOT / "data" / "odte_0dte", "NDX": ROOT / "data" / "odte_0dte_ndxp", "RUT": ROOT / "data" / "odte_0dte_rutw"}
TIMES = tuple(f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30))
TRADES = {"fly": {"structure": "iron_fly", "wing": 1.5},
          "condor": {"structure": "iron_condor", "short_k": 1.0, "wing": 0.5},
          "put_spread": {"structure": "put_credit_spread", "short_k": 0.8, "wing": 0.5},
          "call_spread": {"structure": "call_credit_spread", "short_k": 0.8, "wing": 0.5},
          "long_straddle": {"structure": "long_straddle"}}


def fly_mid(grid: dict, t, K: float, W: float) -> float:
    try:
        b, a = grid["bid"].loc[t], grid["ask"].loc[t]
        m = lambda r, k: (b[(r, k)] + a[(r, k)]) / 2
        return float(m("C", K) + m("P", K) - m("C", K + W) - m("P", K - W))
    except KeyError:
        return float("nan")


def flow_features(grid: dict, day, hhmm: str, K: float, W: float) -> dict:
    """Backward-looking flow at t (known at t): change of the fly's mid over the last 5 and 15 minutes."""
    t = day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
    now = fly_mid(grid, t, K, W)
    out = {}
    for m in (5, 15):
        past = fly_mid(grid, t - pd.Timedelta(minutes=m), K, W)
        out[f"fly_chg{m}"] = (now / past - 1) if np.isfinite(now) and np.isfinite(past) and past > 0 else np.nan
    return out


def today_rel_spread(sym: str, files: list, settle: pd.Series) -> float:
    vals = []
    for p in files:
        d = pd.Timestamp(p.stem)
        if zoo.COST_WINDOW[0] <= d <= zoo.COST_WINDOW[1] and d in settle.index:
            x = zoo.Day(d, pd.read_parquet(p), float(settle[d]), np.nan, {}).straddle_rel_spread(d + pd.Timedelta(hours=10))
            if np.isfinite(x):
                vals.append(x)
    return float(np.median(vals)) if vals else 0.01


def build_index(sym: str, limit_days: list | None = None) -> pd.DataFrame:
    from odte.data import calendar as cal
    from odte.data import context as ctx
    from odte.eval.expiry_edge import implied_forward, load_early_closes, minute_grid, session_rows
    folder = DIRS[sym]
    settle = pd.read_csv(folder / "settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
    files = sorted(p for p in folder.glob("20??-??-??.parquet"))
    files = [p for p in files if pd.Timestamp(p.stem) < eg.EXAM_FROM]           # development only
    if limit_days is not None:
        files = [p for p in files if pd.Timestamp(p.stem) in set(pd.to_datetime(limit_days))]
    eg.check([p.stem for p in files])
    events, wide, early = cal.load(), ctx.load(), load_early_closes()
    timed = events[events["session_time"].notna()]
    marks_by_day = {pd.Timestamp(d): sorted(set(g["session_time"])) for d, g in timed.groupby("session")}
    R_today = today_rel_spread(sym, files, settle)
    rows, prev_ratio = [], np.nan
    for p in files:
        day = pd.Timestamp(p.stem)
        if day not in settle.index:
            continue
        prev = settle[settle.index < day]
        prev_settle = float(prev.iloc[-1]) if len(prev) else np.nan
        chain = pd.read_parquet(p)
        grid = minute_grid(chain)
        F = implied_forward(grid)
        close = early.get(day, "16:00")
        marks = [day + pd.Timedelta(hours=int(h[:2]), minutes=int(h[3:])) for h in marks_by_day.get(day, [])]
        srows = session_rows(grid, F, float(settle[day]), prev_settle, prev_ratio,
                             times=tuple(t for t in TIMES if t < close), event_marks=marks, close_hhmm=close)
        if not srows:
            continue
        zd = zoo.Day(day, chain, float(settle[day]), prev_settle, {})
        r10 = zd.straddle_rel_spread(day + pd.Timedelta(hours=10))
        k_day = float(np.clip(R_today / r10, 0.1, 2.0)) if np.isfinite(r10) and r10 > 0 else 1.0
        oi = None
        if sym == "SPX":
            from odte.data import gamma
            oi = gamma._load(day)
        ctx_row = ctx.features_for(wide, day)
        for r in srows:
            r["index"], r["day"] = sym, day
            r.update(cal.features_for(events, day, r["hhmm"]))
            r.update(ctx_row)
            ann = (r["implied_move"] / 0.8) / math.sqrt(max(r["minutes_to_close"], 1.0) / (390 * 252)) * 100.0
            v = r.get("vix1d_prev")
            r["iv_vs_vix1d"] = ann / v if v and np.isfinite(v) and v != 0 else np.nan
            if sym == "SPX":
                from odte.data import gamma
                r.update(gamma.features_for(oi, day, r["F"], r["implied_move"], r["minutes_to_close"]))
            if np.isfinite(r.get("W", np.nan)):
                r.update(flow_features(grid, day, r["hhmm"], float(r["K"]), float(r["W"])))
            # strategy-rule signals (every rule tested so far, as an input)
            r["sig_event_day"] = float(any(r.get(k, 0) for k in ("ev_cpi", "ev_nfp", "ev_fomc", "ev_fomc_sep")))
            r["sig_morning_dir"] = float(np.sign(r.get("move_so_far", np.nan))) if np.isfinite(r.get("move_so_far", np.nan)) else np.nan
            r["sig_calm5"] = float(r.get("fly_chg5", np.nan) <= 0) if np.isfinite(r.get("fly_chg5", np.nan)) else np.nan
            for name, rule in TRADES.items():
                for share, tag in ((0.5, "half"), (1.0, "touch")):
                    tr = zoo.trade(rule, zd, r["hhmm"], k_day, share)
                    r[f"y_{name}_{tag}"] = tr["ret"] if tr else np.nan
            rows.append(r)
        r10row = [r for r in srows if r["hhmm"] == "10:00"]
        prev_ratio = float(r10row[0]["ratio"]) if r10row and np.isfinite(r10row[0].get("ratio", np.nan)) else np.nan
    d = pd.DataFrame(rows)
    d["R_today"] = R_today
    return d


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "build":
        sym = sys.argv[2]
        d = build_index(sym)
        OUT.mkdir(parents=True, exist_ok=True)
        d.to_parquet(OUT / f"library_{sym}.parquet", index=False)
        print(f"{sym}: {len(d):,} rows, {d['day'].nunique():,} days, {d.shape[1]} columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
