"""Edge #9: overnight option premium (pre-registered 2026-11-10, docs/expiry_edge_plan.md).

Sell the next-day iron fly at day d's close, buy it back at 09:45 on d+1.

  python3 -m odte.research.overnight
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NEXT = ROOT / "data" / "odte_next"
ZERO = ROOT / "data" / "odte_0dte"
EXAM_FROM = pd.Timestamp("2026-09-08")
ERA_SPLIT = pd.Timestamp("2022-01-01")
COST_WINDOW = (pd.Timestamp("2025-09-01"), pd.Timestamp("2026-09-04"))
EXIT_HHMM, WING_MULT = "09:45", 1.5
FEES8, BUDGET = 8 * 1.20, 1000.0
N_BOOT, SEED = 10_000, 0


def _q(grid, side, t, right, k):
    from odte.eval.expiry_edge import _leg
    return _leg(grid, side, t, right, k)


def entry_fly(grid: dict, t: pd.Timestamp, F: float) -> dict | None:
    """ATM body and 1.5x-straddle wings from the two-sided quotes at t (the bot's rule)."""
    from odte.eval.expiry_edge import _nearest, _two_sided
    q = _two_sided(grid, t)
    (cb, ca), (pb, pa) = q["C"], q["P"]
    both = cb.index.intersection(pb.index)
    K = _nearest(both, F) if np.isfinite(F) else None
    if K is None:
        return None
    st_bid, st_ask = cb[K] + pb[K], ca[K] + pa[K]
    st_mid = (st_bid + st_ask) / 2
    if not st_mid > 0:
        return None
    c_d = set(np.round(cb.index.to_numpy(float) - K, 4))
    p_d = set(np.round(K - pb.index.to_numpy(float), 4))
    dists = sorted(d for d in (c_d & p_d) if d >= WING_MULT * st_mid)
    if not dists:
        return None
    W = dists[0]
    wings_mid = (cb[K + W] + ca[K + W] + pb[K - W] + pa[K - W]) / 2
    return {"K": float(K), "W": float(W), "st_mid": float(st_mid), "st_bid": float(st_bid),
            "credit_mid": float(st_mid - wings_mid), "credit_touch": float(st_bid - (ca[K + W] + pa[K - W]))}


def exit_value(grid: dict, t: pd.Timestamp, K: float, W: float) -> dict | None:
    """Cost to buy the fly back at t: mid and touch (body at the ask, wings at the bid)."""
    legs = {(r, k): (_q(grid, "bid", t, r, k), _q(grid, "ask", t, r, k))
            for r, k in (("C", K), ("P", K), ("C", K + W), ("P", K - W))}
    v = np.array([x for pair in legs.values() for x in pair], float)
    if not np.isfinite(v).all() or (v < 0).any():
        return None
    mid = lambda r, k: sum(legs[(r, k)]) / 2
    body_mid, wings_mid = mid("C", K) + mid("P", K), mid("C", K + W) + mid("P", K - W)
    st_ask = legs[("C", K)][1] + legs[("P", K)][1]
    st_mid = body_mid
    return {"value_mid": float(body_mid - wings_mid),
            "value_touch": float(st_ask - (legs[("C", K + W)][0] + legs[("P", K - W)][0])),
            "st_mid": float(st_mid), "st_ask": float(st_ask)}


def nightly_pnl(credit_mid, value_mid, W, g_in, g_out, share=0.5):
    """(pnl, risk) in dollars for one short fly held overnight."""
    credit = np.asarray(credit_mid) * (1 - share * g_in)
    buyback = np.asarray(value_mid) * (1 + share * g_out)
    pnl = (credit - buyback) * 100 - FEES8
    risk = (np.asarray(W) - credit) * 100 + FEES8
    return pnl, risk


def nights() -> pd.DataFrame:
    from odte.eval.expiry_edge import implied_forward, load_early_closes, minute_grid
    sessions = pd.DatetimeIndex(pd.read_parquet(ROOT / "data" / "calendar" / "sessions.parquet")["session"])
    early = load_early_closes()
    rows, skipped = [], {"not_next_session": 0, "no_file": 0, "no_entry": 0, "no_exit": 0}
    for p in sorted(NEXT.glob("20??-??-??.parquet")):
        d = pd.Timestamp(p.stem)
        later = sessions[sessions > d]
        if not len(later):
            continue
        d1 = later[0]
        if d1 >= EXAM_FROM:
            continue
        ch = pd.read_parquet(p)
        if ch.empty or pd.Timestamp(ch["expiration"].iloc[0]).normalize() != d1:
            skipped["not_next_session"] += 1
            continue
        z = ZERO / f"{d1:%Y-%m-%d}.parquet"
        if not z.exists():
            skipped["no_file"] += 1
            continue
        hh = early.get(d, "16:00")
        t0 = d + pd.Timedelta(hours=int(hh[:2]), minutes=int(hh[3:])) - pd.Timedelta(minutes=1)
        g0 = minute_grid(ch)
        F = implied_forward(g0)
        F0 = F[F.index <= t0].dropna()
        e = entry_fly(g0, t0, float(F0.iloc[-1]) if len(F0) else np.nan)
        if e is None or not e["credit_mid"] > 0:
            skipped["no_entry"] += 1
            continue
        t1 = d1 + pd.Timedelta(hours=int(EXIT_HHMM[:2]), minutes=int(EXIT_HHMM[3:]))
        x = exit_value(minute_grid(pd.read_parquet(z)), t1, e["K"], e["W"])
        if x is None:
            skipped["no_exit"] += 1
            continue
        rows.append({"day": d, "next": d1, **e, **{f"x_{k}": v for k, v in x.items()}})
    out = pd.DataFrame(rows)
    out.attrs["skipped"] = skipped
    return out


def mean_ci(x, n=N_BOOT, seed=SEED) -> dict:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "nights": int(len(x)), "total": float(x.sum()), "win_rate": float((x > 0).mean())}


def main() -> int:
    from odte.data import calendar as cal
    n = nights()
    w = n[n["day"].between(*COST_WINDOW)]
    g_in = float(((w["credit_mid"] - w["credit_touch"]) / w["credit_mid"]).clip(lower=0).median())
    g_out = float(((w["x_value_touch"] - w["x_value_mid"]) / w["x_value_mid"].where(w["x_value_mid"] > 0))
                  .clip(lower=0).median())
    res = {"nights": int(len(n)), "skipped": n.attrs["skipped"], "gap_in": g_in, "gap_out": g_out}
    out = {}
    for name, share in (("half", 0.5), ("touch", 1.0)):
        pnl, risk = nightly_pnl(n["credit_mid"], n["x_value_mid"], n["W"], g_in, g_out, share)
        n[f"ret_{name}"] = np.where(risk > 0, pnl / np.where(risk > 0, risk, 1) * BUDGET, np.nan)
        x = n[f"ret_{name}"].dropna()
        e1 = n.loc[x.index, "day"] < ERA_SPLIT
        out[name] = {**mean_ci(x), "2016-21": float(x[e1].sum()), "2022-26": float(x[~e1].sum())}
    res["fly"] = out
    st = (n["st_mid"] - n["x_st_mid"]) / n["st_mid"]                    # straddle, at mid, % of premium
    res["straddle_mid_return_pct_of_premium"] = mean_ci(st.to_numpy() * 100)
    ev = cal.load()
    timed = ev[ev["session_time"].notna() & (ev["session_time"] < EXIT_HHMM)]
    ev_days = set(pd.to_datetime(timed["session"]).dt.normalize())
    n["event_morning"] = n["next"].isin(ev_days)
    n["weekend"] = (n["next"] - n["day"]).dt.days > 1
    for col in ("weekend", "event_morning"):
        res[f"by_{col}"] = {str(k): mean_ci(g["ret_half"].dropna()) for k, g in n.groupby(col)}
    r = out["half"]
    res["passes"] = bool(r["ci"][0] > 0 and r["2016-21"] > 0 and r["2022-26"] > 0)
    p = ROOT / "reports" / "factory" / "overnight.json"
    p.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
