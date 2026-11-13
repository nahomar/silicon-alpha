"""Edge #3b: 0DTE / next-expiry calendar at the same strike.

Pre-registered 2026-11-10 (docs/expiry_edge_plan.md). Seen SPX data only
(2016-01 .. 2026-09-04); a pass earns a shadow slot, not money.

  python3 -m odte.research.calendar
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import factory as fx

ROOT = Path(__file__).resolve().parents[2]
CHAINS0 = ROOT / "data" / "odte_0dte"
CHAINS1 = ROOT / "data" / "odte_next"
FEES6, BUDGET = 6 * 1.20, 1000.0
COST_WINDOW = (pd.Timestamp("2025-09-01"), pd.Timestamp("2026-09-04"))
PCTL, LOOKBACK, MIN_HIST = 0.70, 250, 120
Q = (0.0125, 0.9875)                      # 97.5% two-sided (Bonferroni over 2 strategies)
N_BOOT, SEED = 10_000, 0


def straddle_quote(grid: dict, t: pd.Timestamp, K: float):
    """(bid, ask) of the K straddle at minute t, or None unless both legs are two-sided."""
    from odte.eval.expiry_edge import _leg
    cb, ca = _leg(grid, "bid", t, "C", K), _leg(grid, "ask", t, "C", K)
    pb, pa = _leg(grid, "bid", t, "P", K), _leg(grid, "ask", t, "P", K)
    v = np.array([cb, ca, pb, pa], float)
    if not np.isfinite(v).all() or cb <= 0 or pb <= 0 or ca < cb or pa < pb:
        return None
    return cb + pb, ca + pa


def quotes(d: pd.DataFrame) -> pd.DataFrame:
    """Per decision: 0DTE straddle at t, next-expiry straddle at t and at close - 1 min."""
    from odte.eval.expiry_edge import minute_grid
    out = []
    for day, g in d.groupby("day"):
        p0, p1 = CHAINS0 / f"{day:%Y-%m-%d}.parquet", CHAINS1 / f"{day:%Y-%m-%d}.parquet"
        if not (p0.exists() and p1.exists()):
            continue
        g0, g1 = minute_grid(pd.read_parquet(p0)), minute_grid(pd.read_parquet(p1))
        for i, r in g.iterrows():
            t = day + pd.Timedelta(hours=int(r["hhmm"][:2]), minutes=int(r["hhmm"][3:]))
            t_exit = (t + pd.Timedelta(minutes=float(r["minutes_to_close"]))
                      - pd.Timedelta(minutes=1)).floor("min")
            K = float(r["K"])
            q = {"i": i, "t_exit": t_exit}
            for name, grid, when in (("s0", g0, t), ("s1", g1, t), ("sx", g1, t_exit)):
                bq = straddle_quote(grid, when, K)
                q[f"{name}_bid"], q[f"{name}_ask"] = bq if bq else (np.nan, np.nan)
            out.append(q)
    return pd.DataFrame(out).set_index("i")


def gap_fracs(q: pd.DataFrame, day: pd.Series) -> dict:
    """Today's typical gap / mid per leg, median over the cost window."""
    w = q[day.reindex(q.index).between(*COST_WINDOW)]
    mid = lambda n: (w[f"{n}_bid"] + w[f"{n}_ask"]) / 2
    return {"s0_sell": float(((mid("s0") - w["s0_bid"]) / mid("s0")).median()),
            "s1_buy": float(((w["s1_ask"] - mid("s1")) / mid("s1")).median()),
            "sx_sell": float(((mid("sx") - w["sx_bid"]) / mid("sx")).median())}


def calendar_pnl(s0_mid, s1_mid, sx_mid, settle, K, f: dict, share: float = 0.5):
    """(pnl, risk) per one calendar in dollars; share = part of today's gap paid (0.5 half, 1 touch)."""
    credit0 = np.asarray(s0_mid) * (1 - share * f["s0_sell"])
    debit1 = np.asarray(s1_mid) * (1 + share * f["s1_buy"])
    exit1 = np.asarray(sx_mid) * (1 - share * f["sx_sell"])
    intrinsic0 = np.abs(np.asarray(settle) - np.asarray(K))
    pnl = (credit0 - intrinsic0 + exit1 - debit1) * 100 - FEES6
    risk = (debit1 - credit0) * 100 + FEES6
    return pnl, risk


def signal_threshold(ts_spread: pd.Series, day: pd.Series, hhmm: pd.Series) -> pd.Series:
    """70th percentile of ts_spread at the same time over the previous 250 sessions (not today)."""
    out = pd.Series(np.nan, index=ts_spread.index)
    for h, idx in hhmm.groupby(hhmm).groups.items():
        s = ts_spread.loc[idx].sort_index(key=lambda ix: day.loc[ix])
        out.loc[s.index] = s.shift(1).rolling(LOOKBACK, min_periods=MIN_HIST).quantile(PCTL).to_numpy()
    return out


def mean_ci(x: np.ndarray, n: int = N_BOOT, seed: int = SEED) -> dict:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci": [float(np.quantile(bs, Q[0])), float(np.quantile(bs, Q[1]))],
            "days": int(len(x))}


def main() -> int:
    from odte.eval.expiry_edge import paired_diff
    from odte.research import term_structure as ts
    d = fx.load()
    d = d.join(ts.build(d)[["ts_spread"]], how="left")
    post = (d["min_since_event"] <= 60).fillna(False)
    q = quotes(d)
    d = d.join(q, how="left")
    n0 = len(d)
    ok = d[[f"{n}_{s}" for n in ("s0", "s1", "sx") for s in ("bid", "ask")]].notna().all(axis=1)
    f = gap_fracs(d[ok], d["day"])
    mid = lambda n: (d[f"{n}_bid"] + d[f"{n}_ask"]) / 2
    res = {"gap_fracs_today": f, "decisions": n0, "skipped_no_quotes": int((~ok).sum())}
    for name, share in (("half", 0.5), ("touch", 1.0)):
        pnl, risk = calendar_pnl(mid("s0"), mid("s1"), mid("sx"), d["settle"], d["K"], f, share)
        d[f"pnl_{name}"], d[f"risk_{name}"] = pnl, risk
        d[f"ror_{name}"] = np.where(risk > 0, pnl / np.where(risk > 0, risk, 1), np.nan)
    credit = ok & (d["risk_half"] <= 0)
    res["skipped_net_credit"] = int(credit.sum())
    elig = ok & ~credit & ~post
    res["skipped_post_event"] = int((ok & ~credit & post).sum())
    thr = signal_threshold(d["ts_spread"], d["day"], d["hhmm"])
    sig = elig & (d["ts_spread"] > thr) & thr.notna()
    days = pd.Index(sorted(d.loc[elig, "day"].unique()))
    out = {}
    for px in ("half", "touch"):
        a = (d.loc[elig, f"ror_{px}"] * BUDGET).groupby(d.loc[elig, "day"]).sum().reindex(days, fill_value=0.0)
        b = (d.loc[sig, f"ror_{px}"] * BUDGET).groupby(d.loc[sig, "day"]).sum().reindex(days, fill_value=0.0)
        out[px] = {"O1a_always": {**mean_ci(a.to_numpy()), **fx.summarize(a)},
                   "O1b_signal": {**mean_ci(b.to_numpy()), **fx.summarize(b), "trades": int(sig.sum())},
                   "O1b_minus_O1a": paired_diff(b, a)}
        if px == "half":
            daily = {"O1a": a, "O1b": b}
    res["results"] = out
    for k, key in (("O1a", "O1a_always"), ("O1b", "O1b_signal")):
        r = out["half"][key]
        res[f"{k}_passes"] = bool(r["ci"][0] > 0 and r.get("2016-21", -1) > 0 and r.get("2022-26", -1) > 0)
    res["by_time_half"] = {h: {"mean_ror": float(g["ror_half"].mean()), "n": int(len(g)),
                               "win_rate": float((g["pnl_half"] > 0).mean())}
                           for h, g in d[elig].groupby("hhmm")}
    res["avg_debit_dollars"] = float(d.loc[elig, "risk_half"].mean())
    res["eligible_decisions"] = int(elig.sum())
    out_dir = ROOT / "reports" / "factory"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calendar.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
