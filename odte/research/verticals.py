"""Edge #1 screen: choose the side (put vs call credit spread) with the T_dir signal.

Pre-registered 2026-11-09 (docs/expiry_edge_plan.md). Seen data only: a pass
earns a shadow slot, not money.

  python3 -m odte.research.verticals
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import factory as fx

ROOT = Path(__file__).resolve().parents[2]
CHAINS = ROOT / "data" / "odte_0dte"
FEES2, BUDGET = 2 * 1.20, 1000.0
DIR_SIGNALS = ["oi_pcr_0dte", "skew_vs_hist", "gex_0dte", "gex_flip_dist", "skew", "gex_net",
               "minutes_to_close"]
COST_WINDOW = (pd.Timestamp("2025-09-01"), pd.Timestamp("2026-09-04"))


def leg_quotes(d: pd.DataFrame) -> pd.DataFrame:
    """Bid/ask of the four legs (P K, P K-W, C K, C K+W) at each decision minute."""
    from odte.eval.expiry_edge import _leg, minute_grid
    out = []
    for day, g in d.groupby("day"):
        p = CHAINS / f"{day:%Y-%m-%d}.parquet"
        if not p.exists():
            continue
        grid = minute_grid(pd.read_parquet(p))
        for i, r in g.iterrows():
            t = day + pd.Timedelta(hours=int(r["hhmm"][:2]), minutes=int(r["hhmm"][3:]))
            K, W = float(r["K"]), float(r["W"])
            q = {"i": i}
            for name, right, k in (("pk", "P", K), ("pw", "P", K - W), ("ck", "C", K), ("cw", "C", K + W)):
                q[f"{name}_b"] = _leg(grid, "bid", t, right, k)
                q[f"{name}_a"] = _leg(grid, "ask", t, right, k)
            out.append(q)
    return pd.DataFrame(out).set_index("i")


def spreads(d: pd.DataFrame) -> pd.DataFrame:
    q = leg_quotes(d)
    d = d.join(q, how="inner")
    mid = lambda b, a: (d[b] + d[a]) / 2
    d["put_mid"] = mid("pk_b", "pk_a") - mid("pw_b", "pw_a")
    d["put_touch"] = d["pk_b"] - d["pw_a"]
    d["call_mid"] = mid("ck_b", "ck_a") - mid("cw_b", "cw_a")
    d["call_touch"] = d["ck_b"] - d["cw_a"]
    S, K, W = d["settle"], d["K"], d["W"]
    d["put_loss"] = np.minimum(np.maximum(K - S, 0), W)
    d["call_loss"] = np.minimum(np.maximum(S - K, 0), W)
    ok = (d["put_mid"] > 0) & (d["call_mid"] > 0) & np.isfinite(d["put_touch"]) & np.isfinite(d["call_touch"])
    return d[ok].copy()


def today_cost(d: pd.DataFrame) -> dict:
    w = d[d["day"].between(*COST_WINDOW)]
    return {s: float(((w[f"{s}_mid"] - w[f"{s}_touch"]) / w[f"{s}_mid"]).clip(lower=0).median())
            for s in ("put", "call")}


def ror(d: pd.DataFrame, side: str, cost_frac: float) -> np.ndarray:
    c = d[f"{side}_mid"] * (1 - 0.5 * cost_frac)              # half of today's gap
    risk = (d["W"] - c) * 100 + FEES2
    pnl = (c - d[f"{side}_loss"]) * 100 - FEES2
    return np.where(risk > 0, pnl / risk, np.nan)


def main() -> int:
    from odte.eval.expiry_edge import paired_diff
    d = fx.load()
    S = fx.signals(d)
    d = pd.concat([d, S[[c for c in DIR_SIGNALS if c not in d.columns]]], axis=1)
    d = spreads(d)
    cost = today_cost(d)
    print(f"{d['day'].nunique()} days, {len(d):,} decisions; today's vertical gap / mid: "
          f"put {cost['put']:.1%}, call {cost['call']:.1%}", flush=True)
    d["r_put"], d["r_call"] = ror(d, "put", cost["put"]), ror(d, "call", cost["call"])
    pred = fx.walk_forward_ridge(d.reset_index(drop=True), d.reset_index(drop=True)[DIR_SIGNALS],
                                 d.reset_index(drop=True)["T_dir"])
    d = d.reset_index(drop=True)
    d["pred_z"] = pred
    post = (d["min_since_event"] <= 60).fillna(False).to_numpy() if "min_since_event" in d else np.zeros(len(d), bool)
    t = np.isfinite(pred) & ~post
    rng = np.random.default_rng(0)
    coin = rng.integers(0, 2, len(d)).astype(bool)
    strategies = {
        "side_selector": np.where(pred > 0, d["r_put"], d["r_call"]),
        "always_put": d["r_put"].to_numpy(),
        "always_call": d["r_call"].to_numpy(),
        "random_side": np.where(coin, d["r_put"], d["r_call"]),
        "always_fly": d["T_fly"].to_numpy(),
    }
    daily = {k: pd.Series(np.nan_to_num(v[t]) * BUDGET).groupby(d["day"].to_numpy()[t]).sum()
             for k, v in strategies.items()}
    res = {"cost": cost, "days": int(d.loc[t, "day"].nunique()),
           "hit_rate": float((np.sign(pred[t]) == np.sign(d["T_dir"].to_numpy()[t])).mean()),
           "strategies": {k: fx.summarize(x) for k, x in daily.items()}}
    pdiff = paired_diff(daily["side_selector"], daily["always_put"])
    res["selector_minus_always_put"] = pdiff
    sel = res["strategies"]["side_selector"]
    res["passes"] = bool(pdiff["ci"][0] > 0 and sel.get("2016-21", -1) > 0 and sel.get("2022-26", -1) > 0)
    out = ROOT / "reports" / "factory" / "verticals.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"side hit rate {res['hit_rate']:.3f} over {res['days']} days\n")
    for k, v in res["strategies"].items():
        print(f"  {k:14s} total ${v['total']:>10,.0f}  Sharpe {v['sharpe']:+.2f} "
              f"[{v['sharpe_ci'][0]:+.2f}, {v['sharpe_ci'][1]:+.2f}]  2016-21 ${v['2016-21']:>9,.0f}  "
              f"2022-26 ${v['2022-26']:>9,.0f}")
    print(f"\nside selector minus always-put: {pdiff['mean']:+.1f}/day CI [{pdiff['ci'][0]:+.1f}, {pdiff['ci'][1]:+.1f}]")
    print(f"SCREEN (pre-registered): {'PASS' if res['passes'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
