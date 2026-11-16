"""Edge #15: quote-flow signal (pre-registered 2026-11-10, docs/expiry_edge_plan.md).

Burst (fly got expensive within 5 minutes) -> buy the fly at t+5; calm -> sell it at t+5.

  python3 -m odte.research.flow
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
TODAY_COST, FEES4, BUDGET = 0.0169, 4 * 1.20, 1000.0
Q = (0.0125, 0.9875)
N_BOOT, SEED = 10_000, 0


def quotes_at(grid: dict, t: pd.Timestamp, K: float, W: float) -> dict | None:
    """Fly mid, buy price (body ask - wings bid) and sell price (body bid - wings ask) at minute t."""
    from odte.eval.expiry_edge import _leg
    legs = (("C", K), ("P", K), ("C", K + W), ("P", K - W))
    b = [_leg(grid, "bid", t, r, k) for r, k in legs]
    a = [_leg(grid, "ask", t, r, k) for r, k in legs]
    if not np.isfinite(b + a).all():
        return None
    mid = (b[0] + a[0] + b[1] + a[1]) / 2 - (b[2] + a[2] + b[3] + a[3]) / 2
    return {"mid": mid, "buy_touch": a[0] + a[1] - b[2] - b[3], "sell_touch": b[0] + b[1] - a[2] - a[3]}


def trade_pnl(side: str, mid: float, touch: float, k: float, W: float, payout: float, share: float = 0.5):
    """(pnl, risk) $ for one fly at t+5; side 'buy' pays above mid, 'sell' receives below mid."""
    gap = abs(touch - mid) * k
    if side == "buy":
        px = mid + share * gap
        return (payout - px) * 100 - FEES4, px * 100 + FEES4
    px = mid - share * gap
    return (px - payout) * 100 - FEES4, (W - px) * 100 + FEES4


def ci(x, n=N_BOOT, seed=SEED) -> dict:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci": [float(np.quantile(bs, Q[0])), float(np.quantile(bs, Q[1]))],
            "days": int(len(x)), "total": float(x.sum())}


def main() -> int:
    from odte.eval.expiry_edge import minute_grid
    d = fx.load()
    d = d[~(d["min_since_event"] <= 60).fillna(False)]
    d = d[(d["fly_credit_mid"] > 0) & np.isfinite(d["W"])]
    rows, skipped = [], 0
    for day, g in d.groupby("day"):
        p = CHAINS / f"{day:%Y-%m-%d}.parquet"
        if not p.exists():
            skipped += len(g)
            continue
        grid = minute_grid(pd.read_parquet(p))
        for _, r in g.iterrows():
            t5 = day + pd.Timedelta(hours=int(r["hhmm"][:2]), minutes=int(r["hhmm"][3:]) + 5)
            K, W = float(r["K"]), float(r["W"])
            q = quotes_at(grid, t5, K, W)
            if q is None or not q["mid"] > 0:
                skipped += 1
                continue
            cm, ct = float(r["fly_credit_mid"]), float(r["fly_credit"])
            k = (TODAY_COST * cm / (cm - ct)) if cm - ct > 0 else 1.0
            payout = min(abs(float(r["settle"]) - K), W)
            burst = bool(r["fill_mid_short"]) if pd.notna(r["fill_mid_short"]) else False
            rec = {"day": day, "burst": burst}
            for share, tag in ((0.5, "half"), (1.0, "touch")):
                for side, touch in (("buy", q["buy_touch"]), ("sell", q["sell_touch"])):
                    pnl, risk = trade_pnl(side, q["mid"], touch, k if tag == "half" else 1.0, W, payout, share)
                    rec[f"{side}_{tag}"] = pnl / risk * BUDGET if risk > 0 else np.nan
            rows.append(rec)
    x = pd.DataFrame(rows)
    res = {"rows": int(len(x)), "skipped": skipped, "burst_rate": float(x["burst"].mean())}
    era = lambda s: (float(s[s.index < fx.ERA_SPLIT].sum()), float(s[s.index >= fx.ERA_SPLIT].sum()))
    for tag in ("half", "touch"):
        b = x["buy_" + tag].where(x["burst"], 0.0).fillna(0.0)
        s = x["sell_" + tag].where(~x["burst"], 0.0).fillna(0.0)
        strat = {"Q1_follow_burst": b, "Q2_calm_seller": s, "always_buy_t5": x["buy_" + tag].fillna(0.0),
                 "always_sell_t5": x["sell_" + tag].fillna(0.0)}
        out = {}
        daily = {}
        for k_, v in strat.items():
            dd = v.groupby(x["day"]).sum()
            daily[k_] = dd
            e1, e2 = era(dd)
            out[k_] = {**ci(dd), "2016-21": e1, "2022-26": e2}
        out["Q2_minus_always_sell"] = ci((daily["Q2_calm_seller"] - daily["always_sell_t5"]).to_numpy())
        res[tag] = out
    h = res["half"]
    for k_ in ("Q1_follow_burst", "Q2_calm_seller"):
        res[f"{k_}_passes"] = bool(h[k_]["ci"][0] > 0 and h[k_]["2016-21"] > 0 and h[k_]["2022-26"] > 0)
    (ROOT / "reports" / "factory" / "flow.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
