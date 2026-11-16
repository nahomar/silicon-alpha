"""E2: order policies for selling the iron fly, simulated on 1-second NBBO (docs/execution_plan.md).

Fill rule (a LOWER bound on fills): a resting sell of the combo fills at the first second its touch
credit (short legs at the bid, long legs at the ask) reaches the limit. Real complex orders can also
trade inside the NBBO through the complex order book, so true fill rates are at least these.

  python3 -m odte.research.execution          # -> reports/execution/e2.json, e2.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "execution"
ONE_S = {"SPX": ROOT / "data" / "odte_1s" / "spx", "XSP": ROOT / "data" / "odte_1s" / "xsp"}
FEES4 = 4 * 1.20
HORIZON = 300                                   # seconds simulated after the decision
WAITS = (10, 30, 60, 120, 300)
EDGES = (0.25, 0.5)


def snapshot(q: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    s = q[q["quote_datetime"] <= t]
    return s.groupby(["option_type", "strike"]).tail(1)


def leg_series(q: pd.DataFrame, legs: list, t0: pd.Timestamp) -> pd.DataFrame:
    """Per-second bid/ask of each leg over [t0, t0 + HORIZON], last quote carried forward."""
    idx = pd.date_range(t0, t0 + pd.Timedelta(seconds=HORIZON), freq="1s")
    out = {}
    for r, k in legs:
        x = q[(q["option_type"] == r) & (q["strike"] == k)].set_index("quote_datetime")[["bid", "ask"]]
        x = x[~x.index.duplicated(keep="last")].sort_index()
        x = x.reindex(x.index.union(idx)).ffill().reindex(idx)
        out[(r, k, "bid")] = x["bid"].to_numpy()
        out[(r, k, "ask")] = x["ask"].to_numpy()
    return pd.DataFrame(out, index=idx)


def policies(tc: np.ndarray, mid0: float) -> dict:
    """Credit received (or None) and seconds to fill for each policy, given the touch-credit path."""
    touch0 = tc[0]
    gap = mid0 - touch0
    res = {"TOUCH": (touch0, 0)}

    def first(limit, a, b):
        w = np.flatnonzero(tc[a:b + 1] >= limit - 1e-9)
        return (a + int(w[0])) if len(w) else None
    for w in WAITS:
        s = first(mid0, 1, w)
        res[f"MID{w}_skip"] = (mid0, s) if s is not None else (None, None)
        res[f"MID{w}_cross"] = (mid0, s) if s is not None else (tc[w], w)
    lad, s_fill, c_fill = [mid0 - i / 5 * 0.5 * gap for i in range(6)], None, None   # bot: 6 rungs x 30 s
    for i, L in enumerate(lad):
        s = first(L, 1 + 30 * i, 30 * (i + 1))
        if s is not None:
            s_fill, c_fill = s, L
            break
    res["LADDER"] = (c_fill, s_fill)
    for k in EDGES:
        L = mid0 + k * gap
        s = first(L, 1, HORIZON)
        res[f"EDGE{k}"] = (L, s) if s is not None else (None, None)
    return res


def ret_per_1k(credit, W, payoff):
    risk = (W - credit) * 100 + FEES4
    return ((credit - payoff) * 100 - FEES4) / risk * 1000 if risk > 0 else np.nan


def simulate(decisions: pd.DataFrame) -> pd.DataFrame:
    from odte.live import fly
    from odte.live.guards import parity_price
    st = pd.read_csv(ROOT / "data" / "odte_0dte" / "settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
    rows = []
    for day, g in decisions.groupby("day"):
        if day not in st.index:
            continue
        qs = {}
        for sym, folder in ONE_S.items():
            f = folder / f"{day:%Y-%m-%d}.parquet"
            if f.exists():
                q = pd.read_parquet(f, columns=["quote_datetime", "strike", "option_type", "bid", "ask"])
                q["quote_datetime"] = pd.to_datetime(q["quote_datetime"])
                qs[sym] = q[(q["bid"] > 0) & (q["ask"] >= q["bid"])].sort_values("quote_datetime")
        for _, d in g.iterrows():
            t0 = day + pd.Timedelta(hours=int(d["hhmm"][:2]), minutes=int(d["hhmm"][3:]))
            spx_snap = snapshot(qs["SPX"], t0) if "SPX" in qs else None
            F = parity_price(spx_snap) if spx_snap is not None and len(spx_snap) else np.nan
            if not np.isfinite(F):
                continue
            for sym, scale in (("SPX", 1.0), ("XSP", 0.1)):
                if sym not in qs:
                    continue
                snap = spx_snap if sym == "SPX" else snapshot(qs[sym], t0)
                fl = fly.fly_from_quotes(snap, F * scale)
                if not fl or not (fl["fly_credit_mid"] > 0):
                    continue
                K, W = fl["K"], fl["W"]
                legs = [("C", K), ("P", K), ("C", K + W), ("P", K - W)]
                ls = leg_series(qs[sym], legs, t0)
                tc = (ls[("C", K, "bid")] + ls[("P", K, "bid")] - ls[("C", K + W, "ask")] - ls[("P", K - W, "ask")]).to_numpy()
                mid0, payoff = fl["fly_credit_mid"], min(abs(st[day] * scale - K), W)
                rec = {"day": day, "hhmm": d["hhmm"], "sym": sym, "b0_fly": d["trade"] == "fly", "K": K, "W": W,
                       "mid0": mid0, "touch0": tc[0], "gap_rel": (mid0 - tc[0]) / mid0, "ret_at_mid": ret_per_1k(mid0, W, payoff)}
                for name, (c, s) in policies(tc, mid0).items():
                    rec[f"{name}_ret"] = ret_per_1k(c, W, payoff) if c is not None and np.isfinite(c) else 0.0
                    rec[f"{name}_fill"] = c is not None and np.isfinite(c)
                    rec[f"{name}_sec"] = s
                rows.append(rec)
    return pd.DataFrame(rows)


def summarise(x: pd.DataFrame, rng) -> dict:
    pol = [c[:-4] for c in x.columns if c.endswith("_ret") and c != "ret_at_mid"]
    days = x["day"].to_numpy()
    uniq = np.unique(days)
    by_day = {p: x.groupby("day")[f"{p}_ret"].sum().reindex(uniq).to_numpy() for p in pol}
    n_day = x.groupby("day").size().reindex(uniq).to_numpy()
    out = {}
    for p in pol:
        diff = by_day[p] - by_day["LADDER"]
        b = rng.integers(0, len(uniq), (5000, len(uniq)))
        mean_dec = by_day[p].sum() / n_day.sum()
        d_dec = diff.sum() / n_day.sum()
        boot = (diff[b].sum(axis=1) / n_day[b].sum(axis=1))
        filled = x[f"{p}_fill"]
        out[p] = {"per_decision": float(mean_dec), "fill_rate": float(filled.mean()),
                  "vs_LADDER": float(d_dec), "vs_LADDER_ci90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))],
                  "median_sec_to_fill": float(x.loc[filled, f"{p}_sec"].median()) if filled.any() else None,
                  # adverse selection: the same trade at mid, filled vs not filled
                  "at_mid_if_filled": float(x.loc[filled, "ret_at_mid"].mean()) if filled.any() else None,
                  "at_mid_if_unfilled": float(x.loc[~filled, "ret_at_mid"].mean()) if (~filled).any() else None}
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    picks = pd.read_parquet(ROOT / "reports" / "round2" / "b0_spx_picks_1s_window.parquet")
    picks["day"] = pd.to_datetime(picks["day"])
    sim = simulate(picks)
    sim.to_parquet(OUT / "e2.parquet", index=False)
    rng = np.random.default_rng(20261112)
    res = {}
    for scope, x in (("primary_b0_fly", sim[sim["b0_fly"]]), ("secondary_all_times", sim)):
        res[scope] = {}
        for sym in ("XSP", "SPX"):
            y = x[x["sym"] == sym]
            res[scope][sym] = {"decisions": int(len(y)), "days": int(y["day"].nunique()),
                               "median_gap_rel": float(y["gap_rel"].median()), "policies": summarise(y, rng)}
    best = max(res["primary_b0_fly"]["XSP"]["policies"].items(), key=lambda kv: kv[1]["per_decision"])
    lad = res["primary_b0_fly"]["XSP"]["policies"]["LADDER"]
    res["decision"] = {"best_xsp_policy": best[0], "vs_LADDER": best[1]["vs_LADDER"], "ci90": best[1]["vs_LADDER_ci90"],
                       "switch": bool(best[0] != "LADDER" and best[1]["vs_LADDER"] >= 2 and best[1]["vs_LADDER_ci90"][0] > 0),
                       "ladder_per_decision": lad["per_decision"]}
    (OUT / "e2.json").write_text(json.dumps(res, indent=2, default=float))
    for scope in ("primary_b0_fly", "secondary_all_times"):
        for sym in ("XSP", "SPX"):
            r = res[scope][sym]
            print(f"\n== {scope} {sym}: {r['decisions']} decisions on {r['days']} days, median gap {r['median_gap_rel']:.1%}")
            for p, v in sorted(r["policies"].items(), key=lambda kv: -kv[1]["per_decision"]):
                print(f"  {p:12s} ${v['per_decision']:+7.1f}/decision  fill {v['fill_rate']:5.0%}  vs LADDER ${v['vs_LADDER']:+6.1f} "
                      f"[{v['vs_LADDER_ci90'][0]:+.1f}, {v['vs_LADDER_ci90'][1]:+.1f}]  "
                      f"at-mid filled {v['at_mid_if_filled'] if v['at_mid_if_filled'] is None else round(v['at_mid_if_filled'], 1)}"
                      f" / unfilled {v['at_mid_if_unfilled'] if v['at_mid_if_unfilled'] is None else round(v['at_mid_if_unfilled'], 1)}")
    print("\nDECISION:", res["decision"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
