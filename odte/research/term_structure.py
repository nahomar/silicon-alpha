"""Edge #3: same-day vs next-expiration option prices (term structure).

Pre-registered 2026-11-09 (docs/expiry_edge_plan.md). Signals at each fixed
decision time from the next expiration's quotes recorded during the session
(data/odte_next/<session>.parquet), tested against the factory's four targets.

  python3 -m odte.research.term_structure
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import factory as fx

ROOT = Path(__file__).resolve().parents[2]
NEXT = ROOT / "data" / "odte_next"
MIN_YEAR = 390.0 * 252.0
T_MIN = 3.0
SIGNALS = ["ts_ratio", "ts_spread", "ts_ratio_chg", "skew_next", "skew_gap", "straddle_next_chg"]


def _straddle_skew(grid, F, t, K_hint=None):
    """ATM straddle (mid) and skew (OTM put - OTM call one implied move away, / straddle)."""
    from odte.eval.expiry_edge import _two_sided
    q = _two_sided(grid, t)
    (cb, ca), (pb, pa) = q["C"], q["P"]
    both = cb.index.intersection(pb.index)
    if not len(both) or not np.isfinite(F):
        return np.nan, np.nan
    K = float(both[np.argmin(np.abs(both.to_numpy(float) - F))])
    st = (cb[K] + ca[K]) / 2 + (pb[K] + pa[K]) / 2
    if not st > 0:
        return np.nan, np.nan
    kp = pb.index[np.argmin(np.abs(pb.index.to_numpy(float) - (K - st)))]
    kc = cb.index[np.argmin(np.abs(cb.index.to_numpy(float) - (K + st)))]
    skew = ((pb[kp] + pa[kp]) - (cb[kc] + ca[kc])) / 2 / st
    return float(st), float(skew)


def build(d: pd.DataFrame) -> pd.DataFrame:
    from odte.eval.expiry_edge import implied_forward, minute_grid
    sessions = pd.DatetimeIndex(pd.read_parquet(ROOT / "data" / "calendar" / "sessions.parquet")["session"])
    rows = {}
    for day, g in d.groupby("day"):
        p = NEXT / f"{day:%Y-%m-%d}.parquet"
        if not p.exists():
            continue
        ch = pd.read_parquet(p)
        if ch.empty:
            continue
        exp = pd.Timestamp(ch["expiration"].iloc[0]).normalize()
        n_days = int(((sessions > day) & (sessions <= exp)).sum())          # sessions until it expires
        grid = minute_grid(ch)
        F1 = implied_forward(grid)
        first = {}
        for i, r in g.sort_values("hhmm").iterrows():
            t = day + pd.Timedelta(hours=int(r["hhmm"][:2]), minutes=int(r["hhmm"][3:]))
            F = F1.get(t, np.nan)
            st1, sk1 = _straddle_skew(grid, F if np.isfinite(F) else r["F"], t)
            if not np.isfinite(st1):
                continue
            T0 = max(r["minutes_to_close"], 1.0) / MIN_YEAR
            T1 = (r["minutes_to_close"] + 390.0 * n_days) / MIN_YEAR
            iv0 = r["straddle_mid"] / r["F"] / 0.8 / math.sqrt(T0)
            iv1 = st1 / r["F"] / 0.8 / math.sqrt(T1)
            rec = {"ts_ratio": iv0 / iv1, "ts_spread": iv0 - iv1, "skew_next": sk1,
                   "skew_gap": r["skew"] - sk1, "_st1": st1}
            if not first:
                first = rec.copy()
            rec["ts_ratio_chg"] = rec["ts_ratio"] - first["ts_ratio"]
            rec["straddle_next_chg"] = st1 / first["_st1"] - 1.0
            rows[i] = rec
    return pd.DataFrame.from_dict(rows, orient="index")[SIGNALS]


def main() -> int:
    d = fx.load()
    S = build(d)
    d = d.join(S, how="inner")
    print(f"{d['day'].nunique()} days, {len(d):,} decisions with next-expiry quotes", flush=True)
    era1 = d["day"] < fx.ERA_SPLIT
    rows = []
    for t in ("T_fly", "T_dir", "T_size", "T_tail"):
        for s in SIGNALS:
            r = fx.ic_test(d[s], d[t], d["day"])
            r1 = fx.ic_test(d[s][era1], d[t][era1], d["day"][era1])
            r2 = fx.ic_test(d[s][~era1], d[t][~era1], d["day"][~era1])
            rows.append({"target": t, "signal": s, **r, "ic_2016_21": r1["ic"], "ic_2022_26": r2["ic"]})
    tab = pd.DataFrame(rows)
    tab["fdr_pass"] = fx.bh(tab["p"].to_numpy(), fx.FDR_Q)
    tab["survives"] = (tab["fdr_pass"] & (tab["t"].abs() >= T_MIN)
                       & (np.sign(tab["ic_2016_21"]) == np.sign(tab["ic_2022_26"]))
                       & (np.sign(tab["ic"]) == np.sign(tab["ic_2016_21"])))
    out = ROOT / "reports" / "factory"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "term_structure_tests.csv", index=False)
    print(tab.sort_values("p")[["target", "signal", "ic", "t", "ic_2016_21", "ic_2022_26", "fdr_pass",
                                "survives"]].round(4).to_string(index=False))
    print(f"\nSURVIVORS (pre-registered gate): {int(tab['survives'].sum())}")
    (out / "term_structure.json").write_text(json.dumps(
        {"survivors": tab[tab["survives"]][["target", "signal", "ic", "t"]].to_dict("records")}, indent=2,
        default=float))
    return 0


TAIL_COMPOSITE = ["gex_flip_dist", "gex_net", "oi_pcr_0dte", "spx_ret_prev", "gex_0dte", "vix9d_over_vix",
                  "iv_over_rv"]                        # factory T_tail survivors (reports/factory/composites.json)


def incremental() -> dict:
    """Post-gate diagnostic: does ts_spread add to the factory T_tail composite?"""
    from odte.eval.expiry_edge import paired_diff
    d = fx.load()
    S = fx.signals(d)
    d = pd.concat([d, S[[c for c in TAIL_COMPOSITE if c not in d.columns]]], axis=1)
    d = d.join(build(d), how="inner")
    d = d[np.isfinite(d["ts_spread"])].reset_index(drop=True)
    res, sellers = {}, {}
    for name, cols in (("base", TAIL_COMPOSITE), ("base_plus_ts", TAIL_COMPOSITE + ["ts_spread"])):
        pred = fx.walk_forward_ridge(d, d[cols], d["T_tail"])
        ok = np.isfinite(pred)
        res[name] = {"oos_ic": fx.ic_test(pd.Series(pred[ok]), d["T_tail"][ok].reset_index(drop=True),
                                          d["day"][ok].reset_index(drop=True))}
        thr = np.nanquantile(pred, 0.8)
        sell = ok & (pred < thr)
        pnl = np.where(sell, d["T_fly"].to_numpy() * 1000, 0.0)
        sellers[name] = fx.day_series(d[ok], pnl[ok])
        res[name]["seller"] = fx.summarize(sellers[name])
    res["always_sell"] = fx.summarize(fx.day_series(d[ok], d["T_fly"].to_numpy()[ok] * 1000))
    res["plus_ts_minus_base"] = paired_diff(sellers["base_plus_ts"], sellers["base"])
    by_time = {h: fx.ic_test(g["ts_spread"], g["T_tail"], g["day"]) for h, g in d.groupby("hhmm")}
    res["ic_by_time"] = {h: {"ic": r["ic"], "t": r["t"]} for h, r in by_time.items()}
    out = ROOT / "reports" / "factory" / "term_structure_incremental.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    if "--incremental" in sys.argv:
        incremental()
        sys.exit(0)
    sys.exit(main())
