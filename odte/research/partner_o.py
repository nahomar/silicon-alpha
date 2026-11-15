"""Partner O (docs/partner_plan.md): long 1 MES from 16:00 to the next 09:30. Does it make money on its own (O1),
and does it make B0 steadier (O2)?

    python3 -m odte.research.partner_o      # -> reports/partners/o.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research.gamma_momentum import es_bars

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "partners"
B0 = ROOT / "reports" / "stress" / "stress_spx_daily.parquet"
COST, MES, SEED = 0.75, 5.0, 20261115
FIRST, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2026-11-06")
ERA = pd.Timestamp("2022-01-01")


def legs(bars: pd.DataFrame) -> pd.DataFrame:
    """Per session: the night that ends at its 09:30 open and its own day leg, gross points, same contract only."""
    dates = bars.index.get_level_values("date").unique().sort_values()
    rows = []
    for prev, d in zip(dates[:-1], dates[1:]):
        r = {"day": d, "weekend": (d - prev).days > 1, "night": np.nan, "dayleg": np.nan}
        try:
            c, o = bars.loc[(prev, "15:59")], bars.loc[(d, "09:30")]
            if c["instrument_id"] == o["instrument_id"]:
                r["night"] = o["open"] - c["close"]
        except KeyError:
            pass
        try:
            o, c = bars.loc[(d, "09:30")], bars.loc[(d, "15:59")]
            if c["instrument_id"] == o["instrument_id"]:
                r["dayleg"] = c["close"] - o["open"]
        except KeyError:
            pass
        rows.append(r)
    x = pd.DataFrame(rows)
    return x[(x["day"] >= FIRST) & (x["day"] <= END)]


def st(v, cost=COST) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)] - cost
    if len(v) < 3:
        return {"n": int(len(v))}
    eq = np.cumsum(v * MES)
    return {"n": int(len(v)), "mean_pts": float(v.mean()), "t": float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))),
            "usd_per_mes_per_year": float(v.mean() * MES * 252), "hit": float((v > 0).mean()),
            "worst_pts": float(v.min()), "max_dd_usd_per_mes": float((eq - np.maximum.accumulate(eq)).min())}


def main() -> int:
    from odte.research.stress_spx import sharpe, stationary_bootstrap
    OUT.mkdir(parents=True, exist_ok=True)
    x = legs(es_bars())
    n = x.dropna(subset=["night"])
    era = {"2016-2021": st(n.loc[n["day"] < ERA, "night"]), "2022-2026": st(n.loc[n["day"] >= ERA, "night"])}
    res = {"O1": {"all": st(n["night"]), **era}}
    res["O1"]["pass"] = bool(era["2016-2021"]["mean_pts"] > 0 and era["2022-2026"]["mean_pts"] > 0
                             and res["O1"]["all"]["t"] >= 2.0)

    b0 = pd.read_parquet(B0)["pnl_per_1k"]
    b0.index = pd.DatetimeIndex(b0.index)
    o = pd.Series(0.0, index=b0.index)
    nn = n[(n["day"] > b0.index[0] - pd.Timedelta(days=1)) & (n["day"] <= b0.index[-1])]
    pos = np.searchsorted(b0.index.values, nn["day"].values)            # first B0 day on or after the morning
    np.add.at(o.values, pos, ((nn["night"] - COST) * MES).to_numpy())
    early = b0.index <= pd.Timestamp("2021-12-31")
    w = float(b0[early].std(ddof=1) / o[early].std(ddof=1))
    comb = b0 + w * o
    p_loss = {}
    for name, s in (("B0", b0), ("B0+O", comb)):
        yr = stationary_bootstrap(s.to_numpy(), 10000, 252, 20.0, np.random.default_rng(SEED))
        p_loss[name] = float((yr.sum(axis=1) < 0).mean())
    res["O2"] = {"nights_booked": int(len(nn)), "corr": float(np.corrcoef(b0, o)[0, 1]), "weight": w,
                 "sharpe": {"B0": sharpe(b0), "O": sharpe(o), "B0+O": sharpe(comb)},
                 "mean_per_day": {"B0": float(b0.mean()), "O_scaled": float((w * o).mean()), "B0+O": float(comb.mean())},
                 "p_losing_year": p_loss,
                 "by_year": {int(y): {"B0": float(b0[b0.index.year == y].mean()), "O_scaled": float((w * o)[o.index.year == y].mean()),
                                      "B0+O": float(comb[comb.index.year == y].mean())} for y in sorted(set(b0.index.year))}}
    r2 = res["O2"]
    r2["pass"] = bool(r2["corr"] <= 0.10 and r2["sharpe"]["B0+O"] > r2["sharpe"]["B0"] and p_loss["B0+O"] < p_loss["B0"])
    e = res["O1"]
    lead = e["2016-2021"]["mean_pts"] > 0 and e["2022-2026"]["mean_pts"] > 0 and e["all"]["t"] >= 1.0 and r2["pass"]
    res["verdict"] = "PASS" if e["pass"] and r2["pass"] else ("LEAD" if lead else "FAIL")

    d = x.dropna(subset=["dayleg"])
    res["report_only"] = {
        "day_leg": {"all": st(d["dayleg"]), "2016-2021": st(d.loc[d["day"] < ERA, "dayleg"]),
                    "2022-2026": st(d.loc[d["day"] >= ERA, "dayleg"])},
        "costs": {str(c): st(n["night"], c) for c in (0.0, COST, 1.5)},
        "weekend": st(n.loc[n["weekend"], "night"]), "weekday": st(n.loc[~n["weekend"], "night"]),
        "by_year": {int(y): st(z["night"]) for y, z in n.groupby(n["day"].dt.year)},
    }
    (OUT / "o.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
