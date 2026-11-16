"""Partner B (docs/partner_plan.md): plain buy-and-hold, 1 MES long at all times, as the benchmark O and T must beat.

    python3 -m odte.research.partner_b      # -> reports/partners/b.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research.gamma_momentum import es_bars
from odte.research.partner_o import legs
from odte.research.partner_t import book, day_table, nyse_days, windows

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "partners"
B0 = ROOT / "reports" / "stress" / "stress_spx_daily.parquet"
COST, MES, SEED = 0.75, 5.0, 20261115
FIRST, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2026-11-06")
ERA, SCALE_END = pd.Timestamp("2022-01-01"), pd.Timestamp("2021-12-31")


def marks(dt: pd.DataFrame) -> pd.DataFrame:
    """Daily close-to-close points for a position held at all times; roll days marked 0 and charged COST."""
    prev = dt.shift(1)
    roll = dt["inst"] != prev["inst"]
    m = pd.DataFrame({"day": dt.index, "gross": np.where(roll, 0.0, dt["close"] - prev["close"]),
                      "roll": roll.to_numpy()}, index=dt.index)
    m["net"] = m["gross"] - np.where(m["roll"], COST, 0.0)
    return m[(m.index >= FIRST) & (m.index <= END)]


def st(v) -> dict:
    v = np.asarray(v, float)
    eq = np.cumsum(v * MES)
    return {"n": int(len(v)), "mean_pts": float(v.mean()), "t": float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))),
            "usd_per_mes_per_year": float(v.mean() * MES * 252), "worst_pts": float(v.min()),
            "max_dd_usd_per_mes": float((eq - np.maximum.accumulate(eq)).min())}


def main() -> int:
    from odte.research.stress_spx import sharpe, stationary_bootstrap
    OUT.mkdir(parents=True, exist_ok=True)
    bars = es_bars()
    dt = day_table(bars)
    m = marks(dt)
    era = {"2016-2021": st(m.loc[m.index < ERA, "net"]), "2022-2026": st(m.loc[m.index >= ERA, "net"])}
    res = {"rolls": int(m["roll"].sum()), "B1": {"all": st(m["net"]), **era}}
    res["B1"]["pass"] = bool(era["2016-2021"]["mean_pts"] > 0 and era["2022-2026"]["mean_pts"] > 0
                             and res["B1"]["all"]["t"] >= 2.0)

    b0 = pd.read_parquet(B0)["pnl_per_1k"]
    b0.index = pd.DatetimeIndex(b0.index)
    early = b0.index <= SCALE_END

    def wgt(x: pd.Series) -> float:
        return float(b0[early].std(ddof=1) / x[early].std(ddof=1))

    def p_loss(x: pd.Series) -> float:
        yr = stationary_bootstrap(x.to_numpy(), 10000, 252, 20.0, np.random.default_rng(SEED))
        return float((yr.sum(axis=1) < 0).mean())

    b = book(b0.index, m["day"], m["net"] * MES)
    wb = wgt(b)
    comb = b0 + wb * b
    pl = {"B0": p_loss(b0), "B0+B": p_loss(comb)}
    res["B2"] = {"corr": float(np.corrcoef(b0, b)[0, 1]), "weight_mes_per_b0_unit": wb,
                 "sharpe": {"B0": sharpe(b0), "B": sharpe(b), "B0+B": sharpe(comb)},
                 "mean_per_day": {"B0": float(b0.mean()), "B_scaled": float((wb * b).mean()), "B0+B": float(comb.mean())},
                 "p_losing_year": pl,
                 "by_year": {int(y): {"B0": float(b0[b0.index.year == y].mean()),
                                      "B_scaled": float((wb * b)[b.index.year == y].mean()),
                                      "B0+B": float(comb[comb.index.year == y].mean())} for y in sorted(set(b0.index.year))}}
    r2 = res["B2"]
    r2["pass"] = bool(r2["corr"] <= 0.10 and r2["sharpe"]["B0+B"] > r2["sharpe"]["B0"] and pl["B0+B"] < pl["B0"])
    e = res["B1"]
    lead = e["2016-2021"]["mean_pts"] > 0 and e["2022-2026"]["mean_pts"] > 0 and e["all"]["t"] >= 1.0 and r2["pass"]
    res["verdict"] = "PASS" if e["pass"] and r2["pass"] else ("LEAD" if lead else "FAIL")

    # O and T daily series, exactly as booked in partner_o / partner_t
    x = legs(bars).dropna(subset=["night"])
    o = book(b0.index, x["day"], (x["night"] - COST) * MES)
    _, mk = windows(nyse_days(), dt)
    t = book(b0.index, mk["day"], (mk["mark"] - np.where(mk["exit"], COST, 0.0)) * MES)
    wo, wt = wgt(o), wgt(t)
    combos = {"B0": b0, "B0+B": comb, "B0+O": b0 + wo * o, "B0+T": b0 + wt * t,
              "B0+O+T": b0 + wo * o + wt * t, "B0+B+O": b0 + wb * b + wo * o}
    h2h = {k: {"sharpe": sharpe(v), "p_losing_year": pl[k] if k in pl else p_loss(v)} for k, v in combos.items()}
    res["head_to_head"] = h2h
    res["head_to_head_verdict"] = {
        k: bool(h2h[f"B0+{k}"]["sharpe"] > h2h["B0+B"]["sharpe"]
                and h2h[f"B0+{k}"]["p_losing_year"] < h2h["B0+B"]["p_losing_year"]) for k in ("O", "T")}
    res["report_only"] = {"corr_O_B": float(np.corrcoef(o, b)[0, 1]), "corr_T_B": float(np.corrcoef(t, b)[0, 1]),
                          "weights_mes_per_b0_unit": {"B": wb, "O": wo, "T": wt}}
    (OUT / "b.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "B2"} | {"B2": {k: v for k, v in r2.items() if k != "by_year"}},
                     indent=1, default=float))
    for y, v in r2["by_year"].items():
        print(y, {a: round(c, 1) for a, c in v.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
