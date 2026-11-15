"""M: close momentum on short-gamma days, as a partner for B0 (docs/momentum_plan.md).

When dealers are short gamma they hedge with the move, so the index tends to keep going into the close.
Those are the days B0's sold flies lose. The rule: at 15:30, if dealer gamma is negative, hold 1 MES in
the direction of the move since yesterday's close until 16:00.

    python3 -m odte.research.gamma_momentum      # -> reports/momentum/momentum.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research.stress_spx import sharpe, stationary_bootstrap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "momentum"
ES = ROOT / "data" / "futures" / "es_1m.parquet"
LIB = ROOT / "data" / "research" / "library_SPX.parquet"
B0 = ROOT / "reports" / "stress" / "stress_spx_daily.parquet"
COST = 0.75            # index points per MES round trip
MES = 5.0              # $ per index point
SEED = 20261115
ERA_SPLIT = pd.Timestamp("2022-01-01")
SCALE_END = pd.Timestamp("2021-12-31")


def es_bars(path: Path = ES) -> pd.DataFrame:
    """Regular-hours ES minute bars (09:30 <= t < 16:00 ET), one row per (date, minute): the busiest contract."""
    x = pd.read_parquet(path, columns=["instrument_id", "open", "close", "volume", "ts"])
    t = pd.to_datetime(x["ts"], unit="s", utc=True).dt.tz_convert("America/New_York")
    x["date"] = pd.to_datetime(t.dt.date)
    x["hhmm"] = t.dt.strftime("%H:%M")
    x = x[(x["hhmm"] >= "09:30") & (x["hhmm"] < "16:00")]
    x = x.sort_values("volume").drop_duplicates(["date", "hhmm"], keep="last")
    return x.set_index(["date", "hhmm"]).sort_index()[["instrument_id", "open", "close"]]


def prev_close(bars: pd.DataFrame) -> pd.DataFrame:
    """For each date: the previous trading date's last regular-hours close and its contract."""
    last = bars.groupby(level="date").tail(1).reset_index(level="hhmm", drop=True)
    return pd.DataFrame({"prev_close": last["close"].shift(1), "prev_inst": last["instrument_id"].shift(1)},
                        index=last.index)


def trades(bars: pd.DataFrame, gex: pd.Series, sig_end: str = "15:29", entry: str = "15:30",
           exit_: str = "15:59", side: str = "neg") -> pd.DataFrame:
    """One row per traded day: signal r from the previous close to the close of the sig_end bar, enter at the
    open of the entry bar, exit at the close of the exit_ bar. side: 'neg' (gex < 0), 'pos' (>= 0), 'all'."""
    pc = prev_close(bars)
    rows = []
    for day, g in gex.items():
        try:
            s, e, x = bars.loc[(day, sig_end)], bars.loc[(day, entry)], bars.loc[(day, exit_)]
        except KeyError:
            continue
        if day not in pc.index or not np.isfinite(pc.at[day, "prev_close"]):
            continue
        inst = s["instrument_id"]
        if not (pc.at[day, "prev_inst"] == inst == e["instrument_id"] == x["instrument_id"]):
            continue                                            # contract roll in between: skip
        if side == "neg" and not g < 0 or side == "pos" and not g >= 0 or not np.isfinite(g):
            continue
        r = s["close"] / pc.at[day, "prev_close"] - 1.0
        if r == 0:
            continue
        pos = np.sign(r)
        rows.append({"day": day, "r": r, "gex": g, "pos": pos, "gross": pos * (x["close"] - e["open"])})
    return pd.DataFrame(rows)


def stats(t: pd.DataFrame, cost: float = COST) -> dict:
    if t.empty:
        return {"n": 0}
    net = t["gross"].to_numpy() - cost
    out = {"n": int(len(net)), "mean_pts": float(net.mean()),
           "t": float(net.mean() / net.std(ddof=1) * np.sqrt(len(net))),
           "hit": float((net > 0).mean()), "avg_win": float(net[net > 0].mean()),
           "avg_loss": float(net[net <= 0].mean()), "usd_per_mes_per_trade": float(net.mean() * MES)}
    return out


def by_era(t: pd.DataFrame, cost: float = COST) -> dict:
    return {"2016-2021": stats(t[t["day"] < ERA_SPLIT], cost), "2022-2026": stats(t[t["day"] >= ERA_SPLIT], cost)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lib = pd.read_parquet(LIB, columns=["day", "hhmm", "gex_net"])
    lib["day"] = pd.to_datetime(lib["day"])
    lib = lib[lib["day"] <= pd.Timestamp("2026-11-06")]
    g1530 = lib[lib["hhmm"] == "15:30"].set_index("day")["gex_net"].sort_index()
    g1500 = lib[lib["hhmm"] == "15:00"].set_index("day")["gex_net"].sort_index()
    g1000 = lib[lib["hhmm"] == "10:00"].set_index("day")["gex_net"].sort_index()
    bars = es_bars()

    prim = trades(bars, g1530)
    res = {"rule": "gex_net(15:30) < 0; 1 MES with the sign of prev close->15:30; 15:30 open -> 15:59 close",
           "cost_pts": COST, "days_considered": int(len(g1530)), "primary": stats(prim),
           "by_era": by_era(prim),
           "costs": {str(c): stats(prim, c) for c in (0.0, COST, 1.5)},
           "by_year": {int(y): stats(z) for y, z in prim.groupby(prim["day"].dt.year)}}
    e = res["by_era"]
    res["M1"] = {"pass": bool(e["2016-2021"].get("mean_pts", -1) > 0 and e["2022-2026"].get("mean_pts", -1) > 0
                              and res["primary"]["t"] >= 2.0)}

    # reported only
    res["report_only"] = {
        "all_days": by_era(trades(bars, g1530, side="all")),
        "gex_pos": by_era(trades(bars, g1530, side="pos")),
        "gao_first_half_hour": by_era(trades(bars, g1000, sig_end="09:59")),
        "entry_1min_late": by_era(trades(bars, g1530, entry="15:31")),
        "entry_15:00": by_era(trades(bars, g1500, sig_end="14:59", entry="15:00")),
        "entry_15:45": by_era(trades(bars, g1530, sig_end="15:44", entry="15:45")),
    }

    # M2: does it help B0?
    b0 = pd.read_parquet(B0)["pnl_per_1k"]
    b0.index = pd.DatetimeIndex(b0.index)
    m = pd.Series(0.0, index=b0.index)
    net = (prim.set_index("day")["gross"] - COST) * MES
    m.loc[m.index.intersection(net.index)] = net.reindex(m.index.intersection(net.index))
    early = b0.index <= SCALE_END
    w = float(b0[early].std(ddof=1) / m[early].std(ddof=1))
    comb = b0 + w * m
    rngs = [np.random.default_rng(SEED), np.random.default_rng(SEED)]
    p_loss = {}
    for name, x, rng in (("B0", b0, rngs[0]), ("B0+M", comb, rngs[1])):
        yr = stationary_bootstrap(x.to_numpy(), 10000, 252, 20.0, rng)
        p_loss[name] = float((yr.sum(axis=1) < 0).mean())
    both = (b0 != 0) & (m != 0)
    res["M2"] = {
        "days": int(len(b0)), "momentum_trade_days": int((m != 0).sum()), "overlap_days": int(both.sum()),
        "corr": float(np.corrcoef(b0, m)[0, 1]),
        "corr_on_overlap_days": float(np.corrcoef(b0[both], m[both])[0, 1]) if both.sum() > 2 else None,
        "weight_mes_per_b0_unit": w,
        "sharpe": {"B0": sharpe(b0), "M": sharpe(m), "B0+M": sharpe(comb)},
        "mean_per_day": {"B0": float(b0.mean()), "M_scaled": float((w * m).mean()), "B0+M": float(comb.mean())},
        "p_losing_year": p_loss,
        "by_year": {int(y): {"B0": float(b0[b0.index.year == y].mean()),
                             "M_scaled": float((w * m)[m.index.year == y].mean()),
                             "B0+M": float(comb[comb.index.year == y].mean())}
                    for y in sorted(set(b0.index.year))},
    }
    r2 = res["M2"]
    r2["pass"] = bool(r2["corr"] <= 0.10 and r2["sharpe"]["B0+M"] > r2["sharpe"]["B0"]
                      and p_loss["B0+M"] < p_loss["B0"])
    res["verdict"] = "PASS" if res["M1"]["pass"] and r2["pass"] else "FAIL"

    prim.to_parquet(OUT / "momentum_trades.parquet", index=False)
    (OUT / "momentum.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
