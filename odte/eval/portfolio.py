"""Turn signals into a money view: sizing, execution, risk limits, scale.

expiry_edge answers "is there an edge per trade?". This answers "what does
it mean for an account": how big each bet is, how it is filled, what stops
it, and how far it scales.

Sizing (fractional Kelly, walk-forward)
  Before each day, regress realized return-on-risk on signal strength
  |1 - predicted ratio| over that strategy's *past* trades (out-of-sample
  predictions only). For a new trade, mu = a + b * strength, and the
  bet is KELLY x mu / E[R^2] of capital at max loss. It is capped at
  MAX_RISK_PER_TRADE and skipped when mu <= 0. Until MIN_HIST trades exist,
  every trade risks BASE_RISK.

Execution
  cross         buy at the ask / sell at the bid, at the decision minute
  mid_or_cross  rest at mid 5 minutes; filled only if the market trades
                through the mid by a tick, else cross at t+5
  mid_or_skip   rest at mid 5 minutes, otherwise no trade

Risk limits (defined-risk trades held to expiry, so these are hard caps)
  per trade   <= MAX_RISK_PER_TRADE of equity
  per day     total max loss opened <= MAX_RISK_PER_DAY of equity; this
              *is* the daily loss limit, since nothing can lose more than
              its max loss
  drawdown    equity below (1 - DD_STOP) x peak -> stop PAUSE_DAYS
              sessions, then resume with the peak reset

Capacity
  Lots per trade are whole contracts, capped at the displayed size at the
  touch across all four legs (`cap_short` / `cap_long`). That is a floor
  on real liquidity, since SPX market makers refill and quote more in
  the complex book. The report shows how often that cap binds at each
  capital level.

The house configuration (fly, mid_or_cross, Kelly sizing, limits above) was
set before any 12-month result; the other combinations are sensitivity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KELLY = 0.25
BASE_RISK = 0.005
MAX_RISK_PER_TRADE = 0.02
MAX_RISK_PER_DAY = 0.06
DD_STOP, PAUSE_DAYS = 0.15, 10
MIN_HIST = 40
CAPITALS = (100_000, 1_000_000, 10_000_000, 100_000_000)
HOUSE = {"structure": "fly", "execution": "mid_or_cross", "sizing": "kelly"}


def trade_table(ds: pd.DataFrame, act: np.ndarray, pred: np.ndarray | None,
                structure: str = "fly", execution: str = "cross") -> pd.DataFrame:
    """One row per intended trade: return on risk, $ risk per lot, capacity."""
    rows = []
    for i in np.flatnonzero(act != 0):
        r = ds.iloc[i]
        side = "short" if act[i] == 1 else "long"
        if structure == "condor":
            pnl, risk = r.get(f"pnl_{side}_condor"), r.get(f"risk_{side}_condor")
        elif execution == "cross":
            pnl, risk = r[f"pnl_{side}_fly"], r[f"risk_{side}_fly"]
        else:
            if bool(r.get(f"fill_mid_{side}", False)):
                pnl, risk = r[f"pnl_{side}_fly_midx"], r[f"risk_{side}_fly_midx"]
            elif execution == "mid_or_cross":
                pnl, risk = r[f"pnl_{side}_fly_t5"], r[f"risk_{side}_fly_t5"]
            else:
                continue                                   # mid_or_skip: no fill
        if not (np.isfinite(pnl) and np.isfinite(risk)) or risk <= 0:
            continue
        cap = r.get(f"cap_{side}", np.nan)
        strength = abs(1.0 - pred[i]) if pred is not None and np.isfinite(pred[i]) else np.nan
        rows.append({"row": i, "day": r["day"], "t": r["t"], "side": side,
                     "ror": pnl / risk, "unit_risk": risk,
                     "cap": cap if structure == "fly" else np.nan,
                     "strength": strength})
    return pd.DataFrame(rows)


def kelly_fractions(trades: pd.DataFrame) -> np.ndarray:
    """Fraction of equity at max loss per trade, fit on prior days only."""
    f = np.full(len(trades), BASE_RISK)
    if trades.empty or trades["strength"].isna().all():
        return f
    days = trades["day"].to_numpy()
    for d in np.unique(days):
        hist = trades[(days < d) & trades["strength"].notna()]
        today = np.flatnonzero(days == d)
        if len(hist) < MIN_HIST:
            continue
        x, y = hist["strength"].to_numpy(), hist["ror"].to_numpy()
        b = np.polyfit(x, y, 1) if np.ptp(x) > 0 else np.array([0.0, y.mean()])
        resid = y - np.polyval(b, x)
        for i in today:
            s = trades["strength"].iat[i]
            mu = float(np.polyval(b, s)) if np.isfinite(s) else float(y.mean())
            m2 = float(resid.var() + mu * mu)
            f[i] = 0.0 if mu <= 0 or m2 <= 0 else min(KELLY * mu / m2, MAX_RISK_PER_TRADE)
    return f


def simulate(trades: pd.DataFrame, fracs: np.ndarray, capital: float,
             all_days) -> dict:
    """Day-by-day account with whole lots, capacity and risk limits."""
    equity, peak, paused = float(capital), float(capital), 0
    daily, capped, wanted, got, n_trades, pause_events = [], 0, 0.0, 0.0, 0, 0
    max_day_risk = 0.0
    by_day = {d: g for d, g in trades.assign(frac=fracs).groupby("day")} if len(trades) else {}
    for d in all_days:
        pnl = 0.0
        if paused > 0:
            paused -= 1
        elif d in by_day:
            opened = 0.0
            for _, tr in by_day[d].sort_values("t").iterrows():
                if tr["frac"] <= 0:
                    continue
                want = tr["frac"] * equity
                lots = np.floor(want / tr["unit_risk"])
                if np.isfinite(tr["cap"]):
                    if lots > tr["cap"]:
                        capped += 1
                    lots = min(lots, tr["cap"])
                room = MAX_RISK_PER_DAY * equity - opened
                lots = min(lots, np.floor(max(room, 0) / tr["unit_risk"]))
                if lots < 1:
                    continue
                risk = lots * tr["unit_risk"]
                wanted += want
                got += risk
                opened += risk
                pnl += tr["ror"] * risk
                n_trades += 1
            max_day_risk = max(max_day_risk, opened / equity)
        equity += pnl
        daily.append(pnl)
        peak = max(peak, equity)
        if paused == 0 and equity < (1 - DD_STOP) * peak:
            paused, peak, pause_events = PAUSE_DAYS, equity, pause_events + 1
    daily = np.array(daily)
    cum = np.cumsum(daily)
    dd = float((cum - np.maximum.accumulate(np.concatenate([[0], cum]))[1:]).min()) if len(cum) else 0.0
    sd = daily.std(ddof=1) if len(daily) > 1 else 0.0
    return {"capital": capital, "final_equity": equity, "total_pnl": float(daily.sum()),
            "mean_daily_pnl": float(daily.mean()) if len(daily) else 0.0,
            "return_pct": float(daily.sum() / capital * 100),
            "sharpe": float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
            "worst_day": float(daily.min()) if len(daily) else 0.0,
            "max_drawdown": dd, "n_trades": n_trades,
            "pct_trades_capped_by_liquidity": float(capped / max(n_trades, 1) * 100),
            "fill_ratio_of_wanted_risk": float(got / wanted) if wanted else float("nan"),
            "drawdown_pauses": pause_events,
            "max_risk_opened_in_a_day": max_day_risk}


def money_view(ds: pd.DataFrame, strategies: dict, preds: dict, tested: np.ndarray) -> dict:
    """House config at every capital level, plus sensitivity at $1M."""
    all_days = sorted(ds.loc[tested, "day"].unique())
    out = {}
    for name, act in strategies.items():
        pred = preds.get(name)
        res = {"scale": [], "sensitivity": []}
        house = trade_table(ds, act, pred, HOUSE["structure"], HOUSE["execution"])
        fr = kelly_fractions(house) if pred is not None else np.full(len(house), BASE_RISK)
        for cap in CAPITALS:
            res["scale"].append(simulate(house, fr, cap, all_days))
        for structure in ("fly", "condor"):
            for execution in (("cross", "mid_or_cross", "mid_or_skip") if structure == "fly" else ("cross",)):
                tt = trade_table(ds, act, pred, structure, execution)
                for sizing in ("fixed", "kelly"):
                    f = (kelly_fractions(tt) if sizing == "kelly" and pred is not None
                         else np.full(len(tt), BASE_RISK))
                    r = simulate(tt, f, 1_000_000, all_days)
                    r.update(structure=structure, execution=execution, sizing=sizing)
                    res["sensitivity"].append(r)
        mean_rate = res["scale"][1]["mean_daily_pnl"] / CAPITALS[1]
        res["capital_for_1m_per_day_if_linear"] = (1_000_000 / mean_rate) if mean_rate > 0 else None
        out[name] = res
    return out


def print_money(mv: dict, title: str) -> None:
    print(f"\n=== MONEY VIEW - {title} ===")
    print("house config: fly, rest at mid 5 min then cross, quarter-Kelly, "
          f"<= {MAX_RISK_PER_TRADE:.0%}/trade, <= {MAX_RISK_PER_DAY:.0%}/day at risk, "
          f"pause {PAUSE_DAYS}d after a {DD_STOP:.0%} drawdown")
    for name, r in mv.items():
        print(f"\n  {name}")
        print(f"    {'capital':>13s} {'$/day':>10s} {'total':>12s} {'return':>8s} {'sharpe':>7s} "
              f"{'worst day':>11s} {'max DD':>12s} {'trades':>6s} {'liq-capped':>10s}")
        for s in r["scale"]:
            print(f"    {s['capital']:>13,.0f} {s['mean_daily_pnl']:>10,.0f} {s['total_pnl']:>12,.0f} "
                  f"{s['return_pct']:>7.1f}% {s['sharpe']:>+7.2f} {s['worst_day']:>11,.0f} "
                  f"{s['max_drawdown']:>12,.0f} {s['n_trades']:>6d} {s['pct_trades_capped_by_liquidity']:>9.0f}%")
        c = r["capital_for_1m_per_day_if_linear"]
        print("    capital for $1M/day at this edge (ignoring liquidity): "
              + (f"${c:,.0f}" if c else "n/a - no positive edge"))
        print("    sensitivity at $1M:  " + "  ".join(
            f"{x['structure']}/{x['execution']}/{x['sizing']}: {x['total_pnl']:+,.0f}"
            for x in r["sensitivity"]))
