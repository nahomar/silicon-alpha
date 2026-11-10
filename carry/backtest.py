"""Crypto funding-rate carry vs. T-bills, the fixed rules pre-registered in docs/carry_plan.md.

  python3 -m carry.backtest fetch     # download free public data (Binance files, ^IRX, US venues)
  python3 -m carry.backtest check     # data coverage and spot checks
  python3 -m carry.backtest run       # the one pre-registered run

Position: long 1 unit spot, short 1 unit perp; capital = 1.3333 x notional (perp at 3x).
A day d held in the carry earns every funding event with time in (d 00:00, d+1 00:00]; the
switch decision for day d uses only funding settled strictly before d 00:00.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "carry"
OUT = ROOT / "reports" / "carry"
ASSETS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
PERIODS = {"discovery": ("2020-01-01", "2023-12-31"), "confirmation": ("2024-01-01", "2026-10-31")}
FIRST_DAY, LAST_DAY = "2020-01-01", "2026-10-31"
CAP = 1.0 + 1.0 / 3.0                    # capital per unit notional: spot + 1/3 margin at 3x
THRESHOLD = 0.10                         # C2 holds when trailing 7-day annualized funding > 10%
LOOKBACK = pd.Timedelta(days=7)
DAYS_PER_YEAR = 365.0
SLIP = 0.0002                            # 2 bp per trade
BINANCE_SPOT_TAKER, BINANCE_PERP_TAKER = 0.0010, 0.0005
CB_SPOT_TAKER = 0.0090                   # Coinbase Advanced entry tier (secondary source, see plan)
CDE_PERP_PCT, CDE_PER_CONTRACT = 0.0002, 0.15
CDE_CONTRACT = {"BTC": 0.01, "ETH": 0.1}
SEED, N_BOOT, BLOCK = 0, 10_000, 30
CI_LEVEL = 0.975                         # Bonferroni over 2 assets: two-sided 97.5% per asset
MARGIN_RISE, MARGIN_DAYS = 0.25, 7
PLAN_SHA256 = "d4de0345a433b890a8ccfdc4f6cc8a9bae759349234db62a404781a0b020d988"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load() -> dict:
    F = pd.read_parquet(DATA / "binance_funding.parquet")
    K = pd.read_parquet(DATA / "binance_spot_1d.parquet")
    irx = pd.read_parquet(DATA / "irx.parquet")["irx"]
    log = json.loads((DATA / "fetch_log.json").read_text())
    out = {"irx": irx, "log": log, "funding": {}, "spot": {}}
    for a, s in ASSETS.items():
        f = F[F["symbol"] == s].set_index("time")["rate"].sort_index()
        out["funding"][a] = f[~f.index.duplicated()]
        k = K[K["symbol"] == s].set_index("date")[["open", "high", "low", "close"]].sort_index()
        out["spot"][a] = k.asfreq("D")
    return out


def check(d: dict) -> dict:
    """Coverage, gaps and extremes of the funding and spot data (bad-data spot check)."""
    rep = {}
    for a in ASSETS:
        f, k = d["funding"][a], d["spot"][a]
        gaps = f.index.to_series().diff().dt.total_seconds().div(3600)
        r = k["close"].pct_change()
        rep[a] = {"funding_first": str(f.index[0]), "funding_last": str(f.index[-1]), "events": int(len(f)),
                  "interval_hours_counts": {str(h): int(n) for h, n in gaps.round(2).value_counts().head(6).items()},
                  "max_gap_hours": float(gaps.max()), "rate_min": float(f.min()), "rate_max": float(f.max()),
                  "spot_first": str(k.index[0].date()), "spot_last": str(k.index[-1].date()),
                  "spot_missing_days": int(k["close"].isna().sum()),
                  "daily_moves_over_25pct": {str(i.date()): round(float(v), 4) for i, v in r[r.abs() > 0.25].items()}}
    irx = d["irx"]
    rep["irx"] = {"first": str(irx.index[0].date()), "last": str(irx.index[-1].date()), "days": int(len(irx)),
                  "min": float(irx.min()), "max": float(irx.max())}
    return rep


# ---------------------------------------------------------------------------
# rule
# ---------------------------------------------------------------------------

def days(start: str = FIRST_DAY, end: str = LAST_DAY) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="D")


def earned_by_day(f: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Sum of funding rates with time in (d, d+1] for each day d; NaN outside the data's span."""
    owner = f.index.ceil("D") - pd.Timedelta(days=1)          # an event at d+1 00:00 belongs to day d
    s = f.groupby(owner).sum().reindex(index, fill_value=0.0)
    covered = (index + pd.Timedelta(days=1) >= f.index[0]) & (index < f.index[-1])
    return s.where(covered)


def signal(f: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """S(d) = sum of rates with time in [d - 7 days, d) x 365/7, using only funding settled strictly
    before d 00:00. NaN unless the data cover the whole window."""
    c = np.r_[0.0, f.to_numpy().cumsum()]
    before = lambda t: c[f.index.searchsorted(t, side="left")]   # sum of events with time < t
    s = (before(index) - before(index - LOOKBACK)) * DAYS_PER_YEAR / LOOKBACK.days
    ok = (f.index[0] <= index - LOOKBACK) & (f.index[-1] >= index - pd.Timedelta(days=1))
    return pd.Series(np.where(ok, s, np.nan), index=index)


def tbill_daily(irx: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Day d earns ^IRX from the last US trading date strictly before d, / 100 / 365."""
    s = irx.dropna().sort_index()
    pos = s.index.searchsorted(index, side="left") - 1
    v = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(v / 100.0 / DAYS_PER_YEAR, index=index)


def switch_cost(venue: str, asset: str, prev_close: pd.Series) -> pd.Series:
    """Cost of one change of state (spot leg + perp leg), as a fraction of notional, per day."""
    if venue == "binance":
        return pd.Series((BINANCE_SPOT_TAKER + SLIP) + (BINANCE_PERP_TAKER + SLIP), index=prev_close.index)
    if venue == "coinbase":
        per_contract = CDE_PER_CONTRACT / (CDE_CONTRACT[asset] * prev_close)
        return (CB_SPOT_TAKER + SLIP) + (CDE_PERP_PCT + SLIP + per_contract)
    raise ValueError(venue)


def c2_state(sig: pd.Series) -> pd.Series:
    return (sig > THRESHOLD).astype(float)                    # NaN signal compares False: flat


def simulate(earned: pd.Series, rf: pd.Series, state: pd.Series, cost: pd.Series) -> pd.DataFrame:
    """Daily return on capital over one period. Starts flat before the first day and closes after the
    last day; each change of state costs `cost` (fraction of notional) / CAP of capital.
    Held: funding / CAP (the short receives positive funding). Flat: T-bills."""
    st = state.astype(float)
    changes = (st - st.shift(1, fill_value=0.0)).abs()
    changes.iloc[-1] += st.iloc[-1]                           # exit at the end of the period
    carry = st * earned.fillna(0.0) / CAP
    cash = (1.0 - st) * rf
    fee = changes * cost / CAP
    out = pd.DataFrame({"state": st, "changes": changes, "carry": carry, "cash": cash, "cost": fee})
    out["gross"] = carry + cash
    out["total"] = out["gross"] - fee
    out["excess"] = out["total"] - rf
    out["gross_excess"] = out["gross"] - rf
    out["rf"] = rf
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def annualize(r: pd.Series) -> float:
    return float((1.0 + r).prod() ** (DAYS_PER_YEAR / len(r)) - 1.0)


def max_drawdown(r: pd.Series) -> float:
    w = (1.0 + r).cumprod()
    return float((w / w.cummax() - 1.0).min())


def describe(sim: pd.DataFrame) -> dict:
    ex = sim["excess"]
    m = (1.0 + sim["total"]).groupby(sim.index.to_period("M")).prod() - 1.0
    sd = ex.std(ddof=1)
    return {"days": int(len(sim)), "ann_return": annualize(sim["total"]), "ann_tbill": annualize(sim["rf"]),
            "ann_return_minus_tbill": annualize(sim["total"]) - annualize(sim["rf"]),
            "mean_daily_excess_x365": float(ex.mean() * DAYS_PER_YEAR),
            "vol": float(sd * np.sqrt(DAYS_PER_YEAR)),
            "sharpe": float(ex.mean() / sd * np.sqrt(DAYS_PER_YEAR)) if sd > 0 else float("nan"),
            "max_drawdown": max_drawdown(sim["total"]),
            "worst_month": float(m.min()), "worst_month_label": str(m.idxmin()),
            "pct_invested": float(sim["state"].mean()), "state_changes": int(sim["changes"].sum()),
            "gross_ann_return": annualize(sim["gross"]),
            "gross_mean_daily_excess_x365": float(sim["gross_excess"].mean() * DAYS_PER_YEAR),
            "cost_total_pct_of_capital": float(sim["cost"].sum())}


def block_bootstrap_mean(x, block: int = BLOCK, n: int = N_BOOT, seed: int = SEED,
                         level: float = CI_LEVEL) -> dict:
    """Moving-block bootstrap of the mean: block starts uniform on [0, T - block], truncated to T."""
    x = np.asarray(x, float)
    T = len(x)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T]
    means = x[idx].mean(axis=1)
    q = (1.0 - level) / 2.0
    return {"mean": float(x.mean()), "ci": [float(np.quantile(means, q)), float(np.quantile(means, 1 - q))],
            "level": level, "days": T, "block": block, "resamples": n, "seed": seed}


def margin_risk(spot: pd.DataFrame, index: pd.DatetimeIndex, state: pd.Series | None = None) -> dict:
    """Days where the highest high over the next 7 days is > 25% above today's close; flagged days
    less than 7 days apart are one episode."""
    hi, c = spot["high"], spot["close"]
    fwd = pd.concat([hi.shift(-j) for j in range(1, MARGIN_DAYS + 1)], axis=1).max(axis=1, skipna=False)
    rise = (fwd / c - 1.0).reindex(index)
    flagged = rise[rise > MARGIN_RISE]
    episodes, cur = [], None
    for t, v in flagged.items():
        if cur is None or (t - cur["last"]).days >= MARGIN_DAYS:
            cur = {"start": t, "last": t, "max_rise": v, "max_rise_date": t}
            episodes.append(cur)
        else:
            cur["last"] = t
            if v > cur["max_rise"]:
                cur["max_rise"], cur["max_rise_date"] = v, t
    out = {"episodes": len(episodes), "flagged_days": int(len(flagged)),
           "largest_7d_rise": float(rise.max()), "largest_7d_rise_date": str(rise.idxmax().date()),
           "missing_days": int(rise.isna().sum()),
           "list": [{"start": str(e["start"].date()), "end": str(e["last"].date()),
                     "max_rise": round(float(e["max_rise"]), 4)} for e in episodes]}
    if state is not None:
        out["episodes_started_while_c2_invested"] = int(sum(state.get(e["start"], 0.0) > 0 for e in episodes))
    return out


def resize_drag(close: pd.Series, state: pd.Series, cost: pd.Series) -> dict:
    """Report-only: times price moved > 25% from the last sizing while invested, x the cost of
    re-sizing 25% of notional on both legs (fraction of capital, per year)."""
    n, ref, drag = 0, None, 0.0
    prev = close.shift(1)
    for t in state.index:
        if state[t] <= 0:
            ref = None
            continue
        if ref is None:
            ref = prev.get(t, np.nan)
        p = close.get(t, np.nan)
        if np.isfinite(ref) and np.isfinite(p) and abs(p / ref - 1.0) > MARGIN_RISE:
            n += 1
            drag += MARGIN_RISE * cost[t] / CAP
            ref = p
    yrs = len(state) / DAYS_PER_YEAR
    return {"resizes": n, "drag_per_year": drag / yrs}


# ---------------------------------------------------------------------------
# the one run
# ---------------------------------------------------------------------------

def run_asset(f: pd.Series, spot: pd.DataFrame, irx: pd.Series, asset: str, venue: str,
              periods: dict) -> dict:
    idx = days(min(p[0] for p in periods.values()), max(p[1] for p in periods.values()))
    earned, sig, rf = earned_by_day(f, idx), signal(f, idx), tbill_daily(irx, idx)
    cost = switch_cost(venue, asset, spot["close"].shift(1).reindex(idx))
    out = {"first_valid_signal": str(sig.first_valid_index().date()) if sig.notna().any() else None}
    sims = {}
    for name, (a, b) in periods.items():
        p = days(a, b)
        st1 = earned.reindex(p).notna().astype(float)
        st2 = c2_state(sig.reindex(p))
        s1 = simulate(earned.reindex(p), rf.reindex(p), st1, cost.reindex(p))
        s2 = simulate(earned.reindex(p), rf.reindex(p), st2, cost.reindex(p))
        sims[name] = {"C1": s1, "C2": s2}
        boot = block_bootstrap_mean(s2["excess"].to_numpy())
        out[name] = {"C1": describe(s1), "C2": describe(s2),
                     "C2_mean_daily_excess_bootstrap": boot,
                     "funding_ann_mean": float(earned.reindex(p).mean() * DAYS_PER_YEAR),
                     "pct_days_signal_above_10pct": float((sig.reindex(p) > THRESHOLD).mean()),
                     "funding_days_negative": float((earned.reindex(p) < 0).mean()),
                     "margin_risk": margin_risk(spot, p, st2),
                     "resize_drag_C1": resize_drag(spot["close"], st1, cost.reindex(p)),
                     "resize_drag_C2": resize_drag(spot["close"], st2, cost.reindex(p))}
        y = s2.groupby(s2.index.year)
        out[name]["C2_by_year"] = {int(k): {"total": float((1 + g["total"]).prod() - 1),
                                            "tbill": float((1 + g["rf"]).prod() - 1),
                                            "invested": float(g["state"].mean())} for k, g in y}
        y1 = s1.groupby(s1.index.year)
        out[name]["C1_by_year"] = {int(k): {"total": float((1 + g["total"]).prod() - 1),
                                            "tbill": float((1 + g["rf"]).prod() - 1)} for k, g in y1}
    return out, sims


def us_extra(f: pd.Series, end: str = LAST_DAY) -> dict:
    """Report-only for a US venue's public history: mean annualized funding, % days S(d) > 10%,
    C2 gross funding return on capital (no costs, no T-bills)."""
    start = (f.index[0] + LOOKBACK).ceil("D")
    stop = min(pd.Timestamp(end), f.index[-1].floor("D") - pd.Timedelta(days=1))   # last fully covered day
    idx = days(str(start.date()), str(stop.date()))
    if len(idx) < 2:
        return {"days": 0}
    earned, sig = earned_by_day(f, idx), signal(f, idx)
    st = c2_state(sig)
    return {"first_event": str(f.index[0]), "last_event": str(f.index[-1]), "events": int(len(f)),
            "days": int(len(idx)), "funding_ann_mean": float(earned.mean() * DAYS_PER_YEAR),
            "pct_days_signal_above_10pct": float((sig > THRESHOLD).mean()),
            "c2_gross_funding_on_capital_ann": float((st * earned.fillna(0) / CAP).mean() * DAYS_PER_YEAR),
            "c1_gross_funding_on_capital_ann": float((earned.fillna(0) / CAP).mean() * DAYS_PER_YEAR)}


def main_run() -> dict:
    d = load()
    res = {"plan": "docs/carry_plan.md", "plan_sha256_at_writing": PLAN_SHA256,
           "run_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
           "data_fetched": {k: d["log"].get(k) for k in ("started", "finished")},
           "data_check": check(d), "binance": {}}
    for a in ASSETS:
        out, _ = run_asset(d["funding"][a], d["spot"][a], d["irx"], a, "binance", PERIODS)
        res["binance"][a] = out
    res["gate_binance"] = {}
    for a in ASSETS:
        b = res["binance"][a]["confirmation"]["C2_mean_daily_excess_bootstrap"]
        res["gate_binance"][a] = {"mean_daily_excess": b["mean"], "ci_97_5": b["ci"], "passes": bool(b["ci"][0] > 0)}
    cde = d["log"].get("coinbase_derivatives", {})
    res["coinbase"] = {"history_public": bool(cde.get("public")), "probe": cde}
    if not cde.get("public"):
        res["coinbase"]["verdict"] = "cannot be tested: funding history not public (endpoint requires an API key)"
    res["us_report_only"] = {}
    bn = DATA / "bitnomial_funding.parquet"
    if bn.exists():
        B = pd.read_parquet(bn)
        for (base, pid), g in B.groupby(["base_symbol", "product_id"]):
            f = g.set_index("interval_end")["rate"].sort_index()
            res["us_report_only"][f"bitnomial:{base}:{pid}"] = us_extra(f[~f.index.duplicated()])
    ks = DATA / "kalshi_funding.parquet"
    if ks.exists():
        Kf = pd.read_parquet(ks)
        for tk, g in Kf.groupby("market_ticker"):
            f = g.set_index("time")["rate"].sort_index()
            res["us_report_only"][f"kalshi:{tk}"] = us_extra(f[~f.index.duplicated()])
    res["verdict"] = {a: ("mechanism works on Binance" if res["gate_binance"][a]["passes"]
                          else "mechanism does not clear the gate on Binance") for a in ASSETS}
    res["verdict"]["implementable_us_retail"] = (res["coinbase"].get("verdict") or "see coinbase gate")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "carry.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "fetch":
        from carry.fetch import fetch_all
        print(json.dumps(fetch_all(), indent=2, default=str))
    elif cmd == "check":
        print(json.dumps(check(load()), indent=2, default=str))
    elif cmd == "run":
        res = main_run()
        print(json.dumps({k: v for k, v in res.items() if k not in ("data_check",)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
