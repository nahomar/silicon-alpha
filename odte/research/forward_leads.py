"""Forward check (docs/sweep_plan.md, "Forward check"): B7, B9, B16 and A1 on NYSE days from 2026-11-16,
scored from free daily SPX closes. Tracking only: it touches nothing the bot uses.

    python3 -m odte.research.forward_leads baseline   # once: backtest daily net P&L -> reports/forward/baseline.parquet
    python3 -m odte.research.forward_leads            # any evening: score forward days, print the next day's positions
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research.partner_t import nyse_days

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "forward"
CTX = ROOT / "data" / "context" / "daily.parquet"
START = pd.Timestamp("2026-11-16")
CHECK_FROM = pd.Timestamp("2026-10-01")       # closes must have no gaps from here on
MES, HALF, CARRY, SEED, DRAWS = 5.0, 0.375, 0.025, 20261115, 10000
POS_IDEAS = ("B7", "B9", "B16")
IDEAS = POS_IDEAS + ("A1",)
DESC = {"B7": "three down closes -> long next day", "B9": "RSI(2) < 10 -> long next day",
        "B16": "move > 2 sd of last 20 days -> fade it next day", "A1": "long Monday 09:30 -> 16:00"}


def signals(x: pd.Series) -> pd.DataFrame:
    """Position for the NEXT trading day, decided at each close. Same formulas as sweep1.fam_B (before its shift(1))."""
    r = x.pct_change()
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    big = r.abs() > 2 * r.rolling(20).std().shift(1)
    return pd.DataFrame({"B7": ((r < 0) & (r.shift(1) < 0) & (r.shift(2) < 0)).astype(float),
                         "B9": (rsi < 10).astype(float),
                         "B16": (-np.sign(r)).where(big, 0.0)}, index=x.index)


def closes() -> tuple[pd.Series, pd.DatetimeIndex, pd.DatetimeIndex]:
    ctx = pd.read_parquet(CTX)
    ctx.index = pd.DatetimeIndex(pd.to_datetime(ctx.index).date)
    x = ctx[~ctx.index.duplicated(keep="last")]["spx"].dropna().sort_index()
    cal = nyse_days("2012-01-01", "2027-12-31")
    x = x[x.index.isin(cal)]
    span = cal[(cal >= CHECK_FROM) & (cal <= x.index[-1])]
    return x, cal, span.difference(x.index)


def baseline() -> int:
    """Backtest daily net $ per MES for each idea, exactly as sweep 1 booked it, plus a check that this file's
    signals give the sweep's held-day counts."""
    from odte.research.sweep1 import FIRST, END, SP, fam_A, fam_B
    OUT.mkdir(parents=True, exist_ok=True)
    sp = SP()
    got = {i["id"]: i for i in fam_A(sp) + fam_B(sp) if i["id"] in IDEAS}
    base = pd.DataFrame({k: got[k]["g"] - got[k]["c"] for k in IDEAS})
    base.to_parquet(OUT / "baseline.parquet")
    x, _, _ = closes()
    pos = signals(x).shift(1)
    pos = pos[(pos.index >= FIRST) & (pos.index <= END)]
    check = {k: {"sweep_held_days": got[k]["n1"] + got[k]["n2"], "this_file": int((pos[k] != 0).sum())} for k in POS_IDEAS}
    print(json.dumps({"days": len(base), "usd_per_mes_per_year": (base.mean() * 252).round(0).to_dict(),
                      "held_day_check": check}, indent=1))
    return 0


def a1_sessions(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Yahoo ^GSPC open and close on the given days (A1 Mondays)."""
    if len(days) == 0:
        return pd.DataFrame(columns=["open", "close"])
    import yfinance as yf
    h = yf.Ticker("^GSPC").history(start=days.min().strftime("%Y-%m-%d"),
                                    end=(days.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), auto_adjust=False)
    h.index = pd.DatetimeIndex(h.index.date)
    return h[["Open", "Close"]].rename(columns={"Open": "open", "Close": "close"}).reindex(days)


def forward() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    x, cal, missing = closes()
    last = x.index[-1]
    if len(missing):
        print(f"WARNING: no SPX close for {[d.date().isoformat() for d in missing]}; scoring stops before the first one")
        last = min(last, cal[cal < missing[0]][-1])
    fdays = cal[(cal >= START) & (cal <= last)]
    sig = signals(x)
    rows = []
    # position ideas: the forward book starts flat at the 2026-11-13 close
    for k in POS_IDEAS:
        prev_p = 0.0
        for d in fdays:
            i = x.index.get_loc(d)
            p0, p1 = x.iloc[i - 1], x.iloc[i]
            p = float(sig[k].iloc[i - 1])
            carry = p * p0 * CARRY * (d - x.index[i - 1]).days / 365
            gross = p * (p1 - p0) - carry
            cost = abs(p - prev_p) * HALF
            rows.append({"day": d, "idea": k, "pos": p, "gross_pts": gross, "cost_pts": cost,
                         "net_usd": (gross - cost) * MES})
            prev_p = p
    mondays = fdays[fdays.dayofweek == 0]
    s = a1_sessions(mondays)
    for d in fdays:
        if d.dayofweek != 0:
            rows.append({"day": d, "idea": "A1", "pos": 0.0, "gross_pts": 0.0, "cost_pts": 0.0, "net_usd": 0.0})
            continue
        o, c = s.at[d, "open"], s.at[d, "close"]
        if not (np.isfinite(o) and np.isfinite(c)):
            print(f"WARNING: no ^GSPC open/close for {d.date()}; A1 not scored that day")
            continue
        gross = (c - o) - o * CARRY * (6.5 / 24) / 365
        rows.append({"day": d, "idea": "A1", "pos": 1.0, "gross_pts": gross, "cost_pts": 2 * HALF,
                     "net_usd": (gross - 2 * HALF) * MES})
    f = pd.DataFrame(rows, columns=["day", "idea", "pos", "gross_pts", "cost_pts", "net_usd"])
    f.to_csv(OUT / "forward.csv", index=False)

    base = pd.read_parquet(OUT / "baseline.parquet")
    summary = {}
    for k in IDEAS:
        z = f[f["idea"] == k]
        n = len(z)
        out = {"what": DESC[k], "forward_days": n, "held_days": int((z["pos"] != 0).sum()),
               "total_usd_per_mes": round(float(z["net_usd"].sum()), 1)}
        if n:
            draws = np.random.default_rng(SEED).choice(base[k].to_numpy(), size=(DRAWS, n)).sum(axis=1)
            p5 = float(np.percentile(draws, 5))
            out |= {"backtest_median_for_same_days": round(float(np.median(draws)), 1), "fifth_pct": round(p5, 1),
                    "verdict": "BROKEN" if out["total_usd_per_mes"] < p5 else "CONSISTENT"}
            v = z["net_usd"].to_numpy()
            if n >= 3 and v.std(ddof=1) > 0:
                out["t_vs_zero_report_only"] = round(float(v.mean() / v.std(ddof=1) * np.sqrt(n)), 2)
        summary[k] = out
    nxt = cal[cal > x.index[-1]][0]
    tomorrow = {k: float(sig[k].iloc[-1]) for k in POS_IDEAS} | {"A1": 1.0 if nxt.dayofweek == 0 else 0.0}
    res = {"last_close": x.index[-1].date().isoformat(), "scored_through": last.date().isoformat() if len(fdays) else None,
           "ideas": summary, "next_day": nxt.date().isoformat(), "positions_next_day": tomorrow}
    (OUT / "forward.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(baseline() if sys.argv[1:] == ["baseline"] else forward())
