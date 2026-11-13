"""N1: delta-hedged P&L as a low-noise yardstick (docs/execution_plan.md, "N1 clarification").

Selling a 0DTE structure and holding it to the close mixes two things: was the option overpriced
(the edge we want to measure), and where the index happened to land (mostly noise). Hedging the
structure's delta every minute with the forward removes most of the second, leaving "implied vs
realised movement".

    python3 -m odte.research.noise_hedged            # primary: the 61 sessions of data B
    python3 -m odte.research.noise_hedged --all      # secondary: the whole development history
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.special import ndtr as _ncdf

from odte.research import library, zoo

OUT = zoo.ROOT / "reports" / "execution"
SHAPES = ("fly", "condor", "put_spread", "call_spread")
PICKS = zoo.ROOT / "reports" / "round2" / "b0_spx_picks_1s_window.parquet"
HHMM = [f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)]


def _black(F, K, s, call):
    """Black-76 undiscounted price with total vol s = sigma * sqrt(tau)."""
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    c = F * _ncdf(d1) - K * _ncdf(d2)
    return np.where(call, c, c - (F - K))


def implied_total_vol(price, F, K, call, lo=1e-5, hi=0.5, it=60):
    """Vectorised bisection for s = sigma*sqrt(tau). NaN where the mid is outside no-arbitrage bounds."""
    price, F, K, call = map(np.asarray, (price, F, K, call))
    intrinsic = np.where(call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    ok = np.isfinite(price) & np.isfinite(F) & (price > intrinsic + 1e-6)
    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    for _ in range(it):
        m = 0.5 * (a + b)
        high = _black(F, K, m, call) > price
        b = np.where(high, m, b)
        a = np.where(high, a, m)
    return np.where(ok, 0.5 * (a + b), np.nan)


def delta(F, K, s, call):
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    n = _ncdf(d1)
    return np.where(call, n, n - 1.0)


def _tau(minutes: pd.DatetimeIndex) -> np.ndarray:
    """Years to 16:00 (any fixed convention works: only s = sigma*sqrt(tau) enters the price)."""
    left = ((minutes.normalize() + pd.Timedelta(hours=16)) - minutes).total_seconds() / 60.0
    return np.maximum(left, 0.5) / (365.0 * 1440.0)


def score_day(day: pd.Timestamp, settle: float, hhmms=HHMM) -> list[dict]:
    ch = pd.read_parquet(zoo.CHAINS / f"{day:%Y-%m-%d}.parquet")
    d = zoo.Day(day, ch, settle, np.nan, {})
    Fser = d.F.reindex(d.minutes).ffill()
    rows = []
    for hhmm in hhmms:
        t = d.at(hhmm)
        if t not in d.minutes:
            continue
        F0 = d.forward(t)
        _, st = d.straddle(t, F0)
        if not (np.isfinite(st) and st > 0):
            continue
        sigma = st / 0.8
        win = d.minutes[(d.minutes >= t) & (d.minutes <= d.at("15:59"))]
        Fw = Fser.loc[win].to_numpy(float)
        if not np.isfinite(Fw).all():
            continue
        dF = np.append(Fw[1:], settle) - Fw                    # forward change over each hedge interval
        tau = _tau(win)
        for name in SHAPES:
            legs = zoo.legs_for(library.TRADES[name], d, t, F0, sigma, None)
            if not legs:
                continue
            try:
                V = sum(q * float(d.mid.loc[t, (r, k)]) for r, k, q in legs)
            except KeyError:
                continue
            if not np.isfinite(V):
                continue
            n = sum(abs(q) for _, _, q in legs)
            lo, bounded = zoo.worst_payoff(legs, F0)
            risk = -((-V) * 100 - n * zoo.FEE + lo * 100)
            if not (bounded and risk > 0):
                continue
            pnl = (-V + zoo.payoff(legs, settle)) * 100 - n * zoo.FEE
            pos_delta = np.zeros(len(win))
            for r, k, q in legs:
                m = d.mid.loc[win, (r, k)].to_numpy(float) if (r, k) in d.mid.columns else np.full(len(win), np.nan)
                s = implied_total_vol(m, Fw, np.full(len(win), k), np.full(len(win), r == "C"))
                sig = pd.Series(s / np.sqrt(tau)).ffill().bfill().to_numpy()     # carry sigma, not s
                if not np.isfinite(sig).any():
                    sig = np.full(len(win), sigma / Fw[0] / np.sqrt(_tau(pd.DatetimeIndex([t]))[0]))
                s_now = np.maximum(sig * np.sqrt(tau), 1e-6)
                pos_delta += q * delta(Fw, k, s_now, r == "C")
            hedge = float(np.sum(-pos_delta * dF)) * 100
            rows.append({"day": day, "hhmm": hhmm, "shape": name, "risk": risk,
                         "U": pnl / risk * zoo.BUDGET, "H": (pnl + hedge) / risk * zoo.BUDGET,
                         "credit": -V, "entry_delta": float(pos_delta[0])})
    return rows


def _job(a):
    try:
        return score_day(*a)
    except Exception as e:  # noqa: BLE001
        print(f"{a[0]:%Y-%m-%d}: {e}", file=sys.stderr)
        return []


def run(days, settle, workers=8) -> pd.DataFrame:
    jobs = [(d, float(settle[d])) for d in days if d in settle.index]
    with ProcessPoolExecutor(workers) as ex:
        out = [r for rows in ex.map(_job, jobs, chunksize=4) for r in rows]
    return pd.DataFrame(out)


def boot_diff(x: pd.DataFrame, col: str, rng, n=5000) -> tuple:
    """mean(col | picked) - mean(col | not picked), with a 90% day-bootstrap CI."""
    days = np.unique(x["day"])
    pk, v = x["picked"].astype(bool), x[col].astype(float)
    by = lambda s: s.groupby(x["day"]).sum().reindex(days).to_numpy(float)
    sp, npk = by(v.where(pk, 0.0)), by(pk.astype(float))
    su, nu = by(v.where(~pk, 0.0)), by((~pk).astype(float))
    est = sp.sum() / npk.sum() - su.sum() / nu.sum()
    b = rng.integers(0, len(days), (n, len(days)))
    bs = sp[b].sum(1) / npk[b].sum(1) - su[b].sum(1) / nu[b].sum(1)
    lo, hi = np.percentile(bs, [5, 95])
    return float(est), [float(lo), float(hi)]


def summarise(x: pd.DataFrame, rng) -> dict:
    res = {"rows": int(len(x)), "days": int(x["day"].nunique()),
           "std_U": float(x["U"].std()), "std_H": float(x["H"].std()),
           "noise_ratio": float(x["U"].std() / x["H"].std()),
           "mean_U": float(x["U"].mean()), "mean_H": float(x["H"].mean()),
           "corr_U_H": float(x[["U", "H"]].corr().iloc[0, 1]),
           "by_shape": {s: {"n": int(len(z)), "std_U": float(z["U"].std()), "std_H": float(z["H"].std()),
                            "ratio": float(z["U"].std() / z["H"].std()),
                            "mean_U": float(z["U"].mean()), "mean_H": float(z["H"].mean())}
                        for s, z in x.groupby("shape")}}
    if "picked" in x and x["picked"].any():
        x = x.assign(D=x["U"] - x["H"])               # the part the hedge removed (where the index went)
        for col in ("U", "H", "D"):
            est, ci = boot_diff(x, col, rng)
            res[f"skill_{col}"] = {"picked_minus_not": est, "ci90": ci, "ci_width": ci[1] - ci[0]}
        res["skill_ci_width_ratio"] = res["skill_U"]["ci_width"] / res["skill_H"]["ci_width"]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="whole development history (secondary)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(zoo.CHAINS / "settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
    rng = np.random.default_rng(20261112)
    if a.all:
        from odte.research import exam_guard as eg
        days = sorted(pd.Timestamp(p.stem) for p in zoo.CHAINS.glob("20*.parquet")
                      if pd.Timestamp(p.stem) <= pd.Timestamp(eg.dev_end()))
        x = run(days, st, a.workers)
        x.to_parquet(OUT / "n1_hedged_all.parquet", index=False)
        res = summarise(x, rng)
        x["year"] = pd.to_datetime(x["day"]).dt.year
        res["by_year"] = {int(y): {"std_U": float(z["U"].std()), "std_H": float(z["H"].std()),
                                   "ratio": float(z["U"].std() / z["H"].std()), "mean_H": float(z["H"].mean())}
                          for y, z in x.groupby("year")}
        (OUT / "n1_hedged_all.json").write_text(json.dumps(res, indent=2))
    else:
        picks = pd.read_parquet(PICKS)
        picks["day"] = pd.to_datetime(picks["day"])
        x = run(sorted(picks["day"].unique()), st, a.workers)
        x = x.merge(picks, on=["day", "hhmm"], how="left")
        x["picked"] = x["trade"] == x["shape"]
        x.to_parquet(OUT / "n1_hedged.parquet", index=False)
        res = summarise(x, rng)
        res["decision"] = {"noise_ratio": res["noise_ratio"], "rule": ">= 3 -> secondary metric",
                           "adopt": bool(res["noise_ratio"] >= 3)}
        (OUT / "n1_hedged.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_year",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
