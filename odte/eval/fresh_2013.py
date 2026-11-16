"""One-shot fresh-data test of the edge-search hypotheses H1-H4 on SPX 0DTE 2013-04 .. 2015-12.

Pre-registered in docs/expiry_edge_plan.md (2026-11-09) before this data was
bought. Nothing here is fitted: every hypothesis is a fixed rule.

  H1  news-day seller: event days (CPI/NFP/FOMC) minus other days, selling the
      iron fly at the fixed times outside the hour after a release and outside
      stress (yesterday VIX/VIX3M > 1). Mechanism check, stated in advance:
      PPI / jobless-claims days should be no better than ordinary days.
  H2  afternoon seller: per day, (13:00 + 14:00) minus (10:00 + 11:00) fly P&L.
  H3  stress filter: selling on stress days loses (CI entirely below 0).
  H4  intraday momentum: sign(09:30-10:00 move) x (15:30 -> close move) > 0 on the
      index; the 0DTE option trade (buy an ATM call/put at 15:30 at the ask, hold
      to the close) is reported.

Pricing for the fly: today's cost (half of a 1.69%-of-mid gap; touch reported),
$1k worst case per trade. Gate: predicted sign with a 98.75% day-resampled CI
excluding 0 (Bonferroni over four). Possible early-close days (day after
Thanksgiving, Dec 24, Jul 3) are dropped: no verified 2013-15 early-close list.

  python3 -m odte.eval.fresh_2013
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.eval.lockbox import LEDGER, _git, load_ledger

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "odte_0dte_pre2016"
TODAY_COST, FEES4, BUDGET = 0.0169, 4 * 1.20, 1000.0
Q = (0.00625, 0.99375)                    # 98.75% two-sided
EVENTS = ["ev_cpi", "ev_nfp", "ev_fomc", "ev_fomc_sep"]


def maybe_early_close(day: pd.Timestamp) -> bool:
    if day.month == 7 and day.day == 3:
        return True
    if day.month == 12 and day.day == 24:
        return True
    return day.month == 11 and day.dayofweek == 4 and 23 <= day.day <= 29   # Friday after Thanksgiving


def build() -> tuple:
    from odte.data import calendar as cal
    from odte.data import context as ctx
    from odte.eval.expiry_edge import DECISION_TIMES, build_dataset, load_sessions
    chains, settles = load_sessions(DATA)
    chains = {d: c for d, c in chains.items() if not maybe_early_close(pd.Timestamp(d))}
    ev = cal.load()
    timed = ev[ev["session_time"].notna()]
    event_times = {pd.Timestamp(d): sorted(set(g["session_time"])) for d, g in timed.groupby("session")}
    ds = build_dataset(chains, settles, times=DECISION_TIMES, event_times=event_times, shocks=False)
    ds = ds[ds["trigger"] == "grid"].reset_index(drop=True)
    ds["day"] = pd.to_datetime(ds["day"])
    cf = pd.DataFrame([cal.features_for(ev, d, h) for d, h in zip(ds["day"], ds["hhmm"])], index=ds.index)
    wide = ctx.load()
    per = {d: ctx.features_for(wide, d) for d in ds["day"].unique()}
    xf = pd.DataFrame([per[d] for d in ds["day"]], index=ds.index)
    ds = pd.concat([ds, cf[EVENTS + ["ev_ppi", "ev_claims"]], xf[["vix_over_vix3m"]]], axis=1)
    return ds, chains, settles


def fly_pnl(ds: pd.DataFrame) -> pd.DataFrame:
    d = ds[np.isfinite(ds["pnl_short_fly_mid"]) & (ds["fly_credit_mid"] > 0) & np.isfinite(ds["W"])].copy()
    cm, W = d["fly_credit_mid"].to_numpy(), d["W"].to_numpy()
    intrinsic = cm - d["pnl_short_fly_mid"].to_numpy() / 100
    for name, gap in (("half", 0.5 * TODAY_COST), ("touch", TODAY_COST)):
        c = cm * (1 - gap)
        risk = (W - c) * 100 + FEES4
        d[name] = np.where(risk > 0, ((c - intrinsic) * 100 - FEES4) / risk * BUDGET, np.nan)
    d["post"] = (d["min_since_event"] <= 60).fillna(False).to_numpy()
    d["stress"] = (d["vix_over_vix3m"] > 1.0).fillna(False).to_numpy()
    return d


def ci_mean(x, n=8000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), [float("nan")] * 2, len(x)
    rng = np.random.default_rng(seed)
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), [float(np.quantile(bs, Q[0])), float(np.quantile(bs, Q[1]))], len(x)


def ci_diff(a, b, n=8000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = np.array([a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
                   for _ in range(n)])
    return float(a.mean() - b.mean()), [float(np.quantile(bs, Q[0])), float(np.quantile(bs, Q[1]))]


def h4(chains, settles) -> dict:
    """Index momentum and the 15:30 option trade, from each session's minute quotes."""
    from odte.eval.expiry_edge import implied_forward, minute_grid
    caps, opt = [], []
    for day, chain in sorted(chains.items()):
        day = pd.Timestamp(day).normalize()
        s = settles.get(day)
        if s is None or not np.isfinite(s):
            continue
        g = minute_grid(chain)
        F = implied_forward(g).dropna()
        t10, t1530 = day + pd.Timedelta(hours=10), day + pd.Timedelta(hours=15, minutes=30)
        early = F[F.index <= day + pd.Timedelta(hours=9, minutes=35)]
        if early.empty or t10 not in F.index or t1530 not in F.index:
            continue
        r1 = np.log(F.at[t10] / early.iloc[0])
        rl = np.log(s / F.at[t1530])
        if r1 == 0:
            continue
        caps.append(np.sign(r1) * rl * 1e4)
        right = "C" if r1 > 0 else "P"
        ks = [k for (r, k) in g["ask"].columns if r == right]
        if not ks:
            continue
        K = min(ks, key=lambda k: abs(k - F.at[t1530]))
        ask = g["ask"].at[t1530, (right, K)]
        if not np.isfinite(ask) or ask <= 0:
            continue
        payoff = max(s - K, 0.0) if right == "C" else max(K - s, 0.0)
        opt.append((payoff - ask) * 100 - 1.20)
    m, ci, n = ci_mean(caps)
    om, oci, on = ci_mean(opt)
    return {"index_capture_bp": m, "ci": ci, "days": n, "option_trade_per_contract": om, "option_ci": oci,
            "option_days": on, "passes": bool(ci[0] > 0)}


def main() -> int:
    led = load_ledger()
    if any(e["data"] == "odte_0dte_pre2016" for e in led):
        raise SystemExit("REFUSED: 2013-15 data already opened")
    if _git("status", "--porcelain", "--", "odte/eval", "odte/data"):
        raise SystemExit("REFUSED: uncommitted strategy code")
    ds, chains, settles = build()
    d = fly_pnl(ds)
    print(f"{d['day'].nunique()} sessions, {len(d):,} fly decisions", flush=True)
    res = {"sessions": int(d["day"].nunique())}

    # H1: event days vs other days (eligible = not post-release, not stress)
    e = d[~d["post"] & ~d["stress"]]
    ev_day = e[EVENTS].sum(axis=1) > 0
    lowkey = (e[["ev_ppi", "ev_claims"]].sum(axis=1) > 0) & ~ev_day
    h1 = {}
    for px in ("half", "touch"):
        daily = e.groupby(["day"])[px].sum()
        kind = e.groupby("day").apply(lambda g: "event" if (g[EVENTS].sum(axis=1) > 0).any() else
                                      ("lowkey" if (g[["ev_ppi", "ev_claims"]].sum(axis=1) > 0).any() else "other"))
        ev_d, ot_d, lk_d = daily[kind == "event"], daily[kind == "other"], daily[kind == "lowkey"]
        diff, ci = ci_diff(ev_d, ot_d)
        lk = ci_diff(lk_d, ot_d) if len(lk_d) >= 3 else (float("nan"), [float("nan")] * 2)
        h1[px] = {"event_days": int(len(ev_d)), "other_days": int(len(ot_d)), "lowkey_days": int(len(lk_d)),
                  "event_mean": float(ev_d.mean()) if len(ev_d) else float("nan"),
                  "other_mean": float(ot_d.mean()) if len(ot_d) else float("nan"),
                  "diff": diff, "ci": ci, "lowkey_minus_other": lk[0], "lowkey_ci": lk[1]}
    h1["passes"] = bool(h1["half"]["ci"][0] > 0)
    res["H1"] = h1

    # H2: afternoon minus morning, per day
    e2 = d[~d["post"]]
    piv = e2.pivot_table(index="day", columns="hhmm", values="half", aggfunc="sum")
    if {"10:00", "11:00", "13:00", "14:00"} <= set(piv.columns):
        x = (piv["13:00"] + piv["14:00"] - piv["10:00"] - piv["11:00"]).dropna()
        m, ci, n = ci_mean(x)
        res["H2"] = {"afternoon_minus_morning": m, "ci": ci, "days": n, "passes": bool(ci[0] > 0)}

    # H3: stress days lose
    st = d[d["stress"] & ~d["post"]].groupby("day")["half"].sum()
    m, ci, n = ci_mean(st)
    res["H3"] = {"stress_mean_day": m, "ci": ci, "days": n, "passes": bool(n >= 3 and ci[1] < 0)}

    # H4: intraday momentum
    res["H4"] = h4(chains, settles)
    res["passed"] = [h for h in ("H1", "H2", "H3", "H4") if res.get(h, {}).get("passes")]
    (ROOT / "reports" / "fresh_2013.json").write_text(json.dumps(res, indent=2, default=float))
    led.append({"data": "odte_0dte_pre2016", "block": "2013-04..2015-12", "start": "2013-04-01",
                "end": "2015-12-31", "opened_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "commit": _git("rev-parse", "--short", "HEAD"), "strategies": ["H1", "H2", "H3", "H4"],
                "results_fresh_2013": res, "forced": None})
    LEDGER.write_text(json.dumps(led, indent=2, default=str))
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
