"""One-shot test: does selling 0DTE iron flies on CPI/NFP/FOMC days beat other days?

Sealed NDX and RUT, 2025-01-01 .. 2026-09-04. Rule and gate are pre-registered in
docs/expiry_edge_plan.md (2026-11-09). No model is fitted: the dataset is built
with build_dataset + calendar/context features only, and only this rule is scored.

  python3 -m odte.eval.event_seller
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.eval.lockbox import LEDGER, _git, load_ledger

ROOT = Path(__file__).resolve().parents[2]
LO, HI = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-09-04")
DIRS = {"ndx": "data/odte_0dte_ndxp", "rut": "data/odte_0dte_rutw"}
EVENTS = ["ev_cpi", "ev_nfp", "ev_fomc", "ev_fomc_sep"]
FEES4, BUDGET = 4 * 1.20, 1000.0


def build(data_dir) -> pd.DataFrame:
    from odte.data import calendar as cal
    from odte.data import context as ctx
    from odte.eval.expiry_edge import DECISION_TIMES, build_dataset, load_early_closes, load_sessions
    chains, settles = load_sessions(ROOT / data_dir)
    chains = {d: c for d, c in chains.items() if LO <= pd.Timestamp(d) <= HI}
    ev = cal.load()
    timed = ev[ev["session_time"].notna()]
    event_times = {pd.Timestamp(d): sorted(set(g["session_time"])) for d, g in timed.groupby("session")}
    ds = build_dataset(chains, settles, times=DECISION_TIMES, event_times=event_times,
                       shocks=False, early_closes=load_early_closes())
    ds = ds[ds["trigger"] == "grid"].reset_index(drop=True)
    ds["day"] = pd.to_datetime(ds["day"])
    cf = pd.DataFrame([cal.features_for(ev, d, h) for d, h in zip(ds["day"], ds["hhmm"])], index=ds.index)
    wide = ctx.load()
    per = {d: ctx.features_for(wide, d) for d in ds["day"].unique()}
    xf = pd.DataFrame([per[d] for d in ds["day"]], index=ds.index)
    return pd.concat([ds, cf[EVENTS], xf[["vix_over_vix3m"]]], axis=1)


def score(ds: pd.DataFrame) -> pd.DataFrame:
    d = ds[np.isfinite(ds["pnl_short_fly_mid"]) & (ds["fly_credit_mid"] > 0) & np.isfinite(ds["W"])].copy()
    cm, cr, W = d["fly_credit_mid"].to_numpy(), d["fly_credit"].to_numpy(), d["W"].to_numpy()
    intrinsic = cm - d["pnl_short_fly_mid"].to_numpy() / 100
    for name, c in (("mid", cm), ("half", cm - 0.5 * (cm - cr)), ("touch", cr)):
        risk = (W - c) * 100 + FEES4
        d[name] = np.where(risk > 0, ((c - intrinsic) * 100 - FEES4) / risk * BUDGET, np.nan)
    post = (d["min_since_event"] <= 60).fillna(False).to_numpy()
    stress = (d["vix_over_vix3m"] > 1.0).fillna(False).to_numpy()
    d["eligible"] = ~post & ~stress
    d["event_day"] = d[EVENTS].sum(axis=1).to_numpy() > 0
    return d[d["eligible"]]


def boot_diff(a: pd.Series, b: pd.Series, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    A, B = a.to_numpy(), b.to_numpy()
    bs = [A[rng.integers(0, len(A), len(A))].mean() - B[rng.integers(0, len(B), len(B))].mean() for _ in range(n)]
    return float(A.mean() - B.mean()), [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def main() -> int:
    led = load_ledger()
    keys = {("odte_0dte_ndxp", "2025-01..2026-09"), ("odte_0dte_rutw", "2025-01..2026-09")}
    if any((e["data"], e["block"]) in keys for e in led):
        raise SystemExit("REFUSED: NDX/RUT 2025-26 already opened")
    if _git("status", "--porcelain", "--", "odte/eval", "odte/data"):
        raise SystemExit("REFUSED: uncommitted strategy code")
    parts = {}
    for k, dd in DIRS.items():
        parts[k] = score(build(dd))
        print(f"{k}: {parts[k]['day'].nunique()} days, {len(parts[k]):,} eligible decisions", flush=True)
    res, pooled_ev, pooled_ot = {}, [], []
    for k, d in parts.items():
        r = {}
        for px in ("mid", "half", "touch"):
            daily = d.groupby(["day", "event_day"])[px].sum().reset_index()
            e, o = daily[daily["event_day"]][px], daily[~daily["event_day"]][px]
            diff, ci = boot_diff(e, o)
            r[px] = {"event_days": int(len(e)), "other_days": int(len(o)), "event_total": float(e.sum()),
                     "event_mean_day": float(e.mean()), "other_mean_day": float(o.mean()),
                     "diff_mean_day": diff, "diff_ci": ci}
            if px == "mid":
                pooled_ev.append(e)
                pooled_ot.append(o)
        res[k] = r
    pdiff, pci = boot_diff(pd.concat(pooled_ev), pd.concat(pooled_ot))
    res["pooled_mid"] = {"diff_mean_day": pdiff, "diff_ci": pci}
    res["verdict"] = "PASS" if (pci[0] > 0 and all(res[k]["mid"]["diff_mean_day"] > 0 for k in DIRS)) else "FAIL"
    (ROOT / "reports" / "event_seller_sealed.json").write_text(json.dumps(res, indent=2, default=float))
    now, commit = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), _git("rev-parse", "--short", "HEAD")
    for data in ("odte_0dte_ndxp", "odte_0dte_rutw"):
        led.append({"data": data, "block": "2025-01..2026-09", "start": f"{LO:%Y-%m-%d}", "end": f"{HI:%Y-%m-%d}",
                    "opened_at": now, "commit": commit, "strategies": ["event-day seller"],
                    "results_event_seller": res, "forced": None})
    LEDGER.write_text(json.dumps(led, indent=2, default=str))
    for k in DIRS:
        print(f"\n== {k.upper()} (each trade sized to $1k worst case)")
        for px, v in res[k].items():
            print(f"  {px:5s}: event days {v['event_days']:3d}: total ${v['event_total']:>9,.0f}, "
                  f"${v['event_mean_day']:+.0f}/day | other days {v['other_days']:3d}: ${v['other_mean_day']:+.0f}/day | "
                  f"difference ${v['diff_mean_day']:+.0f}/day [{v['diff_ci'][0]:+.0f}, {v['diff_ci'][1]:+.0f}]")
    print(f"\npooled difference at mid: ${pdiff:+.0f}/day [{pci[0]:+.0f}, {pci[1]:+.0f}]")
    print(f"VERDICT (pre-registered): {res['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
