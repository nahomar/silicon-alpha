"""E1 + E4 (docs/execution_plan.md): what the bot's fly costs on SPX vs XSP vs SPY, and the locked
model's fly picks priced natively on each (zoo.trade, quoted spreads, k_day = 1).

  python3 -m odte.research.e1_costs     # -> reports/execution/e1_costs.parquet / .json
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "execution"
TIMES = tuple(f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30))
CHAINS = {"SPX": ROOT / "data" / "odte_0dte", "XSP": ROOT / "data" / "odte_xsp", "SPY": ROOT / "data" / "odte_etf" / "SPY"}
FLY = {"structure": "iron_fly", "wing": 1.5}


def _settles() -> dict:
    spx = pd.read_csv(ROOT / "data/odte_0dte/settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
    spy = pd.read_csv(ROOT / "data/odte_etf/SPY/settlements.csv", parse_dates=["date"]).set_index("date")["close"]
    return {"SPX": spx, "XSP": spx / 10.0, "SPY": spy}


def one(args):
    day, b0 = args
    from odte.live import fly
    from odte.live.guards import parity_price
    from odte.research import zoo
    st = _settles()
    rows = []
    for sym, folder in CHAINS.items():
        f = folder / f"{day:%Y-%m-%d}.parquet"
        if not f.exists() or day not in st[sym].index:
            continue
        ch = pd.read_parquet(f)
        ch["quote_datetime"] = pd.to_datetime(ch["quote_datetime"])
        ch = ch[(ch["bid"] > 0) & (ch["ask"] >= ch["bid"])].sort_values("quote_datetime")
        try:
            zd = zoo.Day(day, ch, float(st[sym][day]), np.nan, {})
        except Exception:
            zd = None
        for hh in TIMES:
            t = day + pd.Timedelta(hours=int(hh[:2]), minutes=int(hh[3:]))
            snap = ch[ch["quote_datetime"] <= t].groupby(["option_type", "strike"]).tail(1)
            if snap.empty:
                continue
            F = parity_price(snap)
            fl = fly.fly_from_quotes(snap, F) if np.isfinite(F) else None
            if not fl or not (fl["fly_credit_mid"] > 0) or fl["W"] <= fl["fly_credit_mid"]:
                continue
            risk = fl["W"] - fl["fly_credit_mid"]
            gap = fl["fly_credit_mid"] - fl["fly_credit"]
            rec = {"day": day, "hhmm": hh, "sym": sym, "W_over_F": fl["W"] / F, "gap_rel": gap / fl["fly_credit_mid"],
                   "cost_half_per_1k": 0.5 * gap / risk * 1000, "cost_touch_per_1k": gap / risk * 1000,
                   "worst_case_per_contract": risk * 100, "b0_fly": (hh in b0)}
            if (hh in b0) and zd is not None:
                for share, tag in ((0.5, "half"), (1.0, "touch")):
                    tr = zoo.trade(FLY, zd, hh, 1.0, share)
                    rec[f"ret_{tag}"] = float(tr["ret"]) if tr else np.nan
            rows.append(rec)
    return rows


def main() -> int:
    import pickle
    from odte.live import frozen
    from odte.live.config import Config
    from odte.research import meta
    OUT.mkdir(parents=True, exist_ok=True)
    b = pickle.loads(frozen.model_path(Config()).read_bytes())
    d = meta.load()
    d = d[(d["index"] == "SPX") & (d["day"] >= "2024-11-11")]
    picks = {}
    for _, r in d.iterrows():
        if frozen.choose(frozen.predict_row(b, r.to_dict()), bool(r["post"]))[0] == "fly":
            picks.setdefault(r["day"], set()).add(r["hhmm"])
    nd = pd.read_parquet(ROOT / "reports/round2/newdays.parquet")
    for _, r in nd[(nd["index"] == "SPX") & (nd["B0_trade"] == "fly")].iterrows():
        picks.setdefault(pd.Timestamp(r["day"]), set()).add(r["hhmm"])
    days = sorted({pd.Timestamp(p.stem) for p in CHAINS["XSP"].glob("20??-??-??.parquet")})
    with ProcessPoolExecutor(10) as ex:
        rows = [x for part in ex.map(one, [(dd, picks.get(dd, set())) for dd in days], chunksize=4) for x in part]
    x = pd.DataFrame(rows)
    x.to_parquet(OUT / "e1_costs.parquet", index=False)
    res = {"cost_by_sym": x.groupby("sym")[["gap_rel", "cost_half_per_1k", "cost_touch_per_1k", "worst_case_per_contract"]].median().round(2).to_dict(orient="index"),
           "cost_half_by_time": x.pivot_table(index="hhmm", columns="sym", values="cost_half_per_1k", aggfunc="median").round(2).to_dict(),
           "cost_half_by_year": x.assign(y=x["day"].dt.year).pivot_table(index="y", columns="sym", values="cost_half_per_1k", aggfunc="median").round(2).to_dict()}
    # the locked model's fly picks on all three, same decisions (paired)
    p = x[x["b0_fly"] & x["ret_half"].notna()].pivot_table(index=["day", "hhmm"], columns="sym", values=["ret_half", "ret_touch"])
    p = p.dropna()
    rng = np.random.default_rng(5)
    res["b0_fly_picks"] = {"decisions": int(len(p)), "days": int(p.index.get_level_values(0).nunique())}
    for tag in ("half", "touch"):
        for s in ("SPX", "XSP", "SPY"):
            v = p[(f"ret_{tag}", s)].to_numpy()
            bt = v[rng.integers(0, len(v), (5000, len(v)))].mean(axis=1)
            res["b0_fly_picks"][f"{s}_{tag}"] = [round(float(v.mean()), 1), round(float(np.percentile(bt, 5)), 1), round(float(np.percentile(bt, 95)), 1)]
    (OUT / "e1_costs.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
