"""Round 4 structure table: for every decision row and each of the 14 shapes, the legs and the
entry terms exactly as zoo.trade computes them (entry cash at half-gap and touch, risk, fees),
plus the entry forward and straddle and the close. Lets any forecast of the close be turned into
an expected value per $1k risk.

Exactness check: re-pricing at the actual close must reproduce the realized labels.

  python3 -m odte.research.struct4      # -> data/research/struct4.parquet
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "research" / "struct4.parquet"
MAX_LEGS = 4


def shapes() -> dict:
    from odte.research import library, menu3
    return {**{k: library.TRADES[k] for k in ("fly", "condor", "put_spread", "call_spread")}, **menu3.NEW}


def price(rule, d, hhmm, k_day, share):
    """zoo.trade's entry terms (no exits): legs, cash_in (points), risk ($), fees ($)."""
    from odte.research import zoo
    t = d.at(hhmm)
    if t not in d.minutes:
        return None
    F = d.forward(t)
    K0, st = d.straddle(t, F)
    if not (np.isfinite(st) and st > 0):
        return None
    legs = zoo.legs_for(rule, d, t, F, st / 0.8, None)
    if not legs:
        return None
    m, h = d.mid.loc[t], d.half.loc[t]
    try:
        V = sum(q * float(m[(r, k)]) for r, k, q in legs)
        cost_in = sum(abs(q) * float(h[(r, k)]) for r, k, q in legs) * k_day * share
    except KeyError:
        return None
    if not (np.isfinite(V) and np.isfinite(cost_in)):
        return None
    n = sum(abs(q) for _, _, q in legs)
    cash_in = -V - cost_in
    lo, bounded = zoo.worst_payoff(legs, F)
    if not bounded:
        return None
    risk = -(cash_in * 100 - n * zoo.FEE + lo * 100)
    if not risk > 0:
        return None
    return {"legs": legs, "cash_in": cash_in, "risk": risk, "fees": n * zoo.FEE, "F": F, "st": st}


def _one(job):
    sym, day, times, R = job
    from odte.research import library, zoo
    folder = library.DIRS[sym]
    st = pd.read_csv(folder / "settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
    zd = zoo.Day(day, pd.read_parquet(folder / f"{day:%Y-%m-%d}.parquet"), float(st[day]), np.nan, {})
    r10 = zd.straddle_rel_spread(day + pd.Timedelta(hours=10))
    k_day = float(np.clip(R / r10, 0.1, 2.0)) if np.isfinite(r10) and r10 > 0 else 1.0
    rows = []
    for hhmm in times:
        for name, rule in shapes().items():
            rec = {"day": day, "index": sym, "hhmm": hhmm, "shape": name, "settle": float(st[day])}
            for share, tag in ((0.5, "half"), (1.0, "touch")):
                p = price(rule, zd, hhmm, k_day, share)
                if p is None:
                    break
                rec[f"cash_{tag}"], rec[f"risk_{tag}"] = p["cash_in"], p["risk"]
                rec["fees"], rec["F"], rec["st"] = p["fees"], p["F"], p["st"]
                for i in range(MAX_LEGS):
                    r, k, q = p["legs"][i] if i < len(p["legs"]) else ("C", 0.0, 0)
                    rec[f"r{i}"], rec[f"k{i}"], rec[f"q{i}"] = (1 if r == "C" else -1), float(k), int(q)
            else:
                rows.append(rec)
    return rows


def payoff(tab: pd.DataFrame, S: np.ndarray) -> np.ndarray:
    """Expiry payoff (points, per 1x) of each table row at prices S: shape (rows,) or (rows, n)."""
    S = np.asarray(S, float)
    S2 = S if S.ndim == 2 else S[:, None]
    out = np.zeros_like(S2)
    for i in range(MAX_LEGS):
        r = tab[f"r{i}"].to_numpy()[:, None]
        k = tab[f"k{i}"].to_numpy(float)[:, None]
        q = tab[f"q{i}"].to_numpy(float)[:, None]
        out += q * np.where(r == 1, np.maximum(S2 - k, 0.0), np.maximum(k - S2, 0.0))
    return out if S.ndim == 2 else out[:, 0]


def build(workers: int = 10) -> pd.DataFrame:
    from odte.research import meta, menu3
    m = pd.read_parquet(meta.MATRIX, columns=["day", "index", "hhmm"])
    R = {s: float(pd.read_parquet(meta.LIB / f"library_{s}.parquet", columns=["R_today"])["R_today"].iloc[0]) for s in meta.INDEXES}
    jobs = [(s, d, sorted(g["hhmm"]), R[s]) for (s, d), g in m.groupby(["index", "day"])]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, part in enumerate(ex.map(_one, jobs, chunksize=8)):
            rows += part
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(jobs)} index-days", flush=True)
    tab = pd.DataFrame(rows)
    # exactness: the realized return at the close must equal the stored labels
    lab = pd.read_parquet(menu3.OUT).merge(pd.read_parquet(meta.MATRIX, columns=["day", "index", "hhmm"] + [
        f"y_{k}_{t}" for k in ("fly", "condor", "put_spread", "call_spread") for t in ("half", "touch")]),
        on=["day", "index", "hhmm"])
    pay = payoff(tab, tab["settle"].to_numpy())
    worst = 0.0
    for tag in ("half", "touch"):
        ret = ((tab[f"cash_{tag}"] + pay) * 100 - tab["fees"]) / tab[f"risk_{tag}"] * 1000
        long = lab.melt(id_vars=["day", "index", "hhmm"], value_vars=[c for c in lab if c.endswith(f"_{tag}")],
                        var_name="col", value_name="y")
        long["shape"] = long["col"].str[2:-(len(tag) + 1)]
        chk = tab[["day", "index", "hhmm", "shape"]].assign(ret=ret.to_numpy()).merge(long, on=["day", "index", "hhmm", "shape"])
        worst = max(worst, float(np.nanmax(np.abs(chk["ret"] - chk["y"]))))
    if worst > 1e-6:
        raise SystemExit(f"exactness check failed: max difference {worst}")
    tab.to_parquet(OUT, index=False)
    print(f"{len(tab):,} (row, shape) entries -> {OUT}; exactness check passed (max diff {worst:.1e})")
    return tab


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
