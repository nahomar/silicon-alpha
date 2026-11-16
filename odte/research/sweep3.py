"""Sweep 3 (docs/sweep_plan.md, "Sweep 3"): P partner B3, X country/sector/currency cross-sections,
E earnings-announcement ideas.

    python3 -m odte.research.sweep3 P          # -> reports/sweep3/P.json
    python3 -m odte.research.sweep3 fetchX     # Yahoo ETFs + FRED currencies -> data/xasset/
    python3 -m odte.research.sweep3 X          # -> reports/sweep3/X.json
    python3 -m odte.research.sweep3 E          # -> reports/sweep3/E.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "sweep3"
XDIR = ROOT / "data" / "xasset"
B0 = ROOT / "reports" / "stress" / "stress_spx_daily.parquet"
SEED, MES = 20261115, 5.0
SCALE_END = pd.Timestamp("2021-12-31")


def tstat(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3 or v.std(ddof=1) == 0:
        return float("nan")
    return float(v.mean() / v.std(ddof=1) * math.sqrt(len(v)))


def p_two(t: float) -> float:
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))) if np.isfinite(t) else 1.0


def verdict(v: pd.Series) -> dict:
    """PASS: t >= 2 and both halves positive; LEAD: p < 0.05 and both halves positive."""
    v = v.dropna()
    mid = v.index[0] + (v.index[-1] - v.index[0]) / 2
    a, b = v[v.index < mid], v[v.index >= mid]
    t = tstat(v)
    both = bool(a.mean() > 0 and b.mean() > 0)
    return {"n": int(len(v)), "start": v.index[0].date().isoformat(), "end": v.index[-1].date().isoformat(),
            "mean": float(v.mean()), "t": t, "p": p_two(t), "half1_mean": float(a.mean()), "half2_mean": float(b.mean()),
            "half1_t": tstat(a), "half2_t": tstat(b),
            "verdict": "PASS" if t >= 2.0 and both else ("LEAD" if p_two(t) < 0.05 and t > 0 and both else "FAIL")}


# ----------------------------------------------------------------------------------------------- P partner
def part_p() -> int:
    from odte.research.partner_b import marks
    from odte.research.partner_t import book
    from odte.research.stress_spx import sharpe, stationary_bootstrap
    from odte.research.sweep1 import SP, fam_B
    OUT.mkdir(parents=True, exist_ok=True)
    sp = SP()
    ideas = {i["id"]: i for i in fam_B(sp) if i["id"] in ("B3", "B7")}
    b0 = pd.read_parquet(B0)["pnl_per_1k"]
    b0.index = pd.DatetimeIndex(b0.index)
    early = b0.index <= SCALE_END
    m = marks(sp.dt)
    ser = {"B": book(b0.index, m["day"], m["net"] * MES)}
    for k in ("B3", "B7"):
        net = ideas[k]["g"] - ideas[k]["c"]
        ser[k] = book(b0.index, pd.Series(net.index, index=net.index), net)
    w = {k: float(b0[early].std(ddof=1) / v[early].std(ddof=1)) for k, v in ser.items()}

    def p_loss(x):
        yr = stationary_bootstrap(x.to_numpy(), 10000, 252, 20.0, np.random.default_rng(SEED))
        return float((yr.sum(axis=1) < 0).mean())

    def mdd(x):
        e = x.cumsum()
        return float((e - e.cummax()).min())

    combos = {"B0": b0, "B0+B": b0 + w["B"] * ser["B"], "B0+B3": b0 + w["B3"] * ser["B3"],
              "B0+B+B7": b0 + w["B"] * ser["B"] + w["B7"] * ser["B7"], "B0+B3+B7": b0 + w["B3"] * ser["B3"] + w["B7"] * ser["B7"]}
    res = {"weights_mes_per_b0_unit": w, "corr_B3_B0": float(np.corrcoef(b0, ser["B3"])[0, 1]),
           "combos": {k: {"sharpe": sharpe(v), "p_losing_year": p_loss(v), "max_dd_per_b0_unit": mdd(v),
                          "mean_per_day": float(v.mean())} for k, v in combos.items()},
           "by_year": {int(y): {k: round(float(v[v.index.year == y].sum()), 1) for k, v in combos.items()}
                       for y in sorted(set(b0.index.year))}}
    c = res["combos"]
    res["P1_replace_B_with_B3"] = bool(c["B0+B3"]["sharpe"] > c["B0+B"]["sharpe"]
                                        and c["B0+B3"]["p_losing_year"] < c["B0+B"]["p_losing_year"])
    (OUT / "P.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))
    return 0


# ----------------------------------------------------------------------------------------------- X other markets
COUNTRY = ["EWJ", "EWG", "EWU", "EWQ", "EWH", "EWY", "EWT", "EWA", "EWC", "EWZ", "EWS", "EWM", "EWI", "EWP", "EWL",
           "EWN", "EWD", "EWW", "EWK", "EWO", "FXI", "EZA", "TUR", "INDA", "SPY"]
SECTOR = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
FX = {"AUD": ("DEXUSAL", False, "IR3TIB01AUM156N"), "CAD": ("DEXCAUS", True, "IR3TIB01CAM156N"),
      "JPY": ("DEXJPUS", True, "IR3TIB01JPM156N"), "GBP": ("DEXUSUK", False, "IR3TIB01GBM156N"),
      "CHF": ("DEXSZUS", True, "IR3TIB01CHM156N"), "EUR": ("DEXUSEU", False, "IR3TIB01EZM156N"),
      "NZD": ("DEXUSNZ", False, "IR3TIB01NZM156N"), "NOK": ("DEXNOUS", True, "IR3TIB01NOM156N"),
      "SEK": ("DEXSDUS", True, "IR3TIB01SEM156N")}          # (spot series, quoted as foreign per USD?, 3m rate)
US_RATE = "IR3TIB01USM156N"


def fred(series: str) -> pd.Series:
    import io
    import urllib.request
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    raw = urllib.request.urlopen(url, timeout=60).read()      # FRED stalls on some custom User-Agents
    d = pd.read_csv(io.BytesIO(raw))
    d.columns = ["date", "v"]
    d["date"] = pd.to_datetime(d["date"])
    return pd.to_numeric(d.set_index("date")["v"], errors="coerce").dropna()


def fetch_x() -> int:
    import yfinance as yf
    XDIR.mkdir(parents=True, exist_ok=True)
    px = {}
    for t in ([] if (XDIR / "etf_close.parquet").exists() else COUNTRY + SECTOR):
        h = yf.Ticker(t).history(period="max", auto_adjust=True)
        if h.empty:
            print(t, "EMPTY")
            continue
        h.index = pd.DatetimeIndex(h.index.date)
        px[t] = h["Close"]
        print(t, h.index.min().date(), len(h))
    if px:
        pd.DataFrame(px).to_parquet(XDIR / "etf_close.parquet")
    fx = {}
    for c, (s, inv, r) in FX.items():
        for name in (s, r):
            try:
                fx[name] = fred(name)
                print(name, fx[name].index.min().date(), fx[name].index.max().date(), len(fx[name]))
            except Exception as e:                               # noqa: BLE001
                print(name, "FAILED", type(e).__name__)
    fx[US_RATE] = fred(US_RATE)
    print(US_RATE, fx[US_RATE].index.min().date(), fx[US_RATE].index.max().date())
    pd.concat(fx, axis=1).to_parquet(XDIR / "fred_fx.parquet")
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"P": part_p, "fetchX": fetch_x}.get(arg, lambda: print(__doc__) or 1)())
