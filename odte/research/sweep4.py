"""Sweep 4 (docs/sweep_plan.md, "Sweep 4"): drift after 28 kinds of SEC filings, from the EDGAR submissions
files downloaded by odte.research.sec_earnings.

    python3 -m odte.research.sweep4            # -> reports/sweep4/sweep4.json
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research.sweep3 import p_two, stock_panel, tstat

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "filings" / "submissions"
OUT = ROOT / "reports" / "sweep4"
LO, HI = "2018-06-01", "2026-08-31"
COST_RT = 0.002
H_MAIN, H_ALT = 20, (5, 60)

FORMS = {"S1 SC 13D (new activist-type stake)": ("form", {"SC 13D", "SCHEDULE 13D"}),
         "S2 SC 13D/A": ("form", {"SC 13D/A", "SCHEDULE 13D/A"}),
         "S3 SC 13G (passive 5% stake)": ("form", {"SC 13G", "SCHEDULE 13G"}),
         "S4 DFAN14A (dissident proxy material)": ("form", {"DFAN14A"}),
         "S5 SC TO-T (third-party tender offer)": ("form", {"SC TO-T"}),
         "S6 Form 144 (planned insider sale)": ("form", {"144"}),
         "S7 Form 3 (new insider)": ("form", {"3"}),
         "S8 424B4 (final prospectus)": ("form", {"424B4"}),
         "S9 424B5 (shelf takedown)": ("form", {"424B5"}),
         "S10 S-3 / S-3ASR (shelf registration)": ("form", {"S-3", "S-3ASR"}),
         "S11 S-8 (employee stock plan)": ("form", {"S-8"}),
         "S12 8-K 3.02 (unregistered equity sale)": ("item", "3.02"),
         "S13 NT 10-K / NT 10-Q (late filing)": ("form", {"NT 10-K", "NT 10-Q"}),
         "S14 8-K 4.02 (results no longer reliable)": ("item", "4.02"),
         "S15 8-K 4.01 (auditor change)": ("item", "4.01"),
         "S16 8-K 3.01 (delisting / listing-rule notice)": ("item", "3.01"),
         "S17 8-K 2.05 (restructuring costs)": ("item", "2.05"),
         "S18 8-K 1.02 (material agreement ended)": ("item", "1.02"),
         "S19 8-K 1.03 (bankruptcy)": ("item", "1.03"),
         "S20 8-K 1.01 (material agreement)": ("item", "1.01"),
         "S21 8-K 2.01 (acquisition or sale completed)": ("item", "2.01"),
         "S22 8-K 2.03 (new debt)": ("item", "2.03"),
         "S23 8-K 5.02 (executive or director change)": ("item", "5.02"),
         "S24 8-K 5.03 (bylaw or charter change)": ("item", "5.03"),
         "S25 8-K 5.07 (shareholder vote results)": ("item", "5.07"),
         "S26 8-K 7.01 (Reg FD disclosure)": ("item", "7.01"),
         "S27 8-K 8.01 (other events)": ("item", "8.01"),
         "S28 8-K 2.02 (earnings release), unsorted": ("item", "2.02")}
NO_FINANCE = {"S1", "S2", "S3"}


def company_map() -> dict:
    """cik -> (most traded symbol, sic), same mapping rule as sec_earnings.fetch."""
    from odte.research.sec_earnings import get, symbols
    tick = json.loads(get("https://www.sec.gov/files/company_tickers.json"))
    cik_of = {v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in tick.values()}
    out = {}
    for s in symbols():
        c = cik_of.get(s.upper()) or cik_of.get(s.upper().replace(".", "-"))
        if c and c not in out:
            out[c] = s
    return out


def filings(cmap: dict) -> pd.DataFrame:
    keep = ["filingDate", "acceptanceDateTime", "form", "items"]
    rows = []
    for cik, sym in cmap.items():
        f = RAW / f"CIK{cik:010d}.json"
        if not f.exists():
            continue
        j = json.loads(f.read_bytes())
        try:
            sic = int(j.get("sic") or 0)
        except ValueError:
            sic = 0
        blocks = [j["filings"]["recent"]]
        for extra in j["filings"].get("files", []):
            g = RAW / extra["name"]
            if g.exists():
                blocks.append(json.loads(g.read_bytes()))
        for b in blocks:
            d = pd.DataFrame({k: b.get(k, []) for k in keep})
            d = d[(d["filingDate"] >= LO) & (d["filingDate"] <= HI)]
            d["cik"], d["sym"], d["sic"] = cik, sym, sic
            rows.append(d)
    return pd.concat(rows, ignore_index=True).drop_duplicates()


def day0_index(df: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    acc = pd.to_datetime(df["acceptanceDateTime"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    loc = acc.dt.tz_localize(None)
    base = loc.dt.normalize() + pd.to_timedelta((loc.dt.hour >= 16).astype(int), unit="D")
    # no acceptance time: treat as after the close of the filing date
    fb = pd.to_datetime(df["filingDate"]) + pd.Timedelta(days=1)
    base = base.where(acc.notna(), fb)
    return np.searchsorted(dates.values, base.values.astype("datetime64[ns]"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ret, uni = stock_panel()
    dates = ret.index
    ue = sorted(uni)
    mkt = pd.Series(np.nan, index=dates)
    for a, b in zip(ue, ue[1:] + [dates[-1] + pd.Timedelta(days=1)]):
        idx = dates[(dates > a) & (dates <= b)]
        cols = [c for c in uni[a] if c in ret.columns]
        mkt[idx] = ret.loc[idx, cols].mean(axis=1)
    R, M = ret.to_numpy(float), mkt.to_numpy(float)
    col = {s: j for j, s in enumerate(ret.columns)}
    cmap = company_map()
    fl = filings(cmap)
    fl = fl[fl["sym"].isin(col)].copy()
    fl["i0"] = day0_index(fl, dates)
    fl = fl[(fl["i0"] >= 1) & (fl["i0"] < len(dates) - 1)]
    fl["day0"] = dates[fl["i0"].to_numpy()]
    ue_arr = np.array(ue, dtype="datetime64[ns]")
    k = np.searchsorted(ue_arr, fl["day0"].values.astype("datetime64[ns]"), side="left") - 1
    fl["mem"] = [kk >= 0 and s in uni[ue[kk]] for kk, s in zip(k, fl["sym"])]
    fl = fl[fl["mem"]]
    is8k = fl["form"] == "8-K"
    items = fl["items"].fillna("")

    def book(ev: pd.DataFrame, H: int) -> tuple[pd.Series, pd.Series]:
        n = len(dates)
        tot, cnt = np.zeros(n), np.zeros(n)
        for i0, s in zip(ev["i0"].to_numpy(), ev["sym"].to_numpy()):
            s0, s1 = i0 + 1, min(i0 + 1 + H, n)
            if s0 >= n:
                continue
            rr = R[s0:s1, col[s]] - M[s0:s1]
            tot[s0:s1] += np.where(np.isfinite(rr), rr, 0.0)
            cnt[s0:s1] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            g = pd.Series(np.where(cnt > 0, tot / cnt, np.nan), index=dates).dropna()
        return g, pd.Series(COST_RT / H, index=g.index)

    def scored(g: pd.Series, c: pd.Series) -> dict:
        tp, tm = tstat(g - c), tstat(-g - c)
        sign = 1 if (tp if np.isfinite(tp) else -9) >= (tm if np.isfinite(tm) else -9) else -1
        v = sign * g - c
        t = tstat(v)
        mid = v.index[0] + (v.index[-1] - v.index[0]) / 2
        a, b = v[v.index < mid], v[v.index >= mid]
        return {"direction": "long" if sign == 1 else "short", "days": int(len(v)), "bps_per_day": float(v.mean() * 1e4),
                "t": t, "p": p_two(t), "half1_t": tstat(a), "half2_t": tstat(b),
                "halves_agree": bool(a.mean() > 0 and b.mean() > 0),
                "gross_abnormal_bps_per_day": float(g.mean() * 1e4)}

    res = {}
    for name, (kind, spec) in FORMS.items():
        sid = name.split()[0]
        m = fl["form"].isin(spec) if kind == "form" else (is8k & items.str.split(",").apply(lambda z: spec in z))
        ev = fl[m]
        if sid in NO_FINANCE:
            ev = ev[~ev["sic"].between(6000, 6799)]
        ev = ev.sort_values(["cik", "day0"])
        gap = ev.groupby("cik")["day0"].diff().dt.days
        ev = ev[gap.isna() | (gap > 30)]
        if len(ev) < 30:
            res[name] = {"events": int(len(ev)), "verdict": "NOT SCORED (< 30 events)"}
            continue
        e = {"events": int(len(ev)), "companies": int(ev["cik"].nunique())}
        e |= scored(*book(ev, H_MAIN))
        for h in H_ALT:
            z = scored(*book(ev, h))
            e[f"H{h}_report"] = {"direction": z["direction"], "t": round(z["t"], 2), "bps_per_day": round(z["bps_per_day"], 2)}
        res[name] = e
    sc = {k: v for k, v in res.items() if "p" in v}
    ks = sorted(sc, key=lambda k: sc[k]["p"])
    prev = 1.0
    for i, k in reversed(list(enumerate(ks, 1))):
        prev = min(prev, sc[k]["p"] * len(ks) / i)
        sc[k]["q"] = prev
    for k, v in sc.items():
        v["verdict"] = ("PASS" if v["t"] >= 2.0 and v["halves_agree"] and v["q"] <= 0.10 else
                        "LEAD" if v["p"] < 0.05 and v["halves_agree"] else "FAIL")
    (OUT / "sweep4.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"{'idea':48} {'events':>6} {'dir':>5} {'bps/d':>6} {'t':>6} {'q':>5} {'h1 t':>5} {'h2 t':>5} {'t5':>5} {'t60':>5} verdict")
    for k, v in res.items():
        if "p" not in v:
            print(f"{k:48} {v['events']:6d}  {v['verdict']}")
            continue
        print(f"{k:48} {v['events']:6d} {v['direction']:>5} {v['bps_per_day']:6.2f} {v['t']:6.2f} {v['q']:5.2f} "
              f"{v['half1_t']:5.2f} {v['half2_t']:5.2f} {v['H5_report']['t']:5.2f} {v['H60_report']['t']:5.2f} {v['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
