import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = Path("/Users/nahom/projects/silicon-alpha")
FLY = {"structure": "iron_fly", "wing": 1.5}
FEE = 1.20
def one(args):
    day, hhs = args
    from odte.research import zoo
    from odte.research.struct4 import price
    st = pd.read_csv(ROOT / "data/odte_etf/SPY/settlements.csv", parse_dates=["date"]).set_index("date")["close"]
    f = ROOT / "data/odte_etf/SPY" / f"{day:%Y-%m-%d}.parquet"
    if not f.exists() or day not in st.index: return []
    ch = pd.read_parquet(f); ch["quote_datetime"] = pd.to_datetime(ch["quote_datetime"])
    ch = ch[(ch["bid"] > 0) & (ch["ask"] >= ch["bid"])]
    zd = zoo.Day(day, ch, float(st[day]), np.nan, {})
    t_x = day + pd.Timedelta(hours=15, minutes=59)
    if t_x not in zd.minutes: return []
    S_x = zd.forward(t_x); S_c = float(st[day])
    m, h = zd.mid.loc[t_x], zd.half.loc[t_x]
    out = []
    for hh in hhs:
        if hh >= "15:59": continue
        r = {"day": day, "hhmm": hh}
        for share, tag in ((0.5, "half"), (1.0, "touch")):
            p = price(FLY, zd, hh, 1.0, share)
            if p is None: r[f"itm_{tag}"] = np.nan; continue
            cash, n_close, rest = p["cash_in"], 0, 0.0
            for right, k, q in p["legs"]:
                itm = (S_x > k) if right == "C" else (S_x < k)
                if itm:                                  # close it at 15:59: pay mid + half-spread*share (q<0 short: buy back)
                    try:
                        mid, hs = float(m[(right, k)]), float(h[(right, k)])
                    except KeyError:
                        mid, hs = np.nan, np.nan
                    if not np.isfinite(mid):
                        mid, hs = max((S_x - k) if right == "C" else (k - S_x), 0.0), 0.0
                    cash += q * mid - abs(q) * hs * share   # selling back a long (q>0) receives mid - hs; buying back a short (q<0) pays mid + hs
                    n_close += abs(q)
                else:                                     # left to expire: worth its intrinsic at the close (0 unless it crossed in the last minute)
                    rest += q * (max(S_c - k, 0.0) if right == "C" else max(k - S_c, 0.0))
            pnl = (cash + rest) * 100 - p["fees"] - n_close * FEE
            r[f"itm_{tag}"] = pnl / p["risk"] * 1000
            r[f"n_close_{tag}"] = n_close
        out.append(r)
    return out
if __name__ == "__main__":
    x = pd.read_parquet(ROOT / "reports/execution/e1_costs.parquet")
    p = x[x["b0_fly"] & (x["sym"] == "SPY") & x["ret_half"].notna()]
    with ProcessPoolExecutor(10) as ex:
        rows = [r for part in ex.map(one, [(d, list(g["hhmm"])) for d, g in p.groupby("day")]) for r in part]
    e = pd.DataFrame(rows).merge(p[["day", "hhmm", "ret_half", "ret_touch"]], on=["day", "hhmm"])
    xs = x[x["b0_fly"] & (x["sym"] == "XSP") & x["ret_half"].notna()][["day", "hhmm", "ret_half", "ret_touch"]].rename(columns={"ret_half": "xsp_half", "ret_touch": "xsp_touch"})
    e = e.merge(xs, on=["day", "hhmm"])
    rng = np.random.default_rng(13)
    def ci(v):
        v = v.dropna().to_numpy(); b = v[rng.integers(0, len(v), (5000, len(v)))].mean(axis=1)
        return f"{v.mean():+7.1f} [{np.percentile(b, 5):+.1f}, {np.percentile(b, 95):+.1f}]"
    print(f"{len(e)} paired decisions (locked model's fly picks, 2024-11 .. 2026-11), $ per $1k risk:")
    print("  XSP held to close (cash-settled)      half", ci(e["xsp_half"]), "| touch", ci(e["xsp_touch"]))
    print("  SPY held to close (share delivery!)   half", ci(e["ret_half"]), "| touch", ci(e["ret_touch"]))
    print("  SPY, close ONLY ITM legs at 15:59     half", ci(e["itm_half"]), "| touch", ci(e["itm_touch"]))
    print("  SPY-ITM minus XSP (paired)            half", ci(e["itm_half"] - e["xsp_half"]), "| touch", ci(e["itm_touch"] - e["xsp_touch"]))
    print("  legs closed per trade:", e["n_close_half"].value_counts().sort_index().to_dict())
    e.to_parquet(ROOT / "reports/execution/e4_spy_itm.parquet", index=False)
