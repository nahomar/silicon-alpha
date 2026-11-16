"""Contenders on development stretches that played no part in choosing them (rounds 2-4 used 2023-07+)."""
import numpy as np, pandas as pd
from odte.research import meta, round2, round3, struct4
KEYS = ["day", "index", "hhmm"]
d = round3.load(); tab = pd.read_parquet(struct4.OUT)
base = tab.drop_duplicates(KEYS)[KEYS + ["F", "st", "settle"]].rename(columns={"F": "F4", "st": "st4", "settle": "close4"})
d = d.merge(base, on=KEYS, how="left", validate="one_to_one")
z = ((d["close4"] - d["F4"]) / d["st4"]).to_numpy(float)
X = round2.features(d, round2.B0_GROUPS)
rid = pd.Series(np.arange(len(d)), index=pd.MultiIndex.from_frame(d[KEYS]))
tab["row"] = rid.reindex(pd.MultiIndex.from_frame(tab[KEYS])).to_numpy()
tab = tab.dropna(subset=["row"]).astype({"row": int}); tab = tab[tab["shape"].isin(round2.MENU)]
days = np.array(sorted(d["day"].unique())); dd = d["day"].to_numpy(); ok = ~d["post"].to_numpy() & np.isfinite(z)
qs = (np.arange(200) + 0.5) / 200; EV = {k: np.full(len(d), np.nan) for k in round2.MENU}
for k0 in range(meta.MIN_DAYS, len(days), meta.REFIT):
    tr = (dd < days[k0]) & ok
    te = (dd >= days[k0]) & ((dd < days[k0 + meta.REFIT]) if k0 + meta.REFIT < len(days) else True)
    if tr.sum() < 1000 or not te.any(): continue
    m = round2.ridge(1000.0).fit(X[tr], np.log(np.abs(z[tr]) + 0.05))
    u = np.quantile(z[tr] / np.exp(m.predict(X[tr])), qs)
    te_rows = np.flatnonzero(te); sc = pd.Series(np.exp(m.predict(X[te])), index=te_rows)
    blk = tab[tab["row"].isin(te_rows)]
    S = blk["F"].to_numpy()[:, None] + blk["st"].to_numpy()[:, None] * sc.reindex(blk["row"]).to_numpy()[:, None] * u[None, :]
    ev = ((blk["cash_half"].to_numpy() + struct4.payoff(blk, S).mean(axis=1)) * 100 - blk["fees"].to_numpy()) / blk["risk_half"].to_numpy() * 1000
    for s, g in pd.DataFrame({"row": blk["row"].to_numpy(), "shape": blk["shape"].to_numpy(), "ev": ev}).groupby("shape"):
        EV[s][g["row"].to_numpy()] = g["ev"].to_numpy()
P0, _ = round2.walk(d, X)
def daily(dec): return dec[dec["tested"]].groupby("day")["ret"].sum()
out = {}
for tag in ("half", "touch"):
    b0 = round3.decide_menu(d, P0, list(round2.MENU), tag)
    c7 = round3.decide_menu(d, P0, list(round2.MENU), tag, drop_late=True)
    nf = b0.copy(); fl = nf["trade"] == "fly"
    nf.loc[fl, "ret"] = np.nan_to_num(d.loc[fl, f"y_fly_narrow_{tag}"].to_numpy(float))
    d4 = round3.decide_menu(d, EV, list(round2.MENU), tag, drop_late=True)
    out[tag] = {"B0": daily(b0), "C7": daily(c7), "NF": daily(nf), "D4L": daily(d4)}
rng = np.random.default_rng(1)
for name, (a, b) in {"Jan 2022 - Jun 2023 (3 indexes from May 2022)": ("2022-01-03", "2023-06-30"),
                     "Jun 2019 - Dec 2021 (SPX only, Mon/Wed/Fri expiries)": ("2019-06-26", "2021-12-31")}.items():
    print(f"\n== {name}")
    for c in ("B0", "C7", "NF", "D4L"):
        h = out["half"][c]; h = h[(h.index >= a) & (h.index <= b)]
        t = out["touch"][c]; t = t[(t.index >= a) & (t.index <= b)]
        line = f"  {c:4s} ${h.mean():+6.0f}/day  Sharpe {round2.sharpe(h):+.2f}  capped {round2.capped(h):+.2f} | touch ${t.mean():+6.0f}/day"
        if c != "B0":
            base0 = out["half"]["B0"]; base0 = base0[(base0.index >= a) & (base0.index <= b)]
            diff = (h - base0).to_numpy(); bt = diff[rng.integers(0, len(diff), (5000, len(diff)))].mean(axis=1)
            line += f" | vs B0 ${diff.mean():+5.0f}/day 90% CI [{np.percentile(bt, 5):+.0f}, {np.percentile(bt, 95):+.0f}]"
        print(line + f"  ({len(h)} days)")
