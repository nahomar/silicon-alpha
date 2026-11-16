"""Round 4 (docs/round2_plan.md): one distribution forecast of the close; every shape priced
from it; take the best expected value if > 0.

  python3 -m odte.research.round4
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from odte.research import meta, round2, round3, struct4

KEYS = ["day", "index", "hhmm"]
NQ = 200
M4 = list(round2.MENU)
M14 = round3.MENU14


def main() -> int:
    d = round3.load()
    tab = pd.read_parquet(struct4.OUT)
    base = tab.drop_duplicates(KEYS)[KEYS + ["F", "st", "settle"]].rename(columns={"F": "F4", "st": "st4", "settle": "close4"})
    d = d.merge(base, on=KEYS, how="left", validate="one_to_one")
    z = ((d["close4"] - d["F4"]) / d["st4"]).to_numpy(float)
    X = round2.features(d, round2.B0_GROUPS)
    row_id = pd.Series(np.arange(len(d)), index=pd.MultiIndex.from_frame(d[KEYS]))
    tab["row"] = row_id.reindex(pd.MultiIndex.from_frame(tab[KEYS])).to_numpy()
    tab = tab.dropna(subset=["row"]).astype({"row": int})
    EV = {k: np.full(len(d), np.nan) for k in M14}
    days = np.array(sorted(d["day"].unique()))
    dd = d["day"].to_numpy()
    ok = ~d["post"].to_numpy() & np.isfinite(z)
    qs = (np.arange(NQ) + 0.5) / NQ
    calib = []
    for k0 in range(meta.MIN_DAYS, len(days), meta.REFIT):
        tr = (dd < days[k0]) & ok
        te = (dd >= days[k0]) & ((dd < days[k0 + meta.REFIT]) if k0 + meta.REFIT < len(days) else True)
        if tr.sum() < 1000 or not te.any():
            continue
        m = round2.ridge(1000.0).fit(X[tr], np.log(np.abs(z[tr]) + 0.05))
        s_tr = np.exp(m.predict(X[tr]))
        u = np.quantile(z[tr] / s_tr, qs)
        s_te = np.exp(m.predict(X[te]))
        te_rows = np.flatnonzero(te)
        sc = pd.Series(s_te, index=te_rows)
        blk = tab[tab["row"].isin(te_rows)]
        S = blk["F"].to_numpy()[:, None] + blk["st"].to_numpy()[:, None] * sc.reindex(blk["row"]).to_numpy()[:, None] * u[None, :]
        epay = struct4.payoff(blk, S).mean(axis=1)
        ev = ((blk["cash_half"].to_numpy() + epay) * 100 - blk["fees"].to_numpy()) / blk["risk_half"].to_numpy() * 1000
        for shape, g in pd.DataFrame({"row": blk["row"].to_numpy(), "shape": blk["shape"].to_numpy(), "ev": ev}).groupby("shape"):
            EV[shape][g["row"].to_numpy()] = g["ev"].to_numpy()
        zt = np.abs(z[te])
        calib.append(np.nanmedian(zt) / np.nanmedian(np.abs(s_te * np.median(np.abs(u)))))
    print(f"scale calibration (median realized |z| / median predicted, per refit): mean {np.nanmean(calib):.2f}, "
          f"range {np.nanmin(calib):.2f}..{np.nanmax(calib):.2f}")
    prev = json.loads((round2.OUT / "candidates.json").read_text())
    c7 = prev["C7_no_late"]
    out = {}
    for name, menu, late in (("D4", M4, False), ("D14", M14, False), ("D4L", M4, True), ("D14L", M14, True)):
        P = {k: EV[k] for k in menu}
        r, h, shapes = round3.evaluate(d, P, menu, late)
        r["vs_C7"] = r["primary_capped_sharpe"] - c7["primary_capped_sharpe"]
        r["beats_C7_both_halves"] = bool(r["half1_capped"] > c7["half1_capped"] and r["half2_capped"] > c7["half2_capped"])
        r["shapes"] = shapes.round(4).to_dict(orient="index")
        out[name] = r
        round2.show(name, r)
        print(f"   vs C7 {r['vs_C7']:+.2f}, both halves: {r['beats_C7_both_halves']}; top shapes by picks: "
              + ", ".join(f"{k} {v['share_picks']:.0%} (${v['mean']:+.0f})" for k, v in
                          sorted(r["shapes"].items(), key=lambda kv: -kv[1]["count"])[:5]))
    # are the expected values honest? predicted vs realized for the D14 picks
    dec = round3.decide_menu(d, {k: EV[k] for k in M14}, M14, "half")
    t = dec[dec["take"] & dec["day"].between(*round2.WIN)].copy()
    t["ev"] = [EV[s][i] for s, i in zip(t["trade"], t.index)]
    t["bin"] = pd.qcut(t["ev"], 5, labels=False, duplicates="drop")
    print("\nD14 picks by predicted EV quintile (predicted vs realized $ per $1k):")
    print(t.groupby("bin").agg(n=("ret", "size"), predicted=("ev", "mean"), realized=("ret", "mean")).round(1).to_string())
    (round2.OUT / "round4.json").write_text(json.dumps(out, indent=2, default=float))
    return 0


FINAL = meta.ROOT / "data" / "live" / "round4_model.pkl"


def train_final() -> dict:
    """D4L's frozen forecast: scale ridge + standardized-move quantiles on all development rows."""
    import pickle
    from odte.research import exam
    d = round3.load()
    tab = pd.read_parquet(struct4.OUT)
    base = tab.drop_duplicates(KEYS)[KEYS + ["F", "st", "settle"]].rename(columns={"F": "F4", "st": "st4", "settle": "close4"})
    d = d.merge(base, on=KEYS, how="left", validate="one_to_one")
    z = ((d["close4"] - d["F4"]) / d["st4"]).to_numpy(float)
    feats = exam.feature_list(d, exam.GROUPS)
    X = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = ~d["post"].to_numpy() & np.isfinite(z)
    m = round2.ridge(1000.0).fit(X[ok], np.log(np.abs(z[ok]) + 0.05))
    u = np.quantile(z[ok] / np.exp(m.predict(X[ok])), (np.arange(NQ) + 0.5) / NQ)
    b = {"name": "D4L", "features": feats, "model": m, "u": u, "menu": M4, "late": list(round2.LATE),
         "dev_through": str(d["day"].max().date()), "rows": int(ok.sum())}
    FINAL.write_bytes(pickle.dumps(b))
    return b


def ev_row(b: dict, x: dict, zd, hhmm: str, k_day: float) -> dict:
    """Expected value per $1k risk (half-gap) of each menu shape at one decision."""
    from odte.research.struct4 import price, shapes
    xv = np.array([[np.nan if x.get(c) is None else float(x[c]) for c in b["features"]]])
    from odte.live.linear import linear_predict
    s = float(np.exp(linear_predict(b["model"], xv)[0]))
    out = {}
    for name in b["menu"]:
        p = price(shapes()[name], zd, hhmm, k_day, 0.5)
        if p is None:
            continue
        legs = pd.DataFrame({k: [v] for i in range(struct4.MAX_LEGS) for k, v in (
            (f"r{i}", 1 if (p["legs"][i][0] if i < len(p["legs"]) else "C") == "C" else -1),
            (f"k{i}", float(p["legs"][i][1]) if i < len(p["legs"]) else 0.0),
            (f"q{i}", int(p["legs"][i][2]) if i < len(p["legs"]) else 0))})
        S = p["F"] + p["st"] * s * b["u"][None, :]
        epay = float(struct4.payoff(legs, S).mean())
        out[name] = ((p["cash_in"] + epay) * 100 - p["fees"]) / p["risk"] * 1000
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        b = train_final()
        import hashlib
        print(f"D4L frozen forecast: {len(b['features'])} inputs, {b['rows']:,} rows through {b['dev_through']} -> {FINAL} "
              f"(sha256 {hashlib.sha256(FINAL.read_bytes()).hexdigest()[:16]})")
    else:
        sys.exit(main())
