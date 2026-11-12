"""Round 2 forward head-to-head (docs/round2_plan.md), scored from the bot's journals.

  B0  the locked model, live as the lead shadow (`frozen_lead`)
  C7  THE CHALLENGER: B0 without the 15:00 and 15:30 decisions (same model file)
Secondary, reported but not deciding:
  C4  B0 with confidence sizing: clip(best prediction / SCALE_B0, 0, 2)
  C5  ridge 1000 on the same 41 inputs, trained on touch labels (all development rows)
  C8  C5 trained on rows from 2022-05-16, confidence-sized, without 15:00/15:30

Every model is scored on the same decisions' research labels (`lead_labels`, half-gap and touch,
$300 worst case), from the first session after the challenger freeze. SPX is primary (recorded
exactly); all three indexes are reported.

  python3 -m odte.research.round2_forward train    # secondary models -> data/live/round2_models.pkl
  python3 -m odte.research.round2_forward report   # the head-to-head so far
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
JOURNALS = ROOT / "data" / "live" / "journal"
MODELS = ROOT / "data" / "live" / "round2_models.pkl"
FIRST = pd.Timestamp("2026-11-12")           # first session after the challenger freeze
N_SESSIONS = 44
MENU = ("fly", "condor", "put_spread", "call_spread")
LATE = ("15:00", "15:30")
ERA = pd.Timestamp("2022-05-16")


def _fit(dev, feats, labels, train_from=None):
    from odte.research import meta
    X = dev[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = ~dev["post"].to_numpy()
    if train_from is not None:
        ok &= (dev["day"] >= train_from).to_numpy()
    models = {}
    for k in MENU:
        y = dev[f"y_{k}_{labels}"].to_numpy(float)
        tr = ok & np.isfinite(y)
        models[k] = meta.make("ridge_a1000").fit(X[tr], y[tr])
    best = np.max(np.column_stack([models[k].predict(X[ok]) for k in MENU]), axis=1)
    return models, float(np.median(best[best > 0]))


def train() -> dict:
    from odte.live import frozen
    from odte.live.config import Config
    from odte.research import exam_guard as eg
    from odte.research import meta
    b0 = pickle.loads(frozen.model_path(Config()).read_bytes())
    feats = b0["features"]
    dev = meta.load()
    dev_b0 = dev[dev["day"] < eg.EXAM_FROM]
    X = dev_b0[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = ~dev_b0["post"].to_numpy()
    best = np.max(np.column_stack([b0["models"][k].predict(X[ok]) for k in MENU]), axis=1)
    c5, _ = _fit(dev, feats, "touch")
    c8, s8 = _fit(dev, feats, "touch", train_from=ERA)
    out = {"features": feats, "scale_b0": float(np.median(best[best > 0])), "c5": c5, "c8": c8, "scale_c8": s8,
           "dev_through": str(dev["day"].max().date())}
    MODELS.write_bytes(pickle.dumps(out))
    return out


def load_journals(first=FIRST, n=N_SESSIONS) -> pd.DataFrame:
    rows = []
    files = sorted(p for p in JOURNALS.glob("20??-??-??.jsonl") if pd.Timestamp(p.stem) >= first)[:n]
    for p in files:
        recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        dec = {(r["index"], r["hhmm"]): r for r in recs if r.get("kind") == "shadow"
               and r.get("strategy") == "frozen_lead" and "preds" in r}
        feats = {(r["index"], r["hhmm"]): r for r in recs if r.get("kind") == "lead_row"}
        labs = {(r["index"], r["hhmm"]): r for r in recs if r.get("kind") == "lead_labels"}
        for key, d in dec.items():
            if key not in labs:
                continue
            rows.append({"day": pd.Timestamp(p.stem), "index": key[0], "hhmm": key[1], "b0_trade": d["trade"],
                         "post": bool(d.get("post")), "preds": d["preds"], "x": feats.get(key, {}), **{
                             k: labs[key].get(k) for k in labs[key] if k.endswith(("_half", "_touch"))}})
    return pd.DataFrame(rows)


def _choose(preds: dict, post: bool) -> str:
    vals = {k: v for k, v in preds.items() if v is not None and np.isfinite(v)}
    if post or not vals:
        return "none"
    best = max(vals, key=vals.get)
    return best if vals[best] > 0 else "none"


def score(df: pd.DataFrame, m: dict) -> pd.DataFrame:
    """Per-row returns ($300 worst case) of every model, at half and touch."""
    feats = m["features"]
    out = []
    for _, r in df.iterrows():
        x = np.array([[np.nan if r["x"].get(c) is None else float(r["x"][c]) for c in feats]])
        b0p = {k: float(r["preds"].get(k, np.nan)) for k in MENU}
        c5p = {k: float(m["c5"][k].predict(x)[0]) for k in MENU}
        c8p = {k: float(m["c8"][k].predict(x)[0]) for k in MENU}
        choice = {"B0": (r["b0_trade"], 1.0), "C7": (r["b0_trade"] if r["hhmm"] not in LATE else "none", 1.0),
                  "C4": (r["b0_trade"], float(np.clip(max(b0p.values()) / m["scale_b0"], 0, 2))),
                  "C5": (_choose(c5p, r["post"]), 1.0),
                  "C8": (_choose(c8p, r["post"]) if r["hhmm"] not in LATE else "none",
                         float(np.clip(max(c8p.values()) / m["scale_c8"], 0, 2)))}
        rec = {"day": r["day"], "index": r["index"], "hhmm": r["hhmm"]}
        for name, (trade, size) in choice.items():
            for tag in ("half", "touch"):
                v = r.get(f"{trade}_{tag}") if trade != "none" else 0.0
                rec[f"{name}_{tag}"] = (float(v) * size) if v is not None and np.isfinite(float(v)) else (0.0 if trade == "none" else np.nan)
        out.append(rec)
    return pd.DataFrame(out)


def head_to_head(s: pd.DataFrame, index: str | None = "SPX") -> dict:
    x = s if index is None else s[s["index"] == index]
    daily = x.groupby("day").sum(numeric_only=True)
    diff = (daily["C7_half"] - daily["B0_half"]).to_numpy()
    rng = np.random.default_rng(20261112)
    res = {"sessions": int(len(daily)), "means_half": {c[:-5]: float(daily[c].mean()) for c in daily if c.endswith("_half")},
           "means_touch": {c[:-6]: float(daily[c].mean()) for c in daily if c.endswith("_touch")}}
    if len(diff) >= 2:
        b = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(axis=1)
        res["c7_minus_b0"] = {"mean": float(diff.mean()), "ci90": [float(np.percentile(b, 5)), float(np.percentile(b, 95))]}
        if len(diff) >= N_SESSIONS:
            m, lo = res["c7_minus_b0"]["mean"], res["c7_minus_b0"]["ci90"][0]
            res["outcome"] = "WIN" if (m > 0 and lo > 0) else "LEAN" if m > 0 else "LOSE"
        else:
            res["outcome"] = f"in progress ({len(diff)}/{N_SESSIONS} sessions)"
    return res


def report() -> dict:
    m = pickle.loads(MODELS.read_bytes())
    df = load_journals()
    if df.empty:
        print("no journal sessions yet from", FIRST.date())
        return {}
    s = score(df, m)
    out = {"SPX (primary)": head_to_head(s, "SPX"), "all indexes": head_to_head(s, None)}
    print(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        o = train()
        print(f"secondary models trained on development through {o['dev_through']}; scale_b0 {o['scale_b0']:.1f}, "
              f"scale_c8 {o['scale_c8']:.1f}; sha256 {hashlib.sha256(MODELS.read_bytes()).hexdigest()[:16]}")
    elif sys.argv[1:] == ["report"]:
        report()
