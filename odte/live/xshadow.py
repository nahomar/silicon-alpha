"""NDX and RUT next to SPX, shadow only: no NDX/RUT orders, ever (from this module).

Why (docs/expiry_edge_plan.md):
  - the event-day seller passed on sealed NDX/RUT 2025-26; recording those
    indexes live roughly triples its live evidence per event day;
  - edge #10 (sell SPX only when the model ranks SPX the most overpriced of the
    three) was a near-miss, +$115/day [-9, +232]; a live shadow is its honest test.

Each index gets its own ridge model (same features and target as the SPX model),
trained on the seen pair-trade development rows 2022-05 .. 2024-12. Quotes use
extra market-data lines (IBKR Quote Booster); without them the shadows log
"no quotes" and the SPX bot is unaffected.

  python -m odte.live.xshadow train      # writes data/live/model_ndx.pkl, model_rut.pkl
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from odte.live.config import Config

ROOT = Path(__file__).resolve().parents[2]
SPEC = {"NDX": {"step": 50.0, "inner": 0.01, "outer": 100.0, "band": 0.025, "lines": 44},
        "RUT": {"step": 5.0, "inner": 0.01, "outer": 10.0, "band": 0.025, "lines": 44}}
TRAIN = {s: ROOT / "reports" / "expiry_edge" / f"decisions-pair-dev-{s.lower()}.parquet" for s in SPEC}
SHADOW_RULES = ("event_seller", "always_sell")


def model_path(cfg: Config, sym: str) -> Path:
    return cfg.model_path.parent / f"model_{sym.lower()}.pkl"


def train(sym: str, cfg: Config = Config(), path: Path | None = None) -> dict:
    from odte.eval.expiry_edge import _make_model
    from odte.live.model import feature_list
    cols = feature_list()
    d = pd.read_parquet(TRAIN[sym])
    d = d[d["on6"]][["day", *cols, "ratio"]].dropna(subset=["ratio"])
    m = _make_model("ml_ridge").fit(d[cols].to_numpy(float), d["ratio"].clip(upper=4.0).to_numpy())
    b = {"model": m, "features": cols, "n_rows": int(len(d)), "name": f"ratio_ridge_{sym.lower()}",
         "kind": "ratio", "days": [str(pd.Timestamp(d["day"].min()).date()), str(pd.Timestamp(d["day"].max()).date())]}
    path = path or model_path(cfg, sym)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(b))
    return b


@dataclass
class Side:
    sym: str
    bundle: dict
    prev: float
    chain: list = field(default_factory=list)
    last_row: dict | None = None


def setup(cfg: Config, broker, day, spx_spot: float, spx_prev: float, j) -> list:
    """Subscribe each shadow index's strikes; any failure only disables that index."""
    from odte.live.broker import strikes_for
    sides = []
    for sym in cfg.shadow_indexes:
        try:
            if not hasattr(broker, "subscribe_extra"):
                raise RuntimeError("broker has no extra-index support")
            mp = model_path(cfg, sym)
            if not mp.exists():
                raise RuntimeError(f"no model at {mp} (run: python -m odte.live.xshadow train)")
            bundle = pickle.loads(mp.read_bytes())
            prev = broker.prev_close_extra(sym, day)
            move = spx_spot / spx_prev if np.isfinite(spx_prev) and spx_prev > 0 else 1.0
            est = prev * move                          # yesterday's close moved like SPX overnight
            sp = SPEC[sym]
            ks = strikes_for(est, sp["inner"], sp["band"], sp["outer"], step=sp["step"], max_lines=sp["lines"])
            broker.subscribe_extra(sym, day, ks)
            sides.append(Side(sym, bundle, prev))
            j.log(broker.now(), "shadow_index", sym=sym, prev_close=round(prev, 2), est_spot=round(est, 2),
                  n_strikes=len(ks), model=bundle["name"])
        except Exception as e:
            j.log(broker.now(), "note", text=f"{sym} shadow off today: {e!r}")
    return sides


def record(sides: list, broker) -> None:
    for s in sides:
        try:
            s.chain.append(broker.snapshot_extra(s.sym))
        except Exception:
            pass


def decide(sides: list, hhmm: str, dctx: dict, recent: bool, now, j, shadows: list) -> dict:
    """Shadow trades on each index's own fly; returns {sym: prediction} for the cross-index rule."""
    from odte.live.features import feature_row
    from odte.live.model import predict
    from odte.live.strategies import STRATEGIES
    preds = {}
    for s in sides:
        rows = [c for c in s.chain if len(c)]
        r = feature_row(pd.concat(rows, ignore_index=True), hhmm, dctx, s.prev) if rows else None
        if r is None:
            j.log(now, "shadow", strategy=f"index:{s.sym}", hhmm=hhmm, side=0, reason="no quotes")
            continue
        s.last_row = r
        p = predict(s.bundle, r)
        preds[s.sym] = p
        for name in SHADOW_RULES:
            side, reason = STRATEGIES[name](0, recent, r)
            shadows.append({"strategy": f"{name}:{s.sym}", "hhmm": hhmm, "side": side, "row": r, "index": s.sym})
            j.log(now, "shadow", strategy=f"{name}:{s.sym}", hhmm=hhmm, side=side, reason=reason,
                  pred=round(p, 4) if np.isfinite(p) else None, K=r["K"], W=r["W"],
                  mid=r["fly_credit_mid"], touch=r["fly_credit"])
    return preds


def xindex_side(spx_pred: float, preds: dict, recent: bool) -> tuple:
    """Edge #10: sell SPX only when its prediction is the lowest of SPX/NDX/RUT."""
    vals = [preds.get(k, np.nan) for k in ("NDX", "RUT")]
    if recent:
        return 0, "within 60 min after the release"
    if not (np.isfinite(spx_pred) and all(np.isfinite(v) for v in vals)):
        return 0, "missing an index prediction"
    return (1, "SPX richest") if spx_pred < min(vals) else (0, "SPX not the richest")


def settles(sides: list, broker, day) -> dict:
    """{sym: (close, source)}: official close if available, else the last implied forward."""
    from odte.eval.expiry_edge import implied_forward, minute_grid
    out = {}
    for s in sides:
        try:
            out[s.sym] = (broker.settlement_extra(s.sym, day), "official")
            continue
        except Exception:
            pass
        rows = [c for c in s.chain if len(c)]
        if rows:
            F = implied_forward(minute_grid(pd.concat(rows, ignore_index=True))).dropna()
            if len(F):
                out[s.sym] = (float(F.iloc[-1]), "last implied forward")
    return out


def save(sides: list, cfg: Config, day, tag: str) -> None:
    for s in sides:
        rows = [c for c in s.chain if len(c)]
        if rows:
            from odte.live.runner import merge_save
            merge_save(rows, cfg.chain_dir / s.sym.lower() / f"{pd.Timestamp(day):%Y-%m-%d}{tag}.parquet")


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        for sym in SPEC:
            b = train(sym)
            print(f"{sym}: {b['n_rows']:,} rows {b['days'][0]} .. {b['days'][1]} -> {model_path(Config(), sym)}")
