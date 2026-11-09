"""The model the bot trades: ridge on base + calendar + context features.

Same pipeline, target and feature list as the backtests' "ml_ridge+cal+ctx"
(odte/eval/expiry_edge._make_model, target realized/implied clipped at 4),
trained once on every seen SPX decision row (fixed six times).

  python -m odte.live.model train      # writes data/live/model.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.live.config import Config

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FILES = [ROOT / "reports" / "expiry_edge" / "decisions-lockbox-odte_0dte-2021H1.parquet",
               ROOT / "reports" / "expiry_edge" / "decisions-dev-gex-es.parquet"]


def feature_list() -> list:
    from odte.data.calendar import CAL_FEATURES
    from odte.data.context import CTX_FEATURES
    from odte.eval.expiry_edge import FEATURES
    return [*FEATURES, *CAL_FEATURES, *CTX_FEATURES, "iv_vs_vix1d"]


def train(cfg: Config = Config(), files=TRAIN_FILES) -> dict:
    from odte.eval.expiry_edge import _make_model
    cols = feature_list()
    parts = []
    for f in files:
        d = pd.read_parquet(f)
        parts.append(d[d["on6"]][["day", *cols, "ratio"]])
    d = pd.concat(parts).dropna(subset=["ratio"])
    m = _make_model("ml_ridge").fit(d[cols].to_numpy(float), d["ratio"].clip(upper=4.0).to_numpy())
    bundle = {"model": m, "features": cols, "n_rows": int(len(d)),
              "days": [str(pd.Timestamp(d["day"].min()).date()), str(pd.Timestamp(d["day"].max()).date())]}
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.model_path.write_bytes(pickle.dumps(bundle))
    return bundle


def load(cfg: Config = Config()) -> dict:
    return pickle.loads(cfg.model_path.read_bytes())


def predict(bundle: dict, row: dict) -> float:
    x = np.array([[float(row.get(c, np.nan)) for c in bundle["features"]]])
    return float(bundle["model"].predict(x)[0])


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        b = train()
        print(f"trained on {b['n_rows']:,} rows, {b['days'][0]} .. {b['days'][1]}, "
              f"{len(b['features'])} features -> {Config().model_path}")
