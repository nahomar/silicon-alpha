"""Short-horizon return predictor.

Two backends, same interface:
  - "gbm"         : sklearn GradientBoosting — fast, CPU, baseline
  - "transformer" : reuses models.transformer.MarketTransformer over
                    token-bucketed microprice moves

Target: next-k-step log return of the microprice.
Output at inference: expected next-k return AND predicted variance
(from GBM residuals or transformer entropy) so the quoter can size.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


FEATURE_COLS = [
    "imbalance", "rel_spread", "micro_dev", "ofi", "vpin", "rv", "vr",
]


@dataclass
class ShortHorizonPredictor:
    horizon_steps: int = 5
    model: Pipeline | None = None
    _resid_std: float = 0.0
    _cols: list = field(default_factory=list)

    def build_features(self, tob_feats: pd.DataFrame,
                       tox: pd.DataFrame | None = None) -> pd.DataFrame:
        X = tob_feats[[c for c in FEATURE_COLS if c in tob_feats.columns]].copy()
        if tox is not None:
            for c in ("vpin", "rv", "vr"):
                if c in tox.columns:
                    X[c] = tox[c].reindex(X.index).ffill().fillna(0.0).values
        return X.fillna(0.0)

    def build_target(self, mid: pd.Series) -> pd.Series:
        return (np.log(mid.shift(-self.horizon_steps) / mid)).rename("y")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ShortHorizonPredictor":
        mask = y.notna() & X.notna().all(axis=1)
        Xf, yf = X.loc[mask], y.loc[mask]
        self._cols = list(Xf.columns)
        self.model = Pipeline([
            ("sc", StandardScaler()),
            ("gbm", GradientBoostingRegressor(max_depth=3, n_estimators=120)),
        ])
        self.model.fit(Xf.values, yf.values)
        resid = yf.values - self.model.predict(Xf.values)
        self._resid_std = float(np.std(resid))
        log.info("ShortHorizonPredictor fit: n=%d  resid_std=%.2e", len(Xf), self._resid_std)
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu = self.model.predict(X[self._cols].fillna(0.0).values)
        sigma = np.full_like(mu, self._resid_std)
        return mu, sigma

    def save(self, path: Path) -> None:
        joblib.dump({"model": self.model, "cols": self._cols,
                     "resid_std": self._resid_std,
                     "horizon": self.horizon_steps}, path)

    @classmethod
    def load(cls, path: Path) -> "ShortHorizonPredictor":
        b = joblib.load(path)
        m = cls(horizon_steps=b["horizon"])
        m.model = b["model"]; m._cols = b["cols"]; m._resid_std = b["resid_std"]
        return m
