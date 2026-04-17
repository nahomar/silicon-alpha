"""Fill-probability model.

For each (quote, book-state) pair, predicts P(fill within Δt).

Inputs (features):
  - distance-to-touch (ticks)
  - book imbalance, spread, microprice deviation
  - short-horizon predicted return
  - toxicity (VPIN)
  - queue ahead (if provided)

Backbone: calibrated gradient-boosted classifier (sklearn) with isotonic
calibration so outputs are true probabilities. Replace with lightgbm or
xgboost for production. Transformer embeddings can be concatenated in.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "distance_ticks", "imbalance", "rel_spread", "micro_dev",
    "predicted_return", "vpin", "queue_ahead",
]


@dataclass
class FillProbabilityModel:
    horizon_steps: int = 50
    model: Pipeline | None = None

    def build_labels(self, df: pd.DataFrame, side: str, distance_ticks: float,
                     tick_size: float) -> pd.Series:
        """Label = 1 if our quote at `distance_ticks` from touch is hit within
        `horizon_steps`. Assumes we're at the back of the queue at that level.
        """
        if side == "bid":
            our_px = df["bid_px"] - distance_ticks * tick_size
            hit = (df["last_px"].rolling(self.horizon_steps)
                   .min().shift(-self.horizon_steps) <= our_px)
        else:
            our_px = df["ask_px"] + distance_ticks * tick_size
            hit = (df["last_px"].rolling(self.horizon_steps)
                   .max().shift(-self.horizon_steps) >= our_px)
        return hit.astype(int)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FillProbabilityModel":
        cols = [c for c in FEATURE_COLS if c in X.columns]
        base = Pipeline([
            ("sc", StandardScaler()),
            ("gbm", GradientBoostingClassifier(max_depth=3, n_estimators=80)),
        ])
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self.model.fit(X[cols].fillna(0.0).values, y.values)
        self._cols = cols
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() first")
        return self.model.predict_proba(X[self._cols].fillna(0.0).values)[:, 1]

    def save(self, path: Path) -> None:
        joblib.dump({"model": self.model, "cols": self._cols,
                     "horizon_steps": self.horizon_steps}, path)

    @classmethod
    def load(cls, path: Path) -> "FillProbabilityModel":
        blob = joblib.load(path)
        m = cls(horizon_steps=blob["horizon_steps"])
        m.model = blob["model"]; m._cols = blob["cols"]
        return m
