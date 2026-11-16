"""Version-proof prediction for the frozen linear models (median impute -> standardise -> ridge).

The frozen exam model and D4L's forecast were saved with scikit-learn 1.6.1; the live bot runs
1.9.1, whose SimpleImputer.transform needs an attribute (_fill_dtype) the old pickles lack, so
pipeline.predict raises there. Their fitted numbers are all present, so the prediction is computed
directly from them: identical to pipeline.predict in the environment that fitted it (checked in
tests/odte/test_linear.py), and independent of the installed scikit-learn.
"""
from __future__ import annotations

import numpy as np


def linear_predict(pipe, X) -> np.ndarray:
    steps = [s for _, s in getattr(pipe, "steps", [])]
    names = [type(s).__name__ for s in steps]
    if names != ["SimpleImputer", "StandardScaler", "Ridge"]:
        return pipe.predict(X)                              # not the frozen shape: use the library
    imp, sc, rg = steps
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    st = np.asarray(imp.statistics_, dtype=float)
    X = np.where(np.isnan(X), st[None, :], X)
    keep = ~np.isnan(st)                                    # all-missing training columns are dropped
    if not getattr(imp, "keep_empty_features", False) and keep.sum() != len(st):
        X = X[:, keep]
    mean = np.asarray(sc.mean_, dtype=float) if getattr(sc, "with_mean", True) else 0.0
    scale = np.asarray(sc.scale_, dtype=float) if getattr(sc, "with_std", True) else 1.0
    Z = (X - mean) / scale
    return Z @ np.asarray(rg.coef_, dtype=float).T + rg.intercept_
