"""Shared estimation machinery for chokepoint probes.

Extracted from `probe/eskom_pgm.py` when a second probe needed the same tools.
Every guard here exists because a specific failure already happened in this
track, and each one is documented with the failure it prevents. They are shared
rather than copied so that a probe cannot quietly omit one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SIGNIFICANCE_T = 2.0
LEAKAGE_ACC = 0.60


@dataclass(frozen=True)
class OLSResult:
    beta: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    resid: np.ndarray
    n: int
    names: list[str]

    def t_of(self, name: str) -> float:
        return float(self.tstat[self.names.index(name)])

    def beta_of(self, name: str) -> float:
        return float(self.beta[self.names.index(name)])


def _nw_lags(n: int) -> int:
    """Newey-West bandwidth, standard rule of thumb: 4(n/100)^(2/9)."""
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def ols(y: np.ndarray, X: np.ndarray, names: list[str],
        hac: bool = True) -> OLSResult:
    """OLS with optional Newey-West HAC standard errors.

    HAC matters whenever the regressor is persistent — loadshedding stage,
    inventory levels, anything with momentum. Classical OLS errors on a
    persistent regressor are understated badly enough to turn noise into a
    'significant' result.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if n <= k:
        raise ValueError(f"not enough observations: n={n}, k={k}")

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    if not hac:
        sigma2 = float(resid @ resid) / (n - k)
        V = sigma2 * XtX_inv
    else:
        L = _nw_lags(n)
        u = X * resid[:, None]              # (n, k) score contributions
        Omega = u.T @ u                     # lag 0
        for lag in range(1, L + 1):
            w = 1.0 - lag / (L + 1.0)       # Bartlett kernel
            G = u[lag:].T @ u[:-lag]
            Omega += w * (G + G.T)
        V = XtX_inv @ Omega @ XtX_inv

    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    return OLSResult(beta=beta, se=se, tstat=t, resid=resid, n=n, names=names)


class InsufficientPower(SystemExit):
    """The test cannot distinguish 'no effect' from 'no experiment'."""


def require_power(x: np.ndarray, split: int, min_events: int = 20,
                  min_std: float = 1e-12) -> None:
    """Refuse to render a verdict when a split has no regressor variation.

    Written after the Eskom probe printed a confident "NO TRANSMISSION LAG
    DETECTED" with every out-of-sample t-statistic exactly 0.00, because the
    eval split had landed inside South Africa's ten-month loadshedding-free
    window and the regressor was constant. It had tested nothing.

    A null is only evidence when the test could have detected an effect.
    """
    for name, seg in (("train", x[:split]), ("eval", x[split:])):
        events = int(np.count_nonzero(seg))
        if events < min_events or float(np.std(seg)) <= min_std:
            raise InsufficientPower(
                f"\nINCONCLUSIVE — the {name} split has {events} non-zero "
                f"observations in {len(seg)} (min {min_events}), "
                f"std {float(np.std(seg)):.3g}.\n\n"
                f"  A regressor with no variation yields coefficients of "
                f"exactly zero, so this run would report a confident null "
                f"while testing nothing.\n"
                f"  That is not evidence of no effect; it is the absence of "
                f"an experiment.\n"
            )


def factor_neutralize(target: pd.Series, factors: pd.DataFrame,
                      split: int) -> pd.Series:
    """Residualize `target` against `factors`, fitting betas on TRAIN only.

    Fitting on the full sample would let eval-period information define the
    residual, which is leakage — the residual would be constructed partly from
    the returns we are about to predict.

    Separately, this is what stops beta being mistaken for alpha: a copper
    miner mostly moves with copper, so predicting its RAW return mostly
    rediscovers that. Only the residual is a claim about skill.
    """
    X = np.column_stack([np.ones(len(factors)), factors.to_numpy()])
    y = target.to_numpy()
    fit = ols(y[:split], X[:split], ["const", *factors.columns], hac=False)
    return pd.Series(y - X @ fit.beta, index=target.index,
                     name=f"{target.name}_resid")
