"""QP executor with broker-margin-aware constraints.

Extends odte.executor.DeterministicExecutor to accept a live broker margin
table and fold the required-equity constraint into the QP itself — so the
solver can never produce weights that would be rejected at submission.

Constraint added to the base QP:

    max  μᵀw  −  (λ/2) wᵀΣw
    s.t. ‖w‖₁ ≤ B                                (gross risk budget)
         k_gross · (|w|·notional).sum()          (broker margin:
           + k_delta · (|w|·|δ|·spot).sum()        piecewise linear form
           + k_vega  · (|w|·|𝒱|).sum()             with live broker k's)
           + k_gamma · (|w|·|γ|·spot²).sum() ≤ equity

Warm-starts from the previous solution; falls back to the L1-projection
solver in `odte.executor._solve_projected` if CVXPY is unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .broker_margin import (
    BrokerMarginTable, MarginCoefficients, TableLookup,
)

log = logging.getLogger(__name__)

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


@dataclass
class InstrumentGreeks:
    """Per-contract metadata the QP needs for the margin term."""
    spot: np.ndarray              # (N,)
    delta: np.ndarray             # (N,)
    gamma: np.ndarray             # (N,)
    vega: np.ndarray              # (N,)
    multiplier: np.ndarray        # (N,) contract size, typically 100


@dataclass
class QPResult:
    w: np.ndarray
    required_margin: float
    status: str
    margin_coefs: MarginCoefficients


@dataclass
class QPExecutor:
    """Margin-aware QP executor with warm-start."""
    lam: float = 5.0
    B: float = 1.0
    broker: Optional[BrokerMarginTable] = None
    _prev_w: Optional[np.ndarray] = field(default=None, init=False)

    # -------------------------------------------------------------------
    def solve(self,
              mu: np.ndarray, Sigma: np.ndarray,
              greeks: InstrumentGreeks,
              equity: float,
              lookup: TableLookup,
              B: Optional[float] = None, lam: Optional[float] = None,
              bounds: Optional[Sequence[tuple]] = None) -> QPResult:
        if self.broker is not None:
            self.broker.maybe_reload()
        coefs = self.broker.resolve(lookup) if self.broker is not None \
            else MarginCoefficients.defaults()
        B = float(B if B is not None else self.B)
        lam = float(lam if lam is not None else self.lam)

        if _HAS_CVXPY:
            w, status = self._solve_cvxpy(mu, Sigma, greeks, equity, coefs, B, lam, bounds)
        else:
            w, status = self._solve_projected(mu, Sigma, greeks, equity, coefs, B, lam)

        req = self._required_margin(w, greeks, coefs)
        self._prev_w = w
        return QPResult(w=w, required_margin=float(req),
                        status=status, margin_coefs=coefs)

    # -------- CVXPY path -----------------------------------------------
    def _solve_cvxpy(self, mu, Sigma, g, equity, coefs, B, lam, bounds):
        n = len(mu)
        w = cp.Variable(n)
        if self._prev_w is not None and len(self._prev_w) == n:
            w.value = self._prev_w
        # absolute-value proxies via the convex bound |w| = w⁺ + w⁻
        wp = cp.Variable(n, nonneg=True)
        wn = cp.Variable(n, nonneg=True)
        obj = cp.Maximize(mu @ w - 0.5 * lam * cp.quad_form(w, cp.psd_wrap(Sigma)))

        notional = g.spot * g.multiplier
        delta_dollars = np.abs(g.delta) * g.spot * g.multiplier
        vega_units = np.abs(g.vega) * g.multiplier
        gamma_dollars = np.abs(g.gamma) * g.spot * g.spot * g.multiplier

        margin_expr = (
            coefs.k_gross * (notional @ (wp + wn)) +
            coefs.k_delta * (delta_dollars @ (wp + wn)) +
            coefs.k_vega  * (vega_units @ (wp + wn)) +
            coefs.k_gamma * (gamma_dollars @ (wp + wn))
        )

        cons = [
            w == wp - wn,
            cp.sum(wp + wn) <= B,
            margin_expr <= max(equity - coefs.floor_dollars, 0.0),
        ]
        if bounds is not None:
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            cons += [w >= lo, w <= hi]
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.SCS, warm_start=self._prev_w is not None, verbose=False)
        except Exception as e:
            log.warning("CVXPY failed (%s); projected fallback", e)
            return self._solve_projected(mu, Sigma, g, equity, coefs, B, lam) + ("cvxpy_fail",)
        if w.value is None:
            return np.zeros(n), "infeasible"
        return np.asarray(w.value).reshape(-1), prob.status

    # -------- projected fallback ---------------------------------------
    def _solve_projected(self, mu, Sigma, g, equity, coefs, B, lam, steps: int = 400):
        from odte.executor import _project_l1_ball
        n = len(mu)
        w = np.zeros(n) if self._prev_w is None else self._prev_w.copy()
        marg_slope = (
            coefs.k_gross * g.spot * g.multiplier +
            coefs.k_delta * np.abs(g.delta) * g.spot * g.multiplier +
            coefs.k_vega  * np.abs(g.vega) * g.multiplier +
            coefs.k_gamma * np.abs(g.gamma) * g.spot * g.spot * g.multiplier
        )
        for _ in range(steps):
            grad = -mu + lam * Sigma @ w
            w = w - 0.05 * grad
            w = _project_l1_ball(w, B)
            # Soft margin projection: if current |w| @ marg_slope > equity,
            # rescale uniformly.
            cost = float(np.abs(w) @ marg_slope)
            cap = max(equity - coefs.floor_dollars, 0.0)
            if cost > cap and cost > 0:
                w *= cap / cost
        return w, "projected"

    # -------- diagnostics ----------------------------------------------
    @staticmethod
    def _required_margin(w: np.ndarray, g: InstrumentGreeks,
                         c: MarginCoefficients) -> float:
        aw = np.abs(w)
        return float(
            c.k_gross * (aw * g.spot * g.multiplier).sum() +
            c.k_delta * (aw * np.abs(g.delta) * g.spot * g.multiplier).sum() +
            c.k_vega  * (aw * np.abs(g.vega) * g.multiplier).sum() +
            c.k_gamma * (aw * np.abs(g.gamma) * g.spot * g.spot * g.multiplier).sum()
        )
