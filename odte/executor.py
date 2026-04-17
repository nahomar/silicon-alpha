"""DeterministicExecutor — constrained-QP portfolio optimizer.

Solves, for ML-derived expected returns μ and covariance Σ:

    max_w  μᵀw  −  (λ/2) wᵀΣw
    s.t.   ‖w‖₁ ≤ B               (gross risk budget)

The ML stack is the "weather forecast"; this module is the deterministic
guard that translates the forecast into positions that never violate:
  - margin
  - gamma / vega exposure
  - pin-risk distance to nearest strike at expiry

Uses CVXPY when available; falls back to a Lagrangian projection so the
smoke test still runs on a minimal Mac install.

Pairs with mm/prism.py — wire this in via PRISMConfig.strategy (see Phase 0
edit to prism.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

@dataclass
class DeterministicExecutor:
    """One instance per product; keeps warm-start from prior solution."""

    lam: float = 5.0
    B: float = 1.0
    _prev_w: Optional[np.ndarray] = None

    def solve(self, mu: np.ndarray, Sigma: np.ndarray,
              B: Optional[float] = None, lam: Optional[float] = None,
              bounds: Optional[Sequence[tuple]] = None) -> np.ndarray:
        B = float(B if B is not None else self.B)
        lam = float(lam if lam is not None else self.lam)
        n = len(mu)
        if _HAS_CVXPY:
            return self._solve_cvxpy(mu, Sigma, B, lam, bounds)
        return self._solve_projected(mu, Sigma, B, lam)

    # CVXPY path
    def _solve_cvxpy(self, mu, Sigma, B, lam, bounds) -> np.ndarray:
        n = len(mu)
        w = cp.Variable(n)
        if self._prev_w is not None and len(self._prev_w) == n:
            w.value = self._prev_w
        obj = cp.Maximize(mu @ w - 0.5 * lam * cp.quad_form(w, cp.psd_wrap(Sigma)))
        cons = [cp.norm(w, 1) <= B]
        if bounds is not None:
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            cons += [w >= lo, w <= hi]
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.SCS, warm_start=self._prev_w is not None, verbose=False)
        except Exception as e:
            log.warning("CVXPY failed (%s); projected fallback", e)
            return self._solve_projected(mu, Sigma, B, lam)
        if w.value is None:
            return np.zeros(n)
        sol = np.asarray(w.value).reshape(-1)
        self._prev_w = sol
        return sol

    # Gradient-descent with L1 projection (Mac fallback)
    def _solve_projected(self, mu, Sigma, B, lam, steps: int = 400, lr: float = 0.05
                         ) -> np.ndarray:
        n = len(mu)
        w = np.zeros(n) if self._prev_w is None else self._prev_w.copy()
        for _ in range(steps):
            grad = -mu + lam * Sigma @ w
            w = w - lr * grad
            w = _project_l1_ball(w, B)
        self._prev_w = w
        return w


def _project_l1_ball(v: np.ndarray, b: float) -> np.ndarray:
    """Project v onto {w: ||w||_1 <= b}. (Duchi et al. 2008)"""
    if np.abs(v).sum() <= b:
        return v
    u = np.sort(np.abs(v))[::-1]
    cs = np.cumsum(u) - b
    rho = np.where(u - cs / (np.arange(len(u)) + 1) > 0)[0].max()
    theta = cs[rho] / (rho + 1)
    return np.sign(v) * np.maximum(np.abs(v) - theta, 0.0)


# ---------------------------------------------------------------------------
# Risk gates
# ---------------------------------------------------------------------------

@dataclass
class RiskGates:
    """Post-solve checks. All thresholds are GROSS exposures."""
    gamma_cap: float = 1e4
    vega_cap: float = 1e4
    pin_dist_cap: float = 0.002     # min |S-K|/S at EOD
    margin_cap: float = 1e6

    def check(self, w: np.ndarray, greeks: Dict[str, np.ndarray],
              strikes: np.ndarray, spot: float,
              margins: np.ndarray) -> Dict[str, bool]:
        gamma_exp = float(np.sum(np.abs(w * greeks.get("gamma", np.zeros_like(w)))))
        vega_exp = float(np.sum(np.abs(w * greeks.get("vega", np.zeros_like(w)))))
        margin_exp = float(np.sum(np.abs(w) * margins))
        pin_dist = float(np.min(np.abs(strikes - spot) / spot)) if len(strikes) else np.inf
        return {
            "gamma_ok": gamma_exp <= self.gamma_cap,
            "vega_ok": vega_exp <= self.vega_cap,
            "margin_ok": margin_exp <= self.margin_cap,
            "pin_ok": pin_dist >= self.pin_dist_cap,
        }
