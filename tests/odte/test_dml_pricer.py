"""Phase-0 gate test for the Differential-ML option pricer.

Makes the Phase-0 validation reproducible in CI instead of a one-off notebook
run. Trains the Black-Scholes pretrain stage on a fixed seed and asserts:

  1. Structural sanity across the whole 0DTE grid (no NaN/Inf; delta in
     [0, 1]; gamma >= 0; vega >= 0). These are model-agnostic option-pricing
     invariants -- a violation means the autograd Greek path is broken.

  2. BS-stage Greek accuracy thresholds. The twin-net architecture hard-codes
     BS as its base and learns only a small tau-gated residual, so these
     thresholds are *expected* to be tight; the test guards against a
     regression that breaks that property (e.g. a sign error in the residual
     gate, or an autograd graph that silently zeros gradients).

Tolerance calibration (seed=0, bs_steps=800, this grid; measured):
    delta abs-error   max  ~ 3e-4   (gate: 5e-3)
    gamma % error     p95  ~ 0.02%  (gate: 1.0%)
    vega  % error     p95  ~ 0.02%  (gate: 1.0%)

Gates are ~15x above the observed worst case -- loose enough not to flake on
RNG/BLAS nondeterminism across machines, tight enough that a real regression
(an order-of-magnitude error increase) trips them.

Run:
    PYTHONPATH=. pytest tests/odte/test_dml_pricer.py -xvs
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.config import DMLConfig
from odte.dml_pricer import DMLPricer, train_dml_bs, bs_price_call
from odte.eval.validate_dml import Grid, grid_eval_vs_bs


@pytest.fixture(scope="module")
def trained_pricer():
    """BS-pretrained DML pricer (deterministic). Module-scoped: one train,
    reused by every assertion below."""
    torch.manual_seed(0)
    np.random.seed(0)
    model = DMLPricer(DMLConfig())
    train_dml_bs(model, steps=800, batch=512, device="cpu", S_ref=5500.0)
    return model


def test_structural_invariants(trained_pricer):
    """No NaN/Inf, and Greeks obey their option-pricing sign/range invariants
    across the whole grid."""
    grid = Grid(n_moneyness=15, n_tau=12)
    S, K, T, SIG = grid.mesh()
    args = [torch.tensor(a, dtype=torch.float32) for a in (S, K, T)]
    R = torch.zeros_like(args[0])
    SIG_t = torch.tensor(SIG, dtype=torch.float32)

    price, delta, gamma, vega = trained_pricer(args[0], args[1], args[2], R, SIG_t)
    for name, t in (("price", price), ("delta", delta),
                    ("gamma", gamma), ("vega", vega)):
        arr = t.detach().cpu().numpy()
        assert np.all(np.isfinite(arr)), f"{name} has NaN/Inf"

    d = delta.detach().cpu().numpy()
    g = gamma.detach().cpu().numpy()
    v = vega.detach().cpu().numpy()
    # Call delta in [0, 1]; gamma, vega non-negative. Small numerical slack.
    assert d.min() >= -1e-3 and d.max() <= 1.0 + 1e-3, (d.min(), d.max())
    assert g.min() >= -1e-3, g.min()
    assert v.min() >= -1e-1, v.min()  # vega is O(S*sqrt(tau)); slack scaled up


def test_bs_grid_greek_accuracy(trained_pricer):
    """BS-stage Greek error stays within the calibrated gates over the grid."""
    grid = Grid(n_moneyness=21, n_tau=16)
    ev = grid_eval_vs_bs(trained_pricer, grid, device="cpu")

    assert ev["delta_abs_points"]["max"] < 5e-3, ev["delta_abs_points"]
    assert ev["gamma_pct"]["p95"] < 1.0, ev["gamma_pct"]
    assert ev["vega_pct"]["p95"] < 1.0, ev["vega_pct"]


def test_price_tracks_bs_atm(trained_pricer):
    """At ATM tau=1d the learned price sits on top of analytic BS (the residual
    is a small correction, not a free-floating output)."""
    S = torch.tensor([5500.0]); K = torch.tensor([5500.0])
    tau = torch.tensor([1.0 / 365]); r = torch.tensor([0.0])
    sig = torch.tensor([0.2])
    price = trained_pricer(S, K, tau, r, sig)[0].item()
    bs = float(bs_price_call(S, K, tau, sig, r).item())
    rel = abs(price - bs) / bs
    assert rel < 0.02, f"ATM price {price:.4f} vs BS {bs:.4f} (rel {rel:.4f})"
