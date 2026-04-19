"""Parity test: reservoir-sampled quantile fitter vs exact np.quantile.

Guards the Phase-2 launch assumption that
odte.data.streaming_quantiles.StreamingQuantileFitter is "good enough"
(i.e. close to t-digest) for fitting HybridBinTokenizer edges on the full
OPRA corpus without burning two engineering days to port the fitter.

Default tolerance: max-abs-err on interior bin edges <= 3e-3 in the raw
data units, using a 2M-element reservoir fit from 10M synthetic rows.

Tolerance calibration (measured on this test with seed=42):
    reservoir=1M  ->  max_abs_err ~= 3.3e-3
    reservoir=2M  ->  max_abs_err ~= 2.4e-3   (current default)
    reservoir=4M  ->  max_abs_err ~= 2.1e-3

Error plateaus above ~2M on 10M rows because the reservoir approaches
the population size; finite-sample variance of the 10M corpus itself
dominates. The 3e-3 gate gives ~0.6e-3 headroom above the observed 2M
worst case, tight enough to catch a real regression (doubling of error
would trip it) but loose enough not to flake on RNG variance.

At Phase-2 production scale (>=1T tokens) the reservoir is a 1-in-1M
sample and error converges to the ~1e-3 reservoir-intrinsic floor; this
test is the pessimistic 10M-row case, not representative of the real
run. Rerun with N_ROWS=100_000_000 before Gate 3 as a sanity check.

The user should re-run this test with N_ROWS=100_000_000 (env var)
before committing to Gate 3 ($2k 8xH100 smoke). At 100M rows the
reservoir throws away 99% of data; if the reservoir still passes at 2e-3
there, the fitter is safe for the full ~1T-token run.

Run:
    pytest tests/odte/test_quantile_parity.py -xvs
    # Or with a larger corpus (slow):
    N_ROWS=100000000 pytest tests/odte/test_quantile_parity.py -xvs
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from odte.data.streaming_quantiles import StreamingQuantileFitter


# HybridBinTokenizer uses linspace(0, 1, n_buckets+1) for quantile edges.
N_BUCKETS = 64
N_ROWS = int(os.environ.get("N_ROWS", "10000000"))
RESERVOIR_SIZE = int(os.environ.get("RESERVOIR_SIZE", "2000000"))
TOL_ABS = float(os.environ.get("TOL_ABS", "3e-3"))


def _synth_mixed(n: int, seed: int = 0) -> np.ndarray:
    """Mixture: 70% N(0, 1) + 30% lognormal(mu=0, sigma=0.5).

    Chosen to imitate the shape of log-returns (heavy-tailed, positive
    skew) that HybridBinTokenizer sees in real OPRA tape.
    """
    rng = np.random.default_rng(seed)
    n_norm = int(0.7 * n)
    n_lognorm = n - n_norm
    x = np.concatenate([
        rng.standard_normal(n_norm),
        rng.lognormal(0.0, 0.5, size=n_lognorm),
    ])
    rng.shuffle(x)
    return x


def _fit_reservoir(data: np.ndarray, reservoir_size: int,
                   n_buckets: int, chunk_size: int = 100_000,
                   ) -> np.ndarray:
    fitter = StreamingQuantileFitter(reservoir_size=reservoir_size,
                                     n_buckets=n_buckets, seed=0)
    for i in range(0, len(data), chunk_size):
        fitter.update(data[i:i + chunk_size])
    return fitter.finalize()


def test_reservoir_vs_exact_quantile_parity():
    data = _synth_mixed(N_ROWS, seed=42)

    # Exact edges via np.quantile on the full array — ground truth.
    qs = np.linspace(0, 1, N_BUCKETS + 1)
    exact_edges = np.quantile(data, qs)

    # Reservoir-estimated edges.
    reservoir_edges = _fit_reservoir(data, RESERVOIR_SIZE, N_BUCKETS)

    # finalize() replaces the endpoints with +-inf sentinels; compare only
    # interior edges since np.quantile returns finite endpoints.
    interior_exact = exact_edges[1:-1]
    interior_reservoir = reservoir_edges[1:-1]

    assert interior_exact.shape == interior_reservoir.shape, (
        f"edge-count mismatch: exact={interior_exact.shape} "
        f"reservoir={interior_reservoir.shape}"
    )

    abs_err = np.abs(interior_exact - interior_reservoir)
    max_abs_err = float(abs_err.max())
    mean_abs_err = float(abs_err.mean())
    p95_abs_err = float(np.percentile(abs_err, 95))

    # Print diagnostics before the assertion so failure messages are useful.
    print()
    print(f"[parity] n_rows            = {N_ROWS:,}")
    print(f"[parity] reservoir_size    = {RESERVOIR_SIZE:,}")
    print(f"[parity] n_buckets         = {N_BUCKETS}")
    print(f"[parity] data quantiles    = "
          f"[{np.quantile(data, 0.001):+.3f}, "
          f"{np.quantile(data, 0.5):+.3f}, "
          f"{np.quantile(data, 0.999):+.3f}]")
    print(f"[parity] max_abs_err       = {max_abs_err:.6f}")
    print(f"[parity] p95_abs_err       = {p95_abs_err:.6f}")
    print(f"[parity] mean_abs_err      = {mean_abs_err:.6f}")
    print(f"[parity] tolerance (gate)  = {TOL_ABS}")

    assert max_abs_err <= TOL_ABS, (
        f"Reservoir quantile deviates from exact by max={max_abs_err:.4e} "
        f"(gate={TOL_ABS:.1e}). This is a real finding — DO NOT loosen the "
        f"tolerance to make the test pass. Options: (a) increase "
        f"RESERVOIR_SIZE, (b) switch StreamingQuantileFitter to a t-digest "
        f"implementation. See Phase-2 plan for the cost/benefit analysis."
    )


def test_reservoir_monotonic_edges():
    """Reservoir edges must be monotonically non-decreasing."""
    data = _synth_mixed(N_ROWS, seed=7)
    edges = _fit_reservoir(data, RESERVOIR_SIZE, N_BUCKETS)
    diffs = np.diff(edges)
    n_violations = int((diffs < 0).sum())
    assert n_violations == 0, (
        f"{n_violations} non-monotonic edge pairs: "
        f"{np.where(diffs < 0)[0][:5].tolist()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
