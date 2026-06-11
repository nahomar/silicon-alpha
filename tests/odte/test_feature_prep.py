"""Regression test: returns must be per-instrument, never cross-contract.

The OPRA feed under parent symbology is a single time-ordered stream that
interleaves every child contract. A naive global ``log(mid).diff()`` therefore
computes ``log(mid of contract A) - log(mid of contract B)`` -- a cross-contract
log-ratio, not a return -- which silently corrupts the model's directional
target. This test pins the corrected per-instrument behavior so the bug cannot
return.

Run:
    PYTHONPATH=. pytest tests/odte/test_feature_prep.py -xvs
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from odte.data.datashop_pack import prepare_features


def _row(ts_ns, inst, mid):
    return {"quote_datetime": ts_ns, "bid": mid, "ask": mid,
            "bid_size": 1, "ask_size": 1, "trade_volume": 0, "instrument_id": inst}


def test_returns_are_per_instrument_not_cross_contract():
    # Two contracts (A=1, B=2) interleaved in global time order:
    #   t1 A=100, t2 B=200, t3 A=101, t4 B=198, t5 A=102
    base = pd.Timestamp("2026-04-16 14:30:00").value  # ns
    s = 1_000_000_000  # 1s in ns
    df = pd.DataFrame([
        _row(base + 1 * s, 1, 100.0),
        _row(base + 2 * s, 2, 200.0),
        _row(base + 3 * s, 1, 101.0),
        _row(base + 4 * s, 2, 198.0),
        _row(base + 5 * s, 1, 102.0),
    ])

    out = prepare_features(df)

    # Per-instrument expected within-contract log-returns (first row = 0).
    a = out[out["instrument_id"] == 1].sort_values("ts_ms")
    b = out[out["instrument_id"] == 2].sort_values("ts_ms")
    np.testing.assert_allclose(
        a["ret"].to_numpy(),
        [0.0, math.log(101 / 100), math.log(102 / 101)], atol=1e-12)
    np.testing.assert_allclose(
        b["ret"].to_numpy(),
        [0.0, math.log(198 / 200)], atol=1e-12)

    # And explicitly: none of the corrupted cross-contract values appear.
    forbidden = [math.log(200 / 100), math.log(101 / 200), math.log(198 / 101)]
    for bad in forbidden:
        assert not np.any(np.isclose(out["ret"].to_numpy(), bad, atol=1e-9)), \
            f"cross-contract return {bad:.4f} leaked into the target"


def test_single_instrument_stream_still_works():
    # No instrument column with >1 unique value -> global diff (single series).
    base = pd.Timestamp("2026-04-16 14:30:00").value
    s = 1_000_000_000
    df = pd.DataFrame([
        {"quote_datetime": base + i * s, "bid": 100.0 + i, "ask": 100.0 + i,
         "bid_size": 1, "ask_size": 1, "trade_volume": 0}
        for i in range(4)
    ])
    out = prepare_features(df).sort_values("ts_ms")
    expected = [0.0] + [math.log((100 + i) / (100 + i - 1)) for i in range(1, 4)]
    np.testing.assert_allclose(out["ret"].to_numpy(), expected, atol=1e-12)
