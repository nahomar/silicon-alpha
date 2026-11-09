"""Signal factory statistics: FDR, clustered IC, no look-ahead in history z-scores."""
import numpy as np
import pandas as pd

from odte.research import factory as fx


def test_bh_controls_false_discoveries():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(size=95), [1e-6, 1e-5, 1e-4, 2e-4, 3e-4]])
    passed = fx.bh(p, 0.05)
    assert passed[-5:].all() and passed[:95].sum() <= 2


def test_clustered_ic_is_honest_about_duplicated_rows():
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(300), 6)
    s = pd.Series(np.repeat(rng.normal(size=300), 6))          # one value per day, repeated
    t = pd.Series(np.repeat(rng.normal(size=300), 6))
    r = fx.ic_test(s, t, pd.Series(days))
    assert abs(r["t"]) < 3.5                                   # no fake significance from 6x rows


def test_history_zscore_uses_only_earlier_days():
    d = pd.DataFrame({"hhmm": ["10:00"] * 30, "x": np.arange(30, dtype=float)})
    z1 = fx._hist_z(d, "x")
    d2 = d.copy()
    d2.loc[29, "x"] = 1e6                                      # change the last day only
    z2 = fx._hist_z(d2, "x")
    assert np.allclose(z1[:29], z2[:29], equal_nan=True)
