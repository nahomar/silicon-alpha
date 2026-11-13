import numpy as np
import pandas as pd

from odte.research import combo


def test_walk_forward_never_uses_the_future():
    days = pd.bdate_range("2018-01-01", periods=700)
    d = pd.DataFrame({"day": np.repeat(days, 2)})
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=len(d))})
    y = pd.Series(rng.normal(size=len(d)))
    p1 = combo.walk_forward(d, X, y, "ml_ridge")
    y2 = y.copy()
    y2.iloc[-50:] = 1e6                               # change the last days' outcomes only
    p2 = combo.walk_forward(d, X, y2, "ml_ridge")
    first_pred = np.where(np.isfinite(p1))[0][0]
    assert first_pred == 2 * combo.MIN_DAYS           # nothing predicted before the first fit
    assert np.allclose(p1[:-50], p2[:-50], equal_nan=True)  # earlier predictions unaffected
