import numpy as np
import pandas as pd

from odte.research import xindex as xi


def test_rule_sells_only_when_spx_is_lowest():
    s, n, r = pd.Series([0.8, 0.9, 0.7]), pd.Series([0.85, 0.85, 0.7]), pd.Series([0.9, 0.95, 0.9])
    assert xi.rule_r1(s, n, r).tolist() == [True, False, False]     # ties do not count


def test_spx_ror_matches_hand_calc():
    d = pd.DataFrame({"fly_credit_mid": [10.0], "W": [30.0], "pnl_short_fly_mid": [400.0]})
    r0 = xi.spx_ror(d, 0.0).iloc[0]
    assert np.isclose(r0, ((10 - 6) * 100 - xi.FEES4) / ((30 - 10) * 100 + xi.FEES4))
    assert xi.spx_ror(d, 0.5).iloc[0] < r0
