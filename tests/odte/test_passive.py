import numpy as np
import pandas as pd

from odte.research import passive as pv


def test_unfilled_orders_do_not_trade():
    d = pd.DataFrame({"pnl_short_fly_midx": [100.0, -500.0], "risk_short_fly_midx": [1000.0, 1000.0],
                      "fill_mid_short": [True, False]})
    assert pv.passive_ror(d).tolist() == [0.1, 0.0]
    assert pv.mid_ror(d).tolist() == [0.1, -0.5]
