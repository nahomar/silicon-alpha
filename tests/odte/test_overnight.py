import numpy as np
import pandas as pd

from odte.research import overnight as on


def _grid(t, quotes):
    cols = pd.MultiIndex.from_tuples(list(quotes), names=["right", "strike"])
    bid = pd.DataFrame([[v[0] for v in quotes.values()]], index=[t], columns=cols)
    ask = pd.DataFrame([[v[1] for v in quotes.values()]], index=[t], columns=cols)
    return {"bid": bid, "ask": ask}


def test_entry_uses_the_bot_wing_rule_and_exit_prices_the_same_fly():
    t = pd.Timestamp("2024-03-01 15:59")
    q = {("C", 100.0): (2.0, 2.2), ("P", 100.0): (1.9, 2.1), ("C", 105.0): (0.4, 0.5), ("P", 95.0): (0.5, 0.6),
         ("C", 110.0): (0.1, 0.15), ("P", 90.0): (0.1, 0.15)}
    e = on.entry_fly(_grid(t, q), t, 100.3)
    assert e["K"] == 100.0 and e["W"] == 10.0                 # straddle 4.1 -> wings >= 6.15 -> 10
    assert np.isclose(e["credit_mid"], 4.1 - (0.125 + 0.125))
    x = on.exit_value(_grid(t, q), t, 100.0, 10.0)
    assert np.isclose(x["value_mid"], e["credit_mid"])
    assert x["value_touch"] > x["value_mid"]                   # buying back costs more at the touch


def test_overnight_pnl_and_costs():
    pnl, risk = on.nightly_pnl(3.0, 2.0, 10.0, 0.0, 0.0)
    assert np.isclose(pnl, 100 - on.FEES8) and np.isclose(risk, 700 + on.FEES8)
    a, _ = on.nightly_pnl(3.0, 2.0, 10.0, 0.02, 0.04, share=0.5)
    b, _ = on.nightly_pnl(3.0, 2.0, 10.0, 0.02, 0.04, share=1.0)
    assert pnl > a > b
