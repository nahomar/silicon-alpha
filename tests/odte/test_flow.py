import numpy as np

from odte.research import flow as fl


def test_trade_pnl_buy_and_sell():
    # fly mid 4, touch 4.2 (buy) / 3.8 (sell); expiry payout 10 (big move)
    pb, rb = fl.trade_pnl("buy", 4.0, 4.2, 1.0, 10.0, 10.0, share=0.5)
    assert np.isclose(pb, (10 - 4.1) * 100 - fl.FEES4) and np.isclose(rb, 410 + fl.FEES4)
    ps, rs = fl.trade_pnl("sell", 4.0, 3.8, 1.0, 10.0, 0.0, share=0.5)
    assert np.isclose(ps, 3.9 * 100 - fl.FEES4) and np.isclose(rs, (10 - 3.9) * 100 + fl.FEES4)
    # scaling the gap down (today's tighter spreads) helps both sides
    assert fl.trade_pnl("buy", 4.0, 4.2, 0.5, 10.0, 10.0)[0] > pb
