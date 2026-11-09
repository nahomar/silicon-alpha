"""ES flow reduction: aggressor side mapped correctly, minutes summed."""
import numpy as np

from odte.data.es_flow import reduce_day


def test_reduce_day_sums_buy_and_sell_by_minute():
    a = np.array([(60 * 10**9 + 1, b"B", 5), (60 * 10**9 + 30 * 10**9, b"A", 25),
                  (120 * 10**9, b"B", 30), (120 * 10**9 + 5, b"N", 7)],
                 dtype=[("ts_recv", "u8"), ("side", "S1"), ("size", "u4")])
    m = reduce_day(a).set_index("ts")
    assert m.loc[60, ["buy", "sell", "big_buy", "big_sell", "n"]].tolist() == [5, 25, 0, 25, 2]
    assert m.loc[120, ["buy", "sell", "big_buy", "n"]].tolist() == [30, 0, 30, 2]
