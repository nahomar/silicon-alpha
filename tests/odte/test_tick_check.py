"""Tick gate: chunk-safe reduction, no look-ahead in features, honest taker P&L."""
import numpy as np
import pandas as pd

from odte.eval import tick_check as tc

DAY = "2026-03-18"


def _ns(hhmm_ss):
    return int(pd.Timestamp(f"{DAY} {hhmm_ss}").tz_localize(tc.TZ).tz_convert("UTC").value)


def _arr(rows):
    ts, iid, b, a, bs, as_ = map(np.array, zip(*rows))
    return ts.astype(np.int64), iid.astype(np.int64), b.astype(float), a.astype(float), bs.astype(float), as_.astype(float)


def test_reducer_drops_size_only_updates_across_chunk_boundaries():
    r = tc.Reducer()
    r.add(*_arr([(_ns("10:00:00"), 1, 1.0, 1.1, 5, 5), (_ns("10:00:01"), 1, 1.0, 1.1, 9, 2)]))
    r.add(*_arr([(_ns("10:00:02"), 1, 1.0, 1.1, 1, 1), (_ns("10:00:03"), 1, 1.05, 1.1, 1, 1)]))
    f = r.frame()
    assert list(f["ts"]) == [_ns("10:00:00"), _ns("10:00:03")]


def test_features_ignore_the_future_and_pnl_crosses_the_spread():
    rows = [(_ns("10:00:00"), 1, 1.00, 1.10, 10, 10),
            (_ns("10:00:10"), 1, 1.05, 1.15, 10, 10),
            (_ns("10:00:20"), 1, 1.20, 1.30, 10, 10)]
    ev = pd.DataFrame(dict(zip(["ts", "iid", "bid", "ask", "bsz", "asz"], _arr(rows))))
    a = tc.day_events(ev, DAY)
    later = ev.copy()
    later.loc[2, ["bid", "ask"]] = [5.0, 5.1]                 # change only the future row
    b = tc.day_events(later, DAY)
    pd.testing.assert_frame_equal(a.loc[[0, 1], tc.FEATURES], b.loc[[0, 1], tc.FEATURES])
    # buy at 1.10, sell 30 s later at the 10:00:20 bid 1.20
    assert np.isclose(a.loc[0, "long_pnl"], 1.20 - 1.10 - tc.FEES)
    assert np.isclose(a.loc[0, "short_pnl"], 1.00 - 1.30 - tc.FEES)
    assert a.loc[1, "n60"] == 1 and np.isnan(a.loc[0, "r1"])


def test_exam_days_are_excluded():
    d = pd.read_csv("configs/tradefm/tick_days.csv", parse_dates=["date"])
    assert (d["date"] >= tc.EXAM_FROM).sum() == 4
