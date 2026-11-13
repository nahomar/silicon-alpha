import numpy as np
import pandas as pd

from filings import lazy


def test_split_like_jumps_are_not_returns():
    px = pd.Series([100.0, 101.0, 50.5, 51.0, 255.0])     # 2-for-1 split, then a 1-for-5 reverse split
    r = lazy.split_adjusted_returns(px)
    assert r.iloc[2] == 0.0 and r.iloc[4] == 0.0 and abs(r.iloc[1] - 0.01) < 1e-12


def test_similarity_pairs_need_9_to_15_months():
    mh = np.arange(256, dtype=np.uint32).tobytes()
    mh2 = np.concatenate([np.arange(128), np.arange(1000, 1128)]).astype(np.uint32).tobytes()
    f = pd.DataFrame({"cik": [1, 1, 1], "filed": pd.to_datetime(["2020-03-01", "2021-03-01", "2021-05-01"]),
                      "ticker": ["A", "A", "A"], "minhash": [mh, mh2, mh]})
    s = lazy.similarities(f)
    assert len(s) == 1 and abs(s.iloc[0]["sim"] - 0.5) < 1e-9           # the 2-month gap pair is skipped


def test_delisted_stock_earns_its_last_price():
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    daily = pd.DataFrame({"date": days[:40], "symbol": "X", "close": np.linspace(10, 12, 40), "volume": 1e6})
    me = pd.DatetimeIndex(["2020-01-31", "2020-02-29", "2020-03-31"])
    p = lazy.monthly_panel(daily, me)
    jan = p.loc[(pd.Timestamp("2020-01-31"), "X")]                       # held over Feb; stops trading Feb 25
    px = daily.set_index("date")["close"]
    assert abs(jan["ret"] - (px.iloc[-1] / px[px.index <= "2020-01-31"].iloc[-1] - 1)) < 1e-9
    assert (pd.Timestamp("2020-02-29"), "X") not in p.index              # not selectable after it stopped


def test_newey_west_matches_plain_t_without_autocorrelation():
    x = np.random.default_rng(0).normal(0.01, 0.05, 400)
    plain = x.mean() / (x.std() / np.sqrt(len(x)))
    assert abs(lazy.newey_west_t(x, 0) - plain) < 1e-6
