"""Live bot: features equal the backtest's, ladder and sizing behave, live is gated."""
import numpy as np
import pandas as pd
import pytest

from odte.live import fly
from odte.live.config import Config


def test_ladder_starts_at_mid_and_stops_at_the_concession():
    lad = fly.ladder(mid=1.62, touch=1.40, steps=5, max_concession=0.5, tick=0.05)
    assert lad[0] == 1.60 and lad[-1] >= 1.51 - 1e-9 and lad == sorted(lad, reverse=True)


def test_sizing_respects_the_dollar_limit():
    row = {"K": 5165.0, "W": 30.0, "fly_credit_mid": 16.10, "fly_credit": 15.20}
    o = fly.build(row, "XSP", max_loss=300.0, steps=5, max_concession=0.5, tick=0.05)
    assert o.K == 516.5 and o.W == 3.0 and o.qty >= 1 and o.worst_case <= 300.0
    assert fly.build(row, "SPX", 300.0, 5, 0.5, 0.05) is None        # one SPX fly risks > $300


def test_settlement_pnl():
    assert fly.settle_pnl(1.60, 2, 516.5, 3.0, 516.5) == pytest.approx((1.60 * 100 - 4.8) * 2)
    assert fly.settle_pnl(1.60, 1, 516.5, 3.0, 530.0) == pytest.approx((1.60 - 3.0) * 100 - 4.8)


def test_live_mode_is_refused_without_explicit_permission():
    with pytest.raises(SystemExit, match="REFUSED"):
        Config(mode="live").check()


def test_live_features_equal_the_backtest_rows():
    """Same day through the bot's feature path and the stored backtest rows."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    dec_p = root / "reports/expiry_edge/decisions-dev-gex-es.parquet"
    if not dec_p.exists():
        pytest.skip("local backtest output not present")
    from odte.live.features import day_context, feature_row
    from odte.live.model import feature_list
    day = "2024-03-13"
    chain = pd.read_parquet(root / f"data/odte_0dte/{day}.parquet")
    dec = pd.read_parquet(dec_p)
    dec = dec[(dec.day == pd.Timestamp(day)) & dec.on6 & (dec.trigger == "grid")].set_index("hhmm")
    st = pd.read_csv(root / "data/odte_0dte/settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
    dctx = day_context(day)
    cols = feature_list() + ["K", "W", "fly_credit", "fly_credit_mid"]
    for hhmm in dec.index:
        r = feature_row(chain, hhmm, dctx, float(st[st.index < day].iloc[-1]), float(dec.loc[hhmm, "prev_ratio"]))
        a = np.array([float(r.get(c, np.nan)) for c in cols])
        b = dec.loc[hhmm, cols].to_numpy(float)
        assert np.allclose(a, b, rtol=1e-6, equal_nan=True), hhmm
