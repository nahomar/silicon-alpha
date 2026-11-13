import numpy as np
import pandas as pd

from odte.research import meta


def test_decide_takes_the_best_positive_prediction_and_never_trades_post_release():
    d = pd.DataFrame({"day": pd.to_datetime(["2023-01-03"] * 3), "index": "SPX", "post": [False, False, True]})
    for k in meta.TRADES:
        d[f"y_{k}_half"] = [0.1, 0.2, 0.3]
    d["y_put_spread_half"] = [0.5, -0.4, 0.9]
    preds = {k: np.array([-0.1, -0.2, 0.5]) for k in meta.TRADES}
    preds["put_spread"] = np.array([0.3, -0.1, 0.8])
    dec = meta.decide(d, preds)
    assert dec["trade"].tolist() == ["put_spread", "none", "none"]   # row 2 all negative; row 3 post-release
    assert dec["ret"].tolist() == [0.5, 0.0, 0.0]


def test_whitelist_excludes_outcome_columns():
    cols = ["implied_move", "settle", "ratio", "pnl_short_fly", "fill_mid_short", "credit_t5", "y_fly_half",
            "x_iv_over_rv_vs_others", "f_skew", "vix_prev", "fly_chg5", "tod", "is_SPX"]
    g = meta.feature_groups(cols)
    used = {c for v in g.values() for c in v}
    assert not used & {"settle", "ratio", "pnl_short_fly", "fill_mid_short", "credit_t5", "y_fly_half"}
    assert {"implied_move", "x_iv_over_rv_vs_others", "f_skew", "vix_prev", "fly_chg5", "tod", "is_SPX"} <= used


def test_mlp_learns_a_simple_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    y = 0.3 * X[:, 0] + rng.normal(scale=0.05, size=2000)
    m = meta.TorchMLP(epochs=30).fit(X, y)
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.9
