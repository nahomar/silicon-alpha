import numpy as np

from odte.research import paths as pt


def _path(trend: float, n=390, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 1e-4, n)
    for i in range(1, n):
        r[i] += trend * r[i - 1]
    p = np.full((n, 3), np.nan, np.float32)
    p[:, 0] = np.log(5000) + np.cumsum(r)
    p[:, 1] = np.linspace(0.006, 0.002, n)
    p[:, 2] = 0.01
    return p


def test_trend_vs_reversion_signals():
    up = pt.path_features(_path(0.5), 200, 0.004)
    dn = pt.path_features(_path(-0.5), 200, 0.004)
    assert up["pth_ac1"] > 0.3 and dn["pth_ac1"] < -0.3
    assert up["pth_vr5"] > 1 > dn["pth_vr5"]


def test_features_use_only_the_past():
    p = _path(0.0)
    a = pt.path_features(p, 120, 0.004)
    p2 = p.copy()
    p2[121:, 0] += 1.0                      # change the future only
    b = pt.path_features(p2, 120, 0.004)
    assert a == b


def test_vol_surprise_is_one_when_straddle_decays_like_sqrt_time():
    p = _path(0.0)
    left = np.arange(390, 0, -1)
    p[:, 1] = 0.006 * np.sqrt(left / 390)
    f = pt.path_features(p, 300, 0.004)
    assert abs(f["pth_volsurp"] - 1) < 0.02
