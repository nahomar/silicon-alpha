import numpy as np
import pytest

from odte.research import seq


def _path(n=390, seed=0):
    rng = np.random.default_rng(seed)
    p = np.full((n, 3), np.nan, np.float32)
    p[:, 0] = np.log(5000) + np.cumsum(rng.normal(0, 1e-4, n))
    p[:, 1] = 0.006 * np.sqrt(np.arange(n, 0, -1) / n)
    p[:, 2] = 0.02
    return p


def test_decision_minute():
    assert seq.decision_minute("09:31") == 0
    assert seq.decision_minute("10:00") == 29
    assert seq.decision_minute("15:30") == 359


def test_window_uses_only_the_past():
    p = _path()
    a = seq.window(p, 200, 0.004)
    p2 = p.copy()
    p2[201:] *= 3.0
    assert np.array_equal(a, seq.window(p2, 200, 0.004))


def test_window_padding_and_anchor():
    p = _path()
    w = seq.window(p, 29, 0.004)               # 10:00: 30 real minutes, 150 padded
    assert w[:, 4].sum() == 30 and w[:seq.W - 30].sum() == 0
    assert w[-1, 0] == 0.0                     # position is measured from now
    assert abs(w[-1, 2]) < 1e-6                # decay-adjusted straddle relative to now
    w2 = seq.window(p, 300, 0.004)
    assert w2[:, 4].sum() == seq.W


def test_network_shapes():
    torch = pytest.importorskip("torch")
    net = seq._net(7)
    out = net(torch.zeros(4, seq.W, seq.N_CH), torch.zeros(4, 7))
    assert tuple(out.shape) == (4, 5)
