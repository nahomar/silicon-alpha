import numpy as np

from odte.research import noise_hedged as nh


def test_implied_vol_round_trip_and_bounds():
    F = np.full(4, 7000.0)
    K = np.array([6950, 7000, 7050, 7100.0])
    call = np.array([True, False, True, False])
    s = np.array([0.004, 0.006, 0.005, 0.007])
    p = nh._black(F, K, s, call)
    assert np.abs(nh.implied_total_vol(p, F, K, call) - s).max() < 1e-9
    # a mid at or below intrinsic has no implied vol
    assert np.isnan(nh.implied_total_vol(np.array([40.0]), np.array([7050.0]), np.array([7000.0]), np.array([True]))[0])


def test_put_call_delta_parity():
    F, K, s = np.array([7000.0]), np.array([7020.0]), np.array([0.005])
    assert abs(nh.delta(F, K, s, True)[0] - nh.delta(F, K, s, False)[0] - 1.0) < 1e-12


def test_minute_hedge_removes_most_noise_in_a_black_world():
    """Short straddle under constant vol: the minute-hedged P&L varies far less than the unhedged one."""
    rng = np.random.default_rng(0)
    n_min, sig_day, F0, K = 360, 0.008, 7000.0, 7000.0
    tau = np.arange(n_min, 0, -1) / n_min * sig_day ** 2          # remaining total variance
    s = np.sqrt(tau)
    U, H = [], []
    for _ in range(400):
        z = rng.standard_normal(n_min)
        F = F0 * np.exp(np.cumsum(np.r_[0.0, sig_day * np.sqrt(1 / n_min) * z[:-1]]))
        ST = F[-1] * np.exp(sig_day * np.sqrt(1 / n_min) * z[-1])
        credit = nh._black(F0, K, s[0], True) + nh._black(F0, K, s[0], False)
        pnl = credit - abs(ST - K)
        pos = -(nh.delta(F, K, s, True) + nh.delta(F, K, s, False))
        dF = np.append(F[1:], ST) - F
        U.append(pnl)
        H.append(pnl - np.sum(pos * dF))
    assert np.std(U) / np.std(H) > 4
    assert abs(np.mean(H)) < 3 * np.std(H) / np.sqrt(len(H))   # no edge when implied = realised
