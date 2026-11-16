import numpy as np

from odte.research import skill


def test_auc_perfect_random_and_strata():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.3
    st = np.where(np.arange(2000) % 2 == 0, "a", "b")
    assert skill.strat_auc(y.astype(float), y, st) == 1.0
    assert abs(skill.strat_auc(rng.random(2000), y, st) - 0.5) < 0.05


def test_stratification_removes_base_rate_confound():
    # stratum b has more events AND higher scores, but within each stratum the score is noise
    rng = np.random.default_rng(1)
    st = np.repeat(["a", "b"], 3000)
    y = np.concatenate([rng.random(3000) < 0.1, rng.random(3000) < 0.6])
    score = np.concatenate([rng.random(3000), rng.random(3000) + 5])
    assert abs(skill.strat_auc(score, y, st) - 0.5) < 0.04


def test_pick_lift():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0], bool)
    st = np.array(["a"] * 8)
    pick = np.array([1, 0, 1, 0, 0, 0, 0, 0], bool)
    assert abs(skill.pick_lift(pick, y, st) - 1.0 / (3 / 8)) < 1e-9
