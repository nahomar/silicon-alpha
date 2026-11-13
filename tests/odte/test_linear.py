import pickle
from pathlib import Path

import numpy as np
import pytest

from odte.live.linear import linear_predict

ROOT = Path(__file__).resolve().parents[2]


def test_matches_sklearn_on_a_fresh_pipeline():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6)); X[rng.random(X.shape) < 0.1] = np.nan; X[:, 5] = np.nan   # an all-missing column
    y = rng.normal(size=400)
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)).fit(X, y)
    Xt = rng.normal(size=(50, 6)); Xt[rng.random(Xt.shape) < 0.2] = np.nan
    assert np.allclose(linear_predict(m, Xt), m.predict(Xt), atol=1e-10)


@pytest.mark.parametrize("name", ["model_frozen.pkl", "round4_model.pkl"])
def test_frozen_bundles_predict_in_this_environment(name):
    p = ROOT / "data" / "live" / name
    if not p.exists():
        pytest.skip("model file not present")
    b = pickle.loads(p.read_bytes())
    models = b["models"] if "models" in b else {"m": b["model"]}
    x = np.full((3, len(b["features"])), np.nan); x[1] = 0.0
    for m in models.values():
        assert np.isfinite(linear_predict(m, x)).all()
