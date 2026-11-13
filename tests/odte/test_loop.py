"""Weekly loop: promotion / demotion rules, and an end-to-end week on replayed days."""
import json

import numpy as np
import pandas as pd
import pytest

from odte.live import loop


def _ev(champ, chall, days=60, trades=60, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    keys = [f"2027-01-{i:02d}" if i <= 31 else f"2027-02-{i - 31:02d}" for i in range(1, days + 1)]

    def r(series):
        return {"days": days, "trades": trades, "total": float(sum(series)), "mean_day": float(np.mean(series)),
                "daily": dict(zip(keys, series))}
    base = rng.normal(0, 2500, days)
    c = base + champ
    h = base + chall + rng.normal(0, noise, days)
    return {"status": "tested", "results": {"ratio_ridge_full": r(c), "profit_ridge_full": r(h),
                                            "always_sell": r(base)}}


def test_promotion_needs_a_statistically_clear_advantage():
    reg = {"champion": "ratio_ridge_full"}
    # +$25/day on a $2,500/day-noise P&L: the old rule promoted this; it is luck-sized
    assert loop.decide(reg, _ev(10, 35, noise=800))["champion"] == "ratio_ridge_full"
    # a large, consistent paired advantage clears 1.5 SE
    assert loop.decide(reg, _ev(10, 400, noise=800))["champion"] == "profit_ridge_full"
    # too few days or trades never promote
    assert loop.decide(reg, _ev(10, 400, noise=800, days=30))["champion"] == "ratio_ridge_full"
    assert loop.decide(reg, _ev(10, 400, noise=800, trades=10))["champion"] == "ratio_ridge_full"


def test_no_automatic_on_off_switching():
    reg = {"champion": "ratio_ridge_full"}
    out = loop.decide(reg, _ev(-300, -400, noise=800))
    assert "ml_orders_enabled" not in out


def test_collecting_until_enough_live_days():
    ev = loop.evaluate(pd.DataFrame(), pd.DataFrame(), {})
    assert ev["status"].startswith("collecting")


def test_week_end_to_end_on_replayed_days(tmp_path, monkeypatch):
    """Three replayed days -> ingest -> test window -> decision -> new model saved."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    if not all((root / f"data/odte_0dte/{d}.parquet").exists() for d in ("2026-08-12", "2026-08-13", "2026-08-14")):
        pytest.skip("local data not present")
    from odte.live.broker import ReplayBroker
    from odte.live.config import Config
    from odte.live.runner import run_day
    cfg = Config(mode="replay", journal_dir=tmp_path / "journal", chain_dir=tmp_path / "chains",
                 kill_switch=tmp_path / "STOP", model_path=tmp_path / "model.pkl")
    import shutil
    shutil.copy(Config().model_path, cfg.model_path)
    for d in ("2026-08-12", "2026-08-13", "2026-08-14"):
        run_day(cfg, ReplayBroker(d, data_dir=str(root / "data/odte_0dte")), d, prev_settle=7700.0)
    monkeypatch.setattr(loop, "MIN_LIVE_DAYS", 2)
    monkeypatch.setattr(loop, "WINDOW", 2)
    monkeypatch.setattr(loop, "MIN_TRADES", 1)
    monkeypatch.setattr(loop, "PROMOTE_SE", 1.5)
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    out = loop.weekly(cfg, today="2026-08-15")
    live = pd.read_parquet(tmp_path / "dataset.parquet")
    assert live["day"].nunique() == 3 and {"ratio", "ror_half", "ev_cpi", "vix_prev"} <= set(live.columns)
    assert out["eval"]["status"] == "tested" and set(out["eval"]["results"]) >= {"ratio_ridge_full", "always_sell"}
    assert json.loads((tmp_path / "registry.json").read_text())["champion"]
    assert (tmp_path / "reports" / "live_weekly" / "2026-08-15.md").exists()


def test_rank_rule_and_bonferroni_bar():
    from odte.live.model import signal
    b = loop.fit(pd.DataFrame({"x": np.arange(100.0), "abs_z": np.arange(100.0) / 50}), "abs_z", ["x"], "rank")
    assert 1.4 < b["threshold"] < 1.7                      # 80th percentile of training predictions
    assert signal(b, 0.5) == 1 and signal(b, 1.9) == 0      # sell unless among the riskiest 20%
    assert loop.promote_z(1) == 1.5 and 2.0 < loop.promote_z(6) < 2.3
