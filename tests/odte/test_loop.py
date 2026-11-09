"""Weekly loop: promotion / demotion rules, and an end-to-end week on replayed days."""
import json

import pandas as pd
import pytest

from odte.live import loop


def _ev(champ, chall, always, days=40, trades=60):
    r = lambda m: {"days": days, "trades": trades, "total": m * days, "mean_day": m}
    return {"status": "tested", "results": {"ratio_ridge_full": r(champ), "profit_ridge_full": r(chall),
                                            "always_sell": r(always)}}


def test_promotion_needs_a_clear_margin():
    reg = {"champion": "ratio_ridge_full", "ml_orders_enabled": True}
    assert loop.decide(reg, _ev(10, 30, 0))["champion"] == "ratio_ridge_full"     # +20 < margin
    assert loop.decide(reg, _ev(10, 40, 0))["champion"] == "profit_ridge_full"    # +30 >= margin
    assert loop.decide(reg, _ev(10, 40, 0, trades=5))["champion"] == "ratio_ridge_full"   # too few trades


def test_demotion_switches_model_orders_off_and_back_on():
    reg = {"champion": "ratio_ridge_full", "ml_orders_enabled": True}
    off = loop.decide(reg, _ev(-20, -30, 5))
    assert off["ml_orders_enabled"] is False
    assert loop.decide(off, _ev(15, 0, 5))["ml_orders_enabled"] is True


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
    monkeypatch.setattr(loop, "ROOT", tmp_path)
    out = loop.weekly(cfg, today="2026-08-15")
    live = pd.read_parquet(tmp_path / "dataset.parquet")
    assert live["day"].nunique() == 3 and {"ratio", "ror_half", "ev_cpi", "vix_prev"} <= set(live.columns)
    assert out["eval"]["status"] == "tested" and set(out["eval"]["results"]) >= {"ratio_ridge_full", "always_sell"}
    assert json.loads((tmp_path / "registry.json").read_text())["champion"]
    assert (tmp_path / "reports" / "live_weekly" / "2026-08-15.md").exists()
