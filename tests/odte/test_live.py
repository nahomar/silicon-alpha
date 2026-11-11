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


def test_replay_day_order_fill_and_settlement(tmp_path):
    """Full day through the runner with a broker that fills on the 2nd rung."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    if not (root / "data/odte_0dte/2026-09-04.parquet").exists() or not Config().model_path.exists():
        pytest.skip("local data / model not present")
    from odte.live.broker import ReplayBroker
    from odte.live.runner import run_day

    class SecondRung(ReplayBroker):
        def modify(self, h, price):
            super().modify(h, price)
            o = self.orders[h]
            o.update(filled=o["order"].qty, avg=price, live=False)

    cfg = Config(mode="replay", journal_dir=tmp_path / "j", chain_dir=tmp_path / "c",
                 kill_switch=tmp_path / "STOP", max_trades_per_day=1)
    s = run_day(cfg, SecondRung("2026-09-04", data_dir=str(root / "data/odte_0dte"), settle=7718.6),
                "2026-09-04", prev_settle=7700.0)
    assert s["fills"] == 1 and s["decisions"] == 6
    log = (tmp_path / "j" / "2026-09-04.jsonl").read_text()
    assert '"kind": "fill"' in log and '"kind": "settle"' in log and "max filled trades today" in log
    assert 0 <= s["share_of_gap_paid"][0] <= 0.5 + 1e-9


def test_shadow_strategies_and_pricing():
    from odte.live.strategies import STRATEGIES, shadow_pnl
    assert STRATEGIES["sell_only"](1, False)[0] == 1
    assert STRATEGIES["sell_only"](1, True)[0] == 0              # no short right after news
    assert STRATEGIES["full_model"](-1, True)[0] == -1           # buying is allowed then
    assert STRATEGIES["always_sell"](-1, False)[0] == 1
    row = {"K": 7740.0, "W": 35.0, "fly_credit_mid": 17.9, "fly_credit": 17.7, "fly_debit": 18.1}
    calm = shadow_pnl(1, row, 7740.0)
    assert calm["mid"] > calm["half"] > calm["touch"] > 0        # paying more of the gap earns less
    crash = shadow_pnl(1, row, 7600.0)
    assert crash["touch"] == pytest.approx(-300.0)               # the loss stops at the budget
    assert shadow_pnl(-1, row, 7600.0)["mid"] > 0                # the bought fly gains on the big move


def test_event_seller_rule():
    from odte.live.strategies import event_seller
    cpi = {"ev_cpi": 1.0, "vix_over_vix3m": 0.9}
    assert event_seller(-1, False, cpi)[0] == 1                           # no model needed
    assert event_seller(0, True, cpi)[0] == 0                             # inside the post-release hour
    assert event_seller(0, False, {"ev_cpi": 1.0, "vix_over_vix3m": 1.05})[0] == 0     # stress
    assert event_seller(1, False, {"ev_cpi": 0.0, "vix_over_vix3m": 0.9})[0] == 0      # ordinary day


def test_model_signal_rules():
    from odte.live.model import signal
    assert signal({"kind": "ratio"}, 0.80) == 1 and signal({"kind": "ratio"}, 1.20) == -1
    assert signal({"kind": "ratio"}, 1.0) == 0
    assert signal({"kind": "profit"}, 0.05) == 1 and signal({"kind": "profit"}, -0.01) == 0


def test_previous_day_carries_settle_and_ratio(tmp_path):
    import json as _j
    from odte.live.runner import previous_day
    cfg = Config(journal_dir=tmp_path)
    (tmp_path / "2026-11-10.jsonl").write_text(_j.dumps({"kind": "day_summary", "settle": 6800.5, "ratio": 0.71}) + "\n")
    (tmp_path / "2026-11-12.jsonl").write_text(_j.dumps({"kind": "day_summary", "settle": 6900.0, "ratio": 1.2}) + "\n")
    assert previous_day(cfg, "2026-11-11") == (6800.5, 0.71)        # never a later day


def test_fly_from_xsp_quotes_and_line_budget():
    q = pd.DataFrame([(k, r, b, b + 0.05) for k, r, b in [
        (676.0, "C", 1.60), (676.0, "P", 1.40), (679.0, "C", 0.30), (673.0, "P", 0.35),
        (680.0, "C", 0.20), (672.0, "P", 0.25), (677.0, "C", 1.05), (675.0, "P", 0.95),
        (681.0, "C", 0.10), (671.0, "P", 0.15)]],
        columns=["strike", "option_type", "bid", "ask"])
    f = fly.fly_from_quotes(q, F=676.1)
    assert f["K"] == 676.0 and f["W"] == 5.0                    # smallest distance >= 1.5 x 3.05 = 4.6
    assert f["fly_credit"] == pytest.approx(1.60 + 1.40 - 0.15 - 0.20)
    assert f["fly_credit"] < f["fly_credit_mid"] < f["fly_debit"]
    o = fly.build(f, "XSP", 300.0, 5, 0.5, 0.01, native=True)
    assert o.K == 676.0 and o.ladder[0] == pytest.approx(round(f["fly_credit_mid"], 2))
    assert Config(symbol="XSP").spx_lines() == 95 - 34 and Config(symbol="SPX").spx_lines() == 95


def test_vertical_quotes_and_payoff():
    from odte.live.strategies import vertical_pnl
    q = pd.DataFrame([("P", 100.0, 2.0, 2.2), ("P", 95.0, 0.5, 0.6), ("C", 100.0, 2.1, 2.3), ("C", 105.0, 0.6, 0.7)],
                     columns=["option_type", "strike", "bid", "ask"])
    v = fly.vertical_from_quotes(q, 100.0, 5.0)
    assert v["put_mid"] == pytest.approx(2.1 - 0.55) and v["put_touch"] == pytest.approx(2.0 - 0.6)
    up = vertical_pnl("put", v, settle=103.0)        # close above K: the put spread keeps its credit
    down = vertical_pnl("put", v, settle=90.0)       # close below K - W: max loss, = -budget at the touch
    assert up["mid"] > up["touch"] > 0 and down["touch"] == pytest.approx(-300.0)


def test_trading_day_calendar():
    from odte.live.runner import is_trading_day
    assert not is_trading_day("2026-11-26")          # Thanksgiving
    assert is_trading_day("2026-11-27")              # half day, still a session
    assert not is_trading_day("2026-11-28")          # Saturday
    assert not is_trading_day("2026-12-25")
    assert not is_trading_day("2027-03-26")          # Good Friday
    assert is_trading_day("2026-11-10")


def test_crash_cancels_the_working_order_and_saves_quotes(tmp_path):
    """A failure mid-day leaves no order working, keeps the quotes, and alerts."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    if not (root / "data/odte_0dte/2026-09-04.parquet").exists() or not Config().model_path.exists():
        pytest.skip("local data / model not present")
    from odte.live.broker import ReplayBroker
    from odte.live.runner import run_day

    class Disconnect(ReplayBroker):
        cancelled = []

        def filled(self, h):
            raise ConnectionError("gateway lost")

        def cancel(self, h):
            self.cancelled.append(h)
            super().cancel(h)

    alerts = []
    cfg = Config(mode="replay", journal_dir=tmp_path / "j", chain_dir=tmp_path / "c",
                 kill_switch=tmp_path / "STOP")
    b = Disconnect("2026-09-04", data_dir=str(root / "data/odte_0dte"), settle=7718.6)
    with pytest.raises(ConnectionError) as ei:
        run_day(cfg, b, "2026-09-04", prev_settle=7700.0, alert=lambda t, x: alerts.append((t, x)))
    assert getattr(ei.value, "_alerted", False)
    assert len(b.cancelled) == 1
    log = (tmp_path / "j" / "2026-09-04.jsonl").read_text()
    assert '"kind": "crash"' in log and '"cancelled": true' in log
    assert (tmp_path / "c" / "2026-09-04.parquet").exists()
    assert alerts and alerts[0][0] == "bot stopped" and "Working order cancelled" in alerts[0][1]


def test_notify_passes_text_as_arguments_not_script(tmp_path, monkeypatch):
    import sys
    from odte.live import notify as nt
    calls = []
    monkeypatch.setattr(nt.subprocess, "run", lambda args, **kw: calls.append(args))
    monkeypatch.setattr(sys, "platform", "darwin")
    text = 'he said "x" & do shell script "rm"'
    nt.notify("bot FAILED", text, log=tmp_path / "alerts.log")
    args = calls[0]
    assert args[-2:] == ["bot FAILED", text]                  # data, not code
    assert all(text not in a for a in args[:-2])
    assert "bot FAILED" in (tmp_path / "alerts.log").read_text()


def test_day_text():
    from odte.live.notify import day_text
    s = {"day": "2026-09-04", "orders": 6, "fills": 1, "pnl": -12.5,
         "shadow": {"event_seller": {"half": 602.3}, "always_sell": {"half": 1.0}}}
    t = day_text(s)
    assert "6 orders, 1 filled, P&L $-12.50" in t and "event_seller $+602" in t and "always_sell" not in t


def test_reconcile_flags_every_kind_of_mismatch():
    from odte.live.runner import reconcile_problems
    ok = {"open_orders": 0, "orders_with_fills": 1, "later_expiry_positions": 0}
    assert reconcile_problems(ok, 1) == []
    assert "still open" in reconcile_problems({**ok, "open_orders": 1}, 1)[0]
    assert "journal has 0" in reconcile_problems(ok, 0)[0]
    assert "expiring after today" in reconcile_problems({**ok, "later_expiry_positions": 2}, 1)[0]


def test_replay_day_reconciles_and_logs_the_code_version(tmp_path):
    import json
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    if not (root / "data/odte_0dte/2026-09-04.parquet").exists() or not Config().model_path.exists():
        pytest.skip("local data / model not present")
    from odte.live.broker import ReplayBroker
    from odte.live.runner import run_day
    cfg = Config(mode="replay", journal_dir=tmp_path / "j", chain_dir=tmp_path / "c",
                 kill_switch=tmp_path / "STOP")
    s = run_day(cfg, ReplayBroker("2026-09-04", data_dir=str(root / "data/odte_0dte"), settle=7718.6),
                "2026-09-04", prev_settle=7700.0)
    assert s["reconcile_ok"] is True
    recs = [json.loads(l) for l in (tmp_path / "j" / "2026-09-04.jsonl").read_text().splitlines()]
    assert recs[0]["kind"] == "start" and recs[0]["code"] not in ("", None)
    rec = [r for r in recs if r["kind"] == "reconcile"][0]
    assert rec["ok"] and rec["orders_with_fills"] == rec["journal_fills"] == s["fills"]


def test_watch_finished_and_backup(tmp_path):
    import json
    import subprocess
    from odte.live import watch
    cfg = Config(mode="paper", journal_dir=tmp_path / "live/journal", chain_dir=tmp_path / "live/chains")
    cfg.journal_dir.mkdir(parents=True)
    assert not watch.finished(cfg, "2026-11-11")
    p = cfg.journal_dir / "2026-11-11.jsonl"
    p.write_text(json.dumps({"kind": "start"}) + "\n")
    assert not watch.finished(cfg, "2026-11-11")
    p.write_text(p.read_text() + json.dumps({"kind": "day_summary", "pnl": 0}) + "\n")
    assert watch.finished(cfg, "2026-11-11")
    (cfg.journal_dir / "2026-11-11-replay.jsonl").write_text("{}\n")      # tests are not backed up
    dest = tmp_path / "records"
    subprocess.run(["git", "init", "-q", str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(dest), "config", "user.name", "t"], check=True)
    assert "1 file(s)" in watch.backup(cfg, dest, push=False)
    assert (dest / "live/journal/2026-11-11.jsonl").exists()
    assert not (dest / "live/journal/2026-11-11-replay.jsonl").exists()
    assert watch.backup(cfg, dest, push=False) == "backup: nothing new"


def test_xindex_rule():
    from odte.live.xshadow import xindex_side
    assert xindex_side(0.7, {"NDX": 0.8, "RUT": 0.9}, False)[0] == 1
    assert xindex_side(0.85, {"NDX": 0.8, "RUT": 0.9}, False)[0] == 0
    assert xindex_side(0.7, {"NDX": 0.8}, False)[0] == 0                 # an index missing: no trade
    assert xindex_side(0.7, {"NDX": 0.8, "RUT": 0.9}, True)[0] == 0       # after a release


def test_replay_day_with_ndx_rut_shadows(tmp_path):
    """NDX/RUT are recorded and shadow-traded on their own quotes; SPX decisions are unchanged."""
    import json
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    need = [root / f"data/{d}/2026-09-04.parquet" for d in ("odte_0dte", "odte_0dte_ndxp", "odte_0dte_rutw")]
    if not all(p.exists() for p in need) or not Config().model_path.exists():
        pytest.skip("local data / model not present")
    from odte.live import xshadow
    from odte.live.broker import ReplayBroker
    from odte.live.runner import run_day
    mdir = tmp_path / "m"
    for sym in ("NDX", "RUT"):
        xshadow.train(sym, path=mdir / f"model_{sym.lower()}.pkl")
    (mdir / "model.pkl").write_bytes(Config().model_path.read_bytes())
    base = dict(mode="replay", chain_dir=tmp_path / "c", kill_switch=tmp_path / "STOP", model_path=mdir / "model.pkl")
    a = run_day(Config(**base, journal_dir=tmp_path / "ja", shadow_indexes=()),
                ReplayBroker("2026-09-04", data_dir=str(root / "data/odte_0dte"), settle=7718.6), "2026-09-04",
                prev_settle=7700.0)
    b = run_day(Config(**base, journal_dir=tmp_path / "jb"),
                ReplayBroker("2026-09-04", data_dir=str(root / "data/odte_0dte"), settle=7718.6), "2026-09-04",
                prev_settle=7700.0)
    assert (a["orders"], a["fills"], a["pnl"]) == (b["orders"], b["fills"], b["pnl"])   # SPX path untouched
    recs = [json.loads(l) for l in (tmp_path / "jb" / "2026-09-04.jsonl").read_text().splitlines()]
    kinds = {r["kind"] for r in recs}
    assert {"shadow_index", "settle_index"} <= kinds
    names = {r.get("strategy") for r in recs if r["kind"] == "shadow_settle"}
    assert {"event_seller:NDX", "event_seller:RUT", "always_sell:RUT"} <= names      # Sep 4: jobs report
    assert any(r.get("strategy") == "xindex_spx" for r in recs if r["kind"] == "shadow")
    assert (tmp_path / "c" / "ndx" / "2026-09-04.parquet").exists()


def test_index_shadows_fail_safe_without_models(tmp_path):
    import json
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    if not (root / "data/odte_0dte/2026-09-04.parquet").exists() or not Config().model_path.exists():
        pytest.skip("local data / model not present")
    from odte.live.broker import ReplayBroker
    from odte.live.runner import run_day
    mdir = tmp_path / "m"
    mdir.mkdir()
    (mdir / "model.pkl").write_bytes(Config().model_path.read_bytes())    # no NDX/RUT models here
    s = run_day(Config(mode="replay", journal_dir=tmp_path / "j", chain_dir=tmp_path / "c",
                       kill_switch=tmp_path / "STOP", model_path=mdir / "model.pkl"),
                ReplayBroker("2026-09-04", data_dir=str(root / "data/odte_0dte"), settle=7718.6), "2026-09-04",
                prev_settle=7700.0)
    assert s["decisions"] == 6
    notes = [json.loads(l)["text"] for l in (tmp_path / "j" / "2026-09-04.jsonl").read_text().splitlines()
             if json.loads(l)["kind"] == "note"]
    assert any("NDX shadow off today" in t for t in notes)
