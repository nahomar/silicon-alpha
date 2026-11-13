"""Lead shadow (frozen exam model): rule, and a replayed day against the research matrix."""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from odte.live import frozen
from odte.live.config import Config

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_choose_rule():
    p = {"fly": 12.0, "condor": 30.0, "put_spread": -5.0, "call_spread": 1.0}
    assert frozen.choose(p, post=False) == ("condor", "best predicted > 0")
    assert frozen.choose(p, post=True)[0] == "none"
    assert frozen.choose({k: -1.0 for k in p}, post=False)[0] == "none"
    assert frozen.choose({k: float("nan") for k in p}, post=False)[0] == "none"


def _aligned_broker():
    from odte.live.broker import ReplayBroker

    class Aligned(ReplayBroker):
        """Live records at hh:mm:58 the quote research stamps at the next minute."""
        def _at(self, fn):
            t = self.t
            self.t = t + pd.Timedelta(seconds=2)
            try:
                out = fn()
            finally:
                self.t = t
            out = out.copy()
            out["quote_datetime"] = t
            return out

        def snapshot(self):
            return self._at(lambda: ReplayBroker.snapshot(self))

        def snapshot_extra(self, sym):
            return self._at(lambda: ReplayBroker.snapshot_extra(self, sym))

        def snapshot_next(self):
            return self._at(lambda: ReplayBroker.snapshot_next(self))
    return Aligned


def test_replayed_day_matches_research_for_spx(tmp_path):
    day = "2026-08-20"
    need = [ROOT / f"data/odte_0dte/{day}.parquet", ROOT / f"data/odte_next/{day}.parquet",
            ROOT / "data/research/meta_matrix.parquet", frozen.model_path(Config())]
    if not all(p.exists() for p in need):
        pytest.skip("local data / frozen model not present")
    import os
    os.chdir(ROOT)
    from odte.live.runner import run_day
    from odte.research import meta
    st = pd.read_csv(ROOT / "data/odte_0dte/settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
    cfg = Config(mode="replay", journal_dir=tmp_path / "j", chain_dir=tmp_path / "c", kill_switch=tmp_path / "STOP")
    s = run_day(cfg, _aligned_broker()(day, data_dir=str(ROOT / "data/odte_0dte")), day,
                prev_settle=float(st[st.index < day].iloc[-1]))
    assert "frozen_lead:SPX" in s["shadow"]
    recs = [json.loads(x) for x in (tmp_path / "j" / f"{day}.jsonl").read_text().splitlines()]
    rows = {r["hhmm"]: r for r in recs if r["kind"] == "lead_row" and r["index"] == "SPX"}
    dec = {r["hhmm"]: r["trade"] for r in recs if r["kind"] == "shadow" and r.get("strategy") == frozen.NAME
           and r.get("index") == "SPX" and "preds" in r}
    lab = {r["hhmm"]: r for r in recs if r["kind"] == "lead_labels" and r["index"] == "SPX"}
    import pickle
    b = pickle.loads(frozen.model_path(Config()).read_bytes())
    m = meta.load()
    m = m[(m["day"] == day) & (m["index"] == "SPX")]
    assert len(m) == 12 and len(rows) == 12
    for _, r in m.iterrows():
        live = rows[r["hhmm"]]
        for c in b["features"]:
            a = float(pd.to_numeric(r[c], errors="coerce"))
            v = np.nan if live.get(c) is None else float(live[c])
            assert (np.isnan(a) and np.isnan(v)) or abs(a - v) <= 1e-6 * max(1.0, abs(a)), (r["hhmm"], c, a, v)
        want, _ = frozen.choose(frozen.predict_row(b, r.to_dict()), bool(r["post"]))
        assert dec[r["hhmm"]] == want, r["hhmm"]
        for k in b["menu"]:
            y = r[f"y_{k}_half"]
            if np.isfinite(y):
                assert abs(y * frozen.BUDGET / 1000 - lab[r["hhmm"]][f"{k}_half"]) < 0.01, (r["hhmm"], k)
