"""Lead shadow: the frozen exam model (docs/master_plan.md, freeze ebadab6; WEAK PASS).

Paper evidence only. This module never places orders.

At every half-hour decision time, for SPX, NDX and RUT, it builds the research inputs from the
bot's own recorded quotes with the research code:
  calendar  calendar.features_for (via live.features.feature_row)
  gex       SPX pre-open open interest (the bot's existing OI input); NaN for NDX/RUT, as in training
  flow      library.flow_features on the quotes so far (fly mid change over the last 5 / 15 min)
  term      SPX: term_structure.build on next-expiry quotes the bot records (about 12 lines,
            re-centred two minutes before each decision); NDX/RUT NaN, as in training
  identity  is_SPX / is_NDX / is_RUT, minute of day
Then it applies odte/research/exam.py's rule: the best predicted of fly / condor / put spread / call
spread if > 0, never within 60 minutes after a release.

After the close, each pick is scored with zoo.trade on the day's recorded chain (half-gap and
touch): the research labels, scaled from $1k to the shadow book's $300 worst case. Labels for all
four trades are logged for every decision, so studies can compare any rule later.

  python3 -m odte.live.frozen train        # writes data/live/model_frozen.pkl (development rows only)
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FREEZE = "ebadab696ba01cf27951dfb32b7769869b725f02"
NAME = "frozen_lead"
BUDGET = 300.0                       # shadow book: $ worst case per trade (research labels are per $1k)


def model_path(cfg) -> Path:
    return cfg.model_path.parent / "model_frozen.pkl"


def train(cfg=None, path: Path | None = None) -> dict:
    """Fit exactly as exam.py's fit_predict on all development rows; save the bundle."""
    from odte.live.config import Config
    from odte.research import exam, meta
    cfg = cfg or Config()
    from odte.research import exam_guard as eg
    dev = meta.load()
    dev = dev[dev["day"] < eg.EXAM_FROM].reset_index(drop=True)   # the locked spec: rows before the exam only
    feats = exam.feature_list(dev, exam.GROUPS)
    X = dev[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    post = dev["post"].to_numpy()
    models = {}
    for k in exam.MENU:
        y = dev[f"y_{k}_half"].to_numpy(float)
        ok = ~post & np.isfinite(y)
        models[k] = meta.make(exam.MODEL).fit(X[ok], y[ok])
    R = {s: float(pd.read_parquet(meta.LIB / f"library_{s}.parquet", columns=["R_today"])["R_today"].iloc[0])
         for s in meta.INDEXES}
    b = {"name": NAME, "freeze": FREEZE, "models": models, "features": feats, "menu": list(exam.MENU),
         "R_today": R, "n_rows": int(len(dev)), "days": [str(dev["day"].min().date()), str(dev["day"].max().date())]}
    path = path or model_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(b))
    return b


def predict_row(bundle: dict, row: dict) -> dict:
    x = np.array([[float(row.get(c, np.nan)) if row.get(c) is not None else np.nan for c in bundle["features"]]])
    return {k: float(m.predict(x)[0]) for k, m in bundle["models"].items()}


def choose(preds: dict, post: bool) -> tuple:
    """exam.py's rule: best predicted trade if > 0, never in the post-release hour."""
    best = max(preds, key=lambda k: preds[k] if np.isfinite(preds[k]) else -np.inf)
    if post:
        return "none", "within 60 min after a release"
    if not np.isfinite(preds[best]) or preds[best] <= 0:
        return "none", "no trade predicted > 0"
    return best, "best predicted > 0"


def next_session(day) -> pd.Timestamp:
    s = pd.DatetimeIndex(pd.read_parquet(ROOT / "data" / "calendar" / "sessions.parquet")["session"])
    return s[s > pd.Timestamp(day).normalize()][0]


class Lead:
    def __init__(self, cfg, day, bundle: dict, j):
        self.cfg, self.day, self.b, self.j = cfg, pd.Timestamp(day).normalize(), bundle, j
        self.next_exp = next_session(self.day)
        self.next_chain: list = []
        self.spx_rows: list = []          # SPX decision rows so far (term structure needs the day's first)
        self.decisions: list = []
        self.centred: set = set()
        self.refined: set = set()
        self._cons: list = []
        self.st1 = np.nan                 # last next-expiry ATM straddle seen

    @classmethod
    def start(cls, cfg, day, j, now):
        p = model_path(cfg)
        if not p.exists():
            raise RuntimeError(f"no model at {p} (run: python3 -m odte.live.frozen train)")
        b = pickle.loads(p.read_bytes())
        lead = cls(cfg, day, b, j)
        j.log(now, "lead_shadow", name=NAME, freeze=b["freeze"][:7], features=len(b["features"]),
              menu=b["menu"], next_expiry=f"{lead.next_exp:%Y-%m-%d}", text="paper evidence only, no orders")
        return lead

    # ---- next-expiry quotes (term structure) -------------------------------------------------
    # Two minutes before a decision: subscribe ATM +/-5 (calls and puts) and three strikes around
    # each wing at the expected next-expiry straddle. One minute before: re-centre the wings on the
    # straddle just observed. Research picks wing strikes from every listed strike; this keeps the
    # live pick on the same strike while holding 12 quote lines.
    def tick(self, broker, now, times, done, spot: float, vix: float | None, last_quotes=None) -> None:
        for h in times:
            if h in done:
                continue
            t = self.day + pd.Timedelta(hours=int(h[:2]), minutes=int(h[3:]))
            if h not in self.centred and t - pd.Timedelta(minutes=2, seconds=5) <= now < t:
                self.centred.add(h)
                self._subscribe(broker, h, spot, self._st_guess(h, spot, vix, last_quotes), now, "centre")
            elif h in self.centred and h not in self.refined and t - pd.Timedelta(minutes=1, seconds=5) <= now < t:
                self.refined.add(h)
                st = self._latest_st1()
                if np.isfinite(st):
                    self.st1 = st
                    self._subscribe(broker, h, spot, st, now, "refine")

    def _st_guess(self, hhmm: str, spot: float, vix: float | None, last_quotes=None) -> float:
        """Last observed next-expiry straddle; else today's ATM straddle stretched to tomorrow's
        horizon (same vol); else VIX."""
        if np.isfinite(self.st1):
            return self.st1
        t = self.day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
        left = max((self.day + pd.Timedelta(hours=16) - t).total_seconds() / 60, 1.0)
        try:
            q = last_quotes.set_index(["option_type", "strike"])
            mid = ((q["bid"] + q["ask"]) / 2)
            ks = sorted(set(mid.loc["C"].index) & set(mid.loc["P"].index))
            k = min(ks, key=lambda x: abs(x - spot))
            st0 = float(mid.loc[("C", k)] + mid.loc[("P", k)])
            if np.isfinite(st0) and st0 > 0:
                return st0 * math.sqrt((left + 390) / left)
        except Exception:
            pass
        v = vix if vix and np.isfinite(vix) and vix > 0 else 15.0
        return spot * 0.8 * (v / 100) / math.sqrt(252) * math.sqrt(1 + left / 390)

    def _latest_st1(self) -> float:
        from odte.eval.expiry_edge import implied_forward, minute_grid
        from odte.research import term_structure as ts
        if not self.next_chain:
            return np.nan
        nc = pd.concat(self.next_chain[-3:], ignore_index=True)
        nc["quote_datetime"] = pd.to_datetime(nc["quote_datetime"])
        g = minute_grid(nc)
        t = nc["quote_datetime"].max().ceil("min")          # quotes become visible the next minute
        F1 = implied_forward(g).get(t, np.nan)
        if not np.isfinite(F1):
            return np.nan
        st, _ = ts._straddle_skew(g, F1, t)
        return st

    def _subscribe(self, broker, hhmm: str, spot: float, st: float, now, step: str) -> None:
        if not (np.isfinite(spot) and spot > 0 and np.isfinite(st)) or not hasattr(broker, "subscribe_next"):
            return
        atm = round(spot / 5) * 5
        kp, kc = round((atm - st) / 5) * 5, round((atm + st) / 5) * 5
        cons = ([(r, atm + d) for d in (-5, 0, 5) for r in ("C", "P")]
                + [("P", kp + d) for d in (-5, 0, 5)] + [("C", kc + d) for d in (-5, 0, 5)])
        if step == "refine" and sorted(cons) == sorted(self._cons):
            return
        self._cons = cons
        n = broker.subscribe_next(self.day, self.next_exp, cons)
        self.j.log(now, "lead_next_quotes", hhmm=hhmm, step=step, atm=atm, put_wing=kp, call_wing=kc,
                   st=round(st, 2), lines=n if n is not None else len(cons))

    def record(self, broker) -> None:
        if hasattr(broker, "snapshot_next"):
            try:
                q = broker.snapshot_next()
                if len(q):
                    self.next_chain.append(q)
            except Exception:
                pass

    # ---- decisions ---------------------------------------------------------------------------
    def _inputs(self, sym: str, r: dict, chain: pd.DataFrame, hhmm: str) -> dict:
        from odte.eval.expiry_edge import minute_grid
        from odte.research.library import flow_features
        x = dict(r)
        t = self.day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
        seen = chain[pd.to_datetime(chain["quote_datetime"]) <= t]
        if np.isfinite(x.get("W", np.nan)) and len(seen):
            x.update(flow_features(minute_grid(seen), self.day, hhmm, float(x["K"]), float(x["W"])))
        if sym != "SPX":
            for c in ("gex_net", "gex_0dte", "gex_flip_dist", "maxoi_dist", "oi_pcr_0dte"):
                x[c] = np.nan
        for s in ("SPX", "NDX", "RUT"):
            x[f"is_{s}"] = float(s == sym)
        x["tod"] = int(hhmm[:2]) * 60 + int(hhmm[3:])
        return x

    def _term(self, x: dict, hhmm: str) -> dict:
        from odte.research import term_structure as ts
        self.spx_rows.append({"day": self.day, "hhmm": hhmm, **{k: x.get(k, np.nan) for k in
                              ("F", "minutes_to_close", "straddle_mid", "skew")}})
        if not self.next_chain:
            return {k: np.nan for k in ts.SIGNALS}
        d = pd.DataFrame(self.spx_rows)
        nc = pd.concat(self.next_chain, ignore_index=True)
        nc["quote_datetime"] = pd.to_datetime(nc["quote_datetime"])
        nc["expiration"] = pd.to_datetime(nc["expiration"])
        out = ts.build(d, next_chains={self.day: nc})
        if len(d) - 1 not in out.index:
            return {k: np.nan for k in ts.SIGNALS}
        row = out.loc[len(d) - 1]
        try:                                                     # remember today's next-expiry straddle
            from odte.eval.expiry_edge import implied_forward, minute_grid
            g = minute_grid(nc)
            t = self.day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
            F1 = implied_forward(g).get(t, np.nan)
            st1, _ = ts._straddle_skew(g, F1 if np.isfinite(F1) else x["F"], t)
            if np.isfinite(st1):
                self.st1 = st1
        except Exception:
            pass
        return {k: float(row[k]) for k in ts.SIGNALS}

    def decide(self, now, hhmm: str, spx_row: dict | None, spx_chain: list, xsides: list, dctx: dict) -> None:
        from odte.live.features import feature_row
        rows = []
        if spx_row is not None and spx_chain:
            ch = pd.concat([c for c in spx_chain if len(c)], ignore_index=True)
            x = self._inputs("SPX", spx_row, ch, hhmm)
            x.update(self._term(x, hhmm))
            rows.append(("SPX", x))
        else:
            self.j.log(now, "shadow", strategy=NAME, index="SPX", hhmm=hhmm, trade="none", reason="no SPX row")
        for s in xsides:
            parts = [c for c in s.chain if len(c)]
            if not parts:
                self.j.log(now, "shadow", strategy=NAME, index=s.sym, hhmm=hhmm, trade="none", reason="no quotes")
                continue
            ch = pd.concat(parts, ignore_index=True)
            r = feature_row(ch, hhmm, dctx, s.prev)
            if r is None:
                self.j.log(now, "shadow", strategy=NAME, index=s.sym, hhmm=hhmm, trade="none", reason="no row")
                continue
            rows.append((s.sym, self._inputs(s.sym, r, ch, hhmm)))
        for sym, x in rows:
            preds = predict_row(self.b, x)
            mse = x.get("min_since_event", np.nan)
            post = bool(np.isfinite(mse) and mse <= 60)
            trade, why = choose(preds, post)
            missing = [c for c in self.b["features"] if not np.isfinite(float(x.get(c, np.nan)) if x.get(c) is not None else np.nan)]
            self.decisions.append({"index": sym, "hhmm": hhmm, "trade": trade, "preds": preds})
            self.j.log(now, "shadow", strategy=NAME, index=sym, hhmm=hhmm, trade=trade, reason=why,
                       side=int(trade != "none"), preds={k: round(v, 2) for k, v in preds.items()},
                       post=post, missing_inputs=missing)
            self.j.log(now, "lead_row", index=sym, hhmm=hhmm,
                       **{c: (float(x[c]) if x.get(c) is not None and np.isfinite(float(x[c])) else None)
                          for c in self.b["features"]})

    # ---- end of day --------------------------------------------------------------------------
    def settle(self, now, spx_chain: list, spx_settle: float, spx_prev: float, xsides: list,
               xset: dict, book: dict) -> None:
        from odte.research import zoo
        from odte.research.library import TRADES
        chains = {"SPX": (spx_chain, spx_settle, spx_prev)}
        for s in xsides:
            if s.sym in xset:
                chains[s.sym] = (s.chain, xset[s.sym][0], s.prev)
        scale = BUDGET / 1000.0
        for sym, (parts, settle, prev) in chains.items():
            dec = [d for d in self.decisions if d["index"] == sym]
            parts = [c for c in parts if len(c)]
            if not dec or not parts or not np.isfinite(settle):
                continue
            zd = zoo.Day(self.day, pd.concat(parts, ignore_index=True), float(settle), prev, {})
            t10 = self.day + pd.Timedelta(hours=10)
            r10 = zd.straddle_rel_spread(t10) if t10 in zd.minutes else np.nan
            R = self.b["R_today"].get(sym, np.nan)
            k_day = float(np.clip(R / r10, 0.1, 2.0)) if np.isfinite(r10) and r10 > 0 and np.isfinite(R) else 1.0
            for d in dec:
                labels = {}
                for name in self.b["menu"]:
                    for share, tag in ((0.5, "half"), (1.0, "touch")):
                        tr = zoo.trade(TRADES[name], zd, d["hhmm"], k_day, share)
                        labels[f"{name}_{tag}"] = float(tr["ret"]) * scale if tr else np.nan
                self.j.log(now, "lead_labels", index=sym, hhmm=d["hhmm"], k_day=round(k_day, 3),
                           **{k: (round(v, 2) if np.isfinite(v) else None) for k, v in labels.items()})
                if d["trade"] == "none":
                    continue
                p = {"half": labels[f"{d['trade']}_half"], "touch": labels[f"{d['trade']}_touch"]}
                if not np.isfinite(p["half"]):
                    self.j.log(now, "note", text=f"{NAME} {sym} {d['hhmm']} {d['trade']}: could not price "
                                                 "(a strike was not recorded)")
                    continue
                self.j.log(now, "shadow_settle", strategy=NAME, index=sym, hhmm=d["hhmm"], trade=d["trade"],
                           side=1, **{k: round(v, 2) for k, v in p.items() if np.isfinite(v)})
                for key in (NAME, f"{NAME}:{sym}"):
                    b = book.setdefault(key, {"trades": 0, "half": 0.0, "touch": 0.0})
                    b["trades"] += 1
                    for k, v in p.items():
                        if np.isfinite(v):
                            b[k] = round(b.get(k, 0.0) + v, 2)

    def save(self, cfg, day, tag: str) -> None:
        if self.next_chain:
            from odte.live.runner import merge_save
            merge_save(self.next_chain, cfg.chain_dir / "spx_next" / f"{pd.Timestamp(day):%Y-%m-%d}{tag}.parquet")


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        b = train()
        print(f"{b['name']}: {len(b['features'])} inputs, {b['n_rows']:,} development rows "
              f"{b['days'][0]} .. {b['days'][1]}, freeze {b['freeze'][:7]} -> {model_path(__import__('odte.live.config', fromlist=['Config']).Config())}")
