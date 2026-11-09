"""One trading day, start to finish.

  09:30-close   record the signal symbol's 0DTE quotes once a minute (at :58,
                so a snapshot is visible from the next minute, as in backtests)
  decisions     at each decision time: build the backtest feature row, predict
                realized/implied, SELL the fly if pred < 1 - margin, outside
                60 min after a scheduled event, inside the risk limits
  orders        walk the limit from mid toward at most half the gap; cancel
                if unfilled at the last rung (that outcome is data too)
  after close   settle the fills on the index close, write the journal

  python -m odte.live.runner --mode paper            # IB Gateway paper, today
  python -m odte.live.runner --mode replay --day 2024-03-13
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

from odte.live import fly
from odte.live.config import Config
from odte.live.strategies import STRATEGIES, shadow_pnl
from odte.live.features import day_context, feature_row
from odte.live.model import load as load_model
from odte.live.model import predict


class Journal:
    def __init__(self, cfg: Config, day, tag=""):
        cfg.journal_dir.mkdir(parents=True, exist_ok=True)
        self.path = cfg.journal_dir / f"{pd.Timestamp(day):%Y-%m-%d}{tag}.jsonl"
        self.path.write_text("")

    def log(self, t, kind, **kw):
        rec = {"t": str(t), "kind": kind, **{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                            for k, v in kw.items()}}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        print(f"{t:%H:%M:%S} {kind}: " + ", ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def realized_history(cfg: Config, before) -> pd.Series:
    """Daily realized P&L from earlier journals (for the drawdown pause)."""
    out = {}
    for p in sorted(cfg.journal_dir.glob("20??-??-??.jsonl")):
        if p.stem >= f"{pd.Timestamp(before):%Y-%m-%d}":
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] == "day_summary":
                out[p.stem] = r.get("pnl") or 0.0
    return pd.Series(out, dtype=float)


def paused(cfg: Config, day) -> bool:
    h = realized_history(cfg, day)
    if h.empty:
        return False
    cum = h.cumsum()
    dd = cum - cum.cummax()
    hit = dd[dd <= -cfg.pause_drawdown]
    return bool(len(hit)) and (len(h) - h.index.get_loc(hit.index[-1]) - 1) < cfg.pause_sessions


def run_day(cfg: Config, broker, day, prev_settle=np.nan, prev_ratio=np.nan, tag="") -> dict:
    day = pd.Timestamp(day).normalize()
    j = Journal(cfg, day, tag)
    bundle = load_model(cfg)
    dctx = day_context(day)
    close_hhmm = dctx["close_hhmm"] or "16:00"
    close_t = day + pd.Timedelta(hours=int(close_hhmm[:2]), minutes=int(close_hhmm[3:]))
    times = [h for h in cfg.decision_times if h < close_hhmm]
    events = [day + pd.Timedelta(hours=int(h[:2]), minutes=int(h[3:])) for h in dctx["event_times"]]
    j.log(broker.now(), "start", config={k: str(v) for k, v in asdict(cfg).items()},
          model_rows=bundle["n_rows"], events=dctx["event_times"], close=close_hhmm)
    is_paused = paused(cfg, day)
    if is_paused:
        j.log(broker.now(), "paused", reason=f"drawdown >= ${cfg.pause_drawdown:,.0f}")

    open_t = day + pd.Timedelta(hours=9, minutes=31)
    while broker.now() < open_t:
        broker.sleep(min(30, max(1, (open_t - broker.now()).total_seconds())))
    spot = broker.spot()
    strikes = fly_strikes = broker_strikes(cfg, spot)
    broker.subscribe(day, fly_strikes)
    j.log(broker.now(), "subscribed", spot=round(spot, 2), n_strikes=len(strikes))

    chain, done, active, fills, shadows = [], set(), None, [], []
    opened_worst, n_trades = 0.0, 0
    last_snap = None
    while broker.now() < close_t:
        now = broker.now()
        minute = now.floor("min")
        if now.second >= 58 and last_snap != minute:                 # record once a minute
            chain.append(broker.snapshot())
            last_snap = minute
        hhmm = now.strftime("%H:%M")
        if hhmm in times and hhmm not in done and now.second >= 2:
            done.add(hhmm)
            rows = pd.concat(chain, ignore_index=True) if chain else pd.DataFrame()
            r = feature_row(rows, hhmm, dctx, prev_settle, prev_ratio) if len(rows) else None
            if r is None:
                j.log(now, "decision", hhmm=hhmm, action="skip", reason="no quotes / no fly")
            else:
                pred = predict(bundle, r)
                t_dec = day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
                recent = [e for e in events if 0 <= (t_dec - e).total_seconds() / 60 <= cfg.post_event_min]
                for name, rule in STRATEGIES.items():          # shadow book: no orders
                    side, reason = rule(pred, bool(recent))
                    shadows.append({"strategy": name, "hhmm": hhmm, "side": side, "row": r})
                    j.log(now, "shadow", strategy=name, hhmm=hhmm, side=side, reason=reason)
                why = None
                if pred >= 1 - cfg.margin:
                    why = "model: not overpriced enough"
                elif recent:
                    why = "within 60 min after a scheduled event"
                elif is_paused:
                    why = "paused after drawdown"
                elif cfg.kill_switch.exists():
                    why = "kill switch present"
                elif len(fills) >= cfg.max_trades_per_day:
                    why = "max filled trades today"
                elif n_trades >= cfg.max_attempts_per_day:
                    why = "max order attempts today"
                elif active is not None:
                    why = "an order is still working"
                o = None
                if why is None:
                    budget = min(cfg.max_loss_per_trade, cfg.max_loss_per_day - opened_worst)
                    o = fly.build(r, cfg.symbol, budget, cfg.walk_steps, cfg.max_concession, cfg.tick())
                    if o is None:
                        why = f"one contract risks more than the ${budget:,.0f} left"
                j.log(now, "decision", hhmm=hhmm, pred=round(pred, 4), implied_move=r["implied_move"],
                      K=r["K"], W=r["W"], mid=r["fly_credit_mid"], touch=r["fly_credit"],
                      action="sell" if why is None else "skip", reason=why)
                if o is not None:
                    h = broker.place_fly(o, o.ladder[0])
                    active = {"h": h, "o": o, "rung": 0, "since": now, "hhmm": hhmm}
                    n_trades += 1
                    opened_worst += o.worst_case
                    j.log(now, "order", hhmm=hhmm, symbol=o.symbol, K=o.K, W=o.W, qty=o.qty,
                          price=o.ladder[0], mid=o.mid, touch=o.touch, ladder=o.ladder,
                          worst_case=round(o.worst_case, 2))
        if active is not None:
            q, avg = broker.filled(active["h"])
            o = active["o"]
            if q > 0:
                gap = o.mid - o.touch
                j.log(now, "fill", hhmm=active["hhmm"], qty=q, credit=avg, mid=o.mid, touch=o.touch,
                      share_of_gap_paid=round((o.mid - avg) / gap, 3) if gap > 0 else None,
                      seconds=(now - active["since"]).total_seconds())
                fills.append({"o": o, "qty": q, "credit": avg, "hhmm": active["hhmm"]})
                active = None
            elif (now - active["since"]).total_seconds() >= cfg.walk_wait_s:
                active["rung"] += 1
                if active["rung"] < len(o.ladder):
                    broker.modify(active["h"], o.ladder[active["rung"]])
                    active["since"] = now
                    j.log(now, "reprice", hhmm=active["hhmm"], price=o.ladder[active["rung"]])
                else:
                    broker.cancel(active["h"])
                    opened_worst -= o.worst_case
                    j.log(now, "unfilled", hhmm=active["hhmm"], last_price=o.ladder[-1], mid=o.mid,
                          touch=o.touch)
                    active = None
        broker.sleep(1)

    if active is not None:
        broker.cancel(active["h"])
    if chain:
        cfg.chain_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(chain, ignore_index=True).to_parquet(cfg.chain_dir / f"{day:%Y-%m-%d}{tag}.parquet")
    settle = broker.settlement(day)
    s = 0.1 if cfg.symbol == "XSP" else 1.0
    pnl = 0.0
    for f in fills:
        p = fly.settle_pnl(f["credit"], f["qty"], f["o"].K, f["o"].W, settle * s)
        pnl += p
        j.log(broker.now(), "settle", hhmm=f["hhmm"], settle=settle, pnl=round(p, 2))
    book = {}
    for sh in shadows:
        p = shadow_pnl(sh["side"], sh["row"], settle)
        if p:
            j.log(broker.now(), "shadow_settle", strategy=sh["strategy"], hhmm=sh["hhmm"],
                  side=sh["side"], **{k: round(v, 2) for k, v in p.items()})
            b = book.setdefault(sh["strategy"], {"trades": 0, "mid": 0.0, "half": 0.0, "touch": 0.0})
            b["trades"] += 1
            for k in ("mid", "half", "touch"):
                b[k] = round(b[k] + p[k], 2)
    summary = {"day": f"{day:%Y-%m-%d}", "decisions": len(done), "orders": n_trades, "fills": len(fills),
               "shadow": book,
               "share_of_gap_paid": [round((f["o"].mid - f["credit"]) / (f["o"].mid - f["o"].touch), 3)
                                     if f["o"].mid > f["o"].touch else None for f in fills],
               "pnl": round(pnl, 2), "settle": settle}
    j.log(broker.now(), "day_summary", **summary)
    return summary


def broker_strikes(cfg: Config, spot: float) -> list:
    from odte.live.broker import strikes_for
    return strikes_for(spot, cfg.inner_band, cfg.band, cfg.outer_step, max_lines=cfg.max_lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["paper", "replay"], default="paper")
    ap.add_argument("--day", default=None, help="replay day (YYYY-MM-DD)")
    ap.add_argument("--symbol", choices=["XSP", "SPX"], default="XSP")
    a = ap.parse_args()
    cfg = Config(mode=a.mode, symbol=a.symbol)
    cfg.check()
    if a.mode == "replay":
        from odte.live.broker import ReplayBroker
        day = pd.Timestamp(a.day)
        st = pd.read_csv("data/odte_0dte/settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
        prev = float(st[st.index < day].iloc[-1])
        run_day(cfg, ReplayBroker(day), day, prev_settle=prev, tag="-replay")
    else:
        from odte.live.broker import IBKRBroker
        b = IBKRBroker(cfg)
        day = b.now().normalize()
        st = pd.read_csv("data/odte_0dte/settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
        prev = float(st[st.index < day].iloc[-1]) if (st.index < day).any() else np.nan
        run_day(cfg, b, day, prev_settle=prev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
