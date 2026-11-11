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
from odte.live.strategies import ORDER_STRATEGIES, STRATEGIES, shadow_pnl, vertical_pnl
from odte.live.features import day_context, feature_row
from odte.live.model import load as load_model
from odte.live.model import predict, signal


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


def load_candidates(cfg: Config) -> dict:
    """Every candidate model the weekly loop saved, for the shadow book."""
    import pickle
    d = cfg.journal_dir.parent / "candidates"
    return {p.stem: pickle.loads(p.read_bytes()) for p in sorted(d.glob("*.pkl"))} if d.exists() else {}


def load_side(cfg: Config):
    """The edge #1 side model and its skew history (from the weekly loop)."""
    import pickle
    d = cfg.journal_dir.parent / "side"
    if not (d / "side_dir.pkl").exists():
        return None, pd.DataFrame(columns=["day", "hhmm", "skew"])
    h = pd.read_parquet(d / "skew_history.parquet") if (d / "skew_history.parquet").exists() \
        else pd.DataFrame(columns=["day", "hhmm", "skew"])
    h["day"] = pd.to_datetime(h["day"])
    return pickle.loads((d / "side_dir.pkl").read_bytes()), h


def default_oi(day, spot):
    """Pre-open open interest for the dealer-gamma inputs (history file if present)."""
    from odte.data import gamma
    return gamma._load(day)


def _save_chains(cfg: Config, day, tag, chain, xchain) -> None:
    """The day's recorded quotes (also written on a crash: they are training data)."""
    if chain:
        cfg.chain_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(chain, ignore_index=True).to_parquet(cfg.chain_dir / f"{day:%Y-%m-%d}{tag}.parquet")
    if any(len(x) for x in xchain):
        (cfg.chain_dir / "xsp").mkdir(parents=True, exist_ok=True)
        pd.concat(xchain, ignore_index=True).to_parquet(cfg.chain_dir / "xsp" / f"{day:%Y-%m-%d}{tag}.parquet")


def is_trading_day(day) -> bool:
    """Weekday and not an NYSE full-day holiday (rule-based, works for future days)."""
    from odte.data.calendar import nyse_holidays
    day = pd.Timestamp(day).normalize()
    return day.day_of_week < 5 and day not in nyse_holidays(day, day)


def _no_alert(title, text):
    pass


def run_day(cfg: Config, broker, day, prev_settle=np.nan, prev_ratio=np.nan, tag="",
            oi_provider=default_oi, alert=_no_alert) -> dict:
    day = pd.Timestamp(day).normalize()
    j = Journal(cfg, day, tag)
    bundle = load_model(cfg)
    cands = load_candidates(cfg)
    side_b, skew_hist = load_side(cfg)
    oi, oi_tried = None, False
    dctx = day_context(day)
    close_hhmm = dctx["close_hhmm"] or "16:00"
    close_t = day + pd.Timedelta(hours=int(close_hhmm[:2]), minutes=int(close_hhmm[3:]))
    times = [h for h in cfg.decision_times if h < close_hhmm]
    events = [day + pd.Timedelta(hours=int(h[:2]), minutes=int(h[3:])) for h in dctx["event_times"]]
    j.log(broker.now(), "start", code=code_version(), config={k: str(v) for k, v in asdict(cfg).items()},
          model=bundle.get("name"), model_kind=bundle.get("kind", "ratio"),
          model_rows=bundle["n_rows"], events=dctx["event_times"], close=close_hhmm)
    is_paused = paused(cfg, day)
    order_strats = list(ORDER_STRATEGIES)
    reg = cfg.journal_dir.parent / "registry.json"
    if reg.exists() and not json.loads(reg.read_text()).get("ml_orders_enabled", True):
        order_strats = [s for s in order_strats if s == "event_seller"]
        j.log(broker.now(), "note", text="model-driven orders switched off by the weekly loop")
    if is_paused:
        j.log(broker.now(), "paused", reason=f"drawdown >= ${cfg.pause_drawdown:,.0f}")

    open_t = day + pd.Timedelta(hours=9, minutes=31)
    while broker.now() < open_t:
        broker.sleep(min(30, max(1, (open_t - broker.now()).total_seconds())))
    spot = broker.spot()
    strikes = broker_strikes(cfg, spot)
    xsp_strikes = ([float(k) for k in range(round(spot / 10) - cfg.xsp_halfwidth,
                                            round(spot / 10) + cfg.xsp_halfwidth + 1)]
                   if cfg.symbol == "XSP" else [])
    broker.subscribe(day, strikes, xsp_strikes)
    j.log(broker.now(), "subscribed", spot=round(spot, 2), n_strikes=len(strikes), n_xsp_strikes=len(xsp_strikes))
    from odte.live import xshadow
    xsides = xshadow.setup(cfg, broker, day, spot, prev_settle, j)     # NDX/RUT, shadow only

    chain, xchain, done, active, fills, shadows = [], [], set(), None, [], []
    ratio_ref = None
    opened_worst, n_trades = 0.0, 0
    last_snap = None
    try:
        while broker.now() < close_t:
            now = broker.now()
            minute = now.floor("min")
            if now.second >= 58 and last_snap != minute:                 # record once a minute
                chain.append(broker.snapshot())
                if cfg.symbol == "XSP":
                    xchain.append(broker.snapshot_xsp())
                xshadow.record(xsides, broker)
                last_snap = minute
            hhmm = now.strftime("%H:%M")
            if hhmm in times and hhmm not in done and now.second >= 2:
                done.add(hhmm)
                rows = pd.concat(chain, ignore_index=True) if chain else pd.DataFrame()
                r = feature_row(rows, hhmm, dctx, prev_settle, prev_ratio) if len(rows) else None
                if not oi_tried:                           # once, before the first decision
                    oi_tried = True
                    try:
                        oi = oi_provider(day, spot)
                    except Exception as e:
                        j.log(now, "note", text=f"open interest unavailable: {e}")
                    j.log(now, "open_interest", contracts=0 if oi is None else int(len(oi)))
                if r is not None:
                    from odte.data import gamma
                    r.update(gamma.features_for(oi, day, r["F"], r["implied_move"], r["minutes_to_close"]))
                if r is None:
                    j.log(now, "decision", hhmm=hhmm, action="skip", reason="no quotes / no fly")
                else:
                    pred = predict(bundle, r)
                    sig = signal(bundle, pred, cfg.margin)
                    xfly = (fly.fly_from_quotes(xchain[-1], r["F"] / 10.0)
                            if cfg.symbol == "XSP" and xchain and len(xchain[-1]) else None)
                    if hhmm == "10:00" or ratio_ref is None:
                        ratio_ref = (r["K"], r["straddle_mid"])
                    t_dec = day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
                    recent = [e for e in events if 0 <= (t_dec - e).total_seconds() / 60 <= cfg.post_event_min]
                    sides = {}
                    for name, rule in STRATEGIES.items():          # shadow book: no orders
                        side, reason = rule(sig, bool(recent), r)
                        sides[name] = side
                        shadows.append({"strategy": name, "hhmm": hhmm, "side": side, "row": r, "xfly": xfly})
                        j.log(now, "shadow", strategy=name, hhmm=hhmm, side=side, reason=reason)
                    if side_b is not None:                          # edge #1: pick the side
                        past = skew_hist[(skew_hist["hhmm"] == hhmm) & (skew_hist["day"] < day)].tail(20)
                        r["skew_vs_hist"] = ((r["skew"] - past["skew"].mean()) / past["skew"].std()
                                             if len(past) >= 10 and past["skew"].std() > 0 else np.nan)
                        skew_hist = pd.concat([skew_hist, pd.DataFrame([{"day": day, "hhmm": hhmm,
                                                                         "skew": r["skew"]}])], ignore_index=True)
                        pz = predict(side_b, r)
                        vq = fly.vertical_from_quotes(chain[-1], r["K"], r["W"]) if chain else None
                        structure = "put" if pz > 0 else "call"
                        if vq is not None and np.isfinite(pz) and not recent:
                            shadows.append({"strategy": "side_seller", "hhmm": hhmm, "side": 1,
                                            "structure": structure, "vq": vq, "row": r})
                        j.log(now, "shadow", strategy="side_seller", hhmm=hhmm,
                              side=(structure if vq is not None and not recent else 0),
                              pred_z=round(pz, 4) if np.isfinite(pz) else None)
                    for name, cb in cands.items():                  # every candidate model, shadow only
                        cp = predict(cb, r)
                        side = 1 if signal(cb, cp, cfg.margin) == 1 and not recent else 0
                        shadows.append({"strategy": f"model:{name}", "hhmm": hhmm, "side": side, "row": r,
                                        "xfly": xfly})
                        j.log(now, "shadow", strategy=f"model:{name}", hhmm=hhmm, side=side,
                              pred=round(cp, 4) if np.isfinite(cp) else None)
                    if xsides:                                  # NDX/RUT shadows + edge #10
                        try:
                            xp = xshadow.decide(xsides, hhmm, dctx, bool(recent), now, j, shadows)
                            spx_ratio = pred if bundle.get("kind", "ratio") == "ratio" else (
                                predict(cands["ratio_ridge_full"], r) if "ratio_ridge_full" in cands else np.nan)
                            xs, xwhy = xshadow.xindex_side(spx_ratio, xp, bool(recent))
                            shadows.append({"strategy": "xindex_spx", "hhmm": hhmm, "side": xs, "row": r,
                                            "xfly": xfly})
                            j.log(now, "shadow", strategy="xindex_spx", hhmm=hhmm, side=xs, reason=xwhy,
                                  preds={k: round(v, 4) for k, v in {"SPX": spx_ratio, **xp}.items()
                                         if np.isfinite(v)})
                        except Exception as e:
                            j.log(now, "note", text=f"index shadows failed at {hhmm}: {e!r}")
                    why = None
                    wants = [n for n in order_strats if sides.get(n) == 1]
                    if not wants:
                        why = "no order strategy says sell"
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
                        if xfly is not None:                     # XSP's own book
                            o = fly.build(xfly, cfg.symbol, budget, cfg.walk_steps, cfg.max_concession,
                                          cfg.tick(), native=True)
                        else:                                     # no XSP quotes: SPX / 10 (logged)
                            o = fly.build(r, cfg.symbol, budget, cfg.walk_steps, cfg.max_concession, cfg.tick())
                        if o is not None and o.qty > cfg.max_qty:
                            o.worst_case = o.worst_case / o.qty * cfg.max_qty
                            o.qty = cfg.max_qty
                        if o is None:
                            why = f"one contract risks more than the ${budget:,.0f} left"
                    j.log(now, "decision", hhmm=hhmm, pred=round(pred, 4), wanted_by=wants,
                          priced_on="XSP book" if xfly is not None else "SPX/10",
                          xsp=({k: round(v, 3) for k, v in xfly.items()} if xfly else None),
                          implied_move=r["implied_move"],
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
    except BaseException as e:                       # crash or Ctrl-C: leave nothing working
        cancelled = None
        if active is not None:
            try:
                broker.cancel(active["h"])
                cancelled = True
            except Exception as ce:
                cancelled = False
                j.log(broker.now(), "note", text=f"cancel failed: {ce!r}")
        _save_chains(cfg, day, tag, chain, xchain)
        try:
            xshadow.save(xsides, cfg, day, tag)
        except Exception:
            pass
        j.log(broker.now(), "crash", error=repr(e), working_order=active is not None, cancelled=cancelled,
              fills=len(fills))
        alert("bot stopped", f"{day:%Y-%m-%d} {broker.now():%H:%M}: {e!r}. "
              + ("" if active is None else "Working order cancelled. " if cancelled
                 else "COULD NOT cancel the working order: check IBKR. ")
              + "Any filled fly is defined-risk and settles in cash at the close.")
        try:
            e._alerted = True                          # main() then skips a second alert
        except Exception:
            pass
        raise

    if active is not None:
        broker.cancel(active["h"])
    _save_chains(cfg, day, tag, chain, xchain)
    xshadow.save(xsides, cfg, day, tag)
    if side_b is not None and len(skew_hist) and not tag:      # never let replays touch live history
        sp = cfg.journal_dir.parent / "side" / "skew_history.parquet"
        skew_hist[skew_hist["day"] >= day - pd.Timedelta(days=60)].to_parquet(sp, index=False)
    settle = broker.settlement(day)
    try:
        xset = xshadow.settles(xsides, broker, day) if xsides else {}
    except Exception as e:
        xset = {}
        j.log(broker.now(), "note", text=f"index settlements failed: {e!r}")
    for k, (v, src) in xset.items():
        j.log(broker.now(), "settle_index", sym=k, settle=round(v, 2), source=src)
    s = 0.1 if cfg.symbol == "XSP" else 1.0
    pnl = 0.0
    for f in fills:
        p = fly.settle_pnl(f["credit"], f["qty"], f["o"].K, f["o"].W, settle * s)
        pnl += p
        j.log(broker.now(), "settle", hhmm=f["hhmm"], settle=settle, pnl=round(p, 2))
    book = {}
    for sh in shadows:
        if sh.get("index"):                               # NDX/RUT: their own fly and close
            if sh["index"] not in xset:
                continue
            p = shadow_pnl(sh["side"], sh["row"], xset[sh["index"]][0])
        else:
            p = (vertical_pnl(sh["structure"], sh["vq"], settle) if sh.get("structure")
                 else shadow_pnl(sh["side"], sh["row"], settle))
        if p and sh.get("xfly"):                          # same trade on XSP's own bid/ask
            px = shadow_pnl(sh["side"], sh["xfly"], settle / 10.0)
            p.update({f"xsp_{k}": v for k, v in px.items()})
        if p:
            j.log(broker.now(), "shadow_settle", strategy=sh["strategy"], hhmm=sh["hhmm"],
                  side=sh["side"], **{k: round(v, 2) for k, v in p.items()})
            b = book.setdefault(sh["strategy"], {"trades": 0, "mid": 0.0, "half": 0.0, "touch": 0.0})
            b["trades"] += 1
            for k, v in p.items():
                if np.isfinite(v):
                    b[k] = round(b.get(k, 0.0) + v, 2)
    ratio_today = (abs(settle - ratio_ref[0]) / ratio_ref[1]
                   if ratio_ref and ratio_ref[1] > 0 else None)          # tomorrow's prev_ratio
    summary = {"day": f"{day:%Y-%m-%d}", "decisions": len(done), "orders": n_trades, "fills": len(fills),
               "ratio": round(ratio_today, 4) if ratio_today is not None else None,
               "shadow": book,
               "share_of_gap_paid": [round((f["o"].mid - f["credit"]) / (f["o"].mid - f["o"].touch), 3)
                                     if f["o"].mid > f["o"].touch else None for f in fills],
               "pnl": round(pnl, 2), "settle": settle}
    summary["reconcile_ok"] = _reconcile(broker, day, fills, j, alert)
    j.log(broker.now(), "day_summary", **summary)
    return summary


def reconcile_problems(rec: dict, journal_fills: int) -> list:
    out = []
    if rec["open_orders"]:
        out.append(f"{rec['open_orders']} order(s) still open at IBKR")
    if rec["orders_with_fills"] != journal_fills:
        out.append(f"IBKR shows {rec['orders_with_fills']} filled order(s), journal has {journal_fills}")
    if rec["later_expiry_positions"]:
        out.append(f"{rec['later_expiry_positions']} option position(s) expiring after today")
    return out


def _reconcile(broker, day, fills, j, alert):
    """Funds check their own records against the broker every night; so do we."""
    if not hasattr(broker, "reconcile"):
        return None
    try:
        rec = broker.reconcile(day)
    except Exception as e:
        j.log(broker.now(), "note", text=f"reconcile failed: {e!r}")
        alert("bot reconcile failed", f"{day:%Y-%m-%d}: {e!r}. Check IBKR by hand.")
        return False
    problems = reconcile_problems(rec, len(fills))
    j.log(broker.now(), "reconcile", **rec, journal_fills=len(fills), ok=not problems, problems=problems)
    if problems:
        alert("bot reconcile MISMATCH", f"{day:%Y-%m-%d}: " + "; ".join(problems))
    return not problems


def code_version() -> str:
    """Short git commit of the code that ran (+dirty if odte/ had local edits)."""
    import subprocess
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[2])
    try:
        h = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "odte"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (h or "unknown") + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def load_databento_key() -> None:
    """For scheduled runs: read the key from an owner-only file the user created
    (~/.config/silicon-alpha/databento.env, line `export DATABENTO_API_KEY=...`).
    The key is never printed or written anywhere else."""
    import os
    from pathlib import Path
    if os.environ.get("DATABENTO_API_KEY"):
        return
    p = Path.home() / ".config" / "silicon-alpha" / "databento.env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line.startswith("DATABENTO_API_KEY="):
                os.environ["DATABENTO_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")


def previous_day(cfg: Config, day) -> tuple:
    """(prev_settle, prev_ratio) from the latest earlier journal, else history."""
    for p in sorted(cfg.journal_dir.glob("20??-??-??.jsonl"), reverse=True):
        if p.stem >= f"{pd.Timestamp(day):%Y-%m-%d}":
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] == "day_summary" and r.get("settle"):
                return float(r["settle"]), float(r["ratio"]) if r.get("ratio") is not None else np.nan
    st = pd.read_csv("data/odte_0dte/settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
    prev = float(st[st.index < pd.Timestamp(day)].iloc[-1]) if (st.index < pd.Timestamp(day)).any() else np.nan
    return prev, np.nan


def broker_strikes(cfg: Config, spot: float) -> list:
    from odte.live.broker import strikes_for
    return strikes_for(spot, cfg.inner_band, cfg.band, cfg.outer_step, max_lines=cfg.spx_lines())


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
        from odte.live.notify import day_text, notify
        today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
        if not is_trading_day(today):
            print(f"{today:%Y-%m-%d}: market closed (weekend or NYSE holiday); nothing to do", flush=True)
            return 0
        try:
            summary = _run_scheduled(cfg, today, notify)
        except BaseException as e:
            if not isinstance(e, SystemExit) and not getattr(e, "_alerted", False):
                notify("bot FAILED", f"{today:%Y-%m-%d}: {e!r}. See data/live/logs/daily.err")
            raise
        notify("bot day done", day_text(summary))
    return 0


def _run_scheduled(cfg: Config, today, notify) -> dict:
    from odte.data import context as ctx
    from odte.eval.expiry_edge import load_early_closes
    from odte.live.broker import IBKRBroker
    if not any(d.year == today.year for d in load_early_closes()):
        notify("bot warning", f"no {today.year} early closes in configs/calendar/early_closes.csv: "
                              "half days would be traded as full days")
    try:
        ctx.refresh()                         # yesterday's VIX etc.; only dates before today are used
    except Exception as e:
        print(f"WARNING: context refresh failed ({e}); using the saved closes", flush=True)
    b = IBKRBroker(cfg)
    day = b.now().normalize()
    prev_settle, prev_ratio = previous_day(cfg, day)

    def live_oi(d, spot):
        from odte.data import gamma
        from odte.data.fetch_oi import fetch_today
        load_databento_key()
        fetch_today(d, spot)                          # ~$0.001; pre-open data only
        return gamma._load(d)
    return run_day(cfg, b, day, prev_settle=prev_settle, prev_ratio=prev_ratio, oi_provider=live_oi,
                   alert=notify)


if __name__ == "__main__":
    sys.exit(main())
