"""Weekly learn -> test -> promote loop.

Run once a week (e.g. Saturday) after the bot's trading days:

  1. INGEST: turn each new live day (quotes the bot recorded + its settlement)
     into decision rows with the backtest feature code, appended to
     data/live/dataset.parquet. These are days no model has seen.
  2. RETRAIN every candidate model on history + live days before the test window.
  3. TEST each on the most recent WINDOW live days (unseen by those fits): the
     daily P&L of selling the iron fly when the candidate says so, at half the
     day's own bid-ask gap, $1k worst case per trade, outside the post-release hour.
  4. PROMOTE by the fixed rule below (revised 2026-11-09 after the independent
     review, still before any live day): a challenger replaces the champion only
     if, over >= MIN_LIVE_DAYS shared test days, it has >= MIN_TRADES trades, a
     positive total, and its PAIRED daily advantage over the champion is at least
     PROMOTE_SE standard errors. Daily P&L at a $1k budget swings ~$2,500/day,
     so a fixed $/day margin would swap models on luck. There is no automatic
     on/off switching; the runner's drawdown pause and kill switch handle risk.
  5. REFIT the champion on everything and save it as the bot's model.
  6. REPORT to reports/live_weekly/<date>.md, including every shadow strategy's
     live results from the journals.

  python -m odte.live.loop weekly
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from odte.live.config import Config

ROOT = Path(__file__).resolve().parents[2]
HIST = [ROOT / "reports" / "expiry_edge" / "decisions-sealed-c1.parquet",      # 2016-01 .. 2022-05-13
        ROOT / "reports" / "expiry_edge" / "decisions-dev-gex-es.parquet"]     # 2022-05-16 .. 2026-09-04
EXAM_FROM = pd.Timestamp("2026-09-08")       # never trained on: the final exam stays locked
TIMES = ("10:00", "11:00", "12:00", "13:00", "14:00", "15:00")
FEES4, BUDGET, TODAY_COST = 4 * 1.20, 1000.0, 0.0169

# Fixed before any live day (2026-11-09; revised after the independent review)
WINDOW, MIN_LIVE_DAYS, MIN_TRADES, PROMOTE_SE = 120, 60, 30, 1.5


def _small():
    from odte.eval.expiry_edge import FEATURES
    return list(FEATURES) + ["ev_fomc", "ev_cpi", "ev_nfp", "events_ahead", "minutes_to_next_event",
                             "vix_prev", "vix1d_over_vix", "vix_chg", "spx_ret_prev", "spx_rv5",
                             "iv_vs_vix1d", "megacap_earnings"]


def candidates() -> dict:
    """name -> (target, kind, feature list). New ideas are added here; each must
    earn promotion on live days."""
    from odte.live.model import feature_list
    full = feature_list()
    return {"ratio_ridge_full": ("ratio", "ratio", full),        # the current champion
            "profit_ridge_full": ("ror_half", "profit", full),
            "ratio_ridge_small": ("ratio", "ratio", _small())}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _returns(d: pd.DataFrame, gap_frac=None) -> pd.DataFrame:
    """Half-gap and touch return on risk of selling the fly. gap_frac=None uses
    each row's own quoted gap (live); a number uses today's typical gap (history,
    whose old spreads were far wider than today's)."""
    cm, W = d["fly_credit_mid"].to_numpy(float), d["W"].to_numpy(float)
    touch = (cm * (1 - gap_frac)) if gap_frac is not None else d["fly_credit"].to_numpy(float)
    intrinsic = cm - d["pnl_short_fly_mid"].to_numpy(float) / 100
    for name, c in (("half", cm - 0.5 * (cm - touch)), ("touch", touch)):
        risk = (W - c) * 100 + FEES4
        d[f"ror_{name}"] = np.where(risk > 0, ((c - intrinsic) * 100 - FEES4) / risk, np.nan)
    d["post"] = (d["min_since_event"] <= 60).fillna(False).to_numpy() if "min_since_event" in d else False
    return d


def history() -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in HIST if f.exists()]
    d = pd.concat(parts, ignore_index=True)
    d = d[d["on6"]].copy()
    d["day"] = pd.to_datetime(d["day"])
    d = d[d["day"] < EXAM_FROM].drop_duplicates(["day", "hhmm"]).reset_index(drop=True)
    d["source"] = "history"
    return _returns(d, TODAY_COST)


def ingest_day(cfg: Config, day) -> pd.DataFrame | None:
    """Decision rows (features + outcome) for one live day the bot recorded."""
    from odte.data import calendar as cal
    from odte.eval.expiry_edge import implied_forward, minute_grid, session_rows
    from odte.live.features import day_context
    day = pd.Timestamp(day).normalize()
    chain_p = cfg.chain_dir / f"{day:%Y-%m-%d}.parquet"
    jr = cfg.journal_dir / f"{day:%Y-%m-%d}.jsonl"
    if not chain_p.exists() or not jr.exists():
        return None
    settle = next((json.loads(l).get("settle") for l in jr.read_text().splitlines()
                   if json.loads(l)["kind"] == "day_summary"), None)
    if settle is None:
        return None
    chain = pd.read_parquet(chain_p)
    dctx = day_context(day)
    grid = minute_grid(chain)
    F = implied_forward(grid)
    marks = [day + pd.Timedelta(hours=int(h[:2]), minutes=int(h[3:])) for h in dctx["event_times"]]
    rows = session_rows(grid, F, float(settle), times=TIMES, event_marks=marks,
                        close_hhmm=dctx["close_hhmm"] or "16:00")
    if not rows:
        return None
    d = pd.DataFrame(rows)
    cf = pd.DataFrame([cal.features_for(dctx["events"], day, h) for h in d["hhmm"]])
    xf = pd.DataFrame([dctx["ctx"]] * len(d))
    d = pd.concat([d.reset_index(drop=True), cf, xf], axis=1)
    import math
    d["iv_vs_vix1d"] = [(im / 0.8) / math.sqrt(max(m, 1.0) / (390 * 252)) * 100.0 / v
                        if v and np.isfinite(v) else np.nan
                        for im, m, v in zip(d["implied_move"], d["minutes_to_close"], d["vix1d_prev"])]
    d["day"], d["on6"], d["source"] = day, True, "live"
    return _returns(d)


def live_dataset(cfg: Config) -> pd.DataFrame:
    p = cfg.journal_dir.parent / "dataset.parquet"
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    done = set(pd.to_datetime(have["day"]).dt.strftime("%Y-%m-%d")) if len(have) else set()
    new = []
    for c in sorted(cfg.chain_dir.glob("20??-??-??.parquet")):
        if c.stem not in done:
            r = ingest_day(cfg, c.stem)
            if r is not None:
                new.append(r)
    if new:
        have = pd.concat([have, *new], ignore_index=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        have.to_parquet(p, index=False)
    if len(have):
        have["day"] = pd.to_datetime(have["day"])
    return have


# ---------------------------------------------------------------------------
# fit / test / promote
# ---------------------------------------------------------------------------

def fit(d: pd.DataFrame, target: str, cols: list):
    from odte.eval.expiry_edge import _make_model
    y = d[target].clip(upper=4.0).to_numpy(float) if target == "ratio" else d[target].to_numpy(float)
    ok = np.isfinite(y)
    return _make_model("ml_ridge").fit(d.loc[ok, cols].to_numpy(float), y[ok])


def daily_pnl(d: pd.DataFrame, sell: np.ndarray) -> pd.Series:
    pnl = np.where(sell & ~d["post"].to_numpy(bool), d["ror_half"].to_numpy(float) * BUDGET, 0.0)
    return pd.Series(np.nan_to_num(pnl)).groupby(d["day"].to_numpy()).sum()


def evaluate(hist: pd.DataFrame, live: pd.DataFrame, cands: dict) -> dict:
    from odte.live.model import signal
    days = sorted(live["day"].unique()) if len(live) else []
    test_days = days[-WINDOW:]
    if len(test_days) < MIN_LIVE_DAYS:
        return {"status": f"collecting: {len(days)} live days (need {MIN_LIVE_DAYS})", "results": {}}
    start = test_days[0]
    train = pd.concat([hist, live[live["day"] < start]], ignore_index=True)
    test = live[live["day"] >= start].reset_index(drop=True)
    out = {}
    for name, (target, kind, cols) in cands.items():
        m = fit(train, target, cols)
        pred = m.predict(test[cols].to_numpy(float))
        sell = np.array([signal({"kind": kind}, p) == 1 for p in pred])
        x = daily_pnl(test, sell)
        out[name] = {"days": int(len(x)), "trades": int((sell & ~test["post"].to_numpy(bool)).sum()),
                     "total": float(x.sum()), "mean_day": float(x.mean()),
                     "daily": {str(k.date()): float(v) for k, v in x.items()}}
    a = daily_pnl(test, np.ones(len(test), bool))
    out["always_sell"] = {"days": int(len(a)), "trades": int((~test["post"].to_numpy(bool)).sum()),
                          "total": float(a.sum()), "mean_day": float(a.mean()),
                          "daily": {str(k.date()): float(v) for k, v in a.items()}}
    return {"status": "tested", "window": [str(pd.Timestamp(start).date()), str(pd.Timestamp(days[-1]).date())],
            "results": out}


def paired(chall: dict, champ: dict) -> tuple:
    """(mean daily advantage, its standard error, shared days)."""
    days = sorted(set(chall) & set(champ))
    d = np.array([chall[k] - champ[k] for k in days])
    if len(d) < 2:
        return float("nan"), float("nan"), len(d)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)


def decide(reg: dict, ev: dict) -> dict:
    """Apply the fixed promotion rule. Returns the updated registry."""
    reg = dict(reg)
    res = ev.get("results", {})
    champ = reg.get("champion", "ratio_ridge_full")
    if ev.get("status") != "tested" or champ not in res:
        reg["last_decision"] = ev.get("status")
        return reg
    best, best_t = None, 0.0
    for name, r in res.items():
        if name in (champ, "always_sell"):
            continue
        adv, se, n = paired(r["daily"], res[champ]["daily"])
        if (n >= MIN_LIVE_DAYS and r["trades"] >= MIN_TRADES and r["total"] > 0
                and np.isfinite(se) and se > 0 and adv >= PROMOTE_SE * se and adv / se > best_t):
            best, best_t = name, adv / se
    if best:
        reg.setdefault("history", []).append({"date": str(date.today()), "promoted": best,
                                              "replaced": champ, "t_stat": round(best_t, 2)})
        reg["champion"] = best
    reg["last_decision"] = (f"promoted {best} (paired advantage {best_t:.1f} SE)" if best
                            else f"kept {champ}: no challenger cleared {PROMOTE_SE} SE")
    return reg


def shadow_summary(cfg: Config, since=None) -> pd.DataFrame:
    rows = []
    for p in sorted(cfg.journal_dir.glob("20??-??-??.jsonl")):
        if since and p.stem < since:
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] == "shadow_settle":
                rows.append({"day": p.stem, "strategy": r["strategy"], "mid": r.get("mid"),
                             "half": r.get("half"), "touch": r.get("touch")})
    if not rows:
        return pd.DataFrame()
    s = pd.DataFrame(rows)
    return s.groupby("strategy").agg(days=("day", "nunique"), trades=("mid", "size"),
                                     mid=("mid", "sum"), half=("half", "sum"), touch=("touch", "sum")).round(0)


def weekly(cfg: Config = Config(), today=None) -> dict:
    today = pd.Timestamp(today or date.today()).normalize()
    reg_p = cfg.journal_dir.parent / "registry.json"
    reg = json.loads(reg_p.read_text()) if reg_p.exists() else {"champion": "ratio_ridge_full"}
    hist, live = history(), live_dataset(cfg)
    cands = candidates()
    ev = evaluate(hist, live, cands)
    reg = decide(reg, ev)
    target, kind, cols = cands[reg["champion"]]
    allrows = pd.concat([hist, live], ignore_index=True) if len(live) else hist
    m = fit(allrows, target, cols)
    bundle = {"model": m, "features": cols, "n_rows": int(len(allrows)), "name": reg["champion"],
              "kind": kind, "trained": str(today.date()), "live_days": int(live["day"].nunique()) if len(live) else 0,
              "days": [str(allrows["day"].min().date()), str(allrows["day"].max().date())]}
    cfg.model_path.write_bytes(pickle.dumps(bundle))
    reg_p.parent.mkdir(parents=True, exist_ok=True)
    reg_p.write_text(json.dumps(reg, indent=2))
    sh = shadow_summary(cfg)
    lines = [f"# Weekly loop {today:%Y-%m-%d}", "",
             f"- Live days collected: {bundle['live_days']}; model rows: {bundle['n_rows']:,}",
             f"- Test: {ev['status']}" + (f", window {ev['window'][0]} .. {ev['window'][1]}" if "window" in ev else ""),
             f"- Decision: {reg['last_decision']}", f"- Model now in charge: **{reg['champion']}** ({kind})", ""]
    if ev.get("results"):
        lines += ["## Candidates on unseen live days (half gap, $1k worst case per trade)", "",
                  "| model | days | trades | total | per day |", "|---|---|---|---|---|"]
        lines += [f"| {k} | {v['days']} | {v['trades']} | ${v['total']:,.0f} | ${v['mean_day']:+,.0f} |"
                  for k, v in ev["results"].items()]
        champ_d = ev["results"][reg["champion"]]["daily"]
        lines += ["", f"Paired advantage over the model in charge (promotion needs >= {PROMOTE_SE} SE):", ""]
        for k, v in ev["results"].items():
            if k not in (reg["champion"], "always_sell"):
                adv, se, n = paired(v["daily"], champ_d)
                lines.append(f"- {k}: {adv:+,.0f}/day, SE {se:,.0f} -> {adv / se if se else float('nan'):+.2f} SE over {n} days")
    if len(sh):
        lines += ["", "## Shadow book, all live days ($300 worst case per trade)", "",
                  "| strategy | days | trades | mid | half gap | touch |", "|---|---|---|---|---|---|"]
        lines += [f"| {k} | {int(v['days'])} | {int(v['trades'])} | ${v['mid']:,.0f} | ${v['half']:,.0f} | "
                  f"${v['touch']:,.0f} |" for k, v in sh.iterrows()]
    out = ROOT / "reports" / "live_weekly" / f"{today:%Y-%m-%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return {"registry": reg, "eval": ev, "report": str(out)}


if __name__ == "__main__":
    if sys.argv[1:] == ["weekly"]:
        weekly()
    else:
        print(__doc__)
