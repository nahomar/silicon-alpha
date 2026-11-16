"""Cheap gate before retraining TradeFM: does tick data predict moves that pay the spread?

The old gate ("LightGBM beats 53% on next-tick direction") would pass for
reasons that make no money: book imbalance and bid-ask bounce predict the
next mid change well, but a trader has to cross the spread to use it.
This check asks the money question instead.

Data: cmbp-1 (consolidated best bid/offer, every update) for SPXW 0DTE
contracts within +/-3% of spot, on the days in configs/tradefm/tick_days.csv.
Days on or after the locked exam (2026-09-08) are NOT used, per the lockbox
protocol. The remaining 16 days are split by time: the earliest 12 train,
the latest 4 test.

Events: every change of a contract's best bid or ask price (size-only
updates dropped), stamped at ts_recv, i.e. when we would know it.
Features at event t (contract c), using only rows at or before t:
  imb          (bid size - ask size) / (bid size + ask size)
  spread_rel   (ask - bid) / mid
  log_mid      log mid
  r1..r5       last five log mid changes of c
  log_dt       log seconds since c's previous event
  n60          c's events in the last 60 s
  mins_left    minutes to 16:00
Target over H = 30 s, from c's book at the last event at or before t + 30 s:
  long  taker P&L = bid(t+H) - ask(t) - fees
  short taker P&L = bid(t) - ask(t+H) - fees,   fees = 2 x $1.20 / 100
  direction = sign(mid(t+H) - mid(t)), zero moves dropped for training

Models (trained on up/down): imbalance-only logistic regression, and
LightGBM on all features. On the test days each trades in its predicted
direction on its most confident 1% and 10% of events.

Pre-registered gate (written before the run):
  GO (retrain TradeFM) only if LightGBM's top-1% mean taker P&L is > 0
  with a 95% block-bootstrap CI (30-minute blocks) above 0, AND it beats
  the imbalance-only model's top 1%. Otherwise NO-GO: tick direction may be
  predictable, but not by enough to pay to trade it.

Usage (on Sol, CPU job):
  python3 -m odte.eval.tick_check --tick-dir /scratch/$USER/odte_ticks
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXAM_FROM = pd.Timestamp("2026-09-08")
H_NS = 30 * 10**9
FEES = 2 * 1.20 / 100.0
N_LAGS = 5
FEATURES = ["imb", "spread_rel", "log_mid", *[f"r{k}" for k in range(1, N_LAGS + 1)],
            "log_dt", "n60", "mins_left"]
MAX_EVENTS_PER_DAY = 1_500_000
TZ = "America/New_York"


# ----------------------------------------------------------------------------
# 1. Reduce a stream of book updates to price-change events
# ----------------------------------------------------------------------------
class Reducer:
    """Keeps rows where a contract's best bid or ask price changed.

    Carries each contract's last (bid, ask) across chunks so a chunk boundary
    never fakes a change."""

    def __init__(self):
        self.last = {}
        self.parts = []

    def add(self, ts, iid, bid, ask, bsz, asz):
        ok = (bid > 0) & (ask > bid) & (ask < 1e5)
        ts, iid, bid, ask, bsz, asz = (x[ok] for x in (ts, iid, bid, ask, bsz, asz))
        if not len(ts):
            return
        o = np.argsort(iid, kind="stable")                 # time order kept within contract
        ts, iid, bid, ask, bsz, asz = (x[o] for x in (ts, iid, bid, ask, bsz, asz))
        first = np.r_[True, iid[1:] != iid[:-1]]
        pb, pa = np.r_[np.nan, bid[:-1]], np.r_[np.nan, ask[:-1]]
        for j in np.flatnonzero(first):
            pb[j], pa[j] = self.last.get(int(iid[j]), (np.nan, np.nan))
        keep = (bid != pb) | (ask != pa)
        last_row = np.r_[iid[1:] != iid[:-1], True]
        for j in np.flatnonzero(last_row):
            self.last[int(iid[j])] = (float(bid[j]), float(ask[j]))
        self.parts.append(np.rec.fromarrays(
            [ts[keep], iid[keep], bid[keep], ask[keep], bsz[keep], asz[keep]],
            names="ts,iid,bid,ask,bsz,asz"))

    def frame(self) -> pd.DataFrame:
        if not self.parts:
            return pd.DataFrame(columns=["ts", "iid", "bid", "ask", "bsz", "asz"])
        return pd.DataFrame(np.concatenate(self.parts))


def read_day(path: Path, chunk=20_000_000) -> pd.DataFrame:
    import databento as db
    store = db.DBNStore.from_file(str(path))
    red = Reducer()
    for a in store.to_ndarray(count=chunk):
        px = lambda f: np.where(a[f] == np.iinfo(np.int64).max, -1.0, a[f] * 1e-9)
        red.add(a["ts_recv"].astype(np.int64), a["instrument_id"].astype(np.int64),
                px("bid_px_00"), px("ask_px_00"),
                a["bid_sz_00"].astype(float), a["ask_sz_00"].astype(float))
    return red.frame()


def _shift(x, s):
    out = np.full(len(x), np.nan)
    if s == 0:
        return x.astype(float)
    if s < len(x):
        out[s:] = x[:-s]
    return out


# ----------------------------------------------------------------------------
# 2. Features and targets, per day
# ----------------------------------------------------------------------------
def day_events(ev: pd.DataFrame, day) -> pd.DataFrame:
    day = pd.Timestamp(day)
    t0 = int(day.tz_localize(TZ).tz_convert("UTC").value)
    open_ns = int((day + pd.Timedelta(hours=9, minutes=31)).tz_localize(TZ).tz_convert("UTC").value)
    close_ns = int((day + pd.Timedelta(hours=16)).tz_localize(TZ).tz_convert("UTC").value)
    ev = ev[(ev["ts"] >= t0) & (ev["ts"] < close_ns)].sort_values(["iid", "ts"], kind="stable")
    if ev.empty:
        return pd.DataFrame()
    ts = ev["ts"].to_numpy(np.int64)
    iid = ev["iid"].to_numpy()
    bid, ask = ev["bid"].to_numpy(float), ev["ask"].to_numpy(float)
    bsz, asz = ev["bsz"].to_numpy(float), ev["asz"].to_numpy(float)
    g = np.r_[0, np.cumsum(iid[1:] != iid[:-1])]           # contract index 0..G-1
    key = g.astype(np.int64) * 10**14 + (ts - t0)           # sorted; groups never overlap
    mid = 0.5 * (bid + ask)
    lm = np.log(mid)
    same = np.r_[False, g[1:] == g[:-1]]
    r = np.where(same, lm - np.r_[np.nan, lm[:-1]], np.nan)
    out = {"imb": (bsz - asz) / np.maximum(bsz + asz, 1.0),
           "spread_rel": (ask - bid) / mid, "log_mid": lm}
    gf = g.astype(float)
    for k in range(1, N_LAGS + 1):                          # r1 = the change into t
        out[f"r{k}"] = np.where(_shift(gf, k - 1) == gf, _shift(r, k - 1), np.nan)
    dt = np.where(same, (ts - np.r_[0, ts[:-1]]) / 1e9, np.nan)
    out["log_dt"] = np.log1p(dt)
    out["n60"] = np.arange(len(key)) - np.searchsorted(key, key - 60 * 10**9, side="left")
    out["mins_left"] = (close_ns - ts) / 6e10
    j = np.searchsorted(key, key + H_NS, side="right") - 1   # last state at or before t+H, same contract
    out["long_pnl"] = bid[j] - ask - FEES
    out["short_pnl"] = bid - ask[j] - FEES
    out["move"] = mid[j] - mid
    out["block"] = (ts - t0) // (30 * 60 * 10**9)
    df = pd.DataFrame(out)
    keep = (ts >= open_ns) & (ts + H_NS <= close_ns)
    df = df[keep].reset_index(drop=True)
    df["day"] = day.strftime("%Y-%m-%d")
    return df


# ----------------------------------------------------------------------------
# 3. Fit, trade, judge
# ----------------------------------------------------------------------------
def _trade(p_up, d, q):
    conf = np.abs(p_up - 0.5)
    sel = conf >= np.quantile(conf, 1 - q)
    pnl = np.where(p_up[sel] > 0.5, d["long_pnl"].to_numpy()[sel], d["short_pnl"].to_numpy()[sel])
    return pnl, d.loc[sel, ["day", "block"]].astype(str).agg("|".join, axis=1).to_numpy()


def _block_ci(pnl, blocks, n=2000, seed=0):
    if len(pnl) == 0:
        return [np.nan, np.nan]
    s = pd.Series(pnl).groupby(blocks).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    m = s["sum"].to_numpy()[idx].sum(1) / s["count"].to_numpy()[idx].sum(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    tr = train[train["move"] != 0]
    y = (tr["move"] > 0).astype(int).to_numpy()
    base = LogisticRegression(max_iter=200).fit(tr[["imb"]].fillna(0).to_numpy(), y)
    try:
        import lightgbm as lgb
        gbm = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 verbose=-1).fit(tr[FEATURES], y)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05).fit(tr[FEATURES], y)
    nz = test["move"].to_numpy() != 0
    yt = (test["move"].to_numpy() > 0).astype(int)
    res = {"n_train": int(len(tr)), "n_test": int(len(test)),
           "test_up_share": float(yt[nz].mean()),
           "test_share_any_taker_profit": float(
               (np.maximum(test["long_pnl"], test["short_pnl"]) > 0).mean())}
    for name, p in (("imbalance_only", base.predict_proba(test[["imb"]].fillna(0).to_numpy())[:, 1]),
                    ("lightgbm", gbm.predict_proba(test[FEATURES])[:, 1])):
        r = {"direction_accuracy": float(((p[nz] > 0.5) == yt[nz]).mean())}
        for q in (0.10, 0.01):
            pnl, blocks = _trade(p, test, q)
            r[f"top{int(q * 100)}pct"] = {"n": int(len(pnl)), "mean_pnl_usd": float(pnl.mean() * 100),
                                          "ci_usd": [x * 100 for x in _block_ci(pnl, blocks)],
                                          "win_rate": float((pnl > 0).mean())}
        res[name] = r
    g, b = res["lightgbm"]["top1pct"], res["imbalance_only"]["top1pct"]
    res["verdict"] = ("GO" if g["mean_pnl_usd"] > 0 and g["ci_usd"][0] > 0
                      and g["mean_pnl_usd"] > b["mean_pnl_usd"] else "NO-GO")
    res["note"] = ("imbalance alone pays the spread: a simple rule, not TradeFM"
                   if b["ci_usd"][0] > 0 else "imbalance alone does not pay the spread")
    return res


def _one_day(args):
    path, d, seed = args
    ev = day_events(read_day(path), d)
    if len(ev) > MAX_EVENTS_PER_DAY:
        rng = np.random.default_rng(seed)
        ev = ev.iloc[np.sort(rng.choice(len(ev), MAX_EVENTS_PER_DAY, replace=False))]
    return d, ev


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tick-dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "reports" / "tick_check.json"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    days = pd.read_csv(ROOT / "configs" / "tradefm" / "tick_days.csv", parse_dates=["date"])
    days = days[days["date"] < EXAM_FROM].sort_values("date")["date"].tolist()
    train_days, test_days = days[:-4], days[-4:]
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(Path(a.tick_dir) / f"{d:%Y-%m-%d}.cmbp-1.dbn.zst", d, i) for i, d in enumerate(days)]
    frames, t = {}, time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for d, ev in ex.map(_one_day, jobs):
            frames[d] = ev
            print(f"{d:%Y-%m-%d}: {len(ev):,} events ({time.time() - t:.0f}s elapsed)", flush=True)
    res = evaluate(pd.concat([frames[d] for d in train_days]),
                   pd.concat([frames[d] for d in test_days]))
    res["train_days"] = [f"{d:%Y-%m-%d}" for d in train_days]
    res["test_days"] = [f"{d:%Y-%m-%d}" for d in test_days]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
