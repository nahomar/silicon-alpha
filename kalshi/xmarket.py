"""Edge #12: Kalshi S&P 500 close markets vs SPX 0DTE option fair value.

Pre-registered 2026-11-10 (docs/expiry_edge_plan.md, edge #12).

  python3 -m kalshi.xmarket select      # markets + options fair values (no Kalshi prices, no outcomes)
  python3 -m kalshi.xmarket fetch       # hourly candles for the pre-filtered markets (resumable)
  python3 -m kalshi.xmarket run         # the one run
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi import backtest as kb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "kalshi"
ZERO = ROOT / "data" / "odte_0dte"
OUT = ROOT / "reports" / "kalshi"
TZ = "America/New_York"
SERIES_OLD = ("INX", "INXD", "INXU", "INXDU")
SERIES_NEW = ("KXINX", "KXINXU")
STRIKE_TYPES = ("between", "greater", "greater_or_equal", "less")
FIRST, LAST = pd.Timestamp("2022-05-16"), pd.Timestamp("2026-09-04")
KX_FROM = pd.Timestamp("2024-10-24")
MIN_VOL = 100.0
TIMES = ("11:00", "13:00", "15:00")
P_BAND = (0.05, 0.95)
MARGIN = 0.03
LOOKBACK = 6 * 3600
N_BOOT, SEED = 10_000, 0
CANDIDATES = DATA / "xm_candidates.parquet"
CANDLES = DATA / "xm_candles.jsonl"


# ---------------------------------------------------------------------------
# markets
# ---------------------------------------------------------------------------

def _num_list(s: str) -> list:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", s or "")]


def thresholds(ticker: str, strike_type: str, title: str) -> tuple | None:
    """(kind, a, b): 'between' lo<=S<=hi, 'above' S>T, 'below' S<T; None if it does not parse."""
    try:
        if strike_type == "between":
            nums = _num_list(title)
            return ("between", nums[0], nums[1]) if len(nums) == 2 and nums[0] < nums[1] else None
        m = re.search(r"-T([\d.]+)$", ticker)
        if not m:
            return None
        T = float(m.group(1))
        return ("above", T, np.nan) if strike_type in ("greater", "greater_or_equal") else ("below", T, np.nan)
    except Exception:
        return None


def load_markets(with_result: bool = False) -> pd.DataFrame:
    cols = ["ticker", "event_ticker", "strike_type", "yes_sub_title", "close_time", "volume_fp", "status"]
    if with_result:
        cols.append("result")
    parts = []
    for src, folder in (("hist", "markets_hist"), ("live", "markets_live")):
        for p in sorted(glob.glob(str(DATA / folder / "*.parquet"))):
            d = pd.read_parquet(p, columns=cols)
            d["source"] = src
            parts.append(d)
    m = pd.concat(parts, ignore_index=True).drop_duplicates("ticker", keep="last")
    m["series"] = m["event_ticker"].str.split("-").str[0]
    return m


def select(m: pd.DataFrame) -> pd.DataFrame:
    from odte.eval.expiry_edge import load_early_closes
    early = load_early_closes()
    m = m[m["series"].isin(SERIES_OLD + SERIES_NEW) & m["strike_type"].isin(STRIKE_TYPES)].copy()
    m = m[(m["series"] != "KXINXU") | m["ticker"].str.contains("H1600")]
    m["vol"] = pd.to_numeric(m["volume_fp"], errors="coerce")
    m = m[m["vol"] >= MIN_VOL]
    ct = pd.to_datetime(m["close_time"], utc=True, format="ISO8601").dt.tz_convert(TZ).dt.tz_localize(None)
    m["day"] = ct.dt.normalize()
    m["close_hhmm"] = ct.dt.strftime("%H:%M")
    m = m[m["day"].between(FIRST, LAST)]
    m = m[m["close_hhmm"] == m["day"].map(lambda d: early.get(d, "16:00"))]
    th = [thresholds(t, s, y) for t, s, y in zip(m["ticker"], m["strike_type"], m["yes_sub_title"])]
    m["kind"] = [x[0] if x else None for x in th]
    m["a"] = [x[1] if x else np.nan for x in th]
    m["b"] = [x[2] if x else np.nan for x in th]
    n_bad = int(m["kind"].isna().sum())
    m = m[m["kind"].notna()]
    kx_u = m["series"] == "KXINXU"
    on25 = ((m["a"] + 0.0001) % 25).round(3).isin([0.0, 25.0])
    m = m[~kx_u | on25]
    m.attrs["unparsed"] = n_bad
    return m.reset_index(drop=True)


# ---------------------------------------------------------------------------
# options fair value
# ---------------------------------------------------------------------------

def digital_curve(grid: dict, t: pd.Timestamp, F: float):
    """(midpoints, P(S > midpoint)) from out-of-the-money mids at minute t."""
    from odte.eval.expiry_edge import _two_sided
    q = _two_sided(grid, t)
    (cb, ca), (pb, pa) = q["C"], q["P"]
    cm, pm = (cb + ca) / 2, (pb + pa) / 2
    C = {}
    for k, v in cm.items():
        if k >= F:
            C[float(k)] = float(v)
    for k, v in pm.items():
        if k < F:
            C[float(k)] = float(v) + F - float(k)
    if len(C) < 4:
        return None
    ks = np.array(sorted(C))
    cs = np.array([C[k] for k in ks])
    d = np.clip(-(np.diff(cs) / np.diff(ks)), 0.0, 1.0)
    d = np.minimum.accumulate(d)                                   # non-increasing in strike
    return (ks[:-1] + ks[1:]) / 2, d


def p_above(curve, x: float) -> float:
    mids, d = curve
    if not (mids[0] <= x <= mids[-1]):
        return np.nan
    return float(np.interp(x, mids, d))


def fair(curve, kind: str, a: float, b: float) -> float:
    if curve is None:
        return np.nan
    if kind == "above":
        return p_above(curve, a)
    if kind == "below":
        p = p_above(curve, a)
        return 1.0 - p if np.isfinite(p) else np.nan
    pa, pb = p_above(curve, a), p_above(curve, b)
    return max(pa - pb, 0.0) if np.isfinite(pa) and np.isfinite(pb) else np.nan


def fair_values(sel: pd.DataFrame) -> pd.DataFrame:
    from odte.eval.expiry_edge import implied_forward, minute_grid
    rows = []
    for day, g in sel.groupby("day"):
        p = ZERO / f"{day:%Y-%m-%d}.parquet"
        if not p.exists():
            continue
        grid = minute_grid(pd.read_parquet(p))
        F = implied_forward(grid)
        curves = {}
        for hh in TIMES:
            t = day + pd.Timedelta(hours=int(hh[:2]), minutes=int(hh[3:]))
            Ft = F[F.index <= t].dropna()
            curves[hh] = digital_curve(grid, t, float(Ft.iloc[-1])) if len(Ft) else None
        for i, r in g.iterrows():
            rows.append({"i": i, **{f"p_{hh}": fair(curves[hh], r["kind"], r["a"], r["b"]) for hh in TIMES}})
    fv = pd.DataFrame(rows).set_index("i")
    return sel.join(fv, how="inner")


# ---------------------------------------------------------------------------
# trading rule
# ---------------------------------------------------------------------------

def trade(p_star: float, bid: float, ask: float, maker: bool = False) -> tuple | None:
    """('yes'|'no', price, fee) if the Kalshi quote is cheap by MARGIN after fees, else None.
    Taker (gate): YES at the ask, NO at 1 - bid. Maker (report only, optimistic: assumes a
    fill): YES at the bid, NO at 1 - ask, maker fee."""
    if not (np.isfinite(p_star) and kb.valid_quote(bid, ask)):
        return None
    fee = (lambda x: kb.maker_fee(x)) if maker else kb.taker_fee
    yes_px = bid if maker else ask
    fy = fee(yes_px)
    if yes_px + fy <= p_star - MARGIN:
        return "yes", yes_px, fy
    no_px = round(1.0 - (ask if maker else bid), 6)
    fn = fee(no_px)
    if no_px + fn <= (1.0 - p_star) - MARGIN:
        return "no", no_px, fn
    return None


def day_cluster_ci(x: np.ndarray, days: np.ndarray, n=N_BOOT, seed=SEED) -> list:
    ud, inv = np.unique(days, return_inverse=True)
    sums = np.bincount(inv, weights=x, minlength=len(ud))
    cnts = np.bincount(inv, minlength=len(ud)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ud), size=(n, len(ud)))
    bs = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_select() -> None:
    sel = select(load_markets(with_result=False))
    cand = fair_values(sel)
    pcols = [f"p_{hh}" for hh in TIMES]
    keep = cand[pcols].apply(lambda c: c.between(*P_BAND)).any(axis=1)
    cand = cand[keep].reset_index(drop=True)
    cand.to_parquet(CANDIDATES, index=False)
    print(json.dumps({"selected": int(len(sel)), "unparsed": sel.attrs.get("unparsed"),
                      "with_fair_value_in_band": int(len(cand)), "days": int(cand["day"].nunique()),
                      "by_series": cand["series"].value_counts().to_dict()}, indent=2))


def cmd_fetch() -> None:
    cand = pd.read_parquet(CANDIDATES)
    cand["close"] = pd.to_datetime(cand["close_time"], utc=True, format="ISO8601")

    def window(m):
        day = pd.Timestamp(m.day).tz_localize(TZ)
        return int((day + pd.Timedelta(hours=8)).timestamp()), int((day + pd.Timedelta(hours=16)).timestamp())

    kb.fetch_candles(kb.Client(), cand, {}, f=CANDLES, window=window)
    print("fetch done", flush=True)


def cmd_run() -> dict:
    cand = pd.read_parquet(CANDIDATES)
    res_map = load_markets(with_result=True).set_index("ticker")["result"]
    recs = {t: r.get("candlesticks") or [] for t, r in kb.load_candles(CANDLES).items()}
    settle = pd.read_csv(ZERO / "settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
    trades, cal, maker = [], [], []
    for _, m in cand.iterrows():
        cs = recs.get(m["ticker"])
        if not cs:
            continue
        out = res_map.get(m["ticker"])
        if out not in ("yes", "no"):
            continue
        S = settle.get(m["day"], np.nan)
        cond = (m["a"] <= S <= m["b"]) if m["kind"] == "between" else (S > m["a"] if m["kind"] == "above" else S < m["a"])
        done = done_m = False
        for hh in TIMES:
            t = int((pd.Timestamp(m["day"]) + pd.Timedelta(hours=int(hh[:2]), minutes=int(hh[3:])))
                    .tz_localize(TZ).timestamp())
            q = kb.decision_quote(cs, t, lookback=LOOKBACK)
            p = m[f"p_{hh}"]
            if q is None or not kb.valid_quote(q["bid"], q["ask"]) or not np.isfinite(p):
                continue
            cal.append({"day": m["day"], "hhmm": hh, "kalshi_mid": kb.mid(q["bid"], q["ask"]), "p_star": p,
                        "y": 1.0 if out == "yes" else 0.0})
            if not done_m:
                mk = trade(p, q["bid"], q["ask"], maker=True)
                if mk is not None:
                    pm = 1.0 if mk[0] == out else 0.0
                    maker.append({"day": m["day"], "ret": (pm - mk[1] - mk[2]) / (mk[1] + mk[2])})
                    done_m = True
            if done:
                continue
            tr = trade(p, q["bid"], q["ask"])
            if tr is None:
                continue
            side, price, fee = tr
            payoff = 1.0 if side == out else 0.0
            trades.append({"ticker": m["ticker"], "series": m["series"], "day": m["day"], "hhmm": hh,
                           "side": side, "price": price, "fee": fee, "p_star": p, "payoff": payoff,
                           "ret": (payoff - price - fee) / (price + fee), "vol24": q["vol24"],
                           "mismatch": bool(np.isfinite(S) and (cond != (out == "yes")))})
            done = True
    tr = pd.DataFrame(trades)
    cl = pd.DataFrame(cal)
    res = {"candidates": int(len(cand)), "trades": int(len(tr)), "days": int(tr["day"].nunique()) if len(tr) else 0}
    if len(tr):
        x, dd = tr["ret"].to_numpy(), tr["day"].to_numpy()
        res["mean_ret"] = float(x.mean())
        res["ci"] = day_cluster_ci(x, dd)
        era = tr["day"] < KX_FROM
        res["era_old_mean"] = float(tr.loc[era, "ret"].mean()) if era.any() else float("nan")
        res["era_kx_mean"] = float(tr.loc[~era, "ret"].mean()) if (~era).any() else float("nan")
        res["n_old"], res["n_kx"] = int(era.sum()), int((~era).sum())
        res["by_side"] = {k: {"n": int(len(g)), "mean": float(g["ret"].mean()), "win": float(g["payoff"].mean())}
                          for k, g in tr.groupby("side")}
        tr["bucket"] = pd.cut(tr["p_star"], [0, .1, .25, .5, .75, .9, 1.0])
        res["by_p_star"] = {str(k): {"n": int(len(g)), "mean": float(g["ret"].mean())}
                            for k, g in tr.groupby("bucket", observed=True)}
        res["by_time"] = {k: {"n": int(len(g)), "mean": float(g["ret"].mean())} for k, g in tr.groupby("hhmm")}
        res["mismatches"] = int(tr["mismatch"].sum())
        res["median_vol24"] = float(tr["vol24"].median())
        res["passes"] = bool(res["ci"][0] > 0 and res["era_old_mean"] > 0 and res["era_kx_mean"] > 0)
    if maker:
        mk = pd.DataFrame(maker)
        res["maker_optimistic"] = {"trades": int(len(mk)), "mean_ret": float(mk["ret"].mean()),
                                   "ci": day_cluster_ci(mk["ret"].to_numpy(), mk["day"].to_numpy())}
    if len(cl):
        res["brier_kalshi_mid"] = float(((cl["kalshi_mid"] - cl["y"]) ** 2).mean())
        res["brier_options"] = float(((cl["p_star"] - cl["y"]) ** 2).mean())
        res["calibration_quotes"] = int(len(cl))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "xmarket.json").write_text(json.dumps(res, indent=2, default=str))
    tr.to_parquet(OUT / "xmarket_trades.parquet", index=False) if len(tr) else None
    print(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    {"select": cmd_select, "fetch": cmd_fetch, "run": cmd_run}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
