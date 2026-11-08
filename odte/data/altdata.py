"""Free alternative data for SPX 0DTE: prediction markets, presidential
posts, global news volume, Bitcoin, buyback blackouts.

Every source is free and timestamped finely enough for intraday decisions.
Each is saved under data/alt/ and turned into point-in-time features by
`features_for` (only data stamped before the decision time t).

  kalshi_inx   Kalshi KXINX: ~30 price-range contracts on where the S&P 500
               closes today, hourly bid/ask history. Their implied
               distribution gives a second, crowd-priced forecast of today's
               move to compare with the options straddle.
  kalshi_fed   Kalshi KXFED: target-rate contracts per FOMC meeting, daily
               history. Their spread of outcomes = how uncertain the next
               decision is.
  truth        Presidential Truth Social posts (CNN's public archive, full
               history, exact timestamps). In 2025 they caused several of
               the largest intraday SPX moves (tariffs).
  gdelt        GDELT global news volume, 15-minute resolution, for fixed
               queries: market, tariffs, the Fed, geopolitics. Uses the
               share of all coverage, so it is not swamped by total volume.
  btc          Coinbase BTC-USD 1-minute bars. It trades 24/7, so it shows
               overnight and weekend risk appetite before the open.
  (buyback)    Derived, no download: share of the eight mega-caps inside
               their buyback blackout (30 days before earnings to 2 after).

Usage:
  python3 -m odte.data.altdata fetch truth|btc|kalshi|gdelt [--start 2025-10-25]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ALT = ROOT / "data" / "alt"
UA = {"User-Agent": "Mozilla/5.0 (silicon-alpha research)"}
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
TRUTH_URL = "https://ix.cnn.io/data/truth-social/truth_archive.json"
COINBASE = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERIES = {          # fixed before any result was seen
    "market": '"stock market" OR "wall street" OR "S&P 500"',
    "tariff": "tariff OR tariffs",
    "fed": '"Federal Reserve" OR "interest rates" OR Powell',
    "geo": "sanctions OR invasion OR missile OR airstrike",
}
TZ = "America/New_York"


def _get_json(url: str, attempts: int = 6, pause: float = 0.0):
    for i in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                out = json.loads(r.read())
            if pause:
                time.sleep(pause)
            return out
        except Exception as e:
            code = getattr(e, "code", None)
            wait = 60 if code == 429 else 5 * (i + 1)
            if i == attempts - 1:
                raise
            print(f"  {str(e)[:70]}; retry in {wait}s", flush=True)
            time.sleep(wait)


def _utc(ts) -> int:
    return int(pd.Timestamp(ts).tz_convert("UTC").timestamp()) if pd.Timestamp(ts).tzinfo \
        else int(pd.Timestamp(ts).tz_localize("UTC").timestamp())


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_truth() -> pd.DataFrame:
    with urllib.request.urlopen(urllib.request.Request(TRUTH_URL, headers=UA), timeout=180) as r:
        data = json.loads(r.read())
    rows = []
    for x in data:
        text = re.sub(r"<[^>]+>", " ", x.get("content") or "")
        text = re.sub(r"\s+", " ", text).strip()
        rows.append((str(x["id"]), _utc(x["created_at"]), text))
    df = pd.DataFrame(rows, columns=["id", "created_utc", "text"]).drop_duplicates("id")
    ALT.mkdir(parents=True, exist_ok=True)
    df.sort_values("created_utc").to_parquet(ALT / "truth.parquet", index=False)
    print(f"truth: {len(df):,} posts, "
          f"{pd.Timestamp(df.created_utc.min(), unit='s'):%Y-%m-%d} .. "
          f"{pd.Timestamp(df.created_utc.max(), unit='s'):%Y-%m-%d}")
    return df


def fetch_btc(start, end=None) -> pd.DataFrame:
    p = ALT / "btc_1m.parquet"
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["ts", "close", "volume"])
    lo = pd.Timestamp(old["ts"].max(), unit="s") if len(old) else pd.Timestamp(start)
    hi = pd.Timestamp(end) if end else pd.Timestamp.utcnow().tz_localize(None).floor("min")
    rows, t = [], lo
    while t < hi:
        t2 = min(t + pd.Timedelta(minutes=300), hi)
        q = urllib.parse.urlencode({"granularity": 60, "start": t.isoformat() + "Z", "end": t2.isoformat() + "Z"})
        data = _get_json(f"{COINBASE}?{q}", pause=0.3)
        rows += [(int(c[0]), float(c[4]), float(c[5])) for c in data]
        t = t2
        if len(rows) and len(rows) % 30000 < 300:
            print(f"  btc through {t:%Y-%m-%d}", flush=True)
            pd.concat([old, pd.DataFrame(rows, columns=["ts", "close", "volume"])]) \
              .drop_duplicates("ts").sort_values("ts").to_parquet(p, index=False)
    df = pd.concat([old, pd.DataFrame(rows, columns=["ts", "close", "volume"])]) \
        .drop_duplicates("ts").sort_values("ts")
    ALT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    print(f"btc: {len(df):,} minutes")
    return df


def _markets(**params) -> list:
    out, cursor = [], None
    while True:
        q = dict(params, limit=1000, **({"cursor": cursor} if cursor else {}))
        d = _get_json(f"{KALSHI}/markets?{urllib.parse.urlencode(q)}", pause=0.2)
        out += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor or not d.get("markets"):
            return out


def _px(side: dict):
    """Candle close price in dollars; live and historical candles name it differently."""
    side = side or {}
    v = side.get("close_dollars", side.get("close"))
    return float(v) if v not in (None, "") else np.nan


def _candles(series: str, ticker: str, lo: int, hi: int, period: int,
             historical: bool = False) -> list:
    q = urllib.parse.urlencode({"start_ts": lo, "end_ts": hi, "period_interval": period})
    path = (f"/historical/markets/{ticker}/candlesticks" if historical
            else f"/series/{series}/markets/{ticker}/candlesticks")
    d = _get_json(f"{KALSHI}{path}?{q}", pause=0.15)
    return [(int(c["end_period_ts"]), _px(c.get("yes_bid")), _px(c.get("yes_ask")))
            for c in d.get("candlesticks", [])]


def _events(series: str, statuses=("settled",)) -> list:
    out = []
    for status in statuses:
        cursor = None
        while True:
            q = dict(series_ticker=series, limit=200, status=status, **({"cursor": cursor} if cursor else {}))
            d = _get_json(f"{KALSHI}/events?{urllib.parse.urlencode(q)}", pause=0.2)
            out += [e["event_ticker"] for e in d.get("events", [])]
            cursor = d.get("cursor")
            if not cursor or not d.get("events"):
                break
    return sorted(set(out))


def _event_markets(ev: str) -> tuple:
    """(markets, historical?) - recent markets are live, older ones archived."""
    m = _get_json(f"{KALSHI}/markets?{urllib.parse.urlencode({'event_ticker': ev, 'limit': 200})}",
                  pause=0.15).get("markets", [])
    if m:
        return m, False
    h = _get_json(f"{KALSHI}/historical/markets?{urllib.parse.urlencode({'event_ticker': ev, 'limit': 200})}",
                  pause=0.15).get("markets", [])
    return h, True


def fetch_kalshi(start) -> None:
    """S&P daily range markets (hourly) and Fed meeting markets (daily).

    Recent markets come from the live API, older ones from /historical (same
    contracts, archived). Events are enumerated from /events, which pages
    back to 2022.
    """
    ALT.mkdir(parents=True, exist_ok=True)
    inx_dir = ALT / "kalshi_inx"
    inx_dir.mkdir(exist_ok=True)
    start_utc = pd.Timestamp(start, tz="UTC")
    evs = [e for e in _events("KXINX") if e.startswith("KXINX-")]
    n_done = 0
    for ev in sorted(evs):
        markets, hist = _event_markets(ev)
        if not markets:
            continue
        close = pd.Timestamp(markets[0]["close_time"])
        if close < start_utc:
            continue
        day = close.tz_convert(TZ).strftime("%Y-%m-%d")
        f = inx_dir / f"{day}.parquet"
        if f.exists():
            continue
        rows = []
        for m in markets:
            lo, hi = _utc(m["open_time"]), _utc(m["close_time"])
            for ts, b, a in _candles("KXINX", m["ticker"], lo, hi, 60, historical=hist):
                rows.append((m["ticker"], m["strike_type"], m.get("floor_strike"),
                             m.get("cap_strike"), ts, b, a))
        pd.DataFrame(rows, columns=["ticker", "strike_type", "floor", "cap", "ts", "bid", "ask"]) \
          .to_parquet(f, index=False)
        n_done += 1
        if n_done % 20 == 0:
            print(f"  kalshi INX: {n_done} sessions, at {day}", flush=True)
    print(f"kalshi INX: done ({len(list(inx_dir.glob('*.parquet')))} sessions on disk)", flush=True)

    # Open meetings too: on early-November days the *next* meeting (Dec) has
    # not settled yet, and it is exactly the one the feature needs.
    fed_rows = []
    for ev in sorted(e for e in _events("KXFED", ("settled", "open", "closed")) if e.startswith("KXFED-")):
        markets, hist = _event_markets(ev)
        if not markets or pd.Timestamp(markets[0]["close_time"]) < start_utc:
            continue
        for m in markets:
            lo = max(_utc(m["open_time"]), _utc(start_utc) - 86400 * 40)
            hi = min(_utc(m["close_time"]), int(time.time()))
            if hi <= lo:
                continue
            for ts, b, a in _candles("KXFED", m["ticker"], lo, hi, 1440, historical=hist):
                fed_rows.append((ev, m["ticker"], m.get("floor_strike"), m["close_time"], ts, b, a))
    pd.DataFrame(fed_rows, columns=["event", "ticker", "threshold", "meeting_close", "ts", "bid", "ask"]) \
      .to_parquet(ALT / "kalshi_fed.parquet", index=False)
    print(f"kalshi FED: {len(fed_rows)} daily candles", flush=True)


GDELT_PACE_S = float(__import__("os").environ.get("GDELT_PACE_S", "60"))
# Stated limit 1 / 5 s. From one address that had been rate-limited, 10 s and
# 30 s still got 429s for hours; a fresh address (e.g. ASU Sol) can use ~6 s.


def _gdelt_get(url: str, attempts: int = 40):
    """GDELT asks for <= 1 request / 5 s and answers 429 or an empty body when
    pushed; both mean wait. Pace at GDELT_PACE_S, back off up to 15 minutes."""
    for i in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                body = r.read()
            out = json.loads(body) if body.strip() else None
            if out is None:
                raise ValueError("empty body")
            time.sleep(GDELT_PACE_S)
            return out
        except Exception as e:
            wait = min(120 * (i + 1), 900)
            print(f"  gdelt busy ({str(e)[:50]}); wait {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("gdelt: too many failures")


def fetch_gdelt(start, end=None) -> pd.DataFrame:
    """15-minute news volume per query, in 3-day windows, saved every window."""
    p = ALT / "gdelt.parquet"
    ALT.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["query", "ts", "value", "norm"])
    hi_all = pd.Timestamp(end) if end else pd.Timestamp.utcnow().tz_localize(None).floor("D")
    for name, q in GDELT_QUERIES.items():
        done = old[old["query"] == name]
        t = (pd.Timestamp(done["ts"].max(), unit="s").floor("D") if len(done) else pd.Timestamp(start))
        while t < hi_all:
            t2 = min(t + pd.Timedelta(days=3), hi_all)
            params = {"query": q, "mode": "timelinevolraw", "format": "json",
                      "startdatetime": t.strftime("%Y%m%d%H%M%S"),
                      "enddatetime": t2.strftime("%Y%m%d%H%M%S")}
            d = _gdelt_get(f"{GDELT}?{urllib.parse.urlencode(params)}")
            rows = [(name, _utc(pd.Timestamp(pt["date"])), int(pt["value"]), int(pt["norm"]))
                    for pt in (d.get("timeline") or [{}])[0].get("data", [])]
            old = pd.concat([old, pd.DataFrame(rows, columns=["query", "ts", "value", "norm"])]) \
                .drop_duplicates(["query", "ts"]).sort_values(["query", "ts"])
            old.to_parquet(p, index=False)
            t = t2
        print(f"gdelt {name}: through {hi_all:%Y-%m-%d}", flush=True)
    return old


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch")
    f.add_argument("source", choices=["truth", "btc", "kalshi", "gdelt"])
    f.add_argument("--start", default="2025-10-25")
    a = ap.parse_args()
    {"truth": lambda: fetch_truth(), "btc": lambda: fetch_btc(a.start),
     "kalshi": lambda: fetch_kalshi(a.start), "gdelt": lambda: fetch_gdelt(a.start)}[a.source]()
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Point-in-time features
# ---------------------------------------------------------------------------

MARKET_WORDS = re.compile(r"\b(tariffs?|china|fed|powell|rates?|market|stocks?|deal|trade|"
                          r"economy|inflation|dow|s&p|nasdaq|jobs)\b", re.I)
ALT_LAYERS = {
    "+kalshi": ["kx_move_ratio", "kx_down_prob", "fed_entropy_next", "fed_entropy_chg5"],
    "+news": ["tr_n_night", "tr_n_2h", "tr_mkt_today", "tr_fear_today",
              "gd_market_1h", "gd_tariff_1h", "gd_fed_1h", "gd_geo_1h", "gd_all_night"],
    "+btc": ["btc_ret_night", "btc_rv_night", "btc_ret_1h", "btc_rv_1h"],
    "+flows": ["bb_blackout_share"],
}
_ZSCORE = {"tr_n_night", "tr_n_2h", "tr_mkt_today", "gd_market_1h", "gd_tariff_1h",
           "gd_fed_1h", "gd_geo_1h", "gd_all_night"}
MEGACAPS = ("NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AVGO", "TSLA")


def _et_utc(day, hhmm) -> int:
    t = pd.Timestamp(f"{pd.Timestamp(day):%Y-%m-%d} {hhmm}").tz_localize(TZ)
    return int(t.tz_convert("UTC").timestamp())


def _count(ts: np.ndarray, a: int, b: int) -> float:
    i, j = np.searchsorted(ts, [a, b], side="left")
    return float(j - i)


def _kalshi_inx(day, t: int, F: float, straddle: float) -> tuple:
    p = ALT / "kalshi_inx" / f"{pd.Timestamp(day):%Y-%m-%d}.parquet"
    if not p.exists() or not (straddle > 0):
        return np.nan, np.nan
    k = pd.read_parquet(p)
    k = k[k["ts"] <= t].sort_values("ts").groupby("ticker").tail(1)
    if k.empty:
        return np.nan, np.nan
    mid = (k["bid"].fillna(0) + k["ask"]) / 2
    x = np.where(k["strike_type"] == "between", (k["floor"] + k["cap"]) / 2,
                 np.where(k["strike_type"] == "less", k["cap"] - 12.5, k["floor"] + 12.5))
    ok = np.isfinite(mid) & np.isfinite(x)
    if ok.sum() < 5 or mid[ok].sum() <= 0:
        return np.nan, np.nan
    pr = mid[ok].to_numpy() / mid[ok].sum()
    xv = x[ok]
    return float((pr * np.abs(xv - F)).sum() / straddle), float(pr[xv < F].sum())


def _fed_entropy(fed: pd.DataFrame, day) -> float:
    if fed is None or fed.empty:
        return np.nan
    cut = _et_utc(day, "09:30")
    meet = pd.to_datetime(fed["meeting_close"], utc=True)
    nxt = fed[meet >= pd.Timestamp(day, tz=TZ).tz_convert("UTC")]
    if nxt.empty:
        return np.nan
    ev = nxt.loc[pd.to_datetime(nxt["meeting_close"], utc=True).idxmin(), "event"]
    g = nxt[(nxt["event"] == ev) & (nxt["ts"] < cut)].sort_values("ts").groupby("ticker").tail(1)
    if len(g) < 3:
        return np.nan
    g = g.assign(mid=(g["bid"].fillna(0) + g["ask"]) / 2).dropna(subset=["threshold", "mid"])
    surv = g.sort_values("threshold")["mid"].clip(0, 1).to_numpy()       # P(rate > threshold)
    probs = np.clip(np.concatenate([[1 - surv[0]], -np.diff(surv), [surv[-1]]]), 0, None)
    if probs.sum() <= 0:
        return np.nan
    probs = probs / probs.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def alt_features(ds: pd.DataFrame) -> pd.DataFrame:
    """ALT_LAYERS columns for each decision row (NaN where a source is missing)."""
    days = sorted(ds["day"].unique())
    prev = {d: (days[i - 1] if i else None) for i, d in enumerate(days)}
    def load(name):
        p = ALT / name
        return pd.read_parquet(p) if p.exists() else None
    truth, gd, btc, fed = load("truth.parquet"), load("gdelt.parquet"), load("btc_1m.parquet"), load("kalshi_fed.parquet")
    tr_ts = truth["created_utc"].to_numpy() if truth is not None else None
    tr_mkt = truth["text"].fillna("").str.contains(MARKET_WORDS).to_numpy() if truth is not None else None
    tr_cum = np.concatenate([[0], np.cumsum(tr_mkt)]) if truth is not None else None
    sc_p = ROOT / "data" / "social" / "nlp" / "truth.parquet"
    tr_sc = pd.read_parquet(sc_p).sort_values("created_utc") if sc_p.exists() else None
    if btc is not None:
        b_ts = btc["ts"].to_numpy() + 60                  # a 1-min bar is known at its end
        b_lr = np.log(btc["close"].to_numpy())
    gq = {}
    if gd is not None:
        for q, g in gd.groupby("query"):
            g = g.sort_values("ts")
            gq[q] = (g["ts"].to_numpy() + 900, np.concatenate([[0], np.cumsum(g["value"] / g["norm"].clip(lower=1))]))
    try:
        from odte.data import calendar as cal
        ev = cal.load()
        earn = ev[ev["event"] == "megacap_earnings"][["session", "detail"]]
    except Exception:
        earn = None
    fed_h = {d: _fed_entropy(fed, d) for d in days}

    rows = []
    for _, r in ds.iterrows():
        d, hh = r["day"], r["hhmm"]
        t = _et_utc(d, hh)
        o = _et_utc(d, "09:30")
        pc = _et_utc(prev[d], "16:00") if prev[d] is not None else None
        f = {c: np.nan for cols in ALT_LAYERS.values() for c in cols}
        f["kx_move_ratio"], f["kx_down_prob"] = _kalshi_inx(d, t, r["F"], r["straddle_mid"])
        f["fed_entropy_next"] = fed_h[d]
        i = days.index(d)
        if i >= 5 and np.isfinite(fed_h[d]) and np.isfinite(fed_h[days[i - 5]]):
            f["fed_entropy_chg5"] = fed_h[d] - fed_h[days[i - 5]]
        if tr_ts is not None and pc is not None and tr_ts[0] <= pc:
            f["tr_n_night"] = _count(tr_ts, pc, o) / max((o - pc) / 3600, 1)
            f["tr_n_2h"] = _count(tr_ts, t - 7200, t)
            a, b = np.searchsorted(tr_ts, [pc, t], side="left")
            f["tr_mkt_today"] = float(tr_cum[b] - tr_cum[a])
            if tr_sc is not None:
                w = tr_sc[(tr_sc["created_utc"] >= pc) & (tr_sc["created_utc"] < t)]
                if len(w) >= 3:
                    f["tr_fear_today"] = float((w["fear"] - w["greed"]).mean())
        for q, (ts_end, cs) in gq.items():
            if ts_end[0] <= t - 3600:
                a, b = np.searchsorted(ts_end, [t - 3600 + 1, t], side="right")
                f[f"gd_{q}_1h"] = float(cs[b] - cs[a])
        if gq and pc is not None:
            tot = 0.0
            for ts_end, cs in gq.values():
                a, b = np.searchsorted(ts_end, [pc + 1, o], side="right")
                tot += cs[b] - cs[a]
            f["gd_all_night"] = tot
        if btc is not None and pc is not None and b_ts[0] <= pc:
            def last(at):
                j = np.searchsorted(b_ts, at, side="right") - 1
                return b_lr[j] if j >= 0 else np.nan
            f["btc_ret_night"] = (last(o) - last(pc)) * 1e4
            a, b = np.searchsorted(b_ts, [pc, o], side="right")
            f["btc_rv_night"] = float(np.diff(b_lr[a:b]).std() * 1e4) if b - a > 30 else np.nan
            f["btc_ret_1h"] = (last(t) - last(t - 3600)) * 1e4
            a, b = np.searchsorted(b_ts, [t - 3600, t], side="right")
            f["btc_rv_1h"] = float(np.diff(b_lr[a:b]).std() * 1e4) if b - a > 30 else np.nan
        if earn is not None and len(earn):
            dd = pd.Timestamp(d)
            inside = 0
            for tk in MEGACAPS:
                e = pd.to_datetime(earn.loc[earn["detail"] == tk, "session"])
                if ((e - dd).dt.days.between(-2, 30)).any():
                    inside += 1
            f["bb_blackout_share"] = inside / len(MEGACAPS)
        rows.append(f)
    out = pd.DataFrame(rows, index=ds.index)
    from social.features import zscore_by_slot
    keys = ds[["day", "hhmm"]]
    zc = [c for c in _ZSCORE if out[c].notna().any()]
    if zc:
        z = zscore_by_slot(out[zc], keys)
        out[zc] = z[zc]
    return out
