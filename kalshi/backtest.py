"""Kalshi longshot fade (K1) and favorite (K2), the fixed rules pre-registered in
docs/kalshi_plan.md. Free public market data only, no API key.

  python3 -m kalshi.backtest fetch     # metadata, settled markets, sample, events, candles
  python3 -m kalshi.backtest run       # the one pre-registered run
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "kalshi"
OUT = ROOT / "reports" / "kalshi"
BASE = "https://external-api.kalshi.com/trade-api/v2"
USER_AGENT = "research-client/1.0"
SLEEP = 0.25                       # seconds between calls (<= 4 requests/s)
MAX_RETRIES = 8                    # backoff 1, 2, 4, ... 64 s on 429 / 5xx

CLOSE_FROM = pd.Timestamp("2024-01-01", tz="UTC")
CLOSE_TO = pd.Timestamp("2026-11-01", tz="UTC")         # exclusive
SPLIT = pd.Timestamp("2025-07-01", tz="UTC")            # H1 < SPLIT <= H2
MIN_VOLUME = 1_000.0
MIN_OPEN_HOURS = 48
LEAD_SECONDS = 24 * 3600                                # decision at close_time - 24 h
WINDOW_SECONDS = 24 * 3600                              # candles fetched over [D - 24 h, D]
EXT_LOOKBACK_SECONDS = 30 * 24 * 3600                   # report-only check (plan note A2)
SETTLED_STATUS = ("finalized", "settled")
K1_BAND = (0.03, 0.15)
K2_BAND = (0.85, 0.97)
TAKER_COEF = 0.07
MAKER_SHARE = {"quadratic": 0.0, "quadratic_with_maker_fees": 0.25,
               "quadratic_with_combo_maker_fees": 0.5}
FEE_STEP = 0.0001                  # direct-member balance precision (gate)
FEE_STEP_CONSERVATIVE = 0.01       # one contract rounded up to the cent (report only)
MAX_SAMPLE = 20_000
SEED, N_BOOT = 0, 10_000
GATE_Q = (0.0125, 0.9875)          # two-sided 97.5% CI (Bonferroni over 2 strategies)
CI95_Q = (0.025, 0.975)
CALIB_EDGES = [0, .03, .05, .10, .15, .25, .35, .45, .55, .65, .75, .85, .90, .95, .97, 1.0]
PARTICIPATION = 0.10               # capacity proxy: 10% of the 24 h volume before D
MARKET_COLS = ["ticker", "event_ticker", "market_type", "status", "result", "created_time",
               "open_time", "close_time", "expected_expiration_time", "latest_expiration_time",
               "settlement_ts", "volume_fp", "open_interest_fp", "can_close_early",
               "strike_type", "yes_sub_title", "mve_collection_ticker", "is_provisional",
               "settlement_value_dollars"]


# ---------------------------------------------------------------------------
# HTTP (public endpoints, no auth)
# ---------------------------------------------------------------------------

class AuthRequired(RuntimeError):
    pass


class Limiter:
    """Process-wide spacing between requests: at most 1 / SLEEP requests per second in total,
    however many worker threads share it."""

    def __init__(self, interval: float = SLEEP):
        self.interval, self.next, self.lock = interval, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.interval
        time.sleep(max(0.0, t - now))


LIMITER = Limiter()


class Client:
    def __init__(self, base: str = BASE, limiter: Limiter = LIMITER):
        import requests
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self.base, self.limiter, self.calls = base, limiter, 0
        self.log = open(DATA / "fetch_log.txt", "a") if DATA.exists() else None

    def get(self, path: str, params: dict | None = None, allow_404: bool = False):
        wait = 1.0
        for attempt in range(MAX_RETRIES + 1):
            self.limiter.wait()
            try:
                r = self.s.get(self.base + path, params=params, timeout=60)
            except Exception as e:                      # network error: back off and retry
                self._note(f"error {path} {e!r}")
                time.sleep(wait)
                wait = min(wait * 2, 64)
                continue
            self.calls += 1
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 403):
                raise AuthRequired(f"{path} returned {r.status_code}: authentication required")
            if r.status_code in (400, 404) and allow_404:      # no data for this market
                self._note(f"{r.status_code} {path} {r.text[:120]}")
                return None
            if r.status_code == 429 or r.status_code >= 500:
                self._note(f"{r.status_code} {path} retry in {wait:.0f}s")
                time.sleep(wait)
                wait = min(wait * 2, 64)
                continue
            raise RuntimeError(f"{path} {params} -> {r.status_code} {r.text[:300]}")
        raise RuntimeError(f"{path} failed after {MAX_RETRIES} retries")

    def _note(self, msg: str):
        line = f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M:%SZ} {msg}"
        print(line, flush=True)
        if self.log:
            self.log.write(line + "\n")
            self.log.flush()


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def fetch_meta(c: Client) -> dict:
    cutoff = c.get("/historical/cutoff")
    (DATA / "cutoff.json").write_text(json.dumps(cutoff, indent=2))
    series = c.get("/series", {"include_volume": "true"})
    (DATA / "series.json").write_text(json.dumps(series))
    fees = c.get("/series/fee_changes", {"show_historical": "true"})
    (DATA / "fee_changes.json").write_text(json.dumps(fees, indent=2))
    return cutoff


def compact(markets: list) -> pd.DataFrame:
    rows = [{k: m.get(k) for k in MARKET_COLS} for m in markets]
    df = pd.DataFrame(rows, columns=MARKET_COLS)
    for k in ("can_close_early", "is_provisional"):
        df[k] = df[k].astype("object")
    return df.astype({k: "string" for k in MARKET_COLS if k not in ("can_close_early", "is_provisional")})


def page_markets(c: Client, path: str, params: dict, folder: Path, pages_per_part: int = 20) -> int:
    """Cursor-page a markets endpoint into parquet parts; resumable via state.json."""
    folder.mkdir(parents=True, exist_ok=True)
    state_f = folder / "state.json"
    st = json.loads(state_f.read_text()) if state_f.exists() else {"cursor": "", "part": 0, "rows": 0,
                                                                     "done": False}
    if st["done"]:
        return st["rows"]
    buf, pages = [], 0
    cursor = st["cursor"]
    while True:
        p = dict(params, limit=1000)
        if cursor:
            p["cursor"] = cursor
        d = c.get(path, p)
        ms = d.get("markets") or []
        buf.extend(ms)
        cursor = d.get("cursor") or ""
        pages += 1
        done = not cursor or not ms
        if pages % pages_per_part == 0 or done:
            if buf:
                compact(buf).to_parquet(folder / f"part-{st['part']:05d}.parquet", index=False)
                st["part"] += 1
                st["rows"] += len(buf)
            buf = []
            st["cursor"], st["done"] = cursor, done
            state_f.write_text(json.dumps(st))
            print(f"{path}: {st['rows']:,} markets, {st['part']} parts, {c.calls} calls", flush=True)
        if done:
            return st["rows"]


UNIVERSE_SOURCES = (("markets_hist", "hist"), ("markets_live", "live"))


def load_universe(sources=UNIVERSE_SOURCES) -> tuple:
    """Apply the universe filters part by part (the full listing does not fit in memory)."""
    keep, counts = [], {}
    for folder, src in sources:
        f = DATA / folder
        for p in sorted(f.glob("part-*.parquet")) if f.exists() else []:
            u, c = universe(pd.read_parquet(p))
            u["source"] = src
            keep.append(u)
            for k, v in c.items():
                counts[k] = counts.get(k, 0) + v
            counts[f"listed_{src}"] = counts.get(f"listed_{src}", 0) + c["listed"]
    df = pd.concat(keep, ignore_index=True)
    df = df[df["mve_collection_ticker"].isna()]                 # combos are out of the universe
    # a market can appear in both tiers while it moves to the archive: keep the archived copy
    n = len(df)
    df = df.drop_duplicates("ticker", keep="first").reset_index(drop=True)
    counts["qualifying_duplicates_removed"] = int(n - len(df))
    counts["qualifying"] = int(len(df))
    return df, counts


def to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")


def universe(df: pd.DataFrame) -> tuple:
    """Pre-registered filters, in order; returns (qualifying markets, drop counts)."""
    df = df.copy()
    df["close"] = to_utc(df["close_time"])
    df["open"] = to_utc(df["open_time"])
    df["volume"] = pd.to_numeric(df["volume_fp"], errors="coerce")
    steps = [
        ("close_time in [2024-01-01, 2026-11-01)", (df["close"] >= CLOSE_FROM) & (df["close"] < CLOSE_TO)),
        ("market_type binary", df["market_type"] == "binary"),
        ("status finalized/settled", df["status"].isin(SETTLED_STATUS)),
        ("result yes/no", df["result"].isin(["yes", "no"])),
        ("volume >= 1000", df["volume"] >= MIN_VOLUME),
        ("open >= 48h before close", df["open"] <= df["close"] - pd.Timedelta(hours=MIN_OPEN_HOURS)),
    ]
    counts = {"listed": int(len(df))}
    keep = pd.Series(True, index=df.index)
    for name, m in steps:
        m = m.fillna(False).astype(bool)
        counts[f"dropped: {name}"] = int((keep & ~m).sum())
        keep &= m
    counts["qualifying"] = int(keep.sum())
    return df[keep].reset_index(drop=True), counts


def stratified_sample(df: pd.DataFrame, n_max: int = MAX_SAMPLE, seed: int = SEED) -> pd.DataFrame:
    """Proportional stratified sample by close month (largest remainder), seed fixed."""
    if len(df) <= n_max:
        return df.sort_values("ticker").reset_index(drop=True)
    month = df["close"].dt.strftime("%Y-%m")
    counts = month.value_counts().sort_index()
    quota = n_max * counts / counts.sum()
    alloc = np.floor(quota).astype(int)
    extra = n_max - int(alloc.sum())
    rem = (quota - alloc).sort_values(ascending=False, kind="mergesort")   # ties: calendar order
    alloc[rem.index[:extra]] += 1
    rng = np.random.default_rng(seed)
    out = []
    for m in counts.index:
        sub = df[month == m].sort_values("ticker")
        idx = np.sort(rng.choice(len(sub), size=int(alloc[m]), replace=False))
        out.append(sub.iloc[idx])
    return pd.concat(out).reset_index(drop=True)


def fetch_events(c: Client, event_tickers: list, batch: int = 100) -> pd.DataFrame:
    f = DATA / "events.parquet"
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["event_ticker", "series_ticker",
                                                                       "category", "title"])
    todo = sorted(set(event_tickers) - set(have["event_ticker"]))
    rows = []
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        d = c.get("/events", {"tickers": ",".join(chunk), "limit": 200})
        for e in d.get("events") or []:
            rows.append({"event_ticker": e.get("event_ticker"), "series_ticker": e.get("series_ticker"),
                         "category": e.get("category"), "title": e.get("title")})
        if (i // batch) % 20 == 19 or i + batch >= len(todo):
            have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("event_ticker")
            have.to_parquet(f, index=False)
            rows = []
            print(f"events: {len(have):,} / {len(set(event_tickers)):,}", flush=True)
    return have


def candle_window(close_ts: int) -> tuple:
    d = int(close_ts) - LEAD_SECONDS
    return d - WINDOW_SECONDS, d


def ext_window(close_ts: int, open_ts: int) -> tuple:
    d = int(close_ts) - LEAD_SECONDS
    return max(int(open_ts), d - EXT_LOOKBACK_SECONDS), d


def fetch_candles(c: Client, sample: pd.DataFrame, series_of: dict, f: Path | None = None,
                  window=None) -> None:
    """One call per market for hourly candles over [D - 24 h, D] (or `window`); JSONL, resumable."""
    f = f or DATA / "candles.jsonl"
    window = window or (lambda m: candle_window(int(m.close.timestamp())))
    done = set()
    if f.exists():
        with open(f) as fh:
            for line in fh:
                done.add(json.loads(line)["ticker"])
    todo = sample[~sample["ticker"].isin(done)]
    with open(f, "a") as fh:
        for i, m in enumerate(todo.itertuples(index=False)):
            start, end = window(m)
            params = {"start_ts": start, "end_ts": end, "period_interval": 60}
            route = "hist" if m.source == "hist" else "live"
            d = None
            for r in ((route, "live") if route == "hist" else (route, "hist")):
                if r == "hist":
                    d = c.get(f"/historical/markets/{m.ticker}/candlesticks", params, allow_404=True)
                else:
                    s = series_of.get(m.event_ticker) or m.event_ticker.split("-")[0]
                    d = c.get(f"/series/{s}/markets/{m.ticker}/candlesticks", params, allow_404=True)
                if d is not None:
                    route = r
                    break
            fh.write(json.dumps({"ticker": m.ticker, "route": route if d is not None else "missing",
                                 "start_ts": start, "end_ts": end,
                                 "candlesticks": (d or {}).get("candlesticks") or []}) + "\n")
            if i % 200 == 199:
                fh.flush()
                print(f"candles: {len(done) + i + 1:,} / {len(sample):,} ({c.calls} calls)", flush=True)


def fetch(step: str = "all") -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    c = Client()
    if step in ("all", "markets"):
        cutoff = fetch_meta(c) if not (DATA / "cutoff.json").exists() else \
            json.loads((DATA / "cutoff.json").read_text())
        page_markets(c, "/historical/markets", {"mve_filter": "exclude"}, DATA / "markets_hist")
        settled_from = int(pd.Timestamp(cutoff["market_settled_ts"]).timestamp()) - 86400
        page_markets(c, "/markets", {"status": "settled", "min_settled_ts": settled_from,
                                     "mve_filter": "exclude"}, DATA / "markets_live")
    if step in ("all", "sample", "candles"):
        uni, counts = load_universe()
        sample = stratified_sample(uni)
        sample.to_parquet(DATA / "sample.parquet", index=False)
        counts["universe_by_month"] = uni["close"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
        counts["sample_by_month"] = sample["close"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
        counts["universe_by_source"] = uni["source"].value_counts().to_dict()
        (DATA / "universe_counts.json").write_text(json.dumps({**counts, "sampled": int(len(sample))},
                                                              indent=2))
        print(json.dumps({**counts, "sampled": int(len(sample))}, indent=2))
        ev = fetch_events(c, sample["event_ticker"].tolist())
        series_of = dict(zip(ev["event_ticker"], ev["series_ticker"]))
        fetch_candles(c, sample, series_of)
    if step in ("all", "gaps"):
        sample = pd.read_parquet(DATA / "sample.parquet")
        ev = pd.read_parquet(DATA / "events.parquet")
        series_of = dict(zip(ev["event_ticker"], ev["series_ticker"]))
        recs = load_candles(DATA / "candles.jsonl")
        gap = [t for t, m in zip(sample["ticker"], sample.itertuples(index=False))
               if t in recs and decision_quote(recs[t]["candlesticks"],
                                               int(m.close.timestamp()) - LEAD_SECONDS) is None]
        print(f"gap markets to refetch with a 30-day lookback: {len(gap):,}", flush=True)
        fetch_candles(c, sample[sample["ticker"].isin(gap)], series_of, DATA / "candles_ext.jsonl",
                      window=lambda m: ext_window(int(m.close.timestamp()), int(m.open.timestamp())))
    print(f"done, {c.calls} calls", flush=True)


# ---------------------------------------------------------------------------
# rule
# ---------------------------------------------------------------------------

def _num(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _bool(x):
    if x is None or x is pd.NA or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, str):
        return x.lower() == "true"
    return bool(x)


def parse_candle(cd: dict) -> dict:
    """Historical candles use open/close/volume; live ones *_dollars / *_fp. Accept both."""
    def close_of(side):
        d = cd.get(side) or {}
        return _num(d.get("close", d.get("close_dollars")))
    return {"end": int(cd["end_period_ts"]), "bid": close_of("yes_bid"), "ask": close_of("yes_ask"),
            "volume": _num(cd.get("volume", cd.get("volume_fp"))) or 0.0,
            "oi": _num(cd.get("open_interest", cd.get("open_interest_fp")))}


def decision_quote(candles: list, decision_ts: int, lookback: int = WINDOW_SECONDS) -> dict | None:
    """Latest candle ending at or before D (and not before D - lookback). Never uses anything later."""
    cs = [parse_candle(x) for x in candles]
    ok = [x for x in cs if decision_ts - lookback <= x["end"] <= decision_ts]
    if not ok:
        return None
    last = max(ok, key=lambda x: x["end"])
    vol24 = sum(x["volume"] for x in ok if decision_ts - WINDOW_SECONDS < x["end"] <= decision_ts)
    return {"end": last["end"], "bid": last["bid"], "ask": last["ask"], "oi_at_d": last["oi"],
            "vol24": vol24, "stale_s": decision_ts - last["end"]}


def valid_quote(bid, ask) -> bool:
    return bid is not None and ask is not None and 0.0 < bid < ask < 1.0


def mid(bid: float, ask: float) -> float:
    return round((bid + ask) / 2.0, 6)


def signal(p_mid: float) -> str | None:
    if K1_BAND[0] <= p_mid <= K1_BAND[1]:
        return "K1"
    if K2_BAND[0] <= p_mid <= K2_BAND[1]:
        return "K2"
    return None


def entry(strategy: str, bid: float, ask: float, maker: bool = False) -> tuple:
    """(side bought, price paid per contract). K1 buys NO, K2 buys YES."""
    if strategy == "K1":
        return "no", round(1.0 - (ask if maker else bid), 6)
    if strategy == "K2":
        return "yes", round(bid if maker else ask, 6)
    raise ValueError(strategy)


def ceil_to(x: float, step: float) -> float:
    return math.ceil(round(x / step, 6)) * step


def quadratic_fee(price: float, coef: float, step: float = FEE_STEP) -> float:
    """coef x P x (1 - P) per contract, rounded up to `step` (one contract)."""
    if coef <= 0:
        return 0.0
    return round(ceil_to(coef * price * (1.0 - price), step), 6)


def taker_fee(price: float, multiplier: float = 1.0, step: float = FEE_STEP) -> float:
    return quadratic_fee(price, TAKER_COEF * multiplier, step)


def maker_fee(price: float, fee_type: str = "quadratic", multiplier: float = 1.0,
              step: float = FEE_STEP) -> float:
    return quadratic_fee(price, TAKER_COEF * MAKER_SHARE.get(fee_type, 0.0) * multiplier, step)


def payoff(side: str, result: str) -> float:
    if result not in ("yes", "no") or side not in ("yes", "no"):
        raise ValueError((side, result))
    return 1.0 if side == result else 0.0


def roc(pay: float, price: float, fee: float) -> float:
    return (pay - price - fee) / (price + fee)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def cluster_bootstrap(r, clusters, n: int = N_BOOT, seed: int = SEED, q=GATE_Q) -> dict:
    """Pooled mean of r; resample clusters with replacement; percentile CI."""
    r = np.asarray(r, float)
    if len(r) == 0:
        return {"n": 0, "clusters": 0, "mean": float("nan"), "ci": [float("nan")] * 2,
                "ci95": [float("nan")] * 2}
    codes, uniq = pd.factorize(pd.Series(clusters).astype(str))
    G = len(uniq)
    S = np.bincount(codes, weights=r, minlength=G)
    N = np.bincount(codes, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    step = max(1, min(n, 2_000_000 // max(G, 1)))
    for i in range(0, n, step):
        k = min(step, n - i)
        draw = rng.integers(0, G, size=(k, G))
        means[i:i + k] = S[draw].sum(axis=1) / N[draw].sum(axis=1)
    return {"n": int(len(r)), "clusters": int(G), "mean": float(r.mean()),
            "ci": [float(np.quantile(means, q[0])), float(np.quantile(means, q[1]))],
            "ci95": [float(np.quantile(means, CI95_Q[0])), float(np.quantile(means, CI95_Q[1]))]}


# ---------------------------------------------------------------------------
# the one run
# ---------------------------------------------------------------------------

def load_candles(f: Path) -> dict:
    out = {}
    if f.exists():
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                out[d["ticker"]] = d
    return out


def load_inputs() -> tuple:
    sample = pd.read_parquet(DATA / "sample.parquet")
    events = pd.read_parquet(DATA / "events.parquet")
    series = pd.DataFrame(json.loads((DATA / "series.json").read_text())["series"])
    return sample, events, series, load_candles(DATA / "candles.jsonl")


def build_trades(sample: pd.DataFrame, events: pd.DataFrame, series: pd.DataFrame,
                 candles: dict, lookback: int = WINDOW_SECONDS) -> tuple:
    """One row per sampled market: quote at D, signal and every variant's return."""
    ev = events.drop_duplicates("event_ticker").set_index("event_ticker")
    se = series.drop_duplicates("ticker").set_index("ticker")
    gaps = {"no candle record": 0, "no candle in [D-24h, D]": 0, "invalid quote": 0}
    rows = []
    for m in sample.itertuples(index=False):
        close_ts = int(m.close.timestamp())
        d_ts = close_ts - LEAD_SECONDS
        rec = candles.get(m.ticker)
        if rec is None:
            gaps["no candle record"] += 1
            continue
        qd = decision_quote(rec["candlesticks"], d_ts, lookback)
        if qd is None:
            gaps["no candle in [D-24h, D]"] += 1
            continue
        if not valid_quote(qd["bid"], qd["ask"]):
            gaps["invalid quote"] += 1
            continue
        s_tk = ev["series_ticker"].get(m.event_ticker) if m.event_ticker in ev.index else None
        s_tk = s_tk or m.event_ticker.split("-")[0]
        in_snap = s_tk in se.index
        mult = float(se.at[s_tk, "fee_multiplier"]) if in_snap else 1.0
        ftype = str(se.at[s_tk, "fee_type"]) if in_snap else "quadratic"
        cat = (se.at[s_tk, "category"] if in_snap else None) or \
              (ev["category"].get(m.event_ticker) if m.event_ticker in ev.index else None) or "unknown"
        p = mid(qd["bid"], qd["ask"])
        row = {"ticker": m.ticker, "event_ticker": m.event_ticker, "series": s_tk, "category": cat,
               "close": m.close, "decision_ts": d_ts, "candle_end": qd["end"], "stale_s": qd["stale_s"],
               "half": "H1" if m.close < SPLIT else "H2", "bid": qd["bid"], "ask": qd["ask"],
               "p_mid": p, "result": m.result, "yes": 1.0 if m.result == "yes" else 0.0,
               "volume": float(m.volume), "oi_market": _num(m.open_interest_fp),
               "oi_at_d": qd["oi_at_d"], "vol24": qd["vol24"],
               "can_close_early": _bool(m.can_close_early),
               "fee_type": ftype, "fee_multiplier": mult, "series_in_snapshot": in_snap,
               "strategy": signal(p)}
        if row["strategy"]:
            side, price = entry(row["strategy"], qd["bid"], qd["ask"])
            pay = payoff(side, m.result)
            f = taker_fee(price, mult)
            row.update(side=side, price=price, payoff=pay, fee=f, r=roc(pay, price, f),
                       r_gross=roc(pay, price, 0.0),
                       r_cent=roc(pay, price, taker_fee(price, mult, FEE_STEP_CONSERVATIVE)))
            _, mp = entry(row["strategy"], qd["bid"], qd["ask"], maker=True)
            mf = maker_fee(mp, ftype, mult)
            row.update(maker_price=mp, maker_fee=mf, r_maker=roc(pay, mp, mf))
        rows.append(row)
    return pd.DataFrame(rows), gaps


def calibration(q: pd.DataFrame) -> list:
    b = pd.cut(q["p_mid"], CALIB_EDGES, right=False, include_lowest=True)
    g = q.groupby(b, observed=False)
    t = pd.DataFrame({"n": g.size(), "mean_p_mid": g["p_mid"].mean(), "yes_rate": g["yes"].mean()})
    t["yes_rate_minus_p"] = t["yes_rate"] - t["mean_p_mid"]
    t.index = [str(i) for i in t.index]
    return t.round(4).reset_index(names="bucket").to_dict(orient="records")


def summarize(t: pd.DataFrame, col: str = "r", q=GATE_Q) -> dict:
    out = {}
    for label, part in (("H1", t[t["half"] == "H1"]), ("H2", t[t["half"] == "H2"]), ("full", t)):
        b = cluster_bootstrap(part[col].to_numpy(), part["event_ticker"].to_numpy(), q=q)
        b["win_rate"] = float(part["payoff"].mean()) if len(part) else float("nan")
        b["mean_price"] = float(part["price"].mean()) if len(part) else float("nan")
        out[label] = b
    return out


def group_table(t: pd.DataFrame, key: str, top: int | None = None) -> list:
    g = t.groupby(key)
    d = pd.DataFrame({"n": g.size(), "events": g["event_ticker"].nunique(), "mean_r": g["r"].mean(),
                      "win_rate": g["payoff"].mean(), "mean_price": g["price"].mean()})
    d = d.sort_values("n", ascending=False)
    if top:
        d = d.head(top)
    return d.round(4).reset_index().to_dict(orient="records")


def capacity(t: pd.DataFrame) -> dict:
    dep = PARTICIPATION * t["vol24"] * t["price"]
    qs = lambda s: {"median": float(s.median()), "p25": float(s.quantile(.25)), "p75": float(s.quantile(.75))}
    return {"n": int(len(t)), "lifetime_volume": qs(t["volume"]), "open_interest_market": qs(t["oi_market"]),
            "open_interest_at_d": qs(t["oi_at_d"].astype(float)), "contracts_24h_before_d": qs(t["vol24"]),
            "deployable_dollars_10pct_of_24h_volume": qs(dep),
            "share_with_zero_24h_volume": float((t["vol24"] <= 0).mean())}


def main_run() -> dict:
    sample, events, series, candles = load_inputs()
    counts = json.loads((DATA / "universe_counts.json").read_text())
    cutoff = json.loads((DATA / "cutoff.json").read_text())
    q, gaps = build_trades(sample, events, series, candles)
    res = {"universe": counts, "cutoff": cutoff, "sample": int(len(sample)), "data_gaps": gaps,
           "valid_quotes": int(len(q)),
           "valid_by_half": q["half"].value_counts().sort_index().to_dict(),
           "candle_routes": pd.Series([v["route"] for v in candles.values()]).value_counts().to_dict(),
           "stale_quote_share_over_1h": float((q["stale_s"] >= 3600).mean()),
           "stale_quote_median_s": float(q["stale_s"].median()),
           "series_not_in_fee_snapshot": int((~q["series_in_snapshot"]).sum())}
    res["calibration"] = calibration(q)
    res["calibration_by_half"] = {h: calibration(q[q["half"] == h]) for h in ("H1", "H2")}
    for k in ("K1", "K2"):
        t = q[q["strategy"] == k]
        g = summarize(t, "r")
        g["passes"] = bool(all(g[h]["n"] > 0 and g[h]["ci"][0] > 0 for h in ("H1", "H2")))
        res[k] = {"gate": g,
                  "maker_optimistic": summarize(t, "r_maker"),
                  "fee_cent_conservative": summarize(t, "r_cent"),
                  "gross_no_fee": summarize(t, "r_gross"),
                  "excl_can_close_early": summarize(t[t["can_close_early"] != True], "r"),
                  "fresh_quote_only": summarize(t[t["stale_s"] < 3600], "r"),
                  "by_category": group_table(t, "category"),
                  "top_series": group_table(t, "series", top=15),
                  "capacity": capacity(t),
                  "mean_fee": float(t["fee"].mean()) if len(t) else float("nan")}
    # report-only (plan note A2): gap markets get the standing quote from up to 30 days before D
    ext = load_candles(DATA / "candles_ext.jsonl")
    if ext:
        merged = {**candles, **{t: v for t, v in ext.items() if v["candlesticks"]}}
        qx, gx = build_trades(sample, events, series, merged, lookback=EXT_LOOKBACK_SECONDS)
        res["extended_lookback"] = {"data_gaps": gx, "valid_quotes": int(len(qx)),
                                    "added_vs_gate_set": int(len(qx) - len(q))}
        for k in ("K1", "K2"):
            res["extended_lookback"][k] = summarize(qx[qx["strategy"] == k], "r")
    res["verdict"] = {k: ("pass" if res[k]["gate"]["passes"] else "fail") for k in ("K1", "K2")}
    res["overall"] = "edge after fees" if any(res[k]["gate"]["passes"] for k in ("K1", "K2")) else "closed"
    OUT.mkdir(parents=True, exist_ok=True)
    q.to_parquet(OUT / "markets_at_decision.parquet", index=False)
    (OUT / "kalshi.json").write_text(json.dumps(res, indent=2, default=_jsonable))
    return res


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "fetch":
        fetch(sys.argv[2] if len(sys.argv) > 2 else "all")
    elif cmd == "run":
        res = main_run()
        short = {k: v for k, v in res.items() if k not in ("calibration_by_half",)}
        for k in ("K1", "K2"):
            short[k] = {kk: vv for kk, vv in res[k].items() if kk not in ("by_category", "top_series")}
        print(json.dumps(short, indent=2, default=_jsonable))
    return 0


if __name__ == "__main__":
    sys.exit(main())
