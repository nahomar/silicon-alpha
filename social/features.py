"""Social-activity features for SPX 0DTE decisions: is the crowd louder than usual?

The text study (nlpalpha/FINDINGS.md) found chatter predicts the *size* of a
move, not its direction. That is the quantity this model forecasts. So
these features measure activity and lean, not meaning:

  Reddit (Arctic Shift hourly counts; no text needed, a year costs a few
  dozen requests)
    rd_wsb_c_1h       WSB comments in the hour before t
    rd_wsb_c_night    WSB comments per hour, previous close -> 09:00
    rd_wsb_c_today    WSB comments per hour, 09:00 -> t
    rd_opt_c_1h       r/options + r/thetagang comments in the hour before t
    rd_wsb_p_1h       WSB posts in the hour before t
  Stocktwits (local recorder data: SPX, ES_F, VIX streams)
    st_idx_2h         messages in the 2 hours before t
    st_idx_bull       (bullish - bearish) / tagged, previous close -> t

Every count is log1p'd and z-scored against the same window on the previous
20 sessions, so "1,200 comments" becomes "busier than usual for 10 a.m.".
The z-score uses prior sessions only.

Point-in-time rules: an hourly bucket [h, h+1) is used only if h+1 <= t,
and a Stocktwits message only if created_utc < t. Missing data is NaN,
never zero.

Usage:
  python3 -m social.features fetch-reddit --start 2025-10-01
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "data" / "social" / "agg"
STW = ROOT / "data" / "social" / "stocktwits"
API = "https://arctic-shift.photon-reddit.com/api"
SUBS = ("wallstreetbets", "options", "thetagang")
TZ = "America/New_York"
SOCIAL_FEATURES = ["rd_wsb_c_1h", "rd_wsb_c_night", "rd_wsb_c_today",
                   "rd_opt_c_1h", "rd_wsb_p_1h", "st_idx_2h", "st_idx_bull"]
_RAW = SOCIAL_FEATURES                     # z-scored in place, same names
_NO_Z = {"st_idx_bull"}                    # already a bounded ratio


# ---------------------------------------------------------------------------
# Reddit hourly counts
# ---------------------------------------------------------------------------

def _agg(kind: str, sub: str, after: int, before: int, attempts: int = 2) -> list:
    q = urllib.parse.urlencode({"subreddit": sub, "after": after, "before": before,
                                "aggregate": "created_utc", "frequency": "hour"})
    url = f"{API}/{kind}/search/aggregate?{q}"
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "silicon-alpha-research/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read())
            if d.get("data") is None:
                raise RuntimeError(d.get("error") or "no data")
            return d["data"]
        except Exception as e:
            last = e
            if isinstance(e, urllib.error.HTTPError):
                try:
                    last = RuntimeError(f"{e} {e.read().decode('utf-8', 'replace')[:120]}")
                except Exception:
                    pass
            time.sleep(15 * (i + 1))
    raise last


def _agg_adaptive(kind, sub, lo: int, hi: int) -> list:
    """One window, split in half on failure down to a single day.

    The archive answers "Timeout. Maybe slow down a bit" both when pushed
    and when one query is too heavy (a month of WSB comments). Halving fixes
    the second case; the pause between calls handles the first.
    """
    if hi - lo <= 3600:
        # Smallest window: keep trying with long waits instead of failing.
        # The archive recovers; giving up would leave a hole in the series.
        for i in range(20):
            try:
                out = _agg(kind, sub, lo, hi, attempts=1)
                time.sleep(3.0)
                return out
            except Exception as e:
                wait = min(60 * (i + 1), 300)
                print(f"  {sub}/{kind}: hour {pd.Timestamp(lo, unit='s'):%Y-%m-%d %H}:00 "
                      f"busy, wait {wait}s", flush=True)
                time.sleep(wait)
        raise RuntimeError(f"{sub}/{kind}: hour {lo} failed 20 times")
    try:
        out = _agg(kind, sub, lo, hi)
        time.sleep(3.0)
        return out
    except Exception as e:
        print(f"  {sub}/{kind}: split {pd.Timestamp(lo, unit='s'):%Y-%m-%d %H}:00 "
              f"({str(e)[:60]})", flush=True)
        time.sleep(20)
        mid = lo + max(3600, ((hi - lo) // 2 // 3600) * 3600)
        return _agg_adaptive(kind, sub, lo, mid) + _agg_adaptive(kind, sub, mid, hi)


def _agg_path(sub: str, kind: str) -> Path:
    return AGG / "reddit_hourly" / f"{sub}_{kind}.parquet"


def fetch_reddit_hourly(start, end=None, subs=None, kinds=("comments", "posts")) -> pd.DataFrame:
    """Hourly comment and post counts per sub, a week at a time, resumable.

    Each (sub, kind) file is extended from its newest saved hour, so a rerun
    continues where the last one stopped. Windows are whole UTC hours, given
    as epoch seconds (date strings are read in the server's time zone).
    """
    end = pd.Timestamp(end) if end else pd.Timestamp.utcnow().tz_localize(None).floor("h")
    for kind in kinds:
        for sub in (subs or SUBS):
            p = _agg_path(sub, kind)
            old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["hour_utc", "count"])
            lo = (pd.Timestamp(old["hour_utc"].max()) - pd.Timedelta(days=1)).floor("D") \
                if len(old) else pd.Timestamp(start)
            rows = []
            t = lo
            # A week of WSB comments is too heavy a query for the archive
            # (it times out); the smaller subs and all posts are fine weekly.
            step = pd.Timedelta(days=1 if (sub, kind) == ("wallstreetbets", "comments") else 7)
            if t >= end:
                continue                               # this slice is already covered
            while t < end:
                hi = min(t + step, end)
                data = _agg_adaptive(kind, sub, int(t.tz_localize("UTC").timestamp()),
                                     int(hi.tz_localize("UTC").timestamp()))
                rows += [(pd.Timestamp(x["created_utc"]).tz_convert("UTC").tz_localize(None),
                          int(x["count"])) for x in data]
                t = hi
                # Save every window: an interruption loses at most one week.
                new = pd.DataFrame(rows, columns=["hour_utc", "count"])
                merged = (pd.concat([old, new]).drop_duplicates("hour_utc", keep="last")
                          .sort_values("hour_utc"))
                p.parent.mkdir(parents=True, exist_ok=True)
                merged.to_parquet(p, index=False)
            print(f"{sub}/{kind}: through {end:%Y-%m-%d %H:%M}Z", flush=True)
    return load_reddit_hourly()


def load_reddit_hourly():
    parts = []
    for p in sorted((AGG / "reddit_hourly").glob("*_*.parquet")):
        sub, kind = p.stem.rsplit("_", 1)
        d = pd.read_parquet(p)
        d["sub"], d["kind"] = sub, kind
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else None


# ---------------------------------------------------------------------------
# Raw window values
# ---------------------------------------------------------------------------

def _et(day: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    """Naive ET wall time -> naive UTC."""
    t = pd.Timestamp(f"{pd.Timestamp(day):%Y-%m-%d} {hhmm}").tz_localize(TZ)
    return t.tz_convert("UTC").tz_localize(None)


class _Hourly:
    """Sum of complete hourly buckets inside [lo, hi), UTC."""

    def __init__(self, s: pd.Series):
        s = s.sort_index()
        self.t = s.index.to_numpy(dtype="datetime64[ns]")
        self.c = np.concatenate([[0], np.cumsum(s.to_numpy(dtype=float))])
        self.lo_cov = self.t.min() if len(self.t) else None
        self.hi_cov = self.t.max() if len(self.t) else None

    def rate(self, lo: pd.Timestamp, hi: pd.Timestamp) -> float:
        """Mean count per hour over buckets with h >= lo and h + 1h <= hi."""
        if self.lo_cov is None:
            return np.nan
        lo64, last_start = np.datetime64(lo), np.datetime64(hi - pd.Timedelta(hours=1))
        if lo64 < self.lo_cov or last_start > self.hi_cov or last_start < lo64:
            return np.nan                          # outside coverage: unknown
        i = np.searchsorted(self.t, lo64, side="left")
        j = np.searchsorted(self.t, last_start, side="right")
        hours = (hi - lo).total_seconds() / 3600.0
        return float(self.c[j] - self.c[i]) / hours


def _load_stocktwits(streams=("SPX", "ES_F", "VIX")) -> pd.DataFrame:
    parts = []
    for s in streams:
        for f in sorted((STW / s).glob("20??-??-??.parquet")):
            parts.append(pd.read_parquet(f, columns=["id", "created_utc", "sentiment"]))
    if not parts:
        return pd.DataFrame(columns=["id", "created_utc", "sentiment"])
    df = pd.concat(parts).drop_duplicates("id").sort_values("created_utc")
    return df


def raw_features(keys: pd.DataFrame, reddit: pd.DataFrame | None,
                 stw: pd.DataFrame | None) -> pd.DataFrame:
    """Unnormalized values for each (day, hhmm, prev_day) row of `keys`."""
    def hourly(sub, kind):
        if reddit is None or reddit.empty:
            return None
        r = reddit[(reddit["sub"].isin(sub)) & (reddit["kind"] == kind)]
        if r.empty:
            return None
        return _Hourly(r.groupby("hour_utc")["count"].sum())

    wsb_c, wsb_p = hourly(["wallstreetbets"], "comments"), hourly(["wallstreetbets"], "posts")
    opt_c = hourly(["options", "thetagang"], "comments")
    if stw is not None and len(stw):
        st_t = stw["created_utc"].to_numpy()
        st_s = stw["sentiment"].fillna("").to_numpy()
        st_lo, st_hi = int(st_t.min()), int(st_t.max())
    out = []
    for day, hhmm, prev in zip(keys["day"], keys["hhmm"], keys["prev_day"]):
        t = _et(day, hhmm)
        open9 = _et(day, "09:00")
        prev_close = _et(prev, "16:00") if pd.notna(prev) else None
        row = {}
        row["rd_wsb_c_1h"] = wsb_c.rate(t - pd.Timedelta(hours=1), t) if wsb_c else np.nan
        row["rd_wsb_c_today"] = wsb_c.rate(open9, t) if wsb_c else np.nan
        row["rd_wsb_c_night"] = (wsb_c.rate(prev_close, open9)
                                 if wsb_c and prev_close is not None else np.nan)
        row["rd_opt_c_1h"] = opt_c.rate(t - pd.Timedelta(hours=1), t) if opt_c else np.nan
        row["rd_wsb_p_1h"] = wsb_p.rate(t - pd.Timedelta(hours=1), t) if wsb_p else np.nan
        if stw is not None and len(stw):
            t_s = int(t.tz_localize("UTC").timestamp())
            lo2 = t_s - 7200
            if lo2 >= st_lo and t_s <= st_hi:
                i, j = np.searchsorted(st_t, [lo2, t_s], side="left")
                row["st_idx_2h"] = float(j - i)
            else:
                row["st_idx_2h"] = np.nan
            if prev_close is not None:
                p_s = int(prev_close.tz_localize("UTC").timestamp())
                if p_s >= st_lo and t_s <= st_hi:
                    i, j = np.searchsorted(st_t, [p_s, t_s], side="left")
                    seg = st_s[i:j]
                    bull, bear = (seg == "Bullish").sum(), (seg == "Bearish").sum()
                    row["st_idx_bull"] = float((bull - bear) / (bull + bear)) if bull + bear >= 5 else np.nan
                else:
                    row["st_idx_bull"] = np.nan
            else:
                row["st_idx_bull"] = np.nan
        else:
            row["st_idx_2h"] = row["st_idx_bull"] = np.nan
        out.append(row)
    return pd.DataFrame(out, index=keys.index)


def zscore_by_slot(raw: pd.DataFrame, keys: pd.DataFrame, window: int = 20,
                   min_obs: int = 5) -> pd.DataFrame:
    """log1p + z against the same hhmm slot on the previous `window` sessions."""
    z = raw.copy()
    for col in raw.columns:
        if col in _NO_Z:
            continue
        x = np.log1p(raw[col].clip(lower=0))
        df = pd.DataFrame({"day": keys["day"], "hhmm": keys["hhmm"], "x": x})
        res = pd.Series(np.nan, index=raw.index)
        for _, g in df.groupby("hhmm"):
            g = g.sort_values("day")
            m = g["x"].shift(1).rolling(window, min_periods=min_obs).mean()
            s = g["x"].shift(1).rolling(window, min_periods=min_obs).std()
            res.loc[g.index] = ((g["x"] - m) / s.replace(0, np.nan)).to_numpy()
        z[col] = res
    return z


def social_features(keys: pd.DataFrame) -> pd.DataFrame:
    """keys: DataFrame with day, hhmm, prev_day. Returns SOCIAL_FEATURES."""
    reddit = load_reddit_hourly()
    stw = _load_stocktwits()
    raw = raw_features(keys, reddit, stw if len(stw) else None)
    return zscore_by_slot(raw[SOCIAL_FEATURES], keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch-reddit")
    f.add_argument("--start", default="2025-10-01")
    f.add_argument("--end", default=None)
    f.add_argument("--subs", nargs="+", default=None)
    f.add_argument("--kinds", nargs="+", default=["comments", "posts"])
    a = ap.parse_args()
    df = fetch_reddit_hourly(a.start, a.end, subs=a.subs, kinds=tuple(a.kinds))
    print(f"{len(df):,} hourly rows -> {AGG/'reddit_hourly'}/")
    print(df.groupby(["sub", "kind"])["count"].agg(["size", "sum"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# NLP window features (from social.nlp transformer scores)
# ---------------------------------------------------------------------------

NLP = ROOT / "data" / "social" / "nlp"
NLP_FEATURES = ["nlp_st_neg_2h", "nlp_st_fear_2h", "nlp_st_fear_night",
                "nlp_rd_neg_today", "nlp_rd_fear_today", "nlp_rd_0dte_today"]
_NLP_NO_Z = {"nlp_st_neg_2h", "nlp_st_fear_2h", "nlp_st_fear_night",
             "nlp_rd_neg_today", "nlp_rd_fear_today"}   # already bounded means
_MIN_MSGS = 5


class _Scored:
    """Scored messages sorted by time, with prefix sums for fast windows."""

    def __init__(self, df: pd.DataFrame):
        df = df.sort_values("created_utc")
        self.t = df["created_utc"].to_numpy(dtype=np.int64)
        self.lo = int(self.t.min()) if len(self.t) else None
        self.hi = int(self.t.max()) if len(self.t) else None
        cols = {"neg": (df["p_neg"] - df["p_pos"]).to_numpy(float),
                "fear": (df["fear"] - df["greed"]).to_numpy(float),
                "dte": df["zero_dte"].astype(float).to_numpy()}
        self.cs = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in cols.items()}

    def window(self, a: int, b: int):
        """(n, means dict) for created_utc in [a, b); None if outside coverage."""
        if self.lo is None or a < self.lo or b > self.hi + 1:
            return None
        i, j = np.searchsorted(self.t, [a, b], side="left")
        n = j - i
        return n, {k: (c[j] - c[i]) / n if n else np.nan for k, c in self.cs.items()}


def nlp_features(keys: pd.DataFrame) -> pd.DataFrame:
    """Transformer-score window features for each (day, hhmm, prev_day)."""
    def load(name, sub=None):
        p = NLP / f"{name}.parquet"
        if not p.exists():
            return None
        d = pd.read_parquet(p)
        if sub is not None and "sub" in d:
            d = d[d["sub"] == sub]
        return _Scored(d) if len(d) else None

    st, rd = load("stocktwits"), load("reddit_posts", "wallstreetbets")
    rows = []
    for day, hhmm, prev in zip(keys["day"], keys["hhmm"], keys["prev_day"]):
        t = int(_et(day, hhmm).tz_localize("UTC").timestamp())
        pc = int(_et(prev, "16:00").tz_localize("UTC").timestamp()) if pd.notna(prev) else None
        o = int(_et(day, "09:30").tz_localize("UTC").timestamp())
        r = {k: np.nan for k in NLP_FEATURES}
        if st is not None:
            w = st.window(t - 7200, t)
            if w and w[0] >= _MIN_MSGS:
                r["nlp_st_neg_2h"], r["nlp_st_fear_2h"] = w[1]["neg"], w[1]["fear"]
            if pc is not None:
                w = st.window(pc, o)
                if w and w[0] >= _MIN_MSGS:
                    r["nlp_st_fear_night"] = w[1]["fear"]
        if rd is not None and pc is not None:
            w = rd.window(pc, t)
            if w and w[0] >= _MIN_MSGS:
                r["nlp_rd_neg_today"], r["nlp_rd_fear_today"] = w[1]["neg"], w[1]["fear"]
                r["nlp_rd_0dte_today"] = w[1]["dte"] * w[0] / max((t - pc) / 3600, 1)
        rows.append(r)
    raw = pd.DataFrame(rows, index=keys.index)
    z = zscore_by_slot(raw[["nlp_rd_0dte_today"]], keys)
    raw["nlp_rd_0dte_today"] = z["nlp_rd_0dte_today"]
    return raw[NLP_FEATURES]
