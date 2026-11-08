"""Scheduled-event calendar for SPX 0DTE sessions.

Options already price *scheduled* events: a CPI morning or an FOMC
afternoon carries a fatter straddle. A model can only add value by knowing
the schedule better than its own inputs do: which events are on today, and
which are still *ahead* of the decision time.

Three sources, each with its provenance in the `source` column:

  verified   configs/calendar/macro_releases.csv: CPI, jobs (NFP), PPI,
             JOLTS, FOMC statements (with/without projections), FOMC minutes,
             the 2026 midterm. Actual release dates, not planned ones; the
             Oct-Nov 2025 shutdown moved or cancelled several.
  fetched    mega-cap earnings (Yahoo, with time of day) and 10/20/30-year
             Treasury auctions (fiscaldata.treasury.gov, with closing time).
  rule       jobless claims (Thursday 08:30, Wednesday in a holiday week),
             ISM manufacturing / services (1st / 3rd session, 10:00), monthly
             opex (3rd Friday, earlier if a holiday), quad witching, VIX
             expiry, month / quarter end, sessions next to a holiday.

Every event is mapped to the **session it affects**: an 08:30 release on a
non-trading day (the Mar-2026 jobs report came out on Good Friday) or
anything at or after 16:00 counts as pre-open news for the next session.

Not included: GDP and PCE. BEA's archive renders client-side, and FRED needs
an API key. Adding them is one fetch once a FRED key exists.

Usage:
  python3 -m odte.data.calendar build --start 2025-10-01 --end 2026-11-30
  python3 -m odte.data.calendar show --day 2026-03-18
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VERIFIED = ROOT / "configs" / "calendar" / "macro_releases.csv"
OUT = ROOT / "data" / "calendar"
MEGACAPS = ("NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AVGO", "TSLA")
AUCTION_TERMS = ("10-Year", "20-Year", "30-Year")
FISCAL = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
          "v1/accounting/od/auctions_query")

# Daily flags, in the order they become feature columns.
FLAG_EVENTS = ("cpi", "nfp", "ppi", "jolts", "claims", "ism_mfg", "ism_svc",
               "fomc", "fomc_sep", "fomc_minutes", "auction_long",
               "opex", "quad_witching", "vix_expiry", "month_end",
               "quarter_end", "pre_holiday", "post_holiday", "election")
CAL_FEATURES = ([f"ev_{e}" for e in FLAG_EVENTS]
                + ["megacap_earnings", "events_ahead", "fomc_ahead",
                   "auction_ahead", "minutes_to_next_event"])


# ---------------------------------------------------------------------------
# Trading sessions
# ---------------------------------------------------------------------------

def trading_sessions(start, end) -> pd.DatetimeIndex:
    """NYSE sessions, taken from the days ^GSPC actually printed a close."""
    import yfinance as yf
    h = yf.Ticker("^GSPC").history(start=pd.Timestamp(start) - pd.Timedelta(days=10),
                                   end=pd.Timestamp(end) + pd.Timedelta(days=10),
                                   auto_adjust=False)
    return pd.DatetimeIndex(pd.to_datetime(h.index).tz_localize(None).normalize()).unique()


def _holidays(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    wd = pd.bdate_range(sessions.min(), sessions.max())
    return wd.difference(sessions)


def _session_on_or_after(sessions: pd.DatetimeIndex, d: pd.Timestamp):
    i = sessions.searchsorted(d)
    return sessions[i] if i < len(sessions) else None


def _session_on_or_before(sessions: pd.DatetimeIndex, d: pd.Timestamp):
    i = sessions.searchsorted(d, side="right") - 1
    return sessions[i] if i >= 0 else None


def map_to_session(sessions, date, time_et) -> tuple:
    """(session affected, time within it or None if pre-open)."""
    d = pd.Timestamp(date).normalize()
    t = str(time_et or "00:00")
    after_close = t >= "16:00"
    if d in sessions and not after_close:
        return d, (t if "09:30" <= t < "16:00" else None)
    nxt = _session_on_or_after(sessions, d + pd.Timedelta(days=1) if (after_close or d in sessions) else d)
    return nxt, None


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def verified_events() -> pd.DataFrame:
    v = pd.read_csv(VERIFIED, dtype=str)
    v["date"] = pd.to_datetime(v["date"])
    v["detail"] = v["ref_period"].fillna("")
    return v[["date", "time_et", "event", "detail", "source"]]


def rule_events(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    hol = set(_holidays(sessions))
    s = pd.Series(sessions, index=sessions)

    # Jobless claims: Thursday 08:30; Wednesday when Thursday is a holiday.
    for d in pd.date_range(sessions.min(), sessions.max(), freq="W-THU"):
        day = d - pd.Timedelta(days=1) if d in hol else d
        rows.append((day, "08:30", "claims", "", "rule: Thursday 08:30"))

    # Month-level events.
    for (y, m), ms in s.groupby([s.index.year, s.index.month]):
        ms = ms.sort_index()
        rows.append((ms.index[0], "10:00", "ism_mfg", "", "rule: 1st session 10:00"))
        if len(ms) >= 3:
            rows.append((ms.index[2], "10:00", "ism_svc", "", "rule: 3rd session 10:00"))
        rows.append((ms.index[-1], None, "month_end", "", "rule: last session of month"))
        if m in (3, 6, 9, 12):
            rows.append((ms.index[-1], None, "quarter_end", "", "rule: last session of quarter"))
        third_fri = pd.Timestamp(y, m, 1) + pd.offsets.WeekOfMonth(week=2, weekday=4)
        opex = _session_on_or_before(sessions, third_fri)
        if opex is not None and opex.month == m:
            rows.append((opex, None, "opex", "", "rule: 3rd Friday (earlier if holiday)"))
            if m in (3, 6, 9, 12):
                rows.append((opex, None, "quad_witching", "", "rule: quarterly opex"))
        # VIX expiry: the Wednesday 30 days before the following month's
        # 3rd Friday (if that Friday is a holiday, the session before it).
        nm = pd.Timestamp(y, m, 1) + pd.offsets.MonthBegin(1)
        nxt_fri = nm + pd.offsets.WeekOfMonth(week=2, weekday=4)
        anchor = nxt_fri if nxt_fri not in hol else _session_on_or_before(sessions, nxt_fri - pd.Timedelta(days=1))
        vexp = anchor - pd.Timedelta(days=30)
        vexp = _session_on_or_before(sessions, vexp) if vexp not in sessions else vexp
        if vexp is not None and sessions.min() <= vexp <= sessions.max():
            rows.append((vexp, None, "vix_expiry", "", "rule: Wed 30d before next 3rd Fri"))

    # Sessions next to a holiday.
    for h in hol:
        before = _session_on_or_before(sessions, h)
        after = _session_on_or_after(sessions, h)
        if before is not None:
            rows.append((before, None, "pre_holiday", str(h.date()), "rule"))
        if after is not None:
            rows.append((after, None, "post_holiday", str(h.date()), "rule"))
    return pd.DataFrame(rows, columns=["date", "time_et", "event", "detail", "source"])


def earnings_events(tickers=MEGACAPS) -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for tk in tickers:
        try:
            e = yf.Ticker(tk).get_earnings_dates(limit=16)
        except Exception as ex:                       # keep going without it
            print(f"  earnings {tk}: {str(ex)[:80]}")
            continue
        for ts in e.index:
            ts = pd.Timestamp(ts).tz_convert("America/New_York").tz_localize(None)
            hhmm = ts.strftime("%H:%M")
            # 00:00 means "time unknown": treat as pre-open, the conservative
            # reading for a feature (it can only make the flag earlier).
            rows.append((ts.normalize(), "16:00" if hhmm >= "16:00" else hhmm,
                         "megacap_earnings", tk, "yahoo earnings dates"))
    return pd.DataFrame(rows, columns=["date", "time_et", "event", "detail", "source"])


def auction_events(start, end) -> pd.DataFrame:
    params = {
        "filter": f"auction_date:gte:{pd.Timestamp(start):%Y-%m-%d},"
                  f"auction_date:lte:{pd.Timestamp(end):%Y-%m-%d}",
        "fields": "auction_date,security_type,original_security_term,closing_time_comp",
        "page[size]": "1000", "sort": "auction_date",
    }
    url = f"{FISCAL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.loads(r.read())["data"]
    rows = []
    for a in data:
        if a["security_type"] in ("Note", "Bond") and a["original_security_term"] in AUCTION_TERMS:
            t = pd.to_datetime(a["closing_time_comp"], format="%I:%M %p").strftime("%H:%M") \
                if a.get("closing_time_comp") else "13:00"
            rows.append((pd.Timestamp(a["auction_date"]), t, "auction_long",
                         a["original_security_term"], "fiscaldata.treasury.gov"))
    return pd.DataFrame(rows, columns=["date", "time_et", "event", "detail", "source"])


# ---------------------------------------------------------------------------
# Build + features
# ---------------------------------------------------------------------------

def build(start, end, fetch: bool = True) -> pd.DataFrame:
    sessions = trading_sessions(start, end)
    parts = [verified_events(), rule_events(sessions)]
    if fetch:
        parts.append(earnings_events())
        parts.append(auction_events(start, end))
    ev = pd.concat(parts, ignore_index=True)
    mapped = [map_to_session(sessions, d, t) for d, t in zip(ev["date"], ev["time_et"])]
    ev["session"] = [m[0] for m in mapped]
    ev["session_time"] = [m[1] for m in mapped]
    ev = ev.dropna(subset=["session"])
    ev = ev[(ev["session"] >= pd.Timestamp(start)) & (ev["session"] <= pd.Timestamp(end))]
    ev = ev.sort_values(["session", "session_time"], na_position="first").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(OUT / "events.parquet", index=False)
    pd.Series(sessions).to_frame("session").to_parquet(OUT / "sessions.parquet", index=False)
    return ev


def load() -> pd.DataFrame:
    p = OUT / "events.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run: python3 -m odte.data.calendar build")
    return pd.read_parquet(p)


def features_for(events: pd.DataFrame, day: pd.Timestamp, hhmm: str) -> dict:
    """Calendar features for a decision at `hhmm` on session `day`.

    Day flags say what is on today. The *_ahead features count scheduled
    events still to come before the close, the ones the market cannot yet
    have reacted to. Only the schedule is used, never the outcome, so every
    value is known before the open.
    """
    day = pd.Timestamp(day).normalize()
    e = events[events["session"] == day]
    out = {f"ev_{k}": 0.0 for k in FLAG_EVENTS}
    for k in e["event"]:
        if k in FLAG_EVENTS:
            out[f"ev_{k}"] = 1.0
    if out["ev_fomc_sep"]:
        out["ev_fomc"] = 1.0                      # an SEP meeting is an FOMC day
    out["megacap_earnings"] = float((e["event"] == "megacap_earnings").sum())
    timed = e[e["session_time"].notna()]
    ahead = timed[timed["session_time"] > hhmm]
    out["events_ahead"] = float(len(ahead))
    out["fomc_ahead"] = float(ahead["event"].isin(["fomc", "fomc_sep"]).any())
    out["auction_ahead"] = float((ahead["event"] == "auction_long").any())
    if len(ahead):
        nxt = min(ahead["session_time"])
        out["minutes_to_next_event"] = float(
            (pd.Timestamp(f"2000-01-01 {nxt}") - pd.Timestamp(f"2000-01-01 {hhmm}")).seconds / 60)
    else:
        out["minutes_to_next_event"] = float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--start", default="2025-10-01")
    b.add_argument("--end", default="2026-11-30")
    s = sp.add_parser("show")
    s.add_argument("--day", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        ev = build(a.start, a.end)
        print(ev.groupby("event").size().to_string())
        print(f"{len(ev)} events over {ev['session'].nunique()} sessions -> {OUT/'events.parquet'}")
    else:
        ev = load()
        print(ev[ev["session"] == pd.Timestamp(a.day)][["event", "session_time", "detail", "source"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
