"""Audit the candidate universe: which tickers return data, and can be traded.

The CSV beside this file is a CANDIDATE list. This turns it into a verified one
by asking two separate questions of every ticker, because they fail differently:

  1. **Does data come back at all?** Many African listings are simply absent
     from free providers. Nairobi, Lagos, Accra, BRVM and Zimbabwe have no
     coverage worth the name. Absence here is a fact about the data vendor, not
     about the company.

  2. **Is there enough liquidity to act on?** This is the question that
     actually matters and the one usually skipped. A ticker that prints a
     price once a week is not tradeable no matter how clean the series looks.
     Median daily dollar volume is the test; a thin name will show a perfectly
     respectable price history and be impossible to exit.

Both are reported per ticker, and the verdict is the conjunction. A ticker that
returns 4,000 rows of history and trades $12,000 a day has passed (1) and
failed (2), and only the second fact matters for a position.

Output is written to `reports/universe_audit.csv` so the recorder consumes a
verified list rather than the candidate list. Nothing downstream should read
the candidate CSV directly.
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:  # pragma: no cover
    yf = None
    _HAS_YF = False

CANDIDATES = Path(__file__).with_name("africa_tickers.csv")
DEFAULT_OUT = Path("reports/universe_audit.csv")

# Median daily USD volume below which a name is not realistically tradeable
# for anything but a token position. Deliberately low -- this is a floor for
# "can be traded at all", not a threshold for institutional size.
MIN_DOLLAR_VOLUME = 50_000

# Minimum history to support any statistical work.
MIN_ROWS = 250

# Liquidity is measured over a RECENT window, not the full history. A name that
# traded heavily in 2011 and barely trades now is not tradeable now, and a
# lifetime median would hide that entirely.
LIQUIDITY_WINDOW = 120

# Currencies quoted in SUBUNITS. This is the trap: London quotes pence and
# Johannesburg quotes cents, so a raw price*volume in those venues overstates
# turnover by 100x before any FX conversion. Missing this made AAL.L appear to
# trade "8.3 billion" against SBSW's "18 million" -- pence against dollars,
# ranked as though they were the same unit.
SUBUNIT = {"GBp": ("GBP", 100.0), "ZAc": ("ZAR", 100.0),
           "ILA": ("ILS", 100.0), "GBX": ("GBP", 100.0)}


@dataclass
class TickerAudit:
    ticker: str
    name: str
    listing: str
    country: str
    commodity: str
    africa_share: float
    ok: bool = False
    rows: int = 0
    first: str = ""
    last: str = ""
    currency: str = ""
    last_close: float = float("nan")
    median_local_vol: float = float("nan")
    median_usd_vol: float = float("nan")
    fx_to_usd: float = float("nan")
    stale_days: int = -1
    reason: str = ""

    @property
    def liquid(self) -> bool:
        return (not np.isnan(self.median_usd_vol)
                and self.median_usd_vol >= MIN_DOLLAR_VOLUME)

    @property
    def tradeable(self) -> bool:
        """Data exists AND there is enough volume AND it still trades."""
        return (self.ok and self.liquid and self.rows >= MIN_ROWS
                and 0 <= self.stale_days <= 10)


def load_candidates(path: Path = CANDIDATES) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    need = {"ticker", "name", "listing", "country", "commodity", "africa_share"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


_FX_CACHE: dict[str, float] = {}


def fx_rate(currency: str) -> float:
    """Units of USD per one unit of `currency`, subunits handled.

    Uses the CURRENT rate, deliberately. The question this audit answers is
    "can I trade this today", so today's dollars are the right denominator.
    A historical-average rate would be the right choice for a returns study and
    the wrong one here -- Egypt's pound went from ~15 to ~50 per USD over the
    sample, so a lifetime average would materially misstate present liquidity.
    """
    if not currency:
        return float("nan")
    if currency in _FX_CACHE:
        return _FX_CACHE[currency]

    code, divisor = SUBUNIT.get(currency, (currency, 1.0))
    if code == "USD":
        _FX_CACHE[currency] = 1.0 / divisor
        return _FX_CACHE[currency]

    rate = float("nan")
    for sym, invert in ((f"{code}USD=X", False), (f"USD{code}=X", True)):
        try:
            h = yf.Ticker(sym).history(period="5d")
            if h is not None and not h.empty:
                px = float(h["Close"].dropna().iloc[-1])
                if px > 0:
                    rate = (1.0 / px) if invert else px
                    break
        except Exception:  # noqa: BLE001
            continue
    if np.isnan(rate):
        log.warning("no FX rate for %s -- USD volume cannot be computed, so "
                    "this ticker cannot pass the liquidity screen", currency)
    _FX_CACHE[currency] = rate / divisor if not np.isnan(rate) else rate
    return _FX_CACHE[currency]


def _audit_one(row: pd.Series, period: str) -> TickerAudit:
    a = TickerAudit(
        ticker=str(row.ticker), name=str(row["name"]),
        listing=str(row.listing), country=str(row.country),
        commodity=str(row.commodity), africa_share=float(row.africa_share),
    )
    try:
        t = yf.Ticker(a.ticker)
        h = t.history(period=period, auto_adjust=True)
    except Exception as exc:  # noqa: BLE001 - vendor errors are expected here
        a.reason = f"fetch error: {type(exc).__name__}"
        return a

    if h is None or h.empty:
        a.reason = "no data returned"
        return a

    h = h.dropna(subset=["Close"])
    if h.empty:
        a.reason = "all closes NaN"
        return a

    a.ok = True
    a.rows = len(h)
    a.first = str(h.index[0].date())
    a.last = str(h.index[-1].date())
    a.last_close = float(h["Close"].iloc[-1])
    a.stale_days = int((pd.Timestamp.now(tz=h.index.tz) - h.index[-1]).days)

    try:
        a.currency = str(t.fast_info.get("currency") or "")
    except Exception:  # noqa: BLE001
        a.currency = ""

    if "Volume" in h:
        recent = h.tail(LIQUIDITY_WINDOW)
        dv = (recent["Close"] * recent["Volume"]).replace(0, np.nan).dropna()
        a.median_local_vol = float(dv.median()) if len(dv) else float("nan")
        a.fx_to_usd = fx_rate(a.currency)
        if not np.isnan(a.median_local_vol) and not np.isnan(a.fx_to_usd):
            a.median_usd_vol = a.median_local_vol * a.fx_to_usd

    if not a.liquid:
        a.reason = "illiquid"
    elif a.rows < MIN_ROWS:
        a.reason = "short history"
    elif a.stale_days > 10:
        a.reason = f"stale ({a.stale_days}d)"
    return a


def run(period: str = "max", pause: float = 0.4,
        out: Path = DEFAULT_OUT) -> pd.DataFrame:
    if not _HAS_YF:
        raise SystemExit("yfinance not installed: pip install yfinance")

    cands = load_candidates()
    log.info("auditing %d candidate tickers", len(cands))
    audits = []
    for i, row in cands.iterrows():
        a = _audit_one(row, period)
        audits.append(a)
        mark = "OK " if a.tradeable else "-- "
        log.info("%s %-10s %-28s rows=%-6d $vol=%s %s",
                 mark, a.ticker, a.name[:28], a.rows,
                 f"{a.median_usd_vol:>12,.0f}" if not np.isnan(a.median_usd_vol) else "         n/a",
                 a.reason)
        time.sleep(pause)  # vendor rate-limits aggressively; 429s are silent

    df = pd.DataFrame([asdict(a) for a in audits])
    df["tradeable"] = [a.tradeable for a in audits]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("wrote %s", out)
    return df


def report(df: pd.DataFrame) -> None:
    ok = df[df.tradeable]
    print(f"\n{'ticker':<10} {'listing':<8} {'country':<14} {'commodity':<14} "
          f"{'ccy':<5} {'rows':>6} {'median $vol(USD)':>14}")
    print("-" * 80)
    for _, r in ok.sort_values("median_usd_vol", ascending=False).iterrows():
        print(f"{r.ticker:<10} {r.listing:<8} {r.country:<14} {r.commodity:<14} "
              f"{r.currency:<5} {r.rows:>6} {r.median_usd_vol:>14,.0f}")

    print(f"\n{len(ok)} of {len(df)} candidates are tradeable "
          f"(data + >= ${MIN_DOLLAR_VOLUME:,} median daily volume + not stale)")

    failed = df[~df.tradeable]
    if len(failed):
        print("\nExcluded:")
        for reason, grp in failed.groupby(failed.reason.replace("", "unknown")):
            print(f"  {reason}: {', '.join(grp.ticker)}")

    by_listing = ok.groupby("listing").size().to_dict()
    print(f"\nBy listing venue: {by_listing}")
    local = ok[ok.listing == "local"]
    if local.empty:
        print("  NOTE: no locally-listed African company survived. The tradeable")
        print("  universe is entirely foreign listings with African assets, which")
        print("  is a weaker form of the exposure than it looks.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--period", default="max")
    ap.add_argument("--pause", type=float, default=0.4)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    report(run(args.period, args.pause, args.out))


if __name__ == "__main__":  # pragma: no cover
    main()
