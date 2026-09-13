"""Record the African universe: backfill history once, snapshot from then on.

Two modes, because history and the present arrive differently:

  **backfill** — pull the longest history the vendor will give for every
  tradeable ticker, once. This is what already exists and can be re-fetched at
  any time, so it is safe to overwrite.

  **snapshot** — capture the current state and APPEND. This is the part that
  cannot be recovered later: free vendors serve adjusted history, so a price you
  do not record today is a price you can only ever see again through the lens of
  every subsequent split, dividend and restatement. Point-in-time data has to be
  captured at the point in time.

That asymmetry is the whole reason to start recording now rather than when the
strategy is ready.

## Storage

Parquet under `data/market/`:

    data/market/history/{ticker}.parquet      full OHLCV, overwritable
    data/market/snapshots/{YYYY-MM-DD}.parquet one row per ticker per run

Snapshots are keyed on (captured_at, ticker) and are append-only within a day,
so running twice in one day records two observations rather than corrupting
one. Re-running is always safe.

## What is deliberately NOT done

No forward-filling, no gap repair, no interpolation. A missing day is recorded
as missing. This track has already been bitten twice by fabricated continuity —
a forward-filled stage series that invented 352 zero-variance sessions, and a
parser that silently dropped eleven months in twelve. Raw means raw.
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:  # pragma: no cover
    yf = None
    _HAS_YF = False

ROOT = Path("data/market")
HISTORY = ROOT / "history"
SNAPSHOTS = ROOT / "snapshots"
AUDIT = Path("reports/universe_audit.csv")


class NoUniverse(RuntimeError):
    """The audit has not been run, so there is no verified universe to record."""


def tradeable_universe(audit_path: Path = AUDIT) -> pd.DataFrame:
    """The VERIFIED universe. Never reads the candidate CSV directly.

    Recording candidates rather than survivors would fill the store with
    tickers that return nothing, and their absence would later be
    indistinguishable from a genuine outage.
    """
    if not audit_path.exists():
        raise NoUniverse(
            f"no audit at {audit_path}. Run:\n"
            f"  PYTHONPATH=. python -m chokepoint.universe.audit\n"
            f"The recorder deliberately will not fall back to the candidate "
            f"list — an unverified ticker that returns nothing is "
            f"indistinguishable from a real outage once it is in the store."
        )
    df = pd.read_csv(audit_path)
    ok = df[df.tradeable].copy()
    if ok.empty:
        raise NoUniverse(f"{audit_path} has no tradeable tickers")
    return ok


@dataclass
class RunReport:
    mode: str
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    rows: int = 0
    errors: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = {}

    def summary(self) -> str:
        s = (f"{self.mode}: {self.succeeded}/{self.attempted} tickers, "
             f"{self.rows:,} rows")
        if self.failed:
            s += f", {self.failed} FAILED"
        return s


def backfill(period: str = "max", pause: float = 0.4) -> RunReport:
    """Pull full history for every tradeable ticker. Safe to re-run."""
    if not _HAS_YF:
        raise SystemExit("yfinance not installed")
    uni = tradeable_universe()
    HISTORY.mkdir(parents=True, exist_ok=True)
    rep = RunReport(mode="backfill", attempted=len(uni))

    for _, r in uni.iterrows():
        tk = str(r.ticker)
        try:
            h = yf.Ticker(tk).history(period=period, auto_adjust=False)
            if h is None or h.empty:
                raise ValueError("empty")
            h = h.reset_index()
            h.columns = [str(c).lower().replace(" ", "_") for c in h.columns]
            h["ticker"] = tk
            h["currency"] = r.currency
            # Filesystem-safe: 'AAL.L' -> 'AAL_L'
            h.to_parquet(HISTORY / f"{tk.replace('.', '_')}.parquet",
                         index=False)
            rep.succeeded += 1
            rep.rows += len(h)
            log.info("  %-10s %6d rows  %s → %s", tk, len(h),
                     h.iloc[0, 0].date(), h.iloc[-1, 0].date())
        except Exception as exc:  # noqa: BLE001
            rep.failed += 1
            rep.errors[tk] = f"{type(exc).__name__}: {exc}"
            log.warning("  %-10s FAILED: %s", tk, exc)
        time.sleep(pause)
    return rep


def snapshot(pause: float = 0.25) -> RunReport:
    """Capture current state for every tradeable ticker and APPEND."""
    if not _HAS_YF:
        raise SystemExit("yfinance not installed")
    uni = tradeable_universe()
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    rep = RunReport(mode="snapshot", attempted=len(uni))
    captured_at = pd.Timestamp.utcnow()
    rows = []

    for _, r in uni.iterrows():
        tk = str(r.ticker)
        try:
            fi = yf.Ticker(tk).fast_info
            rows.append({
                "captured_at": captured_at,
                "ticker": tk,
                "last_price": fi.get("lastPrice"),
                "open": fi.get("open"),
                "day_high": fi.get("dayHigh"),
                "day_low": fi.get("dayLow"),
                "prev_close": fi.get("previousClose"),
                "volume": fi.get("lastVolume"),
                "market_cap": fi.get("marketCap"),
                "currency": fi.get("currency"),
                "exchange": fi.get("exchange"),
            })
            rep.succeeded += 1
        except Exception as exc:  # noqa: BLE001
            rep.failed += 1
            rep.errors[tk] = f"{type(exc).__name__}: {exc}"
        time.sleep(pause)

    if not rows:
        log.error("snapshot captured nothing — not writing an empty file, "
                  "since an empty file would read as 'the market was silent'")
        return rep

    df = pd.DataFrame(rows)
    out = SNAPSHOTS / f"{captured_at.date()}.parquet"
    if out.exists():
        # Append rather than overwrite: two runs in a day are two observations.
        df = pd.concat([pd.read_parquet(out), df], ignore_index=True)
    df.to_parquet(out, index=False)
    rep.rows = len(rows)
    log.info("snapshot -> %s (%d rows this run, %d in file)",
             out, len(rows), len(df))
    return rep


def store_status() -> str:
    hist = sorted(HISTORY.glob("*.parquet")) if HISTORY.exists() else []
    snaps = sorted(SNAPSHOTS.glob("*.parquet")) if SNAPSHOTS.exists() else []
    lines = [f"history:   {len(hist)} tickers"]
    if hist:
        total = sum(len(pd.read_parquet(p, columns=["ticker"])) for p in hist)
        lines.append(f"           {total:,} total rows")
    lines.append(f"snapshots: {len(snaps)} days")
    if snaps:
        lines.append(f"           {snaps[0].stem} → {snaps[-1].stem}")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["backfill", "snapshot", "status"])
    ap.add_argument("--period", default="max")
    ap.add_argument("--pause", type=float, default=0.4)
    args = ap.parse_args()

    try:
        if args.mode == "status":
            print(store_status())
            return
        rep = backfill(args.period, args.pause) if args.mode == "backfill" \
            else snapshot(args.pause)
    except NoUniverse as exc:
        raise SystemExit(f"\n{exc}\n")

    print("\n" + rep.summary())
    if rep.errors:
        print("\nfailures:")
        for tk, err in rep.errors.items():
            print(f"  {tk}: {err}")


if __name__ == "__main__":  # pragma: no cover
    main()
