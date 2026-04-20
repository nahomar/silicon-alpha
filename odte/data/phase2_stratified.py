"""Phase-2 stratified-day selection for the Databento OPRA pull.

Picks ~18 trading days (15 training + 3 held-out eval) spread across
2022-2025, stratified by EVENT TYPE rather than VIX level. Event-type
stratification is more robust than vol-level stratification because:

  (a) event dates are a matter of public record and verifiable
  (b) vol levels on "event" days are downstream of the event — letting
      the model see both the event type and the regime it induces
  (c) a 524M transformer needs to see a diversity of CAUSAL regimes,
      not just a diversity of outcomes

Regime coverage targets:

  - Non-event baseline days   (4) — "normal market" reference
  - FOMC decision days        (4) — reaction to monetary-policy shocks
  - CPI release days          (2) — inflation-print vol spikes
  - OPEX / quad-witching      (2) — position-squaring mechanics
  - Crisis / shock            (3) — tail-regime coverage

Eval set (3 held out, 2025 only; no overlap with training):
  - 1 baseline, 1 event-driven, 1 crisis candidate

Total ~= 18 days * ~$17.78/day (Databento cmbp-1 SPX+SPXW)  =  ~$320

HONEST CAVEATS:
  - Specific dates below are picked from my event memory, not a live
    VIX feed. Before running, eyeball each date against the actual
    calendar (e.g. FOMC dates at federalreserve.gov/newsevents).
  - "Shock" days are labeled by the EVENT that happened, not by
    realized VIX. The model will learn whatever the day actually was.
  - 3-day eval is tight. If Phase-2 training shows overfit signals,
    expand eval first before scaling training days.

Usage:
    # list what would be pulled (no API call)
    python -m odte.data.phase2_stratified --list

    # cost estimate across all days (API call, zero bytes transferred)
    python -m odte.data.phase2_stratified --cost-only

    # actually pull. Enforces --max-spend-usd across the whole batch,
    # not per-day — one typo'd date can't blow the budget.
    python -m odte.data.phase2_stratified --run --max-spend-usd 400
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DayPick:
    date: str               # ISO YYYY-MM-DD
    split: str              # "train" | "eval"
    regime: str             # "baseline" | "fomc" | "cpi" | "opex" | "shock"
    note: str               # one-liner rationale


# ---------------------------------------------------------------------------
# The curated list. Edit in place if a date turns out to be a holiday
# or if the user wants different regime coverage.
# ---------------------------------------------------------------------------

STRATIFIED_DAYS: List[DayPick] = [
    # --- baseline (non-event) training days --------------------------------
    DayPick("2022-01-18", "train", "baseline", "post-MLK, no major calendar event"),
    DayPick("2023-07-12", "train", "baseline", "mid-summer low-event day"),
    DayPick("2024-04-23", "train", "baseline", "ordinary Tuesday, no scheduled release"),
    DayPick("2024-10-08", "train", "baseline", "ordinary October session"),

    # --- FOMC decision days ------------------------------------------------
    DayPick("2022-05-04", "train", "fomc", "+50bps hike (first 50 of the cycle)"),
    DayPick("2023-03-22", "train", "fomc", "+25bps after SVB collapse"),
    DayPick("2024-03-20", "train", "fomc", "hold, dovish Powell"),
    DayPick("2024-09-18", "train", "fomc", "-50bps surprise (first cut of cycle)"),

    # --- CPI release days --------------------------------------------------
    DayPick("2022-09-13", "train", "cpi", "8.3% print shock — SPX -4%"),
    DayPick("2023-10-12", "train", "cpi", "sticky CPI 3.7% YoY"),

    # --- OPEX / quad-witching ----------------------------------------------
    DayPick("2023-06-16", "train", "opex", "quad-witching Friday"),
    DayPick("2024-12-20", "train", "opex", "year-end quad-witching"),

    # --- crisis / shock ----------------------------------------------------
    DayPick("2022-02-24", "train", "shock", "Russia invades Ukraine"),
    DayPick("2023-03-13", "train", "shock", "SVB aftermath Monday"),
    DayPick("2024-08-05", "train", "shock", "yen carry-trade unwind, VIX 65 intraday"),

    # --- held-out eval (2025 only, no training-set overlap) ----------------
    DayPick("2025-01-29", "eval", "fomc", "first FOMC of 2025"),
    DayPick("2025-02-11", "eval", "baseline", "mid-February low-event"),
    DayPick("2025-04-04", "eval", "shock", "post-tariff-announcement vol"),
]


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def print_list() -> None:
    print(f"{'date':<12} {'split':<6} {'regime':<10} note")
    print("-" * 80)
    for p in STRATIFIED_DAYS:
        print(f"{p.date:<12} {p.split:<6} {p.regime:<10} {p.note}")
    train = sum(1 for p in STRATIFIED_DAYS if p.split == "train")
    ev = sum(1 for p in STRATIFIED_DAYS if p.split == "eval")
    print(f"\ntotal: {len(STRATIFIED_DAYS)} days  ({train} train + {ev} eval)")


def cost_survey(symbols: List[str] = ("SPX", "SPXW")) -> dict:
    """Call Databento get_cost() for each day; return total + per-day."""
    from .databento_pack import DatabentoFetcher
    fetcher = DatabentoFetcher()
    total = 0.0
    per_day = []
    for p in STRATIFIED_DAYS:
        # A trading day is 24h in Databento's eyes; end is next calendar day.
        from datetime import datetime, timedelta
        start = p.date
        end = (datetime.strptime(p.date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            est = fetcher.cost_estimate(start, end, list(symbols))
            per_day.append((p, est["cost_usd"], est["gb"]))
            total += est["cost_usd"]
        except Exception as e:
            per_day.append((p, None, None))
            log.warning("cost_estimate failed for %s: %s", p.date, e)
    return {"total_usd": total, "per_day": per_day}


def run_pull(out_dir: Path, raw_dir: Path, symbols: List[str],
             max_spend_usd: float) -> None:
    """Pull every day in STRATIFIED_DAYS and pack into per-day shards.

    Cost is checked in aggregate BEFORE any bytes transfer. If the batch
    total exceeds max_spend_usd, the whole pull is refused — prevents a
    typo from burning the budget.
    """
    from .databento_pack import pack_databento
    survey = cost_survey(symbols)
    total = survey["total_usd"]
    log.info("batch cost estimate: $%.2f across %d days (cap=$%.2f)",
             total, len(STRATIFIED_DAYS), max_spend_usd)
    if total > max_spend_usd:
        raise RuntimeError(
            f"Batch total ${total:.2f} exceeds cap ${max_spend_usd:.2f}. "
            f"Raise --max-spend-usd or reduce STRATIFIED_DAYS."
        )
    # Per-day pull so we get per-day shard files and per-day inspection.
    from datetime import datetime, timedelta
    for p in STRATIFIED_DAYS:
        start = p.date
        end = (datetime.strptime(p.date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        log.info("PULL %s split=%s regime=%s (%s)",
                 p.date, p.split, p.regime, p.note)
        pack_databento(
            start=start, end=end, symbols=symbols,
            raw_dir=raw_dir, out_dir=out_dir / p.split / p.regime / p.date,
            n_buckets=64, shard_rows=1_000_000,
            max_spend_usd=max_spend_usd, skip_cost_check=False,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="print the curated day list and exit")
    ap.add_argument("--cost-only", action="store_true",
                    help="call Databento for per-day cost estimates and totals")
    ap.add_argument("--run", action="store_true",
                    help="actually perform the pull")
    ap.add_argument("--raw-dir", default="data/databento_raw")
    ap.add_argument("--out-dir", default="reports/odte_shards_real")
    ap.add_argument("--symbols", nargs="+", default=["SPX", "SPXW"])
    ap.add_argument("--max-spend-usd", type=float, default=400.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if a.list:
        print_list()
        return
    if a.cost_only:
        survey = cost_survey(a.symbols)
        print(f"{'date':<12} {'split':<6} {'regime':<10} {'$':>8} {'GB':>8}")
        print("-" * 50)
        for p, cost, gb in survey["per_day"]:
            if cost is None:
                print(f"{p.date:<12} {p.split:<6} {p.regime:<10} {'ERR':>8} {'ERR':>8}")
            else:
                print(f"{p.date:<12} {p.split:<6} {p.regime:<10} ${cost:>7.2f} {gb:>7.2f}")
        print(f"\nTOTAL: ${survey['total_usd']:.2f}")
        return
    if a.run:
        run_pull(Path(a.out_dir), Path(a.raw_dir), a.symbols, a.max_spend_usd)
        return
    ap.error("pass one of --list, --cost-only, or --run")


if __name__ == "__main__":
    _cli()
