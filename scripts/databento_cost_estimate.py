"""Estimate Databento cost for a candidate OPRA pull plan.

Runs cost_estimate() on each (day, symbols) pair — no money spent; just
prints the estimate. Edit DAYS + SYMBOLS below, run:

    cd /Users/nahom/silicon-alpha && set -a && . ./.env && set +a && \
    python3 -m scripts.databento_cost_estimate

Output: per-day cost, running total, fit-under-budget check.
"""
from __future__ import annotations

import datetime as _dt
import os

import databento as db

BUDGET_USD = 115.54

# Candidate days — maximum recency. Today is Fri 2026-04-24; Databento
# data horizon is T-1, so 2026-04-23 is the latest available.
# Train: last 5 US business days leading up to today.
# Eval: one day from the previous week, genuinely held out in time.
#
# Notable events: 2026-04-17 is April OPEX (3rd Friday). FOMC April is
# scheduled 2026-04-28/29, so our corpus is pre-FOMC and captures
# "into the event" positioning by market makers.
DAYS = [
    ("2026-04-16", "held-out eval (day before OPEX)"),
    ("2026-04-17", "April OPEX Friday"),
    ("2026-04-20", "post-OPEX Monday"),
    ("2026-04-21", "Tuesday"),
    ("2026-04-22", "Wednesday"),
    ("2026-04-23", "Thursday (latest available)"),
]

# Parent symbology: SPX = monthly index, SPXW = weekly/daily (0DTE).
# Both needed for real 0DTE coverage.
# Metadata API requires explicit `.OPT` suffix (the batch-submit API
# auto-translates but cost_estimate does not).
SYMBOLS = ["SPX.OPT", "SPXW.OPT"]


def main() -> None:
    assert os.environ.get("DATABENTO_API_KEY"), (
        "DATABENTO_API_KEY not set; source .env first"
    )
    client = db.Historical()
    total = 0.0
    lines: list[tuple[str, str, float, float]] = []
    print(f"Checking cost for {len(DAYS)} days x {SYMBOLS} "
          f"(OPRA.PILLAR, cmbp-1, parent stype)...")
    print()
    for day, note in DAYS:
        d0 = _dt.date.fromisoformat(day)
        d1 = d0 + _dt.timedelta(days=1)
        start = f"{d0.isoformat()}T00:00:00"
        end = f"{d1.isoformat()}T00:00:00"
        try:
            est = client.metadata.get_cost(
                dataset="OPRA.PILLAR",
                schema="cmbp-1",
                start=start,
                end=end,
                symbols=SYMBOLS,
                stype_in="parent",
            )
            gb = client.metadata.get_billable_size(
                dataset="OPRA.PILLAR",
                schema="cmbp-1",
                start=start,
                end=end,
                symbols=SYMBOLS,
                stype_in="parent",
            ) / 1e9
        except Exception as e:
            print(f"  {day}  {note!r}  ERROR: {e}")
            continue
        lines.append((day, note, float(est), gb))
        total += float(est)
        fits = "OK" if total <= BUDGET_USD else "OVER"
        print(f"  {day}  {note:<22}  ${float(est):>6.2f}  {gb:>6.2f} GB"
              f"   running=${total:.2f}  {fits}")

    print()
    print(f"TOTAL estimate: ${total:.2f}")
    print(f"BUDGET        : ${BUDGET_USD:.2f}")
    print(f"HEADROOM      : ${BUDGET_USD - total:+.2f}")
    if total > BUDGET_USD:
        drop = total - BUDGET_USD
        print(f"Over budget by ${drop:.2f} — drop last-listed days until fits.")


if __name__ == "__main__":
    main()
