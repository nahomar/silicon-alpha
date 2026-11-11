# Strategy zoo: disclosed 0DTE strategies, one batch (pre-registered 2026-11-10)

Written before any zoo strategy was run on real data. Only a speed check (3 days, trade
counts, no P&L) was done.

## Batch
- The batch is exactly the testable entries of `docs/strategy_zoo_catalog.json`, as
  committed together with this plan's final update. That catalog was compiled from public
  sources by a research agent, plus a few "house" ideas marked `family: house`. Nothing
  may be added, removed or changed after the run. A bug found after the run is
  documented, fixed, rerun, and both runs are reported.
- Engine: `odte/research/zoo.py`.
  - SPX 0DTE minute quotes 2016-01 .. 2026-09-04 (seen; the exam stays locked).
  - Legs at mid plus `share` × each leg's half-spread, scaled to today's level by the
    day's factor (today's median ATM-straddle relative spread / that day's at 10:00,
    clipped to [0.1, 2]).
  - $1.20 per contract per side.
  - Cash settlement unless an exit fires (minute mids).
  - $1,000 of worst-case loss per trade. Multiple entries per day are summed.
  - Days a rule does not trade count as $0.

## Gate (whole batch at once)
- Per rule, the daily P&L series at half-gap (`share` 0.5): t = mean / (sd / √days),
  normal p, BH-FDR q ≤ 0.05 across all rules, mean > 0;
- AND a positive total in both 2016-21 and 2022-26;
- AND a positive total paying the full touch (`share` 1.0), the adverse-selection lesson
  of edge #14.
- **Survivors earn a live shadow slot only.** Real money follows `docs/go_no_go.md`.

## Report
- The full table (mean/day, t, p, totals by era, touch total, trade days), and family-level
  summaries.
- Running count: 341 + number of rules.
