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

## Frozen batch (2026-11-10, before the run)
- `docs/strategy_zoo_catalog.json`: 80 entries; **66 testable rules** (58 from public
  sources, compiled by a research agent, plus 8 `house` ideas from this project's
  findings). sha256 prefix 39295746a84fc88d.
- Sources: `docs/strategy_zoo_sources.md`. Reddit blocks automated reading, so no
  subreddit is covered; Zhihu and Xueqiu were blocked too (not worked around).
  13 entries come from Chinese sources (8 testable).
- Breakeven-stop condors (MEIC/BEIC) are encoded as a whole-position stop of 0.2 × credit
  (the engine has no per-side stops). The literal 1.0 variant is kept as its own entry.
- Running count: 341 + 66 = 407.

## Result (run once, 2026-11-10): 0 of 66 survive
- 2,041 days; today's ATM-straddle relative half-spread 0.67%. FDR passes: 0. Mean > 0:
  18 rules. Positive in both eras: 14. Positive at the touch: 11.
- **Best (none significant; with 66 rules, |t| ≈ 2.4 is the expected best by luck):**
  - `house_with_move_pcs_1200` (credit spread with the morning move at 12:00): +$16.5/day,
    t 2.2, both eras positive, touch +$27.8k. Its mirror `house_against_move_pcs_1200`
    is −$9.7/day.
  - `dev_short_straddle_1430` (Dim-Eraker-Vilkov intraday VRP, as a 3σ-capped fly):
    +$11.3/day, t 1.8. The 13:00 and 15:00 versions are also positive.
  - `vilkov_bull_call_spread_lowvix` +$15.6/day (t 1.2).
  - `house_quiet_morning_ifly_1100` +$9.7/day; `house_low_vix_ifly_1000` +$8.8/day.
- **Popular practitioner rules lose after costs:**
  - tastylive-style ATM iron flies with 10-20% profit targets: −$110 to −$132/day;
  - every OTM iron condor variant (11): negative;
  - MEIC/BEIC: positive at half-gap, negative at the touch.
- **Premium buying loses, as expected:** lottery strangles −$256/day; Chinese "buy OTM at the
  open, exit early" rules −$130 to −$509/day; late-day butterflies negative.
- By family, only the intraday short-straddle (VRP), breakeven multi-entry and house
  families average above zero.
- Verdict: no new shadow slots. Directional credit spreads are already shadowed by
  `side_seller` (edge #1). Running count: 407.
