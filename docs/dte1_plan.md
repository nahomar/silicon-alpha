# 1-day (1DTE) selling: a cost-driven test (fixed 2026-11-11, before any P&L was computed)

- **Why:** on Oct 2025 - Nov 2026 SPX quotes, the 1DTE iron fly cost $5.05 per $1k risk at
  half-gap vs $7.51 for the 0DTE fly ($5.0 vs $11.1 at 15:30). It also collected 12% more
  premium per unit of risk. The catch is overnight gap risk.
- **Trade:** sell the SPX iron fly on the NEXT expiration (wings 1.5 × its straddle), entered at
  each half-hour 10:00-15:30, held to that expiration's settlement.
  - Priced with zoo.trade on the next-expiry quotes (data/odte_next), half-gap and touch, actual
    quoted spreads (k_day = 1), $1k worst case.
  - Only days whose next expiration is the next session.
- **Comparison:** the 0DTE fly at the same entry times, priced the same way (k_day = 1).
- **Primary:** always-sell 1DTE fly at 15:30 (cheapest relative cost, most overnight premium).
  2022-05-16 .. 2026-11-06 = the daily-expiry era. Mean per trade at half-gap with a 90%
  day-bootstrap CI, and vs the 0DTE fly at 15:30 on the same days.
- Reported: every entry time, by year, worst days, touch.
- **PASS** = primary mean > 0 with CI lower > 0 AND above the 0DTE fly at 15:30. Then a model
  for WHEN to sell 1DTE becomes a candidate for round 5 (its own pre-registration). Otherwise
  it is logged and nothing changes.

## Result (2026-11-11): FAIL
- Always-sell 1DTE fly, 2022-05-16 .. 2026-11-06, per $1k risk at half-gap: every entry time near
  $0 (15:30: +$4.9 [−24.8, +33.1]). The 0DTE fly was better midday (13:30: +$41).
- Lower cost did not make 1DTE profitable; overnight risk appears fairly priced.
- **Post-hoc observation:** 15:30 entries on Fridays (expiring Monday) averaged +$88 vs −$18 on
  other days. This matches the documented weekend effect in option prices (Jones & Shemesh, 2018:
  options are overpriced across non-trading days).

## W1: weekend-effect hold-out test (fixed 2026-11-11, before running it)
- **Data never used for this question:** Friday entries 2016-08-22 .. 2022-05-13 (Monday SPXW
  expirations exist from August 2016). The 2022-26 period that suggested it is excluded.
- **Primary:** sell the 1DTE (Monday-expiry) iron fly at 15:30 on Fridays, held to Monday's close.
  Mean per $1k risk at half-gap; PASS = 90% day-bootstrap CI lower bound > 0.
- Reported: every Friday entry time; touch; vs the same 15:30 trade on Monday-Thursday entries in
  the same years; worst trades.
- PASS makes "Friday 1DTE seller" a forward contender (its own freeze), scored from the bot's
  recordings. FAIL means it is logged as a 2022-26 artefact.
