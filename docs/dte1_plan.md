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
