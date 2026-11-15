# Partners for B0: candidates tested one at a time (owner 2026-11-15: "keep testing candidates")

A partner must do two things: make money on its own, and not lose on the same days as B0.
Test M (close momentum, docs/momentum_plan.md) failed the first. Every candidate below is fixed in
writing and committed before its numbers are computed. Development data only (through 2026-11-06).

## G: sell same-day options on gold (GLD) and Treasury bonds (TLT) (fixed 2026-11-15, before computing)

### Why
- Gold and long bonds move for reasons other than stocks (real rates, the dollar, inflation, flight to
  safety), so their bad days need not line up with B0's.
- Option sellers earn a premium in most markets because buyers pay for protection. Whether that premium
  survives the wider spreads of ETF options is the question.
- Data already bought: same-day GLD and TLT options, 1-minute quotes, 2024-11-11 .. 2026-11-06
  (data/odte_etf/{GLD,TLT}/; same-day expiries mostly Mon/Wed/Fri).

### The rule (no fitted parameters)
- On every session with a same-day GLD or TLT expiry, sell the standard iron fly (zoo FLY: iron_fly,
  wings 1.5 sigma, sigma = ATM straddle / 0.8, the same fly B0 trades) at **10:00, 12:00 and 14:00 ET**.
  Each trade sized at $1k worst-case risk (P&L in $ per $1k, as in the stress test).
- Costs: **touch** (full mid-to-touch distance on every leg) plus $1.20 per contract.
- These options settle into shares, so (as in odte/research/e4_spy_itm.py) every leg that is in the money
  at 15:59 is closed then at the touch (+$1.20 per contract); the rest expire worthless (worth their
  intrinsic value at the close if they crossed in the last minute).
- Day P&L = the sum of that day's trades over both underlyings.

### Pass rules
- **G1 (stands on its own):** pooled GLD+TLT daily P&L at the touch:
  - mean > 0 in both halves (2024-11-11 .. 2025-10-31 and 2025-11-01 .. 2026-11-06), AND
  - pooled t-stat of the daily P&L >= 2.0.
- **G2 (helps B0):** on B0's walk-forward days in the same period (reports/stress/stress_spx_daily.parquet;
  G is 0 on days with no GLD/TLT expiry):
  - correlation of the two daily P&Ls <= 0.10, AND
  - the combination, each side scaled to equal daily volatility measured on the first half only, has a
    higher Sharpe than B0 alone AND a lower probability of a losing year (stationary bootstrap: mean
    block 20 days, 10,000 one-year paths of 252 days, same seed for both).
- **Verdict:**
  - PASS (G1 and G2) -> paper contender next to B0 (needs GLD/TLT option quote lines: owner decides).
  - LEAD (both halves > 0, t >= 1.0, and G2) -> shadow-only in paper for forward evidence, no claim of an edge.
  - FAIL otherwise. Reported as it is, no tuning.
- Power, stated up front: two years is short. A real edge with Sharpe 1 reaches t >= 2 only about a third
  of the time, which is why the LEAD tier exists.

### Reported only (not used for the verdict)
- GLD and TLT separately; each decision time; half-gap costs.
- Held to the close ignoring share delivery (upper bound).
- Same rule on USO, EEM and XLF (fewer expiry days).
- Rich-only filter: trade only when the 10:00 straddle exceeds the average |open-to-close move| of the
  previous 20 sessions (both known at 10:00).

Code: odte/research/partner_g.py -> reports/partners/.
