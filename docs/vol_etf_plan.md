# Volatility premium through an ETF (pre-registered 2026-11-10)

Written before any of this data was downloaded or looked at in this project.

## Why
- Every options test so far found the same thing: the volatility risk premium is real,
  but in 0DTE options it sits inside the bid/ask gap (about 1.7% of the fly's price, and
  adverse selection on top).
- SVXY (−0.5× short VIX futures since 2018-02-28, −1× before) earns the same kind of
  premium through the VIX futures roll, at an ETF spread of a few basis points.
- Timing rule from the literature (Simon & Campasano 2014, "The VIX Futures Basis"):
  short volatility only when the VIX curve is in contango. Proxy used here: VIX < VIX3M.

## Data (free, Yahoo)
- SVXY adjusted close from 2011-10 to 2026-10-30; ^VIX, ^VIX3M, SPY; ^IRX for cash.

## Rule (fixed)
- **V1:** at each close d, compute r = VIX(d) / VIX3M(d). Trade at the NEXT close (d + 1,
  a one-day lag, conservative), so the position holds SVXY for the return from close
  d + 1 to close d + 2 if r < 1.0; otherwise T-bills.
- **V2 (report only):** the same with r < 0.90.
- Costs: 5 bp of value each time the position switches.

## Gate (one test)
- V1 net excess return over T-bills, monthly Sharpe, 6-month block bootstrap 95% CI
  lower bound > 0, over **2018-03 .. 2026-10** (the fund as it exists today, −0.5×)
  AND a positive excess return over the full period.

## Report
- CAGR, volatility, Sharpe, maximum drawdown, worst day and worst month for V1, V2,
  SVXY buy-and-hold and SPY buy-and-hold.
- Time in the market.
- The largest losses (e.g. Feb 2018, Mar 2020, Aug 2024).

## Known limits
- SVXY's leverage change in 2018; Yahoo data quality (spot-check big moves).
- The one-day lag misses same-day exits on crash days, on purpose: that is the realistic
  risk.
- Running count: 340 + 1 = 341.

## Result (run once, 2026-11-10)
- SVXY daily moves above 25% (real): 2016-06-24 (Brexit) −26%; 2018-02-05 −32% and
  2018-02-06 −83% (Volmageddon).
- **FAIL.** V1 after 2018-03: CAGR +7.0%, Sharpe 0.29, 95% CI [−0.36, +1.05]; maximum
  drawdown −56%.
- The full period passes only because of 2011-17, the old short-volatility era: Sharpe
  0.57 [0.09, 1.06].
- V2 (ratio < 0.90, report only): avoided the crashes (Feb 2018 +0.1%, Mar 2020 0%,
  Aug 2024 +0.3%) but lost money after 2018 (CAGR −2.1%).
- V1 in the crashes: Feb 2018 −40%, Mar 2020 −18%, Aug 2024 −25%. The last is worse than
  buy-and-hold: it took the drop, then sat out the rebound.
- SPY buy-and-hold beat every variant after 2018: CAGR +14.3%, Sharpe 0.74, maximum
  drawdown −24%.
- Verdict: closed. The cheaper ETF route does not rescue the volatility premium after
  2018. This is the same pattern as trend (after 2012) and crypto carry (after 2023):
  the edge was real and has shrunk since.
