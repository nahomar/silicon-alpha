# Stress test of the SPX fly strategy (fixed 2026-11-14, before any of these numbers were computed)

Owner: run what quant funds run before trusting a backtest: worse costs and fills, Monte Carlo
drawdowns, walk-forward, parameter sensitivity.

## The strategy under test
- The locked model's spec (B0: ridge alpha 1000 on calendar + gex + term + flow + identity, the
  four sell shapes, best predicted > 0, never within 60 min after an event), refitted walk-forward
  on the development matrix (every 63 sessions, only on earlier days; first fit after 500 sessions).
- **Traded:** SPX only, only when the best predicted shape is the **fly**, at most **2 per day**
  (the first two), held to the close. P&L per $1k worst-case risk per trade (one SPX contract is
  about $1.4k). Costs: **touch** (full mid-to-touch distance on every leg) unless stated.
- Code: odte/research/stress_spx.py -> reports/stress/. Development data only (through 2026-11-06).

## Tests and pass rules
- **T1 walk-forward by year:** each year is predicted only by models fitted on earlier days.
  - PASS if positive in at least 70% of full calendar years, AND annualised Sharpe > 0.5 in both
    2018-2021 and 2022-2026.
  - Also reported: the exam window (2026-09-08 .. 11-06), never used to choose the spec.
- **T2 worse costs and fills:**
  - (a) costs 1.5x touch (each trade's return extrapolated linearly in the cost share from its
    half-gap and touch labels);
  - (b) at touch, 20% of the winning trades missed at random (adverse selection: E2 found the
    trades that fill easily are the worse ones), 1,000 draws, median reported;
  - (c) 2x touch, reported only.
  - PASS if the mean daily P&L stays > 0 under (a) and under (b).
- **T3 Monte Carlo (daily P&L at touch):**
  - 10,000 reshuffles of the days: distribution of max drawdown (median, 95th, 99th percentile) vs
    the drawdown that actually happened.
  - 10,000 one-year paths by stationary block bootstrap (mean block 20 trading days, keeps
    calm/volatile spells together): probability of losing money over a year, 5th percentile of
    the year's P&L, 95th and 99th percentile of the year's max drawdown.
  - PASS if P(losing year) <= 20%. The 99th percentile drawdown is the sizing number.
- **T4 parameter sensitivity:** grid around the centre (alpha 1000, threshold 0, 2 per day, wing
  1.5 sigma):
  - alpha {100, 300, 1000, 3000, 10000};
  - threshold {-20, -10, 0, +10, +20, +40} ($ per $1k predicted);
  - max per day {1, 2, 3, all};
  - wing {1.0, 1.5, 2.0} x sigma (the narrow / standard / wide fly labels; the model still decides
    with the standard fly's prediction).
  - One setting changed at a time, plus the alpha x threshold grid.
  - PASS (plateau, not a spike) if every one-step neighbour of the centre has a positive mean and
    Sharpe >= 50% of the centre's.
- **Overall:** all four PASS -> "robust on history"; the strategy goes to paper as the order
  strategy (owner's yes needed). Any FAIL is reported as it is, and names what to fix.
  These are all on development data; the forward paper test remains the judge.
