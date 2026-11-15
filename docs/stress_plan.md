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

## Result (2026-11-14): FAIL on 3 of 4 (reports/stress/stress_spx.json)
- **Centre** (walk-forward picks from 2019-06-26; 1,210 trades on 644 of 1,588 SPX sessions;
  touch costs): **+$25.3/day per $1k-per-trade risk** (+$6.4k/yr), Sharpe **0.59**, max drawdown
  −$18.8k.
- **T1 walk-forward: FAIL.**
  - Positive in 5 of 7 full years, but Sharpe −0.95 in 2018-21 (rule > 0.5); 1.02 in 2022-26.
  - By year, $/day per $1k: 2019 +37.7 (half year), **2020 −35.1, 2021 −71.7**, 2022 +8.0,
    2023 +61.2, 2024 +40.3, 2025 +51.0, 2026 +66.2. Exam window +72.5.
  - It works only since SPX got daily expiries (2022); before that it lost. Regime-dependent.
  - Bug fixed before reading the verdict: the first run counted only years with >= 200 sessions
    (dropping 2020-21, which have ~160 Mon/Wed/Fri sessions). The verdict is FAIL either way.
- **T2 costs and fills: FAIL.**
  - Costs are not the problem: +$29.0 half, +$25.3 touch, +$21.7 at 1.5x, +$18.0 at 2x.
  - Missing 20% of the winning trades: −$15.7/day (median of 1,000 draws). The edge needs nearly
    every winner. The bot's order rule (cross after 5 minutes) is built not to miss trades, so
    this is less of a live risk than the number suggests, but the rule is as written.
- **T3 Monte Carlo: FAIL.**
  - P(losing year) **25%** (rule <= 20%). One-year P&L: median +$6.6k, 5th percentile −$10.1k.
  - Max drawdown: actual −$18.8k; reshuffled days median −$19.2k, 95th −$30.7k, 99th −$36.8k;
    one-year 99th −$23.2k (all per $1k-per-trade sizing; one SPX contract is about 1.4x).
- **T4 sensitivity: PASS (plateau).**
  - Every one-step neighbour is positive with Sharpe >= 50% of the centre's 0.59.
  - Several neighbours are better (alpha 3000-10000, threshold +10-20, more trades per day:
    1/day 0.34, 2/day 0.59, 3/day 0.76, all 1.20). Choosing one of them now would be fitting to
    this table; any change needs its own pre-registered test on unseen days.
- **Correction to E1's absolute numbers:** E1 took its picks from the final locked model, which was
  trained on data through 2026-09-07, so for 2024-11 .. 2026-09 the picks were in-sample. Its SPX vs
  XSP vs SPY cost comparison (same picks on all three, paired) stands; its absolute P&L (SPX
  +$130 per $1k per trade, and the +$32.9k/yr per contract quoted to the owner on 2026-11-14) is
  inflated. The walk-forward figure for 2022-26 here is about +$45.6/day per $1k-per-trade.
- **Verdict:** not robust enough for real money. Paper trading continues as the forward judge.
