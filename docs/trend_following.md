# Second engine: trend following (pre-registered 2026-11-10)

Written and committed **before any of this data was downloaded or looked at**.
Everything below is fixed: no parameter will be changed after seeing results.

## Why this test
- The options bot loses when markets move hard. Trend following has tended to make
  money in exactly those periods (Hurst, Ooi & Pedersen, *A Century of Evidence on
  Trend-Following Investing*, 2017; Moskowitz, Ooi & Pedersen, *Time Series
  Momentum*, JFE 2012).
- The rule is published, so this is one test of a fixed rule, not a search.
- The rule's parameters were public by 2012, so **2013-01 .. 2026-10 is true
  out-of-sample for the rule**. That is the primary period.

## Data (free)
- Daily adjusted closes (dividends included) from Yahoo Finance via yfinance,
  2003-01-01 .. 2026-10-30. Evaluation ends with October 2026, the last full month.
- 20 ETFs, fixed now, chosen for liquidity and asset-class coverage:
  - stocks: SPY QQQ IWM EFA EEM
  - bonds: SHY IEF TLT TIP LQD
  - commodities: GLD SLV USO DBA DBB
  - currencies: UUP FXE FXY FXA
  - real estate: VNQ
- Cash rate: ^IRX (13-week T-bill yield), daily rate = IRX / 100 / 252, from the
  previous day.
- An ETF joins once it has 252 trading days of history and 60 days for the volatility
  estimate.

## Rule
- Rebalance at each month-end. The signal uses the month-end close; trades happen at
  the **next trading day's close** (no look-ahead).
- Signal per ETF: s = mean of sign(1-month), sign(3-month), sign(12-month) excess
  return (21 / 63 / 252 trading days). Values: −1, −1/3, +1/3, +1.
- Volatility: exponentially weighted daily excess returns, centre of mass 60 days,
  annualised.
- **A, fund version (futures-like, long/short, leverage allowed):**
  weight = s × (0.40 / vol) / N, on excess returns.
- **B, small-account version (long/flat, no leverage, no shorting):**
  weight = max(s, 0) / N; the rest sits in T-bills.
- N = number of ETFs eligible that month. Weights are held constant between
  rebalances (a standard approximation).
- Costs: 10 bp of traded value at each rebalance (conservative for these ETFs).

## Benchmarks
- SPY buy and hold.
- Equal-weight buy and hold of the eligible ETFs, rebalanced monthly.
- 60/40 SPY/IEF, rebalanced monthly.

## Statistics
- Monthly returns. Sharpe = mean / sd × √12 on excess returns.
- 95% CI by moving-block bootstrap: 6-month blocks, 10,000 resamples, seed 0.

## Gates
- **G1, trend exists after publication:** A's net Sharpe over 2013-01 .. 2026-10 has a
  CI lower bound > 0.
- **G2, crash protection:** over the full period, A's mean monthly excess return in
  the worst 10% of SPY months is > 0, with an i.i.d. bootstrap 95% CI lower bound
  > 0.
- **B is report-only:** return, volatility, Sharpe and maximum drawdown against the
  equal-weight benchmark.
- **Verdict:**
  - G1 and G2 pass: trend earns a shadow slot as the second engine (paper first; real
    money only on the owner's go).
  - Only G2 passes: "insurance only", reported as such.
  - Neither passes: closed.

## Known limits, stated in advance
- Survivorship: all 20 ETFs exist today.
- ETFs are not futures. Their fees and roll costs (USO, DBA) are inside the prices,
  which works against the strategy (conservative).
- Yahoo data quality: spot-check large daily moves before running.
- Small accounts face a minimum fee per order (about $1 per order at IBKR). That is
  reported separately as a drag by account size, not inside the 10 bp.

## Test count
- These are 2 gated tests (G1, G2) in a new family (different data and strategy).
- Project running count: 288 + 2 = 290.

## Result (run once, 2026-11-10)
- Data: 20 ETFs, all eligible from 2008; evaluation 2005-01 .. 2026-10. Three daily moves
  above 25% were checked and are real: USO 2020-03-09 and 2020-04-21, and SLV
  2026-01-30 (silver crash, GLD −10.3% the same day).
- **G1 FAIL.** Fund version A after publication (2013-01 .. 2026-10): net Sharpe −0.05,
  95% CI [−0.61, +0.43]. Even with zero trading costs it was only +0.21 (gross), so
  costs are not the whole story: trend was weak in this period. Full period: net
  +0.01, gross +0.27.
- **G2 PASS.** In the worst 10% of SPY months (27 months, SPY average −7.7%), A averaged
  +2.2%, CI [+0.07%, +4.7%]. Big years: 2008 +20.3%, 2020 +24.9%, 2022 +23.7%.
  Monthly correlation with SPY: −0.18.
- **B (small account, report-only):** CAGR +3.2%, volatility 3.6%, Sharpe 0.41, maximum
  drawdown −8.4%. That is a lower return than equal weight (+5.9%) and 60/40 (+8.2%,
  Sharpe 0.73). Minimum-fee drag: 11%/yr on a $1k account, 2.3% on $5k, 0.45% on
  $25k.
- Costs: A turns over about 36× a year at about 3.8× gross leverage, so 10 bp costs
  about 3.6%/yr.
- **Verdict: insurance only.** It reliably made money in crashes, but did not make
  money overall after 2013. It does not earn a shadow slot as a money engine. Possible
  later use, only with enough capital for micro futures: a small crash hedge next to
  the options book, sized as insurance, with its expected cost stated up front.
- Running test count: 290.
