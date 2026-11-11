# SEC filings research (pre-registered 2026-11-10)

First principles: profit comes from risk premia, liquidity provision, better information
processing, or trading against forced traders. 407 tests covered the first two at 0DTE.
This program tests **information processing where attention is scarce**: company filings
that few people read, which software can read in full.

Data:
- SEC EDGAR (free; contact User-Agent kept in a private local file, never in this
  public repo);
- daily prices including delisted stocks (Databento XNAS.ITCH ohlcv-1d, 2018-05 ..
  2026-09, about $30; bought only with the owner's explicit approval).

Written before any price data was bought or any return computed. Text features
(below) contain no outcome information.

## H1: "Lazy Prices" (Cohen, Malloy & Nguyen, Journal of Finance 2020)
- **Feature:** for each 10-K, the Jaccard similarity of its word set with the same
  company's previous 10-K (filed 9-15 months earlier). Estimated with a 256-permutation
  MinHash (`filings/text.py`). Ticker and exchange come from the 10-K's inline-XBRL
  cover page (dei:TradingSymbol, required from mid-2019).
- **Universe** at each month-end t:
  - companies with a similarity from a 10-K filed in months t−2 .. t, a ticker priced on
    the Databento daily file, price ≥ $5 at t;
  - median daily dollar volume ≥ $1M over the prior 21 sessions.
- **Portfolios:** quintiles of similarity within that universe each month-end.
  - Long Q5 (least changed) and short Q1 (most changed), equal weight, held one month,
    rebalanced monthly.
  - A stock that stops trading earns its last price (that is, cash).
  - Split-like jumps (overnight ratios within 2% of 2, 3, 4, 5, 10, 1/2, 1/3, 1/4, 1/5,
    1/10) are treated as splits, not returns.
- **Costs:** 20 bp per unit of one-way turnover.
- **Period:** months 2019-09 .. 2026-08 (after publication: the paper circulated from
  2016 and was published 2020).
- **Gate (one test):** the L/S monthly mean after costs > 0 with a Newey-West (3 lags)
  t ≥ 2.0, AND positive in both halves (split 2023-01).
- **Report:**
  - the long-only Q5 minus the equal-weight universe (implementable without shorting);
  - the quintile monotonicity and turnover;
  - the share of the universe with a cover-page ticker.

## Later (each pre-registered separately, before running)
- **H2:** AI-read tone and guidance change in 8-K earnings releases vs. the price reaction
  (underreaction drift).
- **H3:** sector growth / shrink from XBRL financial data (fundamental momentum by sector).

## H1 clarifications (2026-11-11, before any return was computed)
- **Price file:** Databento XNAS.ITCH ohlcv-1d, every symbol (bought 2026-11-10, $30.08). Tickers
  come from the free symbology map: one stable instrument_id per symbol, delisted symbols included.
- **No same-day look-ahead:** a 10-K counts for the month-end formation date t only if it was
  filed strictly before t.
- **Ticker:** only the filing's own cover-page ticker (no carrying forward from older filings).
- **Costs:** 0.20% × Σ|Δweight| for each leg at each rebalance (every share bought or sold
  pays 20 bp).
- **Returns:** each month from the last close on or before month-end t to the last close on or
  before month-end t + 1. Daily ratios within 2% of a split ratio count as splits.

## H1 result (run once, 2026-11-11): FAIL
- Data: 81 formation months (2019-08 .. 2026-07); on average 527 stocks qualified each month.
  55,398 year-over-year 10-K pairs; 28,831 have a cover-page ticker.
- **Long/short (least changed minus most changed), after costs:** +0.25%/month (+3.0%/yr),
  Newey-West t 0.63. Positive in both halves (+0.03% / +0.44% per month), but nowhere near
  significant.
- **The paper's direction does show up, in the "changers" only:** quintile mean monthly
  returns are Q1 (most changed) 0.47%; Q2 1.03%; Q3 1.04%; Q4 0.96%; Q5 1.11%.
  Q1 lags the rest by about 0.55%/month before costs.
- **Costs:** 0.39%/month for the two legs (each month's portfolio is mostly new names),
  which eats most of the spread.
- **Long-only** (Q5 vs. all qualifying stocks): 0.00%/month.
- Verdict: closed as pre-registered. A possible later, separately pre-registered variant:
  "avoid / short the most-changed" with lower turnover (hold 6-12 months). The original
  paper reports the effect lasting that long.
