# Sweep 1: every idea the data on disk can test (fixed 2026-11-15, before computing)

Owner, 2026-11-15: "have you tried every single possible idea ... even if you think the idea won't make
money at first still test it unless the idea is tested with real data don't rule it out".

Already tested earlier (not repeated here): B0 and its ~420 variants; 80 published 0DTE option strategies
(docs/strategy_zoo_plan.md, 0 survive); 1DTE selling; calendars; trend following on 20 ETFs; crypto
funding carry; volatility ETFs; Kalshi long shots and favorites; SEC filings; Eskom and Zambia supply
probes; a gradient-boosted next-minute direction probe; partners M, G, O, T, B (docs/partner_plan.md).

Every idea below is a fixed rule. Any number in a rule (200 days, 20%, 2x ...) is the conventional value
from where the idea comes from, written here before any result. Nothing is tuned after the run.

## Common rules

### Instruments, costs, units
- **S&P ideas** trade 1 MES ($5 per point). Prices from ES minute bars (data/futures/es_1m.parquet;
  regular hours; NYSE calendar from odte/research/partner_t.nyse_days).
  - Intraday or overnight trades: entry and exit prices from the stated bars, same contract at both ends
    (else skipped). Cost 0.75 points per trade.
  - Positions held for whole days: marked close to close (last regular-hours bar). On a day the ES contract
    changes, the mark is the S&P index's own close-to-close move (data/odte_0dte/settlements.csv).
    Cost 0.375 points per unit of position change (a round trip costs 0.75; a long-to-short flip 0.75).
- **Option ideas** use the library labels (data/research/library_SPX.parquet): return per $1k worst case,
  at the touch (full spread), fees included.
- **Stock ideas** (data/prices/xnas_daily.parquet, closes and volumes of US stocks 2018-05 .. 2026-08):
  equal-weight top decile minus bottom decile, $10k a side, 10 bps per side on traded value at each rebalance.
- **Bitcoin ideas** (data/alt/btc_1m.parquet, 2022-04 .. 2026-11): $10k notional, 0.10% per round trip.
- All signals use only information available before entry.

### Scoring (identical for every idea)
- Daily net P&L series over the idea's trading days (0 on days with no position).
- Each rule is written in one direction, but the test also reads the reverse trade (it pays its own costs).
  The better direction's net t-stat t* gives p = 2 x P(T > t*) (two-sided: the direction is chosen after
  looking).
- **PASS (new edge candidate):** Benjamini-Hochberg false-discovery rate q <= 0.10 across ALL ideas in this
  sweep, AND net mean > 0 in both eras in the chosen direction.
- **LEAD:** p < 0.05 and net mean > 0 in both eras, but not surviving the false-discovery rule. Read as
  "probably luck, watch it forward".
- **FAIL:** otherwise. **NOT SCORED:** fewer than 20 trades in either era (reported with its numbers).
- Eras: 2016-01-04 .. 2021-12-31 and 2022-01-01 .. 2026-11-06. Shorter data use halves: stocks 2018-05 ..
  2021-12 / 2022-01 .. 2026-08; Bitcoin 2022-04 .. 2024-06 / 2024-07 .. 2026-11; VIX1D 2023-04 .. 2024-12 /
  2025-01 .. 2026-11; Reddit 2025-09 .. 2026-03 / 2026-04 .. 2026-11.
- **Partner check** for every PASS and LEAD: add it at equal volatility to B0 + B (buy-and-hold, the current
  best pair). Weights from 2019-06-26 .. 2021-12-31, or the idea's first 252 overlapping B0 days if it
  starts later. It helps only if Sharpe rises AND P(losing year) falls (same bootstrap and seed as
  docs/partner_plan.md).
- A PASS earns a forward paper test, nothing more. Testing ~115 ideas means a few will look good by luck;
  the false-discovery rule is there for that.

## The ideas

### A. Calendar (S&P)
- A1-A5: hold the 09:30 -> 16:00 session (09:30 bar open -> 15:59 bar close) on Mon / Tue / Wed / Thu / Fri.
- A6-A10: hold the night (15:59 close -> next 09:30 open) that ends on Mon / Tue / Wed / Thu / Fri.
- A11 pre-holiday: long close to close on the last trading day before each NYSE holiday.
- A12 post-holiday: long close to close on the first trading day after each NYSE holiday.
- A13 Santa Claus rally: long from the close of the 6th-to-last December trading day to the close of the
  2nd January trading day.
- A14 Halloween: long every trading day November-April, flat May-October.
- A15 monthly options-expiry week: long every trading day of the week holding the 3rd Friday.
- A16 the week after: long every trading day of the following week.
- A17 month-end rebalancing: over the last 3 trading days of each month (close of day -4 -> close of day
  -1), hold the opposite sign of the month-to-date S&P move as of day -4.
- A18 quarter-end rebalancing: the same for quarter ends only.
- A19 pre-FOMC drift (Lucca & Moench 2015): long from the 13:59 bar close the trading day before an FOMC
  statement to the 13:59 bar close on statement day (data/calendar/events.parquet, event "fomc").
- A20 FOMC reaction: on statement days, hold the sign of the 13:59 -> 14:29 move from 14:30 -> 15:59 close.
- A21 FOMC cycle (Cieslak, Morse & Vissing-Jorgensen 2019): long on trading days -1..+3, +9..+13,
  +19..+23, +29..+33 counted from each statement day, flat otherwise.
- A22 CPI days, A23 jobs-report (NFP) days, A24 the day after an FOMC statement, A25 quarterly triple-
  witching days, A26 the trading day after a megacap earnings release: long close to close.

### B. Price patterns (S&P)
- B1 200-day trend: long next day when the close is above its 200-day average, else flat.
- B2 200-day trend long/short: long above, short below.
- B3 10-month rule (Faber): long next month when the month-end close is above the average of the last 10
  month-end closes, else flat.
- B4 12-month time-series momentum: long next month if the 12-month return > 0, else short.
- B5 1-day reversal: next day, hold the opposite sign of today's close-to-close move.
- B6 5-day reversal: next day, hold the opposite sign of the 5-day move.
- B7 three down closes in a row: long next day.
- B8 internal bar strength ((close - low) / (high - low) of today's session from minute closes): < 0.2 ->
  long next day, > 0.8 -> short next day.
- B9 RSI(2) < 10: long next day (Connors).
- B10 close at a 252-day high: long next day.
- B11 gap fade: hold the opposite sign of the night move (prior close -> 09:30 open), 09:30 open -> 15:59 close.
- B12 opening drive: hold the sign of the 09:30 open -> 09:59 close move, 10:00 open -> 15:59 close.
- B13 midday: hold the sign of the 09:30 open -> 11:59 close move, 12:00 open -> 15:59 close.
- B14 30-minute opening-range breakout: after 10:00, the first minute that closes above the 09:30-09:59 high
  -> long, below the low -> short (entry at that bar's close), exit 15:59 close; one trade a day at most.
- B15 day to night: hold the sign of the 09:30 -> 15:59 session move over the following night.
- B16 big-move reversal: after |close-to-close| > 2 x the prior 20-day standard deviation, hold the
  opposite sign next day.

### C. Other markets as signals (data/context/daily.parquet, values known at the prior close; trade the next
S&P day close to close)
- C1 VIX below VIX3M (normal term structure): long, else flat.
- C2 VIX9D below VIX: long, else flat.
- C3 VIX spike (close > 1.2 x its 10-day average): long next day.
- C4 VIX above its trailing 252-day median: long, else flat.
- C5 VVIX above its 252-day median: long, else flat.
- C6 SKEW index above its 252-day median: long, else flat.
- C7 variance premium (VIX minus 21-day realized volatility, annualized %) above its 252-day median: long.
- C8 credit leads: hold the sign of HYG's 5-day return.
- C9 dollar: hold the opposite sign of the dollar index's 5-day return.
- C10 rates: hold the opposite sign of the 10-year yield's 5-day change.
- C11 oil: hold the sign of crude's 5-day return.
- C12 gold: hold the opposite sign of gold's 5-day return.
- C13 Bitcoin: hold the sign of Bitcoin's 1-day return (2014+).
- C14 Bitcoin overnight: hold the sign of Bitcoin's 16:00 -> 09:30 ET move over the 09:30 -> 15:59 S&P
  session (btc_1m; Bitcoin halves).
- C15 VIX1D below VIX: long, else flat (VIX1D halves).

### D. Dealer positioning (library_SPX rows on SPX same-day-expiry days)
- D1 sign of gex_net at 15:30: held from today's close to the next trading day's close.
- D2 sign of gex_net at 10:00: held 10:00 open -> 15:59 close.
- D3 0DTE put/call open-interest ratio at 10:00 above its trailing 252-row median: long 10:00 -> 15:59.
- D4 put skew at 10:00 above its trailing 252-row median: long 10:00 -> 15:59.
- D5 toward the largest-open-interest strike: sign of maxoi_dist at 14:00, held 14:00 open -> 15:59 close.
- D6 above the zero-gamma level (gex_flip_dist < 0): long; below: short; 10:00 open -> 15:59 close.

### E. Option trades on event days (B0 skips these days)
For each event type: cpi, nfp, ppi, jolts, claims, ism_mfg, ism_svc, fomc, fomc_minutes, auction_long, opex,
quad_witching, vix_expiry, month_end, quarter_end, pre_holiday, post_holiday, megacap_earnings
(library ev_* flags; megacap_earnings from events.parquet):
- E-<event>-sell: sell the standard iron fly (y_fly_touch) at 10:00; at 14:30 for fomc and fomc_minutes.
- E-<event>-buy: buy the ATM straddle (y_long_straddle_touch) at the same time.
(The reverse direction of an option trade is not scored here: a sold fly's reverse is not a bought fly
after costs. Each listed side is scored one-sided, p = P(T > t).)

### F. US stocks across the market (monthly rebalance unless stated)
Universe at each rebalance: price >= $5, top 1,000 by 20-day average dollar volume, enough history for the
signal. Split handling: a daily move whose price ratio is within 3% of a common split ratio (1/2, 1/3, 2/3,
1/4, 1/5, 1/10 and their inverses up to 1:50) is treated as a split and removed; other large moves stay.
- F1 momentum: return from 252 to 21 trading days ago; long winners, short losers.
- F2 1-month reversal: last 21-day return; long losers, short winners.
- F3 1-week reversal (weekly): last 5-day return; long losers, short winners.
- F4 low volatility: 63-day volatility; long low, short high.
- F5 MAX: largest daily return of the last 21 days; long low, short high.
- F6 volume shock (weekly): last-5-day volume over the prior 50-day average; long high, short low.
- F7 52-week high: close / 252-day high; long nearest, short farthest.
- F8 liquidity within the universe: 20-day dollar volume; long lowest decile, short highest.
- F9 same-month seasonality (Heston & Sadka): average return in the same calendar month of earlier years
  (>= 2 years); long high, short low.

### G. Alternative data and Bitcoin
- G1 Africa unrest spike: GDELT events in African countries with root codes 14, 18, 19, 20 (protest,
  assault, fight, mass violence); daily count > 2 x its trailing 30-day mean -> short an equal-weight basket of
  the African-exposed miners in data/market/history for the next 5 trading days (adjusted closes).
- G2 the same spike -> long gold (context) for the next 5 trading days.
- G3 Reddit crowding: wallstreetbets posts on a day above 1.5 x their 20-day mean -> short the S&P next day
  (Reddit halves; low power, recorded as such).
- G4 Bitcoin weekend: long from Friday 16:00 ET to Monday 09:30 ET.
- G5 Bitcoin during US hours: long 09:30 -> 16:00 ET each weekday.
- G6 Bitcoin outside US hours: long 16:00 -> next 09:30 ET each weekday night.
- G7 Bitcoin 200-day trend (16:00 ET closes): long next day when above, else flat.
- G8 Bitcoin 1-day reversal: hold the opposite sign of the last 24h move for the next 24h (16:00 to 16:00).
- G9 Bitcoin follows the S&P: hold the sign of the S&P 09:30 -> 15:59 session move from 16:00 -> 09:30 ET.

### Not testable with the data on disk (recorded, not ruled out)
- Weather (NASA POWER) -> crops and metals: no commodity futures prices on disk.
- Kalshi Fed markets -> rates or stocks: 15 months of data and no rates futures on disk.
- Stock earnings drift, value, quality, short interest: need fundamentals or earnings dates for all stocks.
- Stock opening/closing auction effects, intraday stock patterns: only daily closes on disk.

Code: odte/research/sweep1.py -> reports/sweep1/ (one table, every idea, nothing dropped).

## Implementation notes (added 2026-11-15 while coding, before any idea was scored)
- Timing of other-market closes: data/context/daily.parquet is stamped by date, but some closes come after
  the 16:00 S&P close (Bitcoin at 00:00 UTC; futures and the dollar index up to 17:00 ET). To keep "known
  at the prior close" honest, C9 (dollar), C10 (10-year yield), C11 (oil), C12 (gold) and C13 (Bitcoin) use
  values two trading days back (d-2); C1-C8 and C15 (Cboe indexes close by 16:15, HYG at 16:00) use d-1.
- G1/G2: GDELT day files (data/africa/gdelt/events_daily, African events, dated when added) are complete
  only after the day ends in UTC, so positions run on the 2nd-6th trading days after the spike day.
  Costs: miners basket 0.20% per round trip, gold 0.05%, both on $10k.
- "Trades" for the 20-per-era minimum: trades for intraday/overnight ideas; days held for day-by-day positions.
- B8 high and low come from the minute bars' opens and closes (the file keeps no minute highs and lows).
- FOMC statement days are rows "fomc" OR "fomc_sep" in events.parquet (the calendar splits meetings with
  and without projections; "fomc" alone is half the meetings). A19-A21, A24 and E-fomc use both. Found in a
  dry run that printed trade counts only.

## Result (2026-11-15): 1 PASS, 4 LEAD, 104 FAIL, 8 not scored (reports/sweep1/sweep1.csv)
117 ideas, 109 scored. The false-discovery rule applied across all 109.

- **PASS: B7, buy the S&P after three down closes in a row, hold one day.** 234 days held in 10.8 years
  (~8.6% of days). +$1,375 a year per MES after costs, t 3.33, q 0.094 (just under 0.10). Both eras
  positive (2016-21 weak: +$2.1 a day averaged over all days; 2022-26 +$9.7). Lost only in 2019 (−$96).
  - Checks made after the result (reported, not used for the verdict):
    - Signal from ES 15:59 closes instead of the 16:00 index close: t 2.98 (+$1,227 a year). The
      exact days differ near zero, and with t 2.98 the false-discovery rule would NOT pass it (q ~0.3):
      the PASS sits on the line.
    - Double costs (1.5 points): t 3.22. Without the best 3 / 5 / 10 days: t 3.01 / 2.75 / 2.05. The median
      held day earns $25-31 per MES and 57-59% of held days win, so it is not one or two rebounds.
    - Held days averaged $65.9 per MES; all other days $3.2.
  - Honest label: short-term mean reversion after selling streaks (Connors-style), long only, so it also
    collects the stock market's premium on the riskiest days. Worst held day −$740 per MES.
- **LEAD (p < 0.05, both eras positive, not surviving the correction):**
  - B9 RSI(2) < 10 -> long next day: t 2.98 (same family as B7: buying after sharp drops).
  - E-claims-buy, buy the ATM 0DTE straddle at 10:00 on Thursdays (jobless-claims days): t 2.12, 27 trades
    before 2022 and 236 after (Thursday expiries began in 2022). The opposite of what B0 does.
  - B16 big-move reversal: t 2.05. A1 Monday 09:30-16:00 session long: t 2.00 (2016-21 ≈ 0).
- **Partner check (vs B0 + buy-and-hold: Sharpe 0.89, P(losing year) 15.2%):** every PASS/LEAD helps.
  B0+B+B7: Sharpe 1.31, P(losing year) 4.6%. B9 1.23 / 6.7%; E-claims-buy 1.21 / 11.0%; A1 1.10 / 10.7%;
  B16 0.99 / 11.7%. Correlations with B0 between −0.09 and 0.
- **Nothing found (all FAIL):** every US stock cross-section idea (F1-F9: momentum, reversal, low volatility,
  MAX, volume, 52-week high, liquidity, seasonality; 2018-26); every Bitcoin idea (G4-G9); the Africa
  unrest signals (G1-G2); dealer-gamma direction (D1-D6); the other-market signals (C1-C15, best C1 VIX
  term structure t 1.94); the remaining calendar effects (pre-FOMC drift t 1.94, Halloween, opex weeks,
  holidays, month and quarter ends); and 30 of 32 scored event-day option trades.
- Not scored (fewer than 20 trades in an era): quad-witching and quarter-end option trades, long-bond
  auction days, triple-witching day, Reddit crowding (15 months of data).
- **Verdict:** B7 earns a forward paper test (it needs only daily closes, so forward days can be scored
  without any bot change); the four LEADs are watched forward. No claim of an edge until forward days agree.

## Forward check of B7 and the LEADs (fixed 2026-11-15, before any forward day exists)
Tracking only: no bot change, nothing traded. Whether anything is ever traded is the owner's decision.
- **Ideas:** B7, B9, B16, A1, with the exact sweep rules. E-claims-buy needs 0DTE option quotes at 10:00
  on Thursdays; it is scored later from the bot's recorded quotes, not by this script.
- **Forward days:** NYSE trading days from Mon 2026-11-16 on. 2026-11-09..13 (after the sweep's last day,
  before this note) are left out: their closes were already on disk.
- **Prices:** SPX index closes from data/context/daily.parquet (Yahoo; the bot refreshes it every morning).
  A1 uses Yahoo's ^GSPC open and close. The backtest used ES futures, and the index stands in for MES. To
  match the futures, a carry charge of SPX(prior close) × 2.5% × calendar days / 365 comes off each long
  position-day (and is credited to a short one), about 0.5 points a day. Costs as in the sweep: 0.375 points
  per unit of position change, so 0.75 per A1 Monday.
- **Judging rule, per idea:** draw as many trading days as the forward period has, with replacement, from the
  idea's backtest daily net P&L (all days, zero days included, 2016-01-04..2026-11-06), 10,000 times, seed
  20261115. Forward total below the 5th percentile of those draws → BROKEN (dropped). Otherwise CONSISTENT.
  CONSISTENT does not prove an edge: B7 holds about 22 days a year, so six months is about 11 held days and
  can only catch an idea that has stopped working. A t-test against zero is reported, not judged.
- **Reviews:** monthly, and a formal review on 2027-05-14. Any rule change after today is labeled post-hoc.
