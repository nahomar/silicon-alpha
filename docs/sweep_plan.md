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

## Sweep 2: the price-pattern ideas on data the sweep never saw (fixed 2026-11-15, before any data is fetched)
Question: B7 passed on the S&P 2016-26. If the effect is real it should show up where it was not selected:
the S&P before 2016, other US indices, and markets abroad. The same test is a fresh search for every
close-only price idea of family B, on markets sweep 1 never looked at.
- **Ideas (rules exactly as in sweep 1):** B1 200-day trend long/flat, B2 200-day long/short, B3 10-month
  rule, B4 12-month momentum, B5 1-day reversal, B6 5-day reversal, B7 three down closes, B8 internal bar
  strength (Yahoo high/low), B9 RSI(2) < 10, B10 252-day high, B16 big-move reversal. Each in its written
  direction (no flipping).
- **Data:** Yahoo daily index prices (price indices, no dividends). Rows whose close equals the previous close
  exactly are dropped (holiday placeholders); rows with a daily move over 25% are dropped and listed.
- **H1, the S&P before the sample:** ^GSPC 1990-01-01..2015-12-31. 1962-1989 is reported only: before index
  futures and fast trading, index closes were partly stale and short-term reversal is not expected there
  (a known feature of old index data, stated here before looking).
- **H2, other US indices (reported, same years as the sweep so not independent):** ^NDX, ^RUT, ^DJI, ^MID,
  from 1990 (or their first Yahoo day) to 2026-11-06.
- **H3, abroad:** ^FTSE, ^GDAXI, ^FCHI, ^STOXX50E, ^N225, ^HSI, ^AXJO, ^GSPTSE, ^KS11, ^BSESN, ^BVSP, ^TWII,
  2000-01-01..2026-11-06, each on its own trading days. Pooled = equal-weight average of the markets that
  traded each date.
- **Scoring:** daily net return = position × next-day return − 1 bp per unit of position change. t over all
  days. Timing t (reported): Welch test of held-day vs other-day gross returns.
- **Verdict per idea:** CONFIRMED if H1 net t ≥ 2.0 AND H3 pooled net t ≥ 2.0 with at least 7 of 12 markets
  positive; PARTLY if one of the two; NOT CONFIRMED if neither. An idea that failed in sweep 1 but is
  CONFIRMED here becomes a new candidate for its own forward test, not a trade. BH q across the 11 pooled
  H3 tests is reported.
- **H4, an option version on the S&P (reported; same dates B7 was selected on, so weak evidence):** on days
  whose B7 / B9 / B16-long signal is on, sell the 0DTE put credit spread at 10:00 (library y_put_spread_touch);
  B16-short days sell the call credit spread. One-sided t vs zero and Welch vs all other days' same trade.
- **Added at the owner's request, after the data fetch and before any result was computed (report only; the
  verdict rule above is unchanged):** results pooled by region, Europe (^FTSE, ^GDAXI, ^FCHI, ^STOXX50E +
  extra), Asia (^N225, ^HSI, ^KS11, ^BSESN, ^TWII + extra); and H3b, more markets on the same rules and
  years: Europe ^SSMI, ^IBEX, FTSEMIB.MI, ^AEX, ^OMXSPI; Asia ^STI, 000001.SS, ^JKSE, ^KLSE, ^NZ50 (Pacific).
  A region "holds up" if its pooled t ≥ 2.0 and more than half its markets are positive.

### Sweep 2 result (2026-11-15; code odte/research/sweep2.py, output reports/sweep2/sweep2.json)
Data: 27 Yahoo indices; 3 rows dropped as >25% moves (^BVSP 1995-03-10, 1999-01-15; ^HSI 1987-10-26).

| idea | S&P 1990-2015 t | S&P 1962-89 t | abroad pooled t | markets + | verdict |
|---|---|---|---|---|---|
| B1 200-day long/flat | 2.72 | 4.03 | 3.77 | 11/12 | CONFIRMED |
| B2 200-day long/short | 1.06 | 2.75 | 1.32 | 10/12 | NOT CONFIRMED |
| B3 10-month rule | 3.27 | 2.74 | 3.85 | 12/12 | CONFIRMED |
| B4 12-month momentum | 2.25 | 1.44 | 1.16 | 9/12 | PARTLY |
| B5 1-day reversal | 0.85 | −12.18 | −2.47 | 6/12 | NOT CONFIRMED |
| B6 5-day reversal | 1.10 | −5.92 | −0.45 | 5/12 | NOT CONFIRMED |
| **B7 three down closes** | **3.04** | −3.16 | **2.13** | **10/12** | **CONFIRMED** |
| B8 internal bar strength | 1.68 | −6.81 | −2.18 | 3/12 | NOT CONFIRMED |
| B9 RSI(2) < 10 | 2.84 | −3.52 | 1.98 | 10/12 | PARTLY |
| B10 252-day high | 0.25 | 2.00 | 1.91 | 6/12 | NOT CONFIRMED |
| B16 big-move reversal | 1.13 | −4.17 | −0.40 | 8/12 | NOT CONFIRMED |

- **B7:** the S&P 1990-2015 agrees (t 3.04; held days beat other days, timing t 2.92), as do the Nasdaq-100
  (t 4.41), Dow (2.56) and MidCap 400 (2.04); not the Russell 2000 (0.51). Before 1990 it reverses (t −3.16),
  as stated in advance. By region (report only): Europe holds up (9/9 positive, pooled t 2.36); Asia does
  not (5/9, t −0.48). Post-hoc: S&P by period t 1.79 (1990s), 2.58 (2000-07), 1.15 (2008-15), 3.15 (2016-26).
  **Warning, post-hoc:** Europe pooled was strong 2000-2015 (t 2.36, timing t 6.40) and faded 2016-2026
  (t 0.59). A real, widely known effect (short-term reversal in large developed indices), not a private
  edge, and one that has weakened abroad.
- **B9:** PARTLY (pooled abroad t 1.98, just short). **B16:** NOT CONFIRMED, the lead is dropped.
- **B1 and B3 (trend, long or flat):** CONFIRMED by the letter of the rule, but the rule asks "does it make
  money", and an idea that is long ~73% of the time collects the market's premium. Their timing t on the S&P
  is ~0 (0.02, 0.77). Post-hoc 2000-2026: B3 raised Sharpe over buy-and-hold in 8 of 8 markets checked and
  about halved the worst drawdown (S&P −57% → −26%, Nasdaq −83% → −36%, DAX −73% → −30%). Their value is
  smaller crashes, not extra return. In sweep 1 (S&P 2016-26, a long bull market) they failed.
- **Asia:** 1-day reversal is strongly negative there (pooled t −7.25: daily moves continue). Probably stale
  index prices (thinly traded members), not a tradeable effect; futures data would be needed to tell.
- **H4, B7 as a 0DTE put credit spread at 10:00:** t 0.59 (184 trades); B9, B16 versions also near zero.
  B7's gain does not show up in a same-day 10:00 option trade.
- **Verdict:** B7 stays the lead candidate and moves from "one sample, on the line" to "confirmed on the S&P
  history it never saw and in Europe, but weaker abroad since 2016". Forward check continues as fixed.
  New candidate: B3 (10-month rule) in place of plain buy-and-hold as B0's partner, for smaller drawdowns;
  it needs its own pre-registered partner test.

## Sweep 3 (fixed 2026-11-15, before any of its data is fetched or computed)
Three new areas, free public data only. Scoring for every idea: net t ≥ 2.0 over its window AND positive net
mean in both halves (split at the window's middle date) → PASS; BH q ≤ 0.10 across all sweep 3 ideas is
reported. p < 0.05 and both halves positive but not PASS → LEAD.

### P. A steadier partner for B0
- P1: B3 (10-month rule on the S&P, long 1 MES or flat, ES marks as in sweep 1) in place of B (always long
  1 MES). Equal-vol weights from 2019-06-26..2021-12-31, the B0 days, the bootstrap as in partner_plan.md.
  B3 replaces B if B0+B3 has a higher Sharpe AND a lower P(losing year) than B0+B (the rule O and T faced).
  Max drawdowns and B0+B3+B7 vs B0+B+B7 are reported.

### X. Other markets (Yahoo adjusted closes with dividends; FRED for currencies)
Monthly rebalance at month-end closes; an asset enters once it has 13 months of history; 10 bps per unit of
turnover for ETFs, 5 bps for currencies.
- Country ETFs (USD): EWJ EWG EWU EWQ EWH EWY EWT EWA EWC EWZ EWS EWM EWI EWP EWL EWN EWD EWW EWK EWO FXI EZA
  TUR INDA SPY, 1996-2026-10.
  - X1 12-1 month momentum: long top third, short bottom third, equal weight.
  - X2 1-month reversal: long bottom third, short top third.
- US sector ETFs: XLB XLE XLF XLI XLK XLP XLU XLV XLY (1999-2026-10; XLRE, XLC join when 13 months old).
  - X3 12-1 month momentum: long top 3, short bottom 3. X4 1-month reversal: long bottom 3, short top 3.
- Currencies vs USD (FRED daily spot DEX*: AUD CAD JPY GBP CHF EUR NZD NOK SEK; FRED monthly 3-month
  interbank rates IR3TIB01*, US included), 1999-2026-10. Monthly return of a long-foreign position = spot
  change + (foreign − US rate)/12 from the prior month's rates.
  - X5 carry: long the 3 highest-rate currencies, short the 3 lowest (vs USD).
  - X6 momentum: 12-1 month excess return; long top 3, short bottom 3.
  - X7 value proxy: 5-year spot change; long the 3 biggest fallers, short the 3 biggest risers.

### E. Company earnings (SEC EDGAR submissions: 8-K filings with Item 2.02, with acceptance time; prices
from data/prices/xnas_daily.parquet, 2018-05..2026-08; universe as sweep 1 family F: price ≥ $5, top 1,000
by 20-day dollar volume)
- Day 0 = the first trading day whose close comes after the 8-K acceptance time (accepted 16:00 ET or later →
  next trading day). Abnormal return AR = stock return over day 0..day 1 minus the universe's equal-weight
  return over the same days. Ticker-to-CIK from SEC's current ticker map (delisted firms drop out: noted).
  - E1 announcement-return drift: at the close of day 1, AR in the top decile of all announcements of the
    prior 91 calendar days (≥ 200 of them) → long; bottom decile → short; hold to the close of day 61.
    Daily return = mean of active longs − mean of active shorts; 10 bps per side on entry and exit.
  - E2 announcement premium: expected date = the announcement ~1 year earlier (within 330-400 days) + 364
    days. Long from the close 6 trading days before the expected date to the close of the expected date + 1,
    minus the universe's equal-weight return; 10 bps per side.
  - E3 the reverse of E1's short side only (big losers bounce?): long the bottom decile, hold days 2-6,
    minus the universe; 10 bps per side.
- **P result (run first, before X and E were coded):** FAIL. B0+B3 Sharpe 0.74 vs B0+B 0.89; P(losing year)
  19.1% vs 15.2% (B3 whipsawed in 2022: B0+B3 −$12.8k vs B0+B −$9.7k per B0 unit). B stays the partner.
  B0+B3+B7 1.29 / 4.1% vs B0+B+B7 1.35 / 3.8%.
- **X implementation notes (written after the fetch, before scoring):** ETF prices start 1996-03 (EWY, EWT,
  EWZ 2000; EZA 2003; FXI 2004; TUR 2008; INDA 2012), sectors 1998-12. A month is scored only when at least
  6 assets are eligible. Months run through 2026-10. Currencies: Japan's 3-month rate starts 2002-04 and
  Switzerland's 1999-07, so they join then; the UK and euro-area rates end 2026-01 and are carried forward
  (at most 12 months). Month m's carry accrual and month-end m−1's ranking both use month m−1's rates
  (market rates seen in real time; OECD's monthly average stands in). X6 ranks on the 12-1 month excess
  return (spot + carry), X7 on the 60-month spot change (so X7 starts about 2004).
- **X result:** X5 currency carry PASS (1999-2026: +3.19% a year, t 2.44, halves t 2.21 / 1.14, Sharpe 0.46;
  BH q 0.15 if it is sweep 3's smallest p, so it does not survive the correction). It is the textbook carry
  premium: −21% in 2008 (Oct 2008 −10.5% in one month), max drawdown −28%, roughly flat 2013-2025.
  Post-hoc: monthly correlation with B0 0.17; B0+carry monthly Sharpe 0.69 vs B0 0.60 (B0+B is better).
  FAIL: X1 country momentum (t −0.25), X2 country reversal (−0.80), X3 sector momentum (−0.07), X4 sector
  reversal (−1.56), X6 currency momentum (−0.31), X7 currency value (0.44).
- **E implementation notes (written before the earnings data finished downloading, before scoring):**
  SEC acceptance times are true UTC (Apple's 16:30 ET releases show 20:30Z in summer, 21:30Z in winter),
  converted to New York time. A company's releases within 30 days of an earlier one are dropped (amendments,
  repeats). A company with several tickers uses its most traded. A held stock with no price that day counts
  as 0. Costs: each leg's 20 bps round trip is spread evenly over its holding days. Scored over days with a
  position (E1: days with both a long and a short). E2: each release forecasts the next 364 days later.

## Sweep 4: what happens to a stock after each kind of SEC filing (fixed 2026-11-15, before extracting)
Source: the EDGAR submissions files downloaded for sweep 3 (1,632 of the most traded US companies), every
filing 2018-06..2026-08, prices and universe as sweep 3 E. Day 0 = the first trading day whose close comes
after the acceptance time. Position: enter at the close of day 0, hold days 1-20, return minus the universe's
equal-weight return, 20 bps per round trip spread over the holding days; calendar-time daily mean over active
events. Two-sided: the better direction is scored with p = 2×P(T > |t|), costs always subtract. PASS: t ≥ 2.0
in that direction, both halves of the window the same sign, AND BH q ≤ 0.10 across all sweep 4 ideas (with
~30 ideas a lone t of 2 is expected by luck). LEAD: p < 0.05 and both halves agree. Days 1-5 and 1-60
reported. Events of the same type for the same company within 30 days are dropped after the first.
- Ownership: S1 SC 13D (new 5%+ stake with intent; activists), S2 SC 13D/A, S3 SC 13G (passive 5%), S4
  DFAN14A (proxy fight material from dissidents), S5 SC TO-T (third-party tender offer). For S1-S3, finance
  companies (SIC 6000-6799) are left out because they file these about other firms.
- Insiders: S6 Form 144 (an insider's notice of a planned sale), S7 Form 3 (a new insider).
- New shares: S8 424B4 (final prospectus, IPO or follow-on), S9 424B5 (shelf takedown), S10 S-3 or S-3ASR
  (shelf registration), S11 S-8 (employee stock plan), S12 8-K 3.02 (unregistered equity sale).
- Trouble: S13 NT 10-K or NT 10-Q (late filing notice), S14 8-K 4.02 (past results no longer reliable),
  S15 8-K 4.01 (auditor change), S16 8-K 3.01 (delisting or listing-rule notice), S17 8-K 2.05 (restructuring
  costs), S18 8-K 1.02 (material agreement ended), S19 8-K 1.03 (bankruptcy).
- Other 8-K items: S20 1.01 (material agreement), S21 2.01 (acquisition or sale completed), S22 2.03 (new
  debt), S23 5.02 (executive or director change), S24 5.03 (bylaw or charter change), S25 5.07 (shareholder
  vote results), S26 7.01 (Reg FD disclosure), S27 8.01 (other events), S28 2.02 alone, days 1-20 (vs E1's
  sorted version).
- Not testable here: insider buys vs sells (needs each Form 4's contents, about 600k downloads, or SEC's
  bulk insider data sets, a separate download the owner would OK).
- **Data fix found in a dry run before E or sweep 4 was scored:** data/prices/xnas_daily.parquet has 150
  non-NYSE dates (Saturdays, holidays; 98-1,419 symbols each vs ~9,100 on trading days). sweep3.stock_panel
  now keeps NYSE days only. Sweep 1's family F used the unfixed panel (all F ideas FAILed); F is re-run with
  the fix and reported as a post-hoc check.
  Post-hoc F re-run on NYSE days only: still all FAIL (t F1 0.67, F2 0.38, F3 −0.31, F4 0.02, F5 0.03,
  F6 −1.28, F7 0.40 reversed, F8 0.88 reversed, F9 1.13 reversed). Sweep 1's conclusion is unchanged.

## Sweep 5 (fixed 2026-11-15, before fetching): short-sale volume and crypto cross-sections
Scoring as sweep 3 (PASS: t ≥ 2.0 and both halves positive; LEAD: p < 0.05 and both halves positive; BH q
across sweep 5 reported). Two-sided: the better direction is scored with p = 2×P(T > |t|).
### Q. FINRA daily short-sale volume (Reg SHO consolidated files CNMSshvolYYYYMMDD.txt, public)
2018-06..2026-08, stocks and universe as sweep 3 E (NYSE days only). SR = short volume / total volume.
- Q1: each Friday close, rank the universe on 5-day average SR; long the lowest decile, short the highest;
  hold one week; equal weight; 10 bps per unit of turnover.
- Q2: as Q1 on abnormal shorting: 5-day average SR minus the prior 60-day average.
- Q3: monthly: 20-day average SR at month end, hold one month.
### K. Crypto (Coinbase Exchange public daily candles, USD pairs, UTC days), 2019-01..2026-11-06
Universe each Sunday: USD pairs with ≥ 60 days of history, stablecoins and wrapped coins left out, top 20 by
30-day average dollar volume. 25 bps per unit of turnover (60 bps reported). Spot cannot be shorted on
Coinbase; long-short books are a test of the effect, not a tradeable book there.
- K1 1-week momentum: long top quarter, short bottom quarter on the last 7 days' return, hold 7 days.
- K2 4-week momentum: as K1 on 28 days.
- K3 low volatility: 30-day volatility, long the lowest quarter, short the highest.
- K4 Bitcoin lead: BTC up on UTC day d → long the equal-weight basket of the other coins on day d+1,
  else short it.
- K5 time-series momentum: each coin long if its 20-day return > 0, else short; equal weight.
- **E result:** 20,279 earnings releases (1,203 companies, 2018-06..2026-06) after the universe and 30-day
  filters, from 48,872 Item 2.02 8-Ks. Check: day 0 has the largest moves (median |return| 3.21% vs 1.11% on a
  random day), so the timing is right; day −1 is also raised (1.83%), consistent with some companies filing
  the 8-K after a morning press release. All FAIL: E1 drift −1.08 bps/day (t −0.62; 2,070 longs, 2,062
  shorts), E2 announcement premium 0.03 bps/day (t 0.01), E3 big-loser bounce +5.3 bps/day (t 0.96).
  Post-hoc E1 with the reaction measured over days −1..+1: t −0.68 (hold 60), −1.21 (20), −1.23 (5). The
  textbook drift after earnings surprises is not present in large US stocks in 2018-2026.
- **Sweep 4 result:** all 27 scored ideas FAIL (S19 bankruptcy: 4 events, not scored); largest |t| 1.74
  (S3 SC 13G, halves disagree); every q 0.99. In the 1,000 most traded US stocks, 2018-2026, no kind of
  SEC filing is followed by a drift worth trading after 20 bps costs.
- **Sweep 4b (fixed now, after sweep 4's null, before running):** the published filing effects sit mostly in
  smaller firms, so the same 28 ideas are re-run once on a wider universe: price ≥ $5 and top 2,500 by
  20-day dollar volume (the SEC files cover about the 2,500 most traded names). Same rules, its own BH
  correction; labeled as a second look.
- **Sweep 4b result, as fixed:** uninformative. Nearly every filing type showed the same "short, ~11 bps a
  day, t ~1.2": the wider universe's equal-weight return is 17.9 bps a day against a median stock's 1.5,
  pulled up by bad prices (149 one-day moves over +100%: reverse splits and bad prints the split filter
  misses; the top-1,000 universe has 36). Only S14 was a LEAD (t 2.03).
- **Post-hoc 1, daily moves over 50% removed as data errors:** top 1,000: all FAIL (sweep 4 unchanged). Top
  2,500: LEADs S3 13G long (t 2.67), S7 Form 3 long (2.07), S22 new debt long (2.04), S14 restatement short
  (2.04); every q ≥ 0.21. The routine earnings release (S28) also "drifted" over 60 days (t 4.28), a sign of
  survivorship: SEC's ticker map lists only companies still trading today.
- **Post-hoc 2, benchmark = the SEC-covered (surviving) companies only:** S3 (1.60), S7 (−1.19), S22
  (−0.78) disappear, so they were survivorship. S14 remains a weak LEAD: after a company says past results
  are no longer reliable (8-K Item 4.02), its stock lags by about 29 bps a day over the next 20 days
  (t 2.29, ~6% in total), but only 77 events in 8 years, q 0.60, second half t 0.34. Not a strategy.

## Sweep 6 (fixed 2026-11-15, before fetching): Treasury auctions, S&P 500 changes, the January effect
Scoring as sweep 3 (PASS: t ≥ 2.0 and both halves positive; LEAD: p < 0.05 and both halves positive), two-
sided where marked; BH q across sweep 6 reported.
### T. Treasury auction cycle (auction dates and results: Treasury's Fiscal Data API "auctions_query";
prices: Yahoo adjusted closes of IEF for 10-year notes and TLT for 30-year bonds, 2003-2026-10; 2 bps per side)
- T1 after the auction: long from the auction-day close to the close 3 trading days later (10-year, IEF).
- T2 before the auction: short from the close 5 trading days before to the close the day before (IEF).
- T3, T4: T1, T2 for 30-year bond auctions with TLT.
- T5 weak auctions (two-sided): bid-to-cover below the average of the same term's previous 6 auctions → short
  IEF/TLT from the auction close for 3 days; strong → long. 10- and 30-year pooled.
### I. S&P 500 changes (Wikipedia's "List of S&P 500 companies" changes table: effective dates and tickers;
prices and universe benchmark as sweep 3 E, NYSE days, moves over 50% removed as data errors, 2018-06..2026-08)
- I1 added stocks, two-sided: close of the effective date to 20 trading days later, minus the universe.
- I2 deleted stocks (that still trade), two-sided, same window.
### J. January effect (Yahoo ^RUT and ^GSPC price indices, 1988-2026)
- J1: long ^RUT, short ^GSPC from the Dec 15 close (or the trading day before) to the Jan 31 close; one value
  per year; 2 bps per side per leg.
- J2: the same over the last 5 trading days of December plus the first 5 of January.
- **Sweep 6 implementation notes (after fetching, before scoring):** auctions: 502 ten-year and 334 thirty-
  year (reopenings included via original_security_term), used from 2003 to 2026-10 (IEF/TLT start 2002-07-30).
  Auctions close at 13:00 ET, so the 16:00 auction-day close is after the result. T2 runs from the close 5
  trading days before the auction to the close the day before (4 days held). One value per auction; halves
  by date. S&P changes: the current Wikipedia page no longer has the changes table, so the revision of
  2026-06-30 (oldid 1361794875) is used: 402 rows, effective dates up to 2026-06-30. Day 0 = the first NYSE
  day on or after the effective date; a ticker with no price on day 0, or fewer than 15 of the 20 days, is
  dropped (renamed tickers drop out). Benchmark: the top-1,000 equal-weight, moves over 50% removed.
- **Sweep 6 result:** T4 PASS by the rule only: short TLT from 5 days before each 30-year auction to the day
  before, +0.22% per auction (260, 2003-2026), t 2.13, halves t 0.71 / 2.28, q 0.27 (does not survive the
  correction). The same idea on 10-year notes (T2) shows nothing (t 0.25), so it is not a general pattern.
  FAIL: T1 after 10-year auctions (t 1.66, second half −0.39), T3 after 30-year (0.66), T5 weak/strong
  auctions (−0.28), J1 January small-cap effect (t 1.87, second half 0.34: faded), J2 turn of the year
  (1.42), I1 S&P additions (131, t −0.22), I2 deletions (93, t 0.73).
- **Sweep 5 data notes (after fetching, before scoring):** FINRA files exist from 2018-08-01 (June-July 2018
  return 403), 2,031 days. Coinbase: 496 USD products (listed and delisted), 520,279 daily rows. K5 rebalances
  weekly on Sundays like K1-K3. K4 counts only the basket's position flips as turnover.
- **Sweep 5 result: all 8 FAIL.** Q1 short ratio weekly −3.4%/yr net (t −0.92); Q2 abnormal short ratio
  −8.7%/yr net (t −3.09; both directions lose after costs); Q3 monthly −0.9%/yr (t −0.25). Post-hoc, before
  costs: Q1 +7.8%/yr gross (t 2.10), Q2 +5.3% (1.90), Q3 +2.0% (0.57): heavily shorted stocks do lag, as
  published, but the information fades within weeks and weekly turnover at 10 bps a side eats all of it.
  Crypto: K1 1-week momentum t 0.46, K2 4-week −0.77, K3 low volatility 0.59, K4 Bitcoin lead −0.42 (best
  direction), K5 time-series momentum 0.59; at 60 bps all are negative.
