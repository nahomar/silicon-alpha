# Kalshi longshot fade / favorite test (pre-registered 2026-11-10 14:05 PST)

Written **before any call that returns settled market results or historical prices**.
Up to this point only metadata endpoints were called: `GET /historical/cutoff`,
`GET /series/KXHIGHNY`, `GET /series?include_volume=true` and
`GET /series/fee_changes?show_historical=true` (fee types, multipliers, series volume; no
prices, no results). Everything below is fixed: no rule, band, fee or gate will change after
seeing results. The honest answer may be "no"; that is a fine outcome.

## Why this test
- The favorite-longshot bias (cheap contracts are overpriced, expensive ones underpriced) is
  one of the oldest findings about betting and prediction markets. If it survives Kalshi's
  taker fee, two fixed, simple rules should make money: fade longshots (K1) and back
  favorites (K2).
- One fixed rule each, two gated tests, no search.

## Facts found (accessed 2026-11-10, 13:58–14:05 PST)
- **API, no auth needed for market data.** Kalshi Trade API v2, base
  `https://external-api.kalshi.com/trade-api/v2` (also `https://api.elections.kalshi.com/trade-api/v2`).
  The OpenAPI spec for the endpoints below has `security: []`, and the quick start says these
  endpoints need no API keys:
  https://docs.kalshi.com/getting_started/quick_start_market_data.md. Unauthenticated
  calls to `/historical/cutoff`, `/series` and `/series/fee_changes` returned HTTP 200.
- **Live vs historical split.** https://docs.kalshi.com/getting_started/historical_data.md:
  markets (and their candlesticks) that settled before `market_settled_ts` are only served by
  `GET /historical/markets` and `GET /historical/markets/{ticker}/candlesticks`. On
  2026-11-10, `GET /historical/cutoff` returned `market_settled_ts = 2026-09-11T00:00:00Z`.
  Markets settled after that are in `GET /markets` (with `status=settled`, `min_settled_ts`).
- **Endpoints used** (docs: https://docs.kalshi.com/api-reference/historical/get-historical-markets.md,
  .../historical/get-historical-market-candlesticks.md, .../market/get-markets.md,
  .../market/get-market-candlesticks.md, .../events/get-events.md, .../market/get-series-list.md):
  - `GET /historical/markets?mve_filter=exclude&limit=1000` (cursor paging; filters are
    tickers / event / series only, so the whole archive is paged and filtered locally);
  - `GET /markets?status=settled&min_settled_ts=<cutoff − 1 day>&mve_filter=exclude&limit=1000`;
  - `GET /historical/markets/{ticker}/candlesticks?start_ts&end_ts&period_interval=60` and
    `GET /series/{series}/markets/{ticker}/candlesticks?...` for markets still in the live set.
    Each candle has `end_period_ts` ("inclusive end of the candlestick period"),
    `yes_bid` and `yes_ask` OHLC in dollars (close = quote at the end of the period),
    `volume` and `open_interest`;
  - `GET /events?tickers=...` (event → series ticker and category);
  - `GET /series?include_volume=true` (series category, `fee_type`, `fee_multiplier`).
- **Rate limits.** https://docs.kalshi.com/getting_started/rate_limits.md documents token
  buckets for authenticated users (Basic tier: 200 read tokens/s, 10 tokens per request ≈ 20
  requests/s) and 429 with exponential backoff. No unauthenticated limit is documented. This
  study stays far below it: at most about 4 requests/s (≥ 0.25 s between calls), exponential
  backoff on 429 and 5xx (1 s doubling to 64 s), no proxies, single IP, single thread.
- **Fees (current).**
  - The official fee schedule is https://kalshi.com/docs/kalshi-fee-schedule.pdf (linked from
    https://help.kalshi.com/en/articles/13823805-fees). Automated download returned HTTP 429
    with a Vercel bot challenge (`x-vercel-mitigated: challenge`); per the rules it was not
    worked around. The formula below is taken from other official sources:
  - Kalshi's CFTC filing of 2026-10-14 (https://www.cftc.gov/filings/orgrules/rules10142633445.pdf,
    "Event Contracts Taker Rebate Program"): "the Base Taker Fee per contract is $0.0175
    (0.07 × 0.50 × 0.50)", "the standard taker fee per contract for that fill under the
    Exchange's generally applicable fee schedule, before any rounding". So the general taker
    fee is **0.07 × C × P × (1 − P)**, P the contract price in dollars.
  - Per-series scaling from the API (`fee_type`, `fee_multiplier`, described in
    https://docs.kalshi.com/api-reference/market/get-series.md): `quadratic` = general table;
    `quadratic_with_maker_fees` = general table plus maker fees; `..._combo_maker_fees` =
    maker multiplier 0.5 instead of 0.25; `fee_multiplier` scales the fee. Snapshot of all
    16,064 series on 2026-11-10: 15,869 `quadratic`×1, 159 `quadratic_with_maker_fees`×1,
    18 `quadratic`×0.5 (MLB props), 1 `quadratic_with_maker_fees`×0.5 (KXMLBGAME),
    14 `quadratic`×0 (fee-free), 3 combo×1, no `flat`.
  - Rounding, https://docs.kalshi.com/getting_started/fee_rounding.md: the trade fee is the
    model fee rounded up to $0.000001; direct-member balances are then aligned to $0.0001
    (FCM customers to $0.01) with a rounding fee and later rebates.
  - Fees changed over time (`/series/fee_changes?show_historical=true`): maker fees added to
    many sports series from 2025-10-04 on; S&P / Nasdaq series (KXINX, KXINXU, KXNASDAQ100,
    ...) moved from a half-rate to the standard multiplier on 2026-07-03; MLB props cut to
    0.5× on 2026-08-07; some long-dated series set to 0. A taker rebate program for
    high-volume self-clearing members starts 2026-10-27 (not applicable to a small account;
    ignored). **This test charges today's fees to every historical trade.**

## Fee used (fixed)
- m = the series' current `fee_multiplier` (2026-11-10 snapshot); series not in the snapshot
  get the general schedule, m = 1, no maker fee. Event-level fee overrides are ignored.
- **Taker fee per contract (gate):** f_T(P) = ceil_0.0001(0.07 × m × P × (1 − P)), i.e. the
  direct-member precision. P is the price paid (no_ask for K1, yes_ask for K2).
- **Maker fee per contract (maker variant only):** coefficient 0.0175 × m for
  `quadratic_with_maker_fees`, 0.035 × m for `quadratic_with_combo_maker_fees`, 0 for
  `quadratic`; same rounding.
- **Report-only, conservative:** one-contract order rounded up to the next cent,
  ceil_0.01(0.07 × m × P × (1 − P)) (what an FCM customer, or the old "round up to the next
  cent" table, would charge on a single contract).

## Universe (fixed)
- Settled markets with `market_type = binary`, `status` in {finalized, settled},
  `result` in {yes, no} (anything else, e.g. void, scalar, empty, is dropped and counted);
- 2024-01-01T00:00:00Z ≤ `close_time` < 2026-11-01T00:00:00Z;
- lifetime `volume_fp` ≥ 1,000 contracts;
- `open_time` ≤ `close_time` − 48 h.
- **Combos (multivariate events, MVE) are excluded** (`mve_filter=exclude`): they are
  parlays created on request and traded mostly through RFQ, so continuous bid/ask quotes are
  not their price formation, and the archive of them is far too large to page politely. This
  is a stated deviation from "all binary markets".
- Markets closed by 2026-10-31 but not yet settled on the fetch date are not in the
  universe (counted).
- **Sampling:** if more than 20,000 markets qualify, take exactly 20,000 by proportional
  stratified sampling on close month (UTC): allocation by largest remainder of
  20,000 × N_month / N; within a month, sort by ticker and draw without replacement with
  `numpy.random.default_rng(0)`, months processed in calendar order. The sample uses no
  prices and no results. The sample is self-weighting, so equal weight per market in the
  sample estimates the equal-weight mean of the universe.

## Decision price (fixed)
- D = `close_time` − 24 h. Fetch hourly candles (`period_interval=60`) with
  start_ts = D − 24 h, end_ts = D. The API includes candles **ending at or before end_ts**,
  so nothing after D can enter.
- Use the **latest candle with `end_period_ts` ≤ D** (the candle for the hour ending at D when
  `close_time` is on the hour; the hour ending just before D otherwise). If candles are only
  emitted for hours with activity, the latest candle's close is still the standing quote at
  D. No candle in [D − 24 h, D] → dropped as a data gap (counted).
- bid = `yes_bid.close`, ask = `yes_ask.close` (dollars). Require both present and
  0 < bid < ask < 1. p_mid = (bid + ask) / 2.

## Rules (fixed)
- **K1, longshot fade:** if 0.03 ≤ p_mid ≤ 0.15, buy NO at no_ask = 1 − bid. Payoff 1 if
  result = no, else 0.
- **K2, favorite:** if 0.85 ≤ p_mid ≤ 0.97, buy YES at ask. Payoff 1 if result = yes.
- One contract per market, equal weight. Return on capital per market
  r = (payoff − P − f_T(P)) / (P + f_T(P)).

## Statistics and gates (fixed)
- Halves: H1 = `close_time` < 2025-07-01T00:00:00Z, H2 = on or after.
- For each strategy and each half: mean r over traded markets; cluster bootstrap over
  `event_ticker` (resample events with replacement, pooled mean of r over all markets in the
  drawn events), 10,000 resamples, seed 0 (`numpy.random.default_rng(0)`, fresh per
  strategy × half). CI = percentile interval.
- **Gate (Bonferroni over 2 strategies): two-sided 97.5% CI, i.e. the 1.25th percentile of
  the bootstrap means, must be > 0 in BOTH halves.** A strategy passes only if both halves
  pass. A half with no trades fails.
- Verdict per strategy: pass / fail. Overall: "edge after fees" if K1 or K2 passes,
  otherwise "closed".

## Report only (no gates)
- Full-period mean and CI for K1, K2.
- Calibration table over every sampled market with a valid quote: p_mid bucket (edges 0, .03,
  .05, .10, .15, .25, .35, .45, .55, .65, .75, .85, .90, .95, .97, 1), count, mean p_mid,
  realized YES rate.
- K1 and K2 by category (series category) and by the 15 largest series.
- **Maker variant (optimistic: a fill is not guaranteed and fills are adversely selected):**
  K1 buys NO at the NO bid = 1 − ask, K2 buys YES at bid, plus the maker fee.
- Conservative fee: cent rounding on one contract (above).
- Gross: no fee.
- Validity checks: (a) excluding markets with `can_close_early = true` (their `close_time` can
  be the actual early close, so D can depend on when the outcome happened); (b) only candles
  ending in (D − 1 h, D] (fresh quote).
- Capacity for traded markets: median lifetime volume, median `open_interest_fp` (market
  record) and open interest at D (candle), contracts traded in the 24 h up to D; typical
  dollars deployable per market = 10% of the contracts traded in [D − 24 h, D] × entry
  price (median and quartiles). Book depth is not in the free data, so this is a proxy.

## Known limits, stated in advance
- Today's fee schedule is applied to 2024–2026 trades; historical fees were sometimes lower
  (half-rate index series, no maker fees) and sometimes zero.
- Survivorship: only markets the API still serves; voided / removed (e.g. provisional)
  markets are not in the archive.
- Hourly close quotes are top-of-book only: no depth, and the quote may be stale if
  candles are sparse.
- `close_time` for early-closing markets may be the actual close (see validity check a).
- Combos excluded; markets settled after the fetch excluded.
- Events in the same series (e.g. daily weather strikes) are correlated across days; the
  bootstrap clusters only within events.

## Test count
- 2 gated tests (K1, K2), each requiring both halves, in a new family (Kalshi event
  contracts). Project running count: 292 + 2 = 294 (the main session should reconcile if
  other tests landed in parallel).

## Procedure
1. This plan (written 2026-11-10 14:05 PST, before any outcome or price call).
2. Code in `kalshi/backtest.py`, unit tests in `tests/kalshi/` (no-look-ahead, fee, payoff).
3. `python3 -m kalshi.backtest fetch` (markets, events, candles into `data/kalshi/`).
4. `python3 -m kalshi.backtest run`, once, after the tests pass. Results appended below; the
   plan above is not edited afterwards. A bug found after the run is documented, fixed and
   rerun, and both runs are reported.

## Note A1 (2026-11-10 14:22 PST, before any result or price was inspected): listing method kept
- At 14:20 I briefly wrote an amendment to switch to per-series listing with parallel
  workers, because I misjudged the listing speed (I estimated about 10 hours). The real
  speed: the full-archive paging listed 1.9 million markets in 12 minutes (about 2.6
  requests/s, reaching back to settlements on 2026-08-15, about 73,000 markets per day), so
  the full listing back to 2024 takes a few hours. That amendment is **withdrawn**; the
  original method above is kept and resumed from its saved cursor.
- For the cost estimate only `ticker`, `event_ticker`, `open_time`, `close_time`,
  `volume_fp` and `settlement_ts` of the listed rows were profiled (79% of listed markets are
  in hourly / 15-minute series; 0.8% of the markets passing the volume and 48-hour filters
  were in them). No `result` field and no price was inspected.

## Note A2 (2026-11-10 15:42 PST, before the run; no result inspected): sparse candles
- Data QA on the first 204 candle records (mechanics only): hourly candles exist only for
  hours in which the quote or a trade changed. 65% of candles have no trade, and every candle
  has a bid close, so a missing hour means the quote stood still. 27% of these markets had no
  candle at all in [D − 24 h, D].
- The gated test is unchanged: latest candle in [D − 24 h, D], otherwise dropped as a data
  gap. Because a 24-hour cap may select more active markets, one **report-only** check is
  added: for markets dropped as gaps, candles are refetched over
  [max(open_time, D − 30 days), D] and the latest candle at or before D (the standing quote) is
  used. K1 and K2 are reported on that extended set, with the same statistics, no gate.

## Result (run once, 2026-11-10, by the main session after the agent's network failure)
- Data: the 20,000-market sample, 16,953 with a valid quote at D − 24 h (505 had no candle
  in the window; 2,542 had an invalid quote). Most trades fall in the second half: Kalshi
  volume grew in 2025-26.
- **K1, fade long shots (buy NO when YES is 3-15¢): FAIL.** Mean return per trade −4.05%,
  97.5% CI [−5.40%, −2.73%], n = 3,052. H1 −5.1% (n = 72); H2 −4.0%.
- **K2, buy favorites (85-97¢): FAIL.** −2.63% [−5.00%, −0.41%], n = 905. H1 −2.9%; H2 −2.6%.
- Without any fee both are still negative (K1 −3.6%, K2 −2.1%): taking the ask, not the
  fee, is what costs money.
- **Maker variant (optimistic, assumes a fill at the bid):** K1 +0.7% [−0.7%, +2.1%];
  K2 +2.5% [+0.0%, +4.9%]. That is only a lead, because resting orders get filled mostly
  when the price is moving against them.
- **Calibration:** Kalshi prices are close to fair in this sample, and there is no
  long-shot bias at the low end. Contracts priced 5-10¢ won 8.5% of the time (priced
  7.2%); the largest gap is at 25-35¢ (+5.2 points).
- Verdict: closed (both K1 and K2 fail).
