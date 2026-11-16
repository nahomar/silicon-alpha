# Crypto funding-rate carry vs. T-bills (pre-registered 2026-11-10)

Plan written **2026-11-10, 14:10–14:30 PST (UTC−8)**, before any funding-rate history, spot
klines or T-bill series for this test was downloaded or looked at. Everything below is
fixed: no rule, constant, cost or gate will change after seeing results. If a bug is found
after the run, it is documented below the results, fixed, rerun, and both runs are reported.

## Question
- Does the funding-rate carry trade (long spot, short perpetual, collect funding) beat
  T-bills after costs?
- Can a US retail person legally run it?
- "No" is an acceptable answer.

## Legal and venue facts (official sources, all accessed 2026-11-10)
Coinbase's own web pages (coinbase.com, help.coinbase.com) sit behind a Cloudflare bot
check that blocks automated reading (HTTP 403 "Just a moment..."). It did not load in a
normal Chrome tab either. So the Coinbase facts below come from Coinbase Derivatives'
CFTC filings (official, on cftc.gov) and Coinbase's developer docs. Points that rest on
secondary sources are marked **(secondary)**.

### (a) Who a US retail person can legally trade perps with
- **Binance (USDⓈ-M perps): not available to US persons.**
  - Binance.com does not service US persons under its Terms of Use. The Block, 2020-11-24,
    quotes Binance support: https://www.theblock.co/news/markets/2020-11-24-binance-blocking-us-users-14-day-deadline-new-email-85589
    **(secondary)**. The binance.com/en/terms page did not render for automated reading.
  - Binance.US has no futures.
  - So **the Binance result is evidence about the mechanism, not a strategy the user can run.**
- **Coinbase Derivatives, LLC (CFTC-registered DCM; cleared by Nodal Clear, a DCO):**
  - Lists *nano Bitcoin Perp Style Futures* (BIP, 0.01 BTC) and *nano Ether Perp Style
    Futures* (ETP, 0.1 ETH), first trade date 2025-07-21.
  - Contract specs and funding formula: CFTC submission #2025-32, 2025-06-26:
    https://www.cftc.gov/sites/default/files/filings/ptc/25/06/ptc06262524818.pdf
  - Submission #2026-67, 2026-10-19, removed the 5-year expiry and the word "style" from
    2026-10-26: https://www.cftc.gov/filings/orgrules/rules10192634774.pdf. It states that
    "the funding rate methodology, daily settlement procedures, margining, and clearing
    remain unchanged".
  - US retail reach these contracts through **Coinbase Financial Markets** (FCM, NFA member):
    - Coinbase developer docs: https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/guides/futures.md
    - Retail perps open to US customers since July 2025 **(secondary)**:
      https://decrypt.co/331180/coinbase-opens-perpetuals-trading-us-customers,
      https://www.theblock.co/post/363796/coinbase-launches-perpetual-futures
    - Leverage up to 10× for crypto **(secondary)**.
- **Kraken Derivatives US** (NinjaTrader Clearing, LLC dba Kraken Derivatives US;
  CFTC-registered FCM, NFA ID 0309379):
  - Perps are listed on **Bitnomial Exchange** (a DCM), including PBTCUC and PETHUI.
  - Open to clients who have unlocked US futures on Kraken Pro.
  - Source: https://support.kraken.com/articles/us-perpetual-futures
- **Bitnomial Exchange** (DCM; FCM Bitnomial Clearing, LLC):
  - Its own BTC perpetual (PBUC) has been offered since about 2025-09 (blog post
    2025-09-03): https://bitnomial.com/blog/us-perpetual-futures
- **Kalshi** (KalshiEX LLC, DCM; clearing by Kalshi Klear LLC):
  - BTCPERP, 0.0001 BTC per contract, up to 6× leverage.
  - "available legally and natively to US residents" after KYC.
  - Source: https://help.kalshi.com/en/articles/14679970-what-are-perpetual-futures
    (dated 2026-07-24). CFTC approval was late May 2026 and trading began June 2026
    **(secondary)**.
- Offshore and DeFi perps (Hyperliquid etc.) are out of scope. They are not a legal path
  for a US retail person.

### (b) Funding mechanism and fees
- **Binance USDⓈ-M** ("Introduction to Binance Futures Funding Rates", updated 2026-03-06):
  https://www.binance.com/en/support/faq/introduction-to-binance-futures-funding-rates-360033525031
  - Funding is every 8 h at 00:00, 08:00 and 16:00 UTC.
  - F = [P + clamp(I − P, −0.05%, +0.05%)] / (8/N), with I = 0.01% per 8 h.
  - BTCUSDT and ETHUSDT caps: ±0.75 × maintenance margin ratio at maximum leverage.
  - Since 2025-05-02 the interval can switch to 1 h after a settlement hits the cap.
  - Positive rate: longs pay shorts. Funding is peer to peer, and "Binance does not charge
    fees on Funding Payments".
  - Fees, Regular user (base tier), taker:
    - **spot 0.100%**: https://www.binance.com/en/fee/schedule
    - **USDⓈ-M futures 0.05%**: https://www.binance.com/en/support/faq/binance-futures-fee-structure-fee-calculations-360033544231
      (updated 2026-05-01)
- **Coinbase Derivatives BIP/ETP** (CFTC #2025-32, Appendix B):
  - Funding is calculated **every hour**:
    - Premium = TWAP over twenty 3-minute samples of (futures mark − spot mark)/spot mark/24.
    - Rate = 0.75 × Premium + 0.25 × previous hour's rate.
    - No interest-rate term. No funding cap is stated; orders are rejected beyond ±5% of
      fair value.
  - Payment = rate × quantity × multiplier × futures mark × −1, so positive funding pays shorts.
  - Hourly amounts are aggregated and cash-adjusted twice a day, at the mid-day and
    end-of-day margin runs.
  - **Exchange fee:** CFTC #2026-46, 2026-08-03: https://www.cftc.gov/filings/orgrules/rules08032613651.pdf
    - BIP and ETP, Non-Professional, electronic: **$0.10 per contract per side**.
  - **FCM commission (Coinbase Financial Markets):** retail perp taker "as low as 0.02%"
    **(secondary)**: https://cryptoslate.com/coinbase-starts-cftc-regulated-perpetuals-for-us-traders-offering-10x-leverage-and-0-02-fees/
    - Coinbase futures pages reportedly add $0.15 per contract for exchange, NFA and
      clearing fees **(secondary)**.
  - **Spot leg, Coinbase Advanced entry tier taker:** 0.90% from 2026-09-16 **(secondary)**:
    https://www.altrady.com/blog/crypto-trading-platforms/coinbase-advanced-trade,
    https://www.diariobitcoin.com/exchanges/coinbase-reduce-sus-comisiones-de-advanced-y-baja-el-umbral-inicial-a-usd-10-000/
    The official help.coinbase.com page is blocked to automated reading.
- **Bitnomial perps** (also what Kraken Derivatives US trades):
  https://bitnomial.com/exchange/docs/market-operations/settlements/digital-asset-perpetual-pricing/
  - FR = avg(P) + clamp(0.01% − avg(P), −0.05%, +0.05%), set every 8 h.
  - Positive: longs pay shorts.
  - Kraken all-in perp fee: $0.0113 per contract per side (updated 2026-10-30):
    https://support.kraken.com/articles/us-futures-fees
- **Kalshi BTCPERP:**
  - Funding every 8 h at 00:00, 08:00 and 16:00 ET.
  - Capped at ±2% per interval; set to 0 when |rate| < 0.01%.
  - Positive: longs pay shorts.

### (c) Is historical funding data public? (from documentation; checked by fetching after this plan)
- **Binance:** yes, as monthly public files at data.binance.vision
  (`data/futures/um/monthly/fundingRate/<SYMBOL>/`).
- **Coinbase Derivatives:** documented as a public, no-auth endpoint
  `GET https://api.exchange.fairx.net/rest/funding-rate?symbol=<SYM>&trading_session_date=YYYY-MM-DD`.
  - The spec sets `security: []`. Fields: funding_rate, future_mark_price,
    spot_mark_price, event_time.
  - Docs: https://docs.cdp.coinbase.com/api-reference/derivatives-api/rest-api/funding-rate/get-historical-funding-rates.md
  - History depth is not stated.
- **Bitnomial (and so Kraken Derivatives US):** documented as public, `GET
  /exchange/api/v1/funding-rates/?base_symbol=…&begin_time=…`. History starts at product
  launch: https://bitnomial.com/exchange/docs/api/rest/funding-rates
- **Kalshi:** `GET https://external-api.kalshi.com/trade-api/v2/margin/funding_rates/historical`,
  with no security scheme documented: https://docs.kalshi.com/margin-rest/funding/get-historical-funding-rates.md
- If an endpoint turns out to need a login or key, or is blocked, that venue is reported
  as "history not public" and dropped. There is no workaround: no proxies, VPNs or other IPs.

## Data (free, public, no account)
- **Binance USDⓈ-M funding**, BTCUSDT and ETHUSDT:
  - Monthly `fundingRate` zips from data.binance.vision, from the earliest month available
    through 2026-10, each checked against its `.CHECKSUM` (SHA-256).
  - If the files start after 2020-01, the earliest data is used and this is said in the results.
- **Binance spot daily klines**, BTCUSDT and ETHUSDT: monthly `1d` zips from
  data.binance.vision, same period. Used for the price path (margin risk) and for
  converting per-contract fees.
- **T-bills:** ^IRX (13-week T-bill yield) via yfinance, as in `trend/backtest.py`.
- **Coinbase Derivatives funding**, BIP and ETP:
  - Every trading-session date from 2025-07-21 through 2026-11-01 from the endpoint above,
    at about one request per second.
  - Symbols are as listed by CDE: BIPZ30/ETPZ30, then their successors after the
    2026-10-26 expiry removal.
- **Bitnomial and Kalshi:** BTC (and ETH where listed) funding history, report-only (see below).
- Funding timestamps are rounded to the nearest minute (Binance stamps carry a few ms).
  Duplicates are dropped.

## Position and accounting (UTC calendar days)
- **Position:** long 1 unit spot and short 1 unit perp.
  - Funding received per interval = rate × notional. Positive rate means the short receives.
  - Capital = spot notional + perp margin at 3× = **1.3333 × notional**.
- **Notional:** kept at capital / 1.3333 each day. This is the standard
  constant-weight approximation; the cost of daily re-sizing trades is not charged (see
  limits).
- **Earnings window:** a position held on day d earns every funding event with time in
  **(d 00:00, d+1 00:00]**.
  - The state is held through the 00:00 settlement that ends the day, and any switch
    trade happens just after it.
  - The day's return on capital is the sum of those rates / 1.3333.
- **Cash:** spot coins and perp margin earn nothing while invested. When flat, all
  capital earns T-bills.
- **T-bill daily rate for day d:** ^IRX on the last US trading date strictly before d,
  / 100 / 365. Calendar days are used, since crypto trades 7 days a week.
- **Prices:** spot and perp prices are assumed equal at trade times. Basis changes at entry
  and exit are not modeled beyond the 2 bp slippage.

## Strategies (fixed)
- **C1, always-on:**
  - Hold every day of the period.
  - Costs: 4 trades, entry and exit once. Entry is buy spot + sell perp on the first day;
    exit is sell spot + buy perp on the last day.
  - Each trade costs the venue's base-tier taker fee + 2 bp of notional.
- **C2, switch:**
  - At each 00:00 UTC, compute S(d) = sum of funding rates with time in
    [d 00:00 − 7 days, d 00:00) × 365/7. This is the trailing 7-day mean, annualized,
    using only funding settled strictly before that time.
  - S(d) is undefined (stay flat) unless the data cover the whole 7-day window.
  - If S(d) > 10%, hold the carry on day d. Otherwise stay flat and earn T-bills.
  - Each change of state costs 2 legs × (taker fee + 2 bp): spot + perp.
- **Periods:** discovery 2020-01-01 .. 2023-12-31; confirmation 2024-01-01 .. 2026-10-31.
  - Each period is run on its own. C1 enters on the first day and exits on the last.
  - C2 starts flat, may use funding from before the period start for its first signal,
    and pays an exit if invested on the last day.
  - The Coinbase window starts on the first day with a full 7-day signal and ends
    2026-10-31. It lies entirely inside the confirmation period.
- **Excess return** = strategy daily return on capital − T-bill daily return.
  Cash earns T-bills when flat.

### Costs per trade (fraction of notional)
| venue | spot leg (taker + 2 bp) | perp leg (taker + 2 bp) | one state change (both legs) |
|---|---|---|---|
| Binance | 0.10% + 0.02% = 0.12% | 0.05% + 0.02% = 0.07% | 0.19% |
| Coinbase (US) | 0.90% + 0.02% = 0.92% | 0.02% + 0.02% + $0.15 / (contract size × price) | ≈ 0.96–1.0% |

- For Coinbase, the price is the Binance spot close of day d−1. The contract size is
  0.01 BTC (BIP) or 0.1 ETH (ETP).
- The $0.15 per contract (exchange $0.10 per CFTC #2026-46, plus NFA and clearing) is
  conservative.

## Statistics and reporting
For each asset, venue, period and strategy:
- annualized net return (compounded)
- T-bill return
- mean daily excess × 365
- volatility of daily excess × √365
- Sharpe (mean/sd × √365)
- maximum drawdown of the daily wealth curve
- worst calendar month
- % of days invested
- number of state changes
- gross (no-cost) annualized return, report-only

## Gate (the only pass/fail)
- **Binance, BTC and ETH separately:**
  - Statistic: C2's mean daily net excess return in the **confirmation period**.
  - Moving-block bootstrap: 30-day blocks, 10,000 resamples, seed 0. Block starts are drawn
    uniformly from [0, T − 30]; each resample is truncated to T days.
  - Bonferroni over 2 assets gives a two-sided **97.5% CI per asset**, so the lower bound
    is the **1.25th percentile**.
  - **Pass if the lower bound > 0.**
- **Coinbase (the implementable result):**
  - The same statistic, bootstrap and 97.5% CI on its window (about 15 months, low power).
  - Pass/fail is reported per asset.
- **Verdict:**
  - "Mechanism works" for an asset if its Binance gate passes.
  - "Implementable for a US retail person" only if that asset's Coinbase gate passes too.
  - If Coinbase history is not public, the implementable verdict is "cannot be tested",
    stated plainly.
- The discovery period is report-only. Nothing is tuned on it; the rules above are fixed.

## Margin risk (report-only)
- For each day t, rise(t) = max(daily high over t+1 .. t+7) / close(t) − 1, from Binance
  spot daily klines.
- A day is flagged if rise(t) > 25%. Flagged days less than 7 days apart form one episode.
- For each asset and period, report:
  - number of episodes and flagged days
  - the largest 7-day rise and its date
  - how many episodes started while C2 was invested
- A 3× short perp has margin of 33% of notional. A rise above 25% needs a margin top-up
  or risks liquidation.

## Report-only extras
- **Bitnomial and Kalshi BTC (ETH where listed):**
  - mean annualized funding over their history
  - % of days with S(d) > 10%
  - C2 gross funding return on capital
  - No gate: history too short (Kalshi from 2026-06).
- **Rough re-sizing drag:** the number of times price moved more than ±25% since the last
  re-size while invested, × cost of re-sizing 25% of notional on both legs.

## Not modeled (stated in advance)
- Exchange, FCM or counterparty failure, and stablecoin (USDT) de-pegs. Binance margin is
  in USDT and returns are treated as USD.
- Liquidation itself: margin risk is counted, not simulated.
- Borrowing, interest on margin cash, and staking yield on spot.
- Taxes: each funding payment is income, and futures are §1256.
- Daily re-sizing costs; funding-time execution slippage beyond 2 bp; basis at entry
  and exit.
- Withdrawal and transfer fees between spot and futures accounts.

## Code and tests
- `carry/backtest.py`:
  - `python3 -m carry.backtest fetch` downloads all data
  - `python3 -m carry.backtest run` is the one run
- `tests/carry/` must pass first, including:
  - a no-look-ahead test (the switch decision for a day uses only funding settled before
    that day starts)
  - a cost test
  - a funding-sign test (positive funding pays the short)
- Outputs go to `reports/carry/carry.json`.

## Test count
- 4 gated tests: Binance BTC and ETH, Coinbase BTC and ETH. This is a new family (crypto
  carry).
- Last count seen: 294 (docs/kalshi_plan.md, which says the main session should
  reconcile). 294 + 4 = 298.

## Result (run once, 2026-11-10 14:27:54 PST)
Everything above this heading is the plan as written at 14:10:02 PST (sha256 `d4de0345…d988`,
checked unchanged before this section was appended).

### Timeline
- Plan written: 14:10:02 PST.
- Data fetched: 14:14:14 – 14:26:15 PST, with a Bitnomial refetch at 14:27:37 PST.
- Unit tests: 13 passed.
- Main run: 14:27:53 PST, once.

### What happened to the data plan
- **Binance:**
  - The funding files start at **2020-01**, so there is no December 2019 history. C2's
    first valid signal is therefore 2020-01-08; it sat in T-bills for 2020-01-01..07.
  - 7,488 funding events per asset, all 8 h apart, with no gaps.
  - Spot had no missing days. The large daily moves are real: 2020-03-12 crash; ETH
    2021-01-03, 2021-05-19 and 2021-05-24.
  - The Binance API (fapi.binance.com) is geo-blocked from this Mac (HTTP 451), so only the
    public files were used. That leaves out the 2026-11-01 00:00 print, so the last day
    (2026-10-31) earns 2 of its 3 funding events. This is small and conservative.
- **Coinbase Derivatives:** the endpoint is documented as no-auth, but both BIPZ30 and
  ETPZ30 returned **HTTP 401 "missing request header: CB-ACCESS-KEY"**. Under the plan, its
  history is **not public**, so the implementable test **cannot be run**.
- **Bitnomial:** the first fetch stopped after one page. The cursor is served URL-encoded and
  was encoded a second time (HTTP 400). This was fixed and refetched before the main run. It
  affects only report-only numbers.
- **Report-only code fix before the run:** the US summary now stops at each series' last
  event (BUC ended 2026-06-13).

### Binance results (net of costs; returns on capital = 1.333 × notional)
| asset | period | strategy | ann. return | T-bill | excess (ann.) | vol | max DD | worst month | invested |
|---|---|---|---|---|---|---|---|---|---|
| BTC | discovery 2020–23 | C1 | 11.80% | 1.86% | +9.31% | 1.05% | −1.11% | −0.44% (2020-03) | 100% |
| BTC | discovery 2020–23 | C2 | 9.83% | 1.86% | +7.54% | 1.14% | −0.68% | −0.39% (2023-08) | 44.7% |
| BTC | confirmation 2024–26/10 | C1 | 5.23% | 4.36% | +0.83% | 0.34% | −0.31% | −0.13% (2026-04) | 100% |
| BTC | confirmation 2024–26/10 | C2 | 5.35% | 4.36% | +0.94% | 0.42% | −0.31% | +0.08% (2025-01) | 20.7% |
| ETH | discovery 2020–23 | C1 | 14.80% | 1.86% | +11.97% | 1.35% | −1.34% | −1.16% (2022-09) | 100% |
| ETH | discovery 2020–23 | C2 | 13.21% | 1.86% | +10.58% | 1.40% | −0.75% | −0.25% (2022-03) | 47.0% |
| ETH | confirmation 2024–26/10 | C1 | 5.22% | 4.36% | +0.82% | 0.36% | −0.44% | −0.22% (2026-02) | 100% |
| ETH | confirmation 2024–26/10 | C2 | 5.16% | 4.36% | +0.77% | 0.47% | −0.27% | −0.09% (2025-01) | 21.2% |

How to read the table:
- Excess = mean daily excess × 365.
- Volatility is tiny because the price is assumed perfectly hedged (see limits). It
  measures funding noise only, so the Sharpe ratios (1.6–8.9) are not meaningful risk
  measures.

Mean funding, annualized on notional:
- BTC: 14.97% in discovery, 6.93% in confirmation.
- ETH: 18.51% in discovery, 6.92% in confirmation.

Days with the trailing signal above 10%:
- 45–47% in discovery.
- 21% in confirmation.

C2 costs:
- 44 (BTC) and 38 (ETH) state changes in discovery cost 6.3% and 5.4% of capital.
- 14 and 20 changes in confirmation cost 2.0% and 2.9%.
- C1 pays 0.29% of capital in total.

Year by year, the carry has faded since 2024:
- **C1:**
  - BTC: 2024 +9.2%, 2025 +3.9%, 2026 to Oct +1.8%. T-bills: 5.1%, 4.1% and 3.1%.
  - ETH: 2024 +10.1%, 2025 +3.8%, 2026 +1.2%.
- **C2** was invested 54% / 5% / 0% of days (BTC) in 2024 / 2025 / 2026. In 2025 and 2026
  it was essentially T-bills less switching costs.
- So in both 2025 and 2026, always-on carry earned **less than T-bills** on both assets.

### Gate (C2, confirmation; mean daily net excess; moving-block bootstrap, 30-day blocks, 10,000 resamples, seed 0; 97.5% CI)
- **BTC: FAIL.**
  - Mean +2.58e-5 per day (+0.94%/yr).
  - CI [−0.97e-5, +7.82e-5] per day, i.e. [−0.35%, +2.85%]/yr.
- **ETH: FAIL.**
  - Mean +2.10e-5 per day (+0.77%/yr).
  - CI [−1.87e-5, +7.66e-5] per day, i.e. [−0.68%, +2.79%]/yr.
- Discovery CIs (report-only) were well above zero for both assets: BTC +2.9% to +12.8%/yr,
  ETH +4.8% to +17.4%/yr. The edge was real in 2020–2023 and has mostly been competed away
  since.

### Margin risk (price rose > 25% within 7 days, using daily highs)
| asset | period | episodes | flagged days | largest 7-day rise | episodes starting while C2 invested |
|---|---|---|---|---|---|
| BTC | discovery | 9 | 50 | +43.0% (from 2021-01-01) | 4 |
| BTC | confirmation | 3 | 8 | +32.0% (from 2024-11-04) | 2 |
| ETH | discovery | 24 | 94 | +76.8% (from 2021-01-01) | 14 |
| ETH | confirmation | 6 | 29 | +51.2% (from 2025-05-07) | 1 |

- A 3× short perp loses its whole initial margin on a +33% move.
- The largest moves (BTC +43%, ETH +77% and +51%) would have liquidated an un-topped-up
  short. The spot leg would then have been left unhedged.
- C1 must survive every episode. Running it safely needs spare cash or a margin rule.
- The rough re-sizing drag (report-only) is small: 0.09–0.34%/yr for C1 and 0.03–0.16%/yr
  for C2.

### US venues (report-only, gross funding, no costs, no T-bill comparison)
- **Coinbase Derivatives (BIP, ETP):** history not public; needs an API key. Not tested.
- **Bitnomial**, also what Kraken Derivatives US trades:
  - BTC perp BUC (2025-04-28 → 2026-06-13): funding averaged +10.9%/yr on notional;
    signal above 10% on 91% of days.
  - BTCUC and ETHUI (from 2026-05-18): +12.0% and +11.0%/yr.
  - 95.5–100% of prints are exactly the 0.01% per 8 h default (the formula's fixed interest
    term), i.e. about 10.95%/yr. That is about 8.2%/yr on capital before spot fees,
    liquidity, basis and T-bill opportunity cost.
  - **This is a post-run observation, not a tested result.** History is 6–18 months and
    nothing here was gated.
- **Kalshi:**
  - KXBTCPERP (from 2026-06-03): +10.0%/yr on notional, signal above 10% on 48% of days.
  - KXETHPERP: +0.2%/yr. 45% (BTC) and 78% (ETH) of prints are zero because of the dead band.

### Verdict
- **Mechanism on Binance:** a large, real carry in 2020–2023 (C1 about +9–12%/yr over
  T-bills). In the confirmation period (2024-01 .. 2026-10), the switch rule's excess over
  T-bills (+0.8–0.9%/yr) is **not distinguishable from zero for BTC or ETH**. Since 2025,
  always-on carry has trailed T-bills. **Gate: fail on both assets.**
- **Can a US retail person run it?**
  - It is legal through Coinbase Financial Markets (Coinbase Derivatives perps), Kraken
    Derivatives US (Bitnomial perps) and Kalshi.
  - Binance, where the long history lives, is not available to US persons. **The Binance
    result is evidence about the mechanism, not a strategy the user can run.**
  - The implementable Coinbase test **could not be run**: its funding history needs an API key.
  - A Coinbase retail cost structure would make things worse: about 0.9% spot taker per side
    versus Binance's 0.1%. One C2 switch would cost about 0.7% of capital.
- **Closed** as a money engine on this evidence.
- The Bitnomial/Kraken fixed-floor funding (about 11%/yr on notional when the perp sits near
  spot) is a separate, untested lead. If it is pursued, it needs its own pre-registration,
  with Kraken spot fees, open-interest and liquidity limits, and a fresh out-of-sample
  window.

### Test count
- 2 gated tests were run (Binance BTC and ETH). The 2 Coinbase tests could not be run (no
  public data), so they are not counted.
- Running count: 294 + 2 = **296**. This builds on docs/kalshi_plan.md's 294; the main
  session should reconcile.
