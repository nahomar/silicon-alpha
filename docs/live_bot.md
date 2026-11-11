# Automated 0DTE bot: setup and running

The bot trades the strategy that held up best on unseen data: **#1 sell-only**.
- A ridge model on today's option prices, the calendar and the market backdrop predicts realized ÷ implied.
- The bot sells an iron fly when the prediction is below 0.85.
- It never sells within 60 minutes after a scheduled release.

The first goal is **measuring execution**: how much of the bid-ask gap a mid-price limit order actually pays. The backtests show the edge depends on it (SPX: about +$38k over 2022-24 when paying half the gap, −$4k when paying all of it).

## Shadow book: every candidate strategy, every day
- Besides placing paper orders, the bot shadow-trades every candidate on the same live quotes (`odte/live/strategies.py`).
  - `sell_only` places the paper orders.
  - `full_model` also buys the fly when it expects a big move (crash protection).
  - `always_sell` has no model; it is the reference to beat.
  - `event_seller` sells on CPI / jobs / Fed days, outside the hour after a release, unless VIX > VIX3M. No model.
    - It is the only strategy so far that passed a test on never-opened data: Nasdaq/Russell 2025-26.
- Paper orders are placed when `event_seller` OR `sell_only` says sell, within the risk limits.
- Each shadow trade is priced on the settlement at mid, half the gap and the full touch, each sized to the same $300 worst case. No orders are placed.
- Paper days are new unseen data. A strategy earns real money only if it keeps beating `always_sell` here.
- On unseen history (2017-2021) none of them was profitable. Sell-only was closest (−$5.3k vs −$36.9k for always-sell on the same days).

## Dealer-gamma inputs (morning open interest)
- Before the first decision, the bot downloads today's pre-open open interest from Databento (`fetch_oi.fetch_today`, ≈$0.001/day).
  - It covers the next 5 SPXW expiries, ±5% strikes.
  - The window ends at 09:29 ET: same-day OPRA data after 09:30 needs a live licence, and the OI statistic is published before the open.
- From it, every decision gets `gex_net`, `gex_0dte`, `gex_flip_dist`, `maxoi_dist` and `oi_pcr_0dte`: the signal-factory family that predicts move size.
- Scheduled (launchd) runs don't see your shell's environment. Put the key in an owner-only file **yourself**:
  `~/.config/silicon-alpha/databento.env` with the line `export DATABENTO_API_KEY=...`, then run `chmod 600` on it.
  The runner reads it, and the key is never printed or copied.
- If OI is unavailable, the bot logs it and the gamma inputs are blank for that day.

## Candidate models (all shadow-traded every day; `loop.candidates()`)
- The model in charge: `ratio_ridge_full`.
- Challengers:
  - `profit_ridge_full`, `ratio_ridge_small`;
  - `ratio_full_gex` (+ dealer gamma);
  - `danger_full_gex` (+ gamma, sell unless the riskiest 20%);
  - `gex_fly` (factory T_fly composite);
  - `size_filter` (factory T_size composite, rank rule).
- Promotion bar: a paired advantage ≥ max(1.5, Bonferroni z at 10% / number of challengers) standard errors, about 2.13 with 6.

## NDX and RUT (shadow only, `odte/live/xshadow.py`)
- Each day the bot also records NDXP and RUTW 0DTE quotes (NDX every 50 points, RUT every 5, about ±2.5%; 22 strikes each). That needs IBKR's Quote Booster (+100 lines, $30/month): SPX + XSP use 95 of the default 100 lines, and the total with NDX/RUT is about 183. Without it, the index shadows log "no quotes" and the SPX bot is unaffected.
- Shadows, no orders:
  - `event_seller:NDX` and `event_seller:RUT`: the event-day seller on each index's own fly. This is the live evidence the go/no-go rules require per index.
  - `always_sell:NDX/RUT`: the baselines.
  - `xindex_spx`: edge #10, sell SPX only when its predicted realized/implied is the lowest of the three. The live models were trained on different periods (SPX 2016-22, NDX/RUT 2022-24), so the levels can differ. Watch its trade rate: it was 25% in the backtest, and if it trades almost always it is just always-sell.
- Models: `python -m odte.live.xshadow train` (ridge, same features as SPX, pair-trade development rows 2022-05 .. 2024-12) writes `data/live/model_ndx.pkl` and `model_rut.pkl`.
- Closes: Yahoo after the close, else the last implied forward (logged as `settle_index` with its source).

## XSP pricing
- The bot records XSP's own quotes (nearest-the-money ±8 strikes) alongside SPX's. The quote-line budget is split: 34 lines XSP, the rest SPX.
- Orders and an extra shadow P&L (`xsp_mid/half/touch`) are priced on XSP's own bid/ask. If XSP quotes are missing, SPX/10 is used, and the journal records `priced_on`.
- The key live metric is `share_of_gap_paid` per fill, on XSP's book.

## Before any real money (binding, from the plan)
- Full rules, including stop rules: `docs/go_no_go.md`.
- The event-day seller must pass the SPX exam AND at least 60 live event days.
- Both are scored at the touch on the traded symbol's own quotes, per index.
- Then one XSP lot per trade until at least a year of live data.

## Safety
- Paper by default. Live mode is refused unless `allow_live=True`, which is set only on the account owner's explicit go.
- Risk limits (`odte/live/config.py`):
  - $300 worst case per trade (XSP, 1/10 SPX);
  - $600 per day;
  - at most 2 filled trades and 6 order attempts per day;
  - pause for 10 sessions after a $1,500 drawdown.
- Kill switch: create the file `data/live/STOP`, and no new orders are placed.
- Crash or lost connection mid-day: the working order is cancelled, the day's quotes are saved, a `crash` line goes in the journal and an alert fires. A fly that already filled is defined-risk and settles in cash at the close, so nothing needs closing by hand. If the cancel itself fails, the alert says so: check IBKR.
- Alerts: a Mac notification plus one line in `data/live/logs/alerts.log` when the day finishes (orders, fills, P&L, shadow results), when the bot fails, and when the early-close list has no entries for the current year.
- Market holidays (NYSE rules, so future years too): the bot exits at once without connecting.
- End-of-day reconciliation: the bot asks IBKR for open orders, today's executions and any option positions expiring after today, and compares them with its journal. A mismatch raises an alert.
- Every journal starts with the exact code version (git commit, `+dirty` if `odte/` had local edits).
- Live market data is shared with the paper account only while the LIVE login is not in use: stay logged out of the live website and app during market hours (IBKR error 10197 otherwise).
- Orders start at the mid credit and step toward the touch, giving up at most half the gap. Unfilled orders are cancelled and logged, which is data too.

## One-time IBKR setup (account owner)
1. Open an IBKR account. In Account Settings, enable the **Paper Trading Account**.
2. Trading permissions: **options**, at the level that allows **spreads**.
3. Market data (live account, then share it with the paper account under User Settings → Paper Trading):
   - US options (OPRA) top of book;
   - CBOE index data (SPX). This is non-professional, a few dollars a month.
4. Install **IB Gateway** (stable). Log in to the **paper** account.
5. In IB Gateway, Configure → Settings → API:
   - enable socket clients;
   - port **4002**;
   - "Read-only API" OFF (needed to place paper orders);
   - trusted IP 127.0.0.1.

## Check, then run
```
cd ~/projects/silicon-alpha
PYTHONPATH=. .venv-live/bin/python -m odte.live.check     # read-only: connection, data, contracts
PYTHONPATH=. .venv-live/bin/python -m odte.live.runner --mode paper   # one trading day
```
- The runner starts any time before 09:31 ET and exits after the close.
- Journal: `data/live/journal/YYYY-MM-DD.jsonl`. Every decision, order, reprice, fill (with `share_of_gap_paid`), cancel and settlement is logged.
- The day's quotes are saved to `data/live/chains/` in the research format, as new data.

Replay a past day (no broker):
```
PYTHONPATH=. .venv-live/bin/python -m odte.live.runner --mode replay --day 2026-09-04
```

## The weekly learning loop (`odte/live/loop.py`)
Every Saturday:
1. Each new live day (quotes the bot recorded + settlement) becomes training rows, built by the same backtest feature code.
2. Every candidate model is retrained on history + live days before the test window.
3. Each is tested on the last 120 live days (unseen by those fits): sell when it says so, at half the day's own bid-ask gap.
4. Fixed rule, set before any live day and revised after the independent review: **promote** a challenger only if all of these hold over the last 120 live days:
   - at least 60 shared days;
   - at least 30 trades;
   - a positive total;
   - a paired daily advantage over the model in charge of at least **1.5 standard errors**.

   There is no automatic on/off switching; the drawdown pause and the kill switch handle risk.
5. The model in charge is refit on everything and saved as the bot's model.
6. A report is written to `reports/live_weekly/<date>.md`, with every shadow strategy's live results.

Candidates are listed in `loop.candidates()`. New ideas are added there and must earn promotion on live days.
The final exam (SPX ≥ 2026-09-08) is never used for training.

## Running it automatically (owner installs; schedules are in `infra/live/`)
- Daily, Mon-Fri 06:00 Pacific (= 09:00 Eastern): the bot starts, waits for the open, trades and settles.
- Weekly, Saturday 08:00 Pacific: the learning loop.
- Evening, Mon-Fri 13:30 Pacific (`odte.live.watch`): alerts if today's run did not finish or the disk has under 10 GB free, then backs up journals, recorded quotes, alerts and the registry to the PRIVATE repo `nahomar/silicon-alpha-records` (local clone `~/projects/silicon-alpha-records`). The quotes are IBKR market data for personal use: keep that repo private.
```
cp infra/live/com.siliconalpha.bot.*.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.siliconalpha.bot.daily.plist
launchctl load ~/Library/LaunchAgents/com.siliconalpha.bot.weekly.plist
launchctl load ~/Library/LaunchAgents/com.siliconalpha.bot.evening.plist
```
- IB Gateway must be running and logged into the paper account.
- The schedules run under `caffeinate -is`, so the Mac does not idle-sleep while the bot runs. A laptop with the lid closed still sleeps unless it is on power with an external display: keep it plugged in with the lid open from 06:00 to 13:15 Pacific.
- IBKR periodically asks you to log in again; turn on IB Gateway's auto-restart.
- Logs are in `data/live/logs/`.
- The morning context refresh (yesterday's VIX etc.) and yesterday's result are handled by the runner.

## Maintenance
- Retrain the model after new data is added: `python -m odte.live.model train`.
- Calendar: verified releases through 2027-03 (BLS/Fed pages read 2026-11-09). JOLTS dates after 2026-12-01 were not listed yet. 2027 FOMC dates are tentative.
- Rebuild the calendar with `python3 -m odte.data.calendar build` after adding dates.
- Early closes: add each new year's half days to `configs/calendar/early_closes.csv` (2026 is listed; the bot warns when the current year is missing).

## Known limits (v1)
- Shock detection (unscheduled big moves) is not in the live post-event rule yet; only scheduled releases are.
- Paper fills are simulated by IBKR and say nothing about real fill quality. Measuring that needs tiny real XSP orders, only with the owner's go.
