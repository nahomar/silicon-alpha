# Automated 0DTE bot: setup and running

The bot trades the strategy that held up best on unseen data: **#1 sell-only**.
- A ridge model on today's option prices, the calendar and the market backdrop predicts realized ÷ implied.
- The bot sells an iron fly when the prediction is below 0.85.
- It never sells within 60 minutes after a scheduled release.

The first goal is **measuring execution**: how much of the bid-ask gap a mid-price limit order actually pays. The backtests show the edge depends on it (SPX: about +$38k over 2022-24 when paying half the gap, −$4k when paying all of it).

## Shadow book: every candidate strategy, every day
- Besides the order-placing strategy, the bot shadow-trades every candidate on the same live quotes (`odte/live/strategies.py`).
  - `sell_only` places the paper orders.
  - `full_model` also buys the fly when it expects a big move (crash protection).
  - `always_sell` has no model; it is the reference to beat.
- Each shadow trade is priced on the settlement at mid, half the gap and the full touch, each sized to the same $300 worst case. No orders are placed.
- Paper days are new unseen data. A strategy earns real money only if it keeps beating `always_sell` here.
- On unseen history (2017-2021) none of them was profitable. Sell-only was closest (−$5.3k vs −$36.9k for always-sell on the same days).

## Safety
- Paper by default. Live mode is refused unless `allow_live=True`, which is set only on the account owner's explicit go.
- Risk limits (`odte/live/config.py`):
  - $300 worst case per trade (XSP, 1/10 SPX);
  - $600 per day;
  - at most 2 filled trades and 6 order attempts per day;
  - pause for 10 sessions after a $1,500 drawdown.
- Kill switch: create the file `data/live/STOP`, and no new orders are placed.
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

## Maintenance
- Retrain the model after new data is added: `python -m odte.live.model train`.
- Calendar: verified releases through 2027-03 (BLS/Fed pages read 2026-11-09). JOLTS dates after 2026-12-01 were not listed yet. 2027 FOMC dates are tentative.
- Rebuild the calendar with `python3 -m odte.data.calendar build` after adding dates.
- Context (VIX etc.) must be refreshed each morning before the open, so the bot sees yesterday's closes: `python3 -m odte.data.context` (see its `--help`).

## Known limits (v1)
- `prev_ratio` (yesterday's realized ÷ implied) is not yet carried between live days; the model imputes it.
- Shock detection (unscheduled big moves) is not in the live post-event rule yet; only scheduled releases are.
- Paper fills are simulated by IBKR and say nothing about real fill quality. Measuring that needs tiny real XSP orders, only with the owner's go.
