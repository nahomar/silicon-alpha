# Master plan: develop on seen data, then ONE final exam (agreed with the owner 2026-11-11)

## Idea (the owner's)
- Search hard for alpha on the data we have already seen:
  - every strategy so far;
  - new mechanisms and patterns;
  - many model types, with the GPU where it helps.
- Combine everything into **one final model**: every strategy and signal (past, present and
  any invented before the freeze) is an input.
- Freeze it, then open the locked period **once**. Measure which features and strategies made
  the money. No new data purchases.

## Data split (fixed)

| Set | SPX | NDX | RUT | Use |
|---|---|---|---|---|
| Development (seen) | 2016-01-08 .. 2026-09-04 (2,044 days) | 2022-05-16 .. 2026-09-04 (1,022) | 2022-05-16 .. 2026-09-04 (901) | anything goes |
| **Exam (locked, one time)** | 2026-09-08 .. 2026-11-06 (44) | same (44) | same (44) | the final test only |
| Live (unseen, growing) | the bot's days from 2026-11-11 | same | same | continuous out-of-sample test |

- The primary development era is **2022-01 .. 2026-09-04** (daily expiries; today's market
  structure). 2016-21 is reported as a robustness check.
- **Guard in code:** `odte/research/exam_guard.py` refuses any date ≥ 2026-09-08 unless the
  environment variable `EXAM_UNLOCK` equals the freeze commit recorded in this file. So
  nothing can read the exam by accident.

## Phase 1: library (development data only)
- **Decision grid:** every 30 minutes, 10:00 .. 15:30 ET (12 per day, as the bot now does),
  for SPX, NDX and RUT.
- **Features, one row per (index, day, time):**
  - session state (move, range, realized vol, implied move, skew, term structure);
  - calendar and events;
  - market context (VIX family, yields, dollar, credit);
  - dealer positioning (SPX open interest);
  - cross-index richness;
  - the factory's signals;
  - every strategy rule tested so far, expressed as a signal: event day, side direction,
    calm vs burst (backward-looking), morning-move direction, VIX regime, time of day.
- **Candidate trades with P&L labels**, all at half of today's gap and at the touch, $1k
  worst case:
  - short iron fly;
  - short iron condor;
  - put credit spread and call credit spread;
  - long straddle;
  - no trade.

## Phase 2: model zoo (walk-forward on development data only)
- Models:
  - ridge / elastic net;
  - LightGBM;
  - a small neural network (PyTorch, Apple GPU);
  - a sequence model on the minute path of the day (GPU, kept only if it beats the
    tabular models in development).
- **One meta-decision:** predict each candidate trade's return; take the best one if its
  prediction is > 0; otherwise no trade. One trade per decision.
- Walk-forward: expanding window, refit every 63 days, evaluated on 2022-01 .. 2026-09-04.

## Phase 3: importance (development data)
- **Group ablation:** remove one feature family or strategy signal, refit the walk-forward,
  and measure the change in P&L.
- Permutation importance as a second view.
- This is the main evidence for "what made money". The exam only confirms the top groups.

## Phase 4: freeze (before the exam)
- One commit fixes:
  - the final model and hyperparameters;
  - the inputs and the decision rule;
  - the selection rule's outcome;
  - the exam report and its pass criteria.
- Its hash is written below. Nothing changes afterwards.
- **Selection rule (fixed now):** the model with the best development-era (2022-26)
  walk-forward Sharpe that is also positive in 2016-21 (SPX). If none is positive there,
  the best 2022-26 Sharpe, flagged.

## Phase 5: the exam (once)
- Train the frozen model on all development data, then run it on the 132 exam index-days.
- **Report:**
  - daily P&L per index and pooled, with a 90% day-bootstrap CI;
  - vs. always-sell and vs. the best single strategy;
  - the ablation of the top groups on the exam.
- **Outcomes, fixed now (44 days is a small sample):**
  - **strong pass:** pooled mean > 0 with the 90% CI lower bound > 0, AND above always-sell;
  - **weak pass:** pooled mean > 0 and above always-sell, CI includes 0. This becomes the
    live bot's lead shadow;
  - **fail:** otherwise.
- **Real money still requires `docs/go_no_go.md`.**

## Freeze record
- Freeze commit: (to be filled at Phase 4)
