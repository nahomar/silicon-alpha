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

## Development log
- **2026-11-11, Phase 2 zoo (walk-forward, 136 inputs, best-of-five trades).** Pooled daily
  P&L at half-gap over 2022-26, summed over indexes and decisions, $1k risk each:
  - **ridge: +$537/day, Sharpe 0.75 [−0.13, 1.55].** By index-day: SPX +$425, NDX +$54,
    RUT +$81. 2018-21 −$59/day. Takes 92% of decisions: fly 33%, long straddle 32%, put
    spread 15%, call spread 14%, condor 6%.
  - MLP (PyTorch): +$97/day, Sharpe 0.20; 2018-21 +$47/day.
  - LightGBM: −$18/day, Sharpe −0.03; 2018-21 −$123/day.
  - elastic net: pending.
  - Baselines, every decision: always fly +$113 (Sharpe 0.16); condor −$140; put spread
    −$95; call spread −$104; long straddle −$1,990.
  - Under the selection rule as written, ridge is excluded (negative 2018-21). Development
    continues before any freeze.
- Phase 3 group ablation for ridge and MLP submitted to Sol (job arrays 1095832, 1095833).
- **2026-11-11, Phase 3 group ablation, development** ($/day change in 2022-26 when the group
  is removed; minus = the group helps):
  - **ridge** (base +$537):
    - helps: identity −69, strategy signals −58, calendar −24, term −21, session −9;
    - neutral: gex ~0;
    - hurts: flow +17, cross +42, factory +71, context +124 (but removing context makes
      2018-21 −$196);
    - **removing term turns 2018-21 positive (+$103) while 2022-26 stays +$516.**
  - **MLP** (base +$97): helps gex −146, context −112, identity −67, term −41; hurts
    calendar +64, strategy +164.
  - **Elastic net** (all inputs): +$517/day, Sharpe 0.73; 2018-21 −$78.
- **Phase 3 search** (Sol array 1096783): every subset of 9 groups (identity always kept) × ridge
  alpha {1, 10, 100, 1000} = 2,048 walk-forward runs on development data. It will be ranked by
  the **worse of the two eras' Sharpe** (robust objective) before any freeze. Caveat recorded
  now: searching 2,048 configurations on the same data overfits it; the exam is the judge.
- **2026-11-11, Phase 3 search result** (2,048 runs, development; reports/meta/search_summary.csv):
  - 410 of 2,048 are positive in both eras.
  - **Leading robust family:** session + calendar + context + flow + strategy + factory
    (± gex, ± cross), ridge alpha 1.
    - 2022-26: Sharpe 0.81, +$579/day.
    - 2018-21: Sharpe 0.64, +$186/day.
    - Touch 2022-26: +$181/day.
    - Its neighbours (adding or removing gex / cross) score almost the same: a plateau, not
      a lone spike.
  - **Term structure appears in none of the top 50 robust configurations.** The best
    2022-26-only configurations all use it (calendar + gex + term + flow, Sharpe 1.74,
    +$1,027/day), and all lose in 2018-21 (Sharpe −0.5 to −0.9).
    - Likely mechanism: before 2022, SPX had Mon/Wed/Fri expiries, so the "next expiry"
      was often two days away, and the term signal meant something else.
  - Top-50 robust: alpha 1 in 27, alpha 10 in 17. Groups: session / context / factory 34
    each, strategy 32, calendar 28, cross 27, gex 26, flow 24, term 0.
  - Caveat: the leader was selected from 2,048 runs on the same data, so its Sharpe is
    optimistic. The exam decides.
- **2026-11-11, robustness of the leading family** (ridge alpha 1; session, calendar, context,
  flow, strategy, factory):
  - **By year:** 2019 +$15.6k, 2020 +$51.5k, 2021 +$7.8k, 2022 +$66.5k, 2023 +$67.5k,
    2024 +$383k, 2025 +$233k, **2026 (to Sep 4) −$89.0k**. Monthly swings in 2025-26 are
    large (e.g. Nov 2025 +$62k, May 2026 −$62k).
  - **By index** (2022-26): SPX Sharpe 1.20 (+$432/index-day), RUT 0.41, NDX 0.23.
  - **By time:** positive through 14:00; 14:30 +$4, 15:00 −$6, 15:30 −$19 per decision.
  - **By chosen trade** (mean $ per decision): fly +48.7, condor +37.4, put spread +11.2,
    straddle +3.4, call spread −9.4.
  - Touch 2022-26: +$181/day, Sharpe 0.26.
  - Entry margin (predicted $ per $1k): 0 → Sharpe 0.81; $25 → 0.89; $100 → 0.97 (39% of
    decisions).
  - It often gains when always-sell loses badly (Nov 2025, Apr and Jun 2026).
  - **Warning recorded before the exam:** 2026 year-to-date is negative.
- **2026-11-11, new ideas, pre-registered before any result.** Base = the leader (ridge alpha 1;
  session, calendar, context, flow, strategy, factory). "Worse era" = min(Sharpe 2022-26,
  Sharpe 2018-21), half-gap; the leader's is 0.64.
  - **N1 path shape** (`odte/research/paths.py`, group `path`, 7 inputs from the minute path
    up to the decision only):
    - inputs: trend vs reversion (1-minute autocorrelation, 5-minute variance ratio); implied-
      vol surprise vs square-root-of-time decay; 15-minute straddle momentum; minutes since
      the day's high and low; drop from the high in implied moves;
    - **kept if** leader + path raises the worse-era Sharpe by ≥ 0.05 and lowers neither era's
      Sharpe by more than 0.05.
  - **N2 sequence model on Sol GPU** (`odte/research/seq.py`):
    - a 1-D convolutional network over the day's minute path (future minutes masked out),
      joined with the leader's inputs, predicting all five trades' returns;
    - same walk-forward (start after 500 days, refit every 63); 3 seeds averaged;
    - **kept if** its worse-era Sharpe beats the leader's; then also tested as a 50/50 blend
      with the leader's predictions.
  - These are the last development candidates for path data. A candidate that is kept joins the
    final model; otherwise the leader stands.
