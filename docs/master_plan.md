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
- Freeze commit: ebadab696ba01cf27951dfb32b7769869b725f02

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
- **2026-11-11, N1 result (path shape): NOT KEPT** (reports/meta/n1.json).
  - leader: 2022-26 Sharpe 0.81 (+$579/day); 2018-21 0.64 (+$186); touch +$181. Worse era 0.64.
  - leader + path: 2022-26 0.63 (+$455); 2018-21 0.78 (+$229); touch +$59. Worse era 0.63.
  - The worse era changes by −0.01 (rule: ≥ +0.05), and 2022-26 drops 0.18 (rule: ≤ 0.05).
  - Path inputs alone: 2022-26 −$134/day, Sharpe −0.34; 2018-21 −0.02.
- **2026-11-11, N2 result (GPU sequence model, 3 seeds on Sol): NOT KEPT** (reports/meta/n2.json).
  - Seed average: 2022-26 Sharpe 0.58 (+$264/day); 2018-21 −0.34 (−$76); touch 2022-26 −$152.
    Worse era −0.34 vs the leader's 0.64.
  - Seeds disagree: 2018-21 Sharpe −0.45 / −0.60 / +0.16.
  - 50/50 blend with the leader: 2022-26 Sharpe 1.07 (+$590), 2018-21 −0.84 (−$203). Not kept.
  - Prediction correlation with the leader: 0.40-0.57.
  - Same pattern as term structure: what adds in 2022-26 subtracts before 2022.
- **Development conclusion:** the leader stands as the final-model candidate (ridge alpha 1;
  session, calendar, context, flow, strategy, factory, identity). Freeze awaits the owner's go.
- **2026-11-11, pre-freeze statistical audit (owner asked "are you mathematically confident").
  Verdict: NOT ready to freeze.**
  - **Deflated Sharpe** (Bailey & López de Prado), 2022-26: uncorrected probability the edge
    is real 97%. Corrected for the search: 66% (10 independent tries), 40% (50), 23% (200),
    8% (2,048).
  - **Concentration:**
    - 2022-26 total +$661k. Without the best 1 / 5 / 10 / 20 of 1,142 days: +$481k /
      +$237k / +$19k (Sharpe 0.03) / −$300k.
    - The best 10 days are long straddles (+$647k), e.g. 2024-04-04, a real afternoon
      sell-off with labels checked by hand.
    - Daily skew 4.3, kurtosis 61.
    - The model's edge is "when to buy straddles", carried by about 10 events.
  - **Exam power** (2,000 contiguous 44-day blocks from 2022-26):
    - if the backtest is true: strong pass 9%, weak-or-strong 44%;
    - if the edge is zero: 3% and 32%;
    - confirming a Sharpe of 0.81 needs about 2,400 trading days; 44 days can only
      confirm a Sharpe of about 6.
    - The exam is therefore a disaster and leak detector, not proof.
  - **Leak audit:** factory regime signals are fit on prior years only, and walk-forward
    labels come strictly from earlier days. Clean.
- **2026-11-11, pre-registered before running: two pre-freeze tests** (owner: "yes").
  - **T1, straddle-timing skill** (`odte/research/skill.py`, leader's walk-forward predictions,
    non-post rows):
    - Score = the leader's predicted long-straddle return.
    - Events: E1 = the straddle is profitable at half-gap (base rate 37% in 2022-26); E2 = it at
      least doubles (y > +$1,000 per $1k; base rate 8.9%).
    - **Metric: stratified AUC** (pairs compared only within the same index and decision time,
      so time of day cannot fake skill), per era (2022-26; 2018-21).
    - 90% CI from a day-block bootstrap (500 resamples of days).
    - Reported alongside:
      - the lift of the model's actual straddle picks (event rate / same-stratum base rate);
      - a one-input benchmark, score = −iv_over_rv (cheap vs. realized so far).
    - **PASS = E1 AUC with a 90% CI lower bound > 0.5 in both eras.** FAIL means the straddle
      profits are not shown to be skill, and the final model must not rely on them.
  - **T2, tail-robust re-ranking** of the same 2,048-point grid (re-run on Sol, saving every
    configuration's daily P&L):
    - Metric = Sharpe of daily P&L with the **best 1% of days capped** at that era's 99th
      percentile (upside only; losing days are never trimmed).
    - Rank by the worse era. Also reported: the Sharpe without the best 10 days.
    - **The new leader is the top configuration**, checked for a plateau (its neighbours ±1
      group score close) and with a deflated Sharpe reported.
- **2026-11-11, T1 result: FAIL** (reports/meta/skill_t1.json).
  - E1 stratified AUC: 2022-26 0.529 [0.514, 0.546]; 2018-21 0.529 [0.497, 0.563], so the
    lower bound misses 0.5 by 0.003.
  - One-input benchmark (−iv_over_rv): 0.516 and 0.536.
  - E2 (straddle doubles): 2022-26 0.544 [0.517, 0.572]; 2018-21 0.511 [0.459, 0.571].
  - Straddle picks' lift on E1: 1.07× [1.03, 1.11] and 1.18× [1.08, 1.27].
  - Reading: a small, consistent timing skill, not proven, about the same as one simple
    input. **Per the rule, the final model's trade menu excludes long straddles.**
- **Sell-only menu** (same leader predictions, straddle removed from the menu):
  - 2022-26: +$497/day, Sharpe 1.15; capped 1.12; without the top 10 days 0.83; touch +$183
    (0.43). Mix: fly 44%, call spread 30%, put spread 21%, condor 6%; takes 73%.
  - 2018-21: −$68/day, Sharpe −0.31, vs always-fly on the same days −$73 (−0.23): an excess
    of only +$4.7/day. (Corrected: an earlier draft compared it with always-fly over a
    different set of days, −$114.)
  - Note for every "2018-21" figure in this log: walk-forward predictions start after 500
    days, so the evaluated pre-2022 window is 2019-06-26 .. 2021-12-31 (402 days).
  - The leader's positive 2018-21 came from straddles (without its top 10 days: −1.00).
- **Amended selection rule, fixed before T2's results** (T2 re-run saves both menus):
  - Menu = sell-only (fly, condor, put spread, call spread, or no trade).
  - **Primary = 2022-26 Sharpe with the best 1% of days capped** (today's daily-expiry
    market, the plan's primary era).
  - **Guard:** the 2018-21 daily P&L must beat always-fly (mean excess > 0), so the model's
    choices add value even in the older market where selling itself lost.
  - Plateau check and deflated Sharpe reported. The 5-trade T2 ranking is reported as
    pre-registered.
  - Honest cost: every change of criterion is one more degree of freedom; the deflated
    Sharpe will include it.
- **2026-11-11, T2 result** (reports/meta/t2_table.csv; odte/research/t2.py):
  - **5-trade menu (as pre-registered):** the top remains the old leader family, worse-era
    capped Sharpe 0.44, without the top 10 days 0.03. It is straddle-dependent.
  - **Sell-only menu (amended rule):**
    - Across all 2,048 configurations, 98% have a positive 2022-26 capped Sharpe (median
      1.06). Index and time of day alone give −0.11, so the market inputs carry it.
    - 672 pass the pre-2022 guard; the guard excesses are tiny ($0-7/day), so it is a weak
      filter.
  - **New leader = task 1594: calendar + gex + term + flow (+ identity), ridge alpha 1000,
    sell-only.**
    - 2022-26: capped Sharpe 1.45 (raw 1.48), +$641/day; without the top 10 days 1.21;
      touch +$264/day.
    - By year: 2022 +$163k (2.03), 2023 +$164k (1.51), 2024 +$195k (1.79), 2025 +$153k
      (1.38), 2026 to Sep 4 +$56k (0.68). 77% of 57 months positive; worst month −$50k
      (Jun 2026 −$33k).
    - By index: SPX 2.23, NDX 0.73, RUT 0.73. Mix: fly 51%, call spread 23%, put spread
      22%, condor 3%; takes 75%. By time: positive 10:00-15:00, 15:30 −$19.
    - Pre-2022 (SPX only, 2019-06-26 .. 2021): Sharpe −0.34 (2019 +$9k, 2020 −$12k, 2021
      −$23k), about equal to always-fly (+$6.7/day). **The edge appears only in the
      daily-expiry era.**
    - Deflated Sharpe probability real: 96% / 89% / 80% / 62% for N = 10 / 50 / 200 / 2,048
      (old leader: 66% / 40% / 23% / 8%).
    - Plateau: 9 one-group neighbours, median 1.21 (0.63-1.50).
    - Daily skew −1.04, kurtosis 5.0 (seller tail).
  - **Leak audit of the leader's inputs (code read):**
    - flow = fly mid change over the 5/15 minutes before t;
    - term = next-expiry quotes at t; the "change" inputs compare with 10:00 of the same day;
    - gex = OPRA open interest received between 00:00 and 09:35 ET of the same day (the
      prior close's OI);
    - calendar = scheduled events.
    - All are known at t.
  - **Exam power for this leader** (2,000 44-day blocks):
    - mean > 0: 81% if real vs 46% if zero edge;
    - weak-or-strong pass: 53% vs 16%; strong pass: 7% vs 1%;
    - confirming it needs about 707 days.
- **Status: the sell-only leader (task 1594) is the freeze candidate. Freeze awaits the
  owner's go.**
- **2026-11-11, owner's goal: "win big when we win." Pre-registered before running: T3,
  confident-straddle sleeve.**
  - Buy a long straddle ($1k risk) only when the model's own predicted straddle return
    clears a fixed bar, decided at the decision time (no look-ahead thresholds).
  - Primary tier: predicted > $0, the model's own break-even. Secondary tiers: > $100,
    > $250.
  - Score = the freeze candidate's straddle predictions (task 1594 inputs, ridge 1000);
    the old leader's predictions are reported as secondary.
  - Per era: trades, win rate, average win as a multiple of risk, mean per trade at half and
    touch, 90% day-bootstrap CI, and the mean without the 10 best days.
  - **PASS (sleeve is added beside the sell-only model)** = primary tier mean per trade at
    half > 0 with a 2022-26 CI lower bound > 0, AND mean > 0 pre-2022, AND still > 0 in
    2022-26 without the 10 best days. Otherwise the sell-only model stays alone.
- **2026-11-11, T3 result: FAIL** (reports/meta/t3.json). Per trade, $1k risk, half-gap.
  - **Primary** (candidate's predicted straddle > $0):
    - 2022-26: 8,497 trades, win rate 43%, average win 0.8× risk, mean +$47 [−4, +98];
      without the top 10 days −$14.
    - Pre-2022: mean −$40. All three criteria fail.
  - Bar > $100: 2022-26 +$102 [+15, +200], but pre-2022 −$161 [−283, −30]; without the top
    10 days −$25.
  - Bar > $250: 2022-26 +$151 [−9, +293]; pre-2022 −$90; without the top 10 days −$147.
  - Old leader's predictions: none significant.
  - Wins above 3× risk are about 1% of trades (81 of 8,497); every straddle averages −$65
    per trade in 2022-26.
  - **The sell-only model stays alone.**
- **2026-11-11, exam readiness audit** (file counts and sizes only; no exam prices or results
  read):
  - All 44 exam sessions have SPX/NDX/RUT chains, next-expiry chains, OI files and
    settlement dates. The event calendar runs to 2027-03-31 (40 events in the window).
  - **Bug found:** the OI files for 2026-11-04/05/06 hold 918/580/280 contracts (normal
    about 1,400), because the bulk fetcher took expiries from the settlement file, which
    ends 2026-11-06. Only 1-3 of the 5 expiries were requested. This affects gex_net and
    gex_flip_dist on 3 of 44 days; the 0DTE-only inputs are unaffected.
  - Fixed in `odte/data/fetch_oi.py` (expiries from the session calendar). The corrected
    request is about 1,500 contracts per day, quoted at **$0.0028 total; the re-fetch awaits
    the owner's purchase approval.** The old files will be moved to data/oi/_partial/, not
    deleted.

## Phase 4: FREEZE (2026-11-11, owner: "approve $0.0028 and go")
- **Frozen spec = `odte/research/exam.py` at the freeze commit.** It fixes:
  - model: ridge alpha 1000 (median impute, standardise), one per trade, trained on all
    development rows except post-event rows;
  - inputs: calendar + gex + term + flow + identity (the column list comes from the
    development matrix, and every column must exist in the exam rows);
  - menu: fly, condor, put spread, call spread, or none;
  - rule: best predicted > 0, never within 60 minutes after a release;
  - cost: half-gap (primary), touch reported;
  - outcome rules, unchanged from Phase 5: STRONG / WEAK / FAIL; "above always-sell" =
    pooled mean > the always-fly mean on the same days;
  - importance: drop each of calendar / gex / term / flow, refit, and report the change in
    the exam mean.
- **Pre-freeze checks:**
  - OI for 2026-11-04/05/06 re-fetched (owner approved $0.0028; spent $0.0028):
    1,506 / 1,384 / 1,320 contracts. Old files kept in data/oi/_partial/.
  - `exam.py dryrun` on the last 44 development days: the exam builder reproduces the
    development matrix exactly (1,584 rows, no differing inputs, labels or flags). The
    exam's cost constant is the development value (the cost window holds no exam days).
  - Dry-run score (development days, not evidence): +$1,813/day vs always-sell +$2,271/day,
    which would be FAIL. A calm stretch can fail the "above always-sell" rule even when the
    model makes money; this was included in the pre-exam power (53% / 16%).
  - Test suite: 197 passed.
  - Input fingerprints: `docs/exam_inputs_manifest.json` (229 files, sha256; manifest
    e90d5d18ca67f5d1).
- After the run, nothing in the spec changes. Any later model is new development and is judged
  by live paper trading, not this exam.
