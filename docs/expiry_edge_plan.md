# 0DTE ML plan: hold-to-expiry, then tick features

Started 2026-11-07. Goal: use ML models to trade SPXW 0DTE, with each
stage gated on the one before it. No real money until Stage 5, and only on
an explicit go from the operator.

## The approach

At fixed times (10:00, 11:00, ... 15:00 ET) a model forecasts
**realized / implied**: how far SPX will move to the close compared with
how far the options are pricing. Below 1 - margin it sells an ATM iron fly,
above 1 + margin it buys one, otherwise it does nothing. Positions are held
to the 16:00 cash settlement, so the spread is paid once, not twice.

Code: `odte/eval/expiry_edge.py` (study, self-test),
`odte/data/fetch_0dte.py` (Databento downloader).

## Stages

| # | Stage | Cost | Gate to move on |
|---|---|---|---|
| 1 | **Smoke test.** 5 weeks of `cbbo-1m` (2026-10-01 → 11-06, 27 sessions). Ridge (primary) + LightGBM walk-forward vs always-short, always-long, flat, random, and an IV/RV rule. | ~$29 | Pipeline runs clean on real data. *Not* evidence about edge: only ~19 tested days. |
| 2 | **Real test.** 6–12 months of `cbbo-1m`. Same study, same pre-registered rule. | $156–$290 | Primary model beats the best baseline (paired day-bootstrap 95% CI > 0) **and** its Sharpe > 0 at 95%. Synthetic power check: ~250 sessions needed. |
| 3 | **Tick-level features.** Only if Stage 2 is promising or close. Buy a few days of tick data (`cmbp-1`, ~$18/day, or `tcbbo` for trades plus BBO) and build order-flow features: put vs call buying pressure over the last 5–30 min, quote imbalance, aggressor flow near ATM. Feed them into the **same slower forecast**, not a tick-by-tick strategy. Later: TradeFM embeddings as features. | ~$18/day | The tick features improve the out-of-sample forecast over the minute-only model. |
| 2b | **Event calendar features.** Run alongside Stage 2. Flags for CPI, jobs report, FOMC days and Fed-chair speeches, plus earnings dates for the largest index weights. | $0 | Improves the Stage-2 forecast out of sample. |
| 3b | **News and filings NLP.** Run alongside Stage 3. Intraday headline and filing flow with exact publish timestamps: SEC EDGAR (free, accepted-time stamps) and GDELT (free) first, a paid newswire only if those show signal. Features: how many market-moving headlines in the last 30–60 min, how surprising they are, macro release surprise vs consensus. Scope is what moves the **index** (macro, Fed, the largest index weights, geopolitics), not every filing from all 500 companies. | $0 → paid feed | Beats the forecast without it, out of sample, with no look-ahead (text must be timestamped before the decision time). |
| 3c | **Social chatter: Reddit and X.** Run alongside Stage 3. Reddit: r/wallstreetbets, r/options, r/thetagang (heavily 0DTE-focused retail flow). X: index, Fed and macro chatter plus key accounts. Features: post and mention volume vs normal for the time of day, sentiment shift, 0DTE-specific mentions ("SPX calls", "puts printing"), all stamped by creation time before the decision time. **History is the bottleneck:** live collection is cheap but backtesting needs past posts, so start a recorder early (same pattern as the CHOKEPOINT market recorder). Reddit needs API credentials; X's API is paid, so check current tiers before committing. | $0 Reddit; X paid | Same test as 3b: improves the forecast out of sample, with no look-ahead. |
| 4 | **Paper trading.** Live quotes, same rule, `odte/exec/paper_broker.py`, risk gates on. | ~$0 | Several weeks of paper fills consistent with the backtest. |
| 5 | **Small real size.** Operator's explicit go only. Defined-risk flies, fixed max loss per trade. | — | — |

## Not doing: tick-by-tick trading

The firms that profit from tick-level prediction (HFT market makers) win on
colocation and microsecond latency, and they **earn** the spread as market
makers rather than paying it. A retail API pays the spread and arrives
milliseconds late; the repo's own synthetic test showed the spread eating 82%
of a *perfect* next-bar forecast. Tick data is used here only as an input to
the minutes-to-hours forecast (Stage 3), never as a strategy horizon.

## Why NLP fits this strategy

`nlpalpha/FINDINGS.md`: text did **not** predict next-day stock direction,
but chatter **did** predict the size of the move, which makes it a
volatility signal. Move size is exactly what this model forecasts. The
catch: options already price *scheduled* events (straddles cost more on CPI
and FOMC days), so news helps only when it tells the model something the
options have not priced yet, such as unscheduled headlines or how intense
the news flow is today.

## On data granularity

P&L is realized the same day, but the data needed depends on how often the
strategy decides and how long it holds. Deciding six times a day and holding
for hours, one-minute quotes capture what matters for the decision. Tick
data earns its place as **inputs** (Stage 3), and only if it improves the
forecast. It is expensive at scale: one day of `cmbp-1` is ~119 GB and ~$18.

## Findings log

- 2026-11-07, synthetic power test: with a planted edge, the model's own
  Sharpe is clearly positive within 60–120 sessions, but beating
  always-short-fly at 95% took ~250 sessions (1 of 3 seeds). Five weeks
  cannot decide anything.
- 2026-11-07, real data check (2026-10-01): parity-implied forward at 16:00
  was 7666.30 vs SPX close 7666.45. cbbo-1m `ts_recv` marks the end of each
  minute.
- 2026-11-07, preview on 14 sessions (6 tested): mean realized/implied 0.82
  (options rich on average), yet always-short-fly lost over the 6 test days.
  Too small to read.
- 2026-11-08, **Stage 1 complete** (27 sessions, 19 tested, $29.28): pipeline
  runs clean on real data, and every settlement came from the SPX close.
  Realized/implied: median 0.71, mean 0.87. Options overpriced the rest of
  the day most of the time, with fat tails. Always-short-fly made +$3,738
  (Sharpe 1.18, CI [-5.8, +10.1]), sized $1,000 max loss per trade. Ridge
  made +$4,215, only +$25/day better than always-short (CI [-892, +1002]),
  forecast IC +0.07. No evidence of ML skill yet, and none expected at 19
  days. **Risk lesson:** the worst day was -$6,000, all six same-day entries
  hitting max loss on one trending day. Same-day entries are one correlated
  bet, not six, so Stage 2 needs a daily risk cap.
- 2026-11-08, X data options: official pay-per-use ~$5 per 1k post reads
  (full-archive access unclear); third-party X APIs claim $0.02–$0.40 per 1k
  (they scrape X, which X's terms forbid); Stocktwits' public symbol stream
  works free (tested: 30 msgs/call, optional user-tagged Bullish/Bearish),
  history depth untested.
- 2026-11-08, **Stage 2 under way.** 224 more sessions (2025-11-07 →
  2026-09-30) priced at $232.50 and downloading, which gives 251 sessions in
  total. Built alongside it, as cumulative feature sets tested against the
  same model on base features:
  - `+cal` (`odte/data/calendar.py`): CPI, NFP, PPI and JOLTS at their
    *actual* dates (the shutdown moved or cancelled several), FOMC with and
    without projections, FOMC minutes, the 2026 midterm, mega-cap earnings,
    10/20/30y auctions, claims, ISM, opex, quad witching, VIX expiry, month
    and quarter end, sessions next to holidays. Also "event still ahead of
    t" flags (FOMC 14:00, auctions 13:00).
  - `+ctx` (`odte/data/context.py`): VIX1D/9D/30D/3M, VVIX and SKEW at the
    prior close, the 10y yield, dollar, crude, gold, HYG, BTC, SPX realized
    vol, and today's straddle vol vs yesterday's VIX1D.
  - `+soc` (`social/features.py`): hourly Reddit activity (Arctic Shift
    aggregates: WSB, r/options, r/thetagang), plus Stocktwits volume and
    bull/bear lean for SPX, ES and VIX. Each is a z-score against the same
    time slot over the prior 20 sessions.
  - Gaps: GDP and PCE release dates (BEA's archive renders client-side; FRED
    needs a key), and Fed speeches.
  - Caution: with 8 model variants, about 0.4 of them should look
    significant by luck. The pre-registered ml_ridge decides the verdict;
    a variant has to repeat on new data.
- 2026-11-08, **Pre-registration for the 12-month run** (written before any
  12-month result):
  - Sessions: 2025-11-07 → 2026-11-06. Walk-forward with
    `--min-train-days 40` for every model, so the transformer is not starved,
    and margin 0.15. Decisions at 10:00 … 15:00, iron fly held to the close,
    sized to $1,000 max loss.
  - Primary verdict: `ml_ridge` on base features vs the best non-ML baseline
    (paired day-bootstrap 95% CI > 0) and Sharpe > 0 at 95%.
  - Secondary, exploratory with the multiple-comparisons caution printed:
    ridge and LightGBM on cumulative sets +cal → +ctx → +soc → +nlp, plus
    `seq_transformer` (minute-path transformer, ~30k params, monthly refits,
    2 seeds, widest static set) vs LightGBM on the same features.
  - NLP: cardiffnlp twitter-roberta sentiment plus MiniLM similarity to the
    fixed FEAR/GREED phrases in `social/nlp.py`. Text is Stocktwits
    SPX/ES/VIX and WSB posts; Reddit comments enter as counts only. No news
    headlines yet (gap).
  - Command: `python3 -m odte.eval.expiry_edge --seq --min-train-days 40 --write`
- 2026-11-08, **Pre-registration amended before any 12-month result:** the
  user asked for 15-minute checks and event reactions. All models now train
  on every decision row; three trading versions are scored from the same
  predictions:
  1. `fixed6`: 10:00 … 15:00 hourly, margin 0.15. **Decides the verdict**
     (unchanged).
  2. `every15`: checks at 10:00, 10:15, … 15:00 (21 a day). Trades only on
     a strong signal (margin 0.30), at most 1 trade a day (the first).
     Compared with short- or long-once-at-10:00 and random-once-a-day.
  3. `events`: decisions 5 and 30 min after scheduled intraday events (FOMC
     14:00, minutes 14:00, 10/20/30y auctions 13:00, 10:00 JOLTS/ISM), plus
     5 min after a market shock. A shock is a 5-min move above the 99.9th
     percentile of the previous 20 sessions' 5-min moves, at most 2 a day,
     30 min apart. (4x the median was tried first and rejected in testing,
     before any real result: it fired ~twice a day on noise.) New inputs:
     minutes since the event, move since it, and change in implied vol
     since just before it (`+evt` layer).
  Versions 2 and 3 are exploratory. Command:
  `python3 -m odte.eval.expiry_edge --seq --min-train-days 40 --write`
- 2026-11-08, **Money view added** (`odte/eval/portfolio.py`), house config
  pre-registered before any 12-month result:
  - Iron fly. Rest at mid for 5 min, filled only if the market trades
    through mid by a tick, else cross at t+5. Quarter-Kelly sizing, fit
    walk-forward on each strategy's own past out-of-sample trades, base 0.5%
    until 40 trades exist.
  - Limits: <= 2% of equity per trade, <= 6% opened per day (the hard daily
    loss limit for defined-risk trades held to expiry), and a 10-session
    pause after a 15% drawdown. Whole lots, capped at displayed touch size.
  - Sensitivity: cross / mid_or_skip execution, fixed sizing, iron condor
    (shorts at +/-0.75x straddle, wings 1x beyond). Reported at $100k, $1M,
    $10M and $100M.
  - First read (55 test days, noise): displayed liquidity caps 72-95% of
    trades at $10M and 100% at $100M. A single SPX 0DTE fly cannot absorb
    size, so scaling needs more underlyings, structures and real liquidity
    beyond the touch. At $100k a 0.5% bet is smaller than one SPX fly; small
    accounts would need XSP (1/10 size).
  - Next underlyings, regular hours, priced 2026-11-08: NDXP ~$0.70/day and
    RUTW ~$0.54/day, so ~$300 for a year of both. Both cash-settled like SPX.
    `fetch_0dte.py --root NDXP|RUTW` is ready. SPY/QQQ are physically
    settled (exercise risk) and not planned.
- 2026-11-08, **GDELT source amended before any 12-month result:** the DOC
  API rate-limited this machine for hours. GDELT's own route for bulk use is
  Google BigQuery (`odte/data/gdelt_bq.sql`, ~120 GB scanned, inside the free
  tier, 7 s). Topics are GKG theme tags, fixed from a one-day theme listing:
  - market = ECON_STOCKMARKET
  - tariff = EPU_CATS_TRADE_POLICY | WB_775_TRADE_POLICY_AND_INTEGRATION
  - fed = ECON_CENTRALBANK | ECON_INTEREST_RATES | EPU_POLICY_INTEREST_RATE(S)
  - geo = ARMEDCONFLICT | SANCTIONS
  Each 15-min batch counts as known 15 min after its stamp. Using another
  machine's IP to get around the DOC API (and Stocktwits) rate limits was
  considered and rejected.
- 2026-11-08, **Objective B pre-registered (before any result):** the
  user wants both objectives tested, A first.
  - A (existing): forecast realized/implied for every decision and
    sell/buy on it.
  - B, "danger detector": classify danger = realized/implied >= 1.25. The
    default is to sell the fly; skip when P(danger) > 2x the training base
    rate. The flip variant also buys the fly when P(danger) > 3x.
  - B models: logistic (C=0.1) and LightGBM classifiers, on base and all
    features, walk-forward daily. Reported as AUC plus P&L, worst day and
    drawdown vs always-short.
  - The 4-year run (2022-05 → 2026-11) tests both objectives. The last 2
    months stay a locked final exam.
- 2026-11-08, **12-month result (objective A), 251 sessions / 211 tested,
  $1k max loss per trade:**
  - Pre-registered `ml_ridge` (base): −$12.6k, Sharpe −0.88. **Verdict FAIL.**
  - Always-short-fly: +$6.2k, Sharpe +0.18, roughly breakeven after costs.
    Realized/implied: mean 0.93, median 0.76.
  - Best variant `ml_ridge+cal+ctx+soc(+nlp+kalshi)`: ≈ +$51k, Sharpe ≈ 2.0
    [-0.1, +4.2]; vs always-short +$214/day CI [-186, +628], not
    significant; picked after ~40 variants.
  - Drop-one-layer (full LightGBM): calendar is the only significant layer,
    −$169/day CI [+24, +331] when removed. LightGBM gain: market context 50%,
    led by vix1d_over_vix, gap, vvix_prev and spx_rv20.
  - Layers beyond kalshi (news, btc, flows, evt) lowered ridge P&L; most
    LightGBM variants lost (too many features for the sample). The sequence
    transformer made +$6k, no better than ridge.
  - every15 / events versions: small and not significant.
  - Next: replicate the chosen configuration on 2022-05 → 2025-10 (unseen
    for selection), add objective B, and keep the last 2 months locked.
- 2026-11-08, **4-year run pre-registered (written while it runs, before any
  output):**
  - Data: 1,125 sessions (2022-05-16 → 2026-11-06, $12.37). **Final exam:**
    sessions from 2026-09-08 on are excluded (`--holdout-from`) and stay
    untouched until the end.
  - Scoring window = **replication window 2022-07-01 → 2025-11-06**, the
    years not used to choose any variant. Models train on all earlier days,
    refit weekly (`--refit-every 5`). Sets: base, +cal, +cal+ctx; the
    social/NLP/Kalshi layers are left out because their data mostly starts
    in late 2024–2025.
  - Objective A: `ml_ridge+cal+ctx` (the 12-month pick, minus layers
    without history). **PASS** if its paired daily P&L vs always-short-fly
    has 95% CI > 0 and its Sharpe > 0 at 95%.
  - Objective B: `danger_logit+cal+ctx skip`. **PASS** if paired daily
    P&L vs always-short-fly has 95% CI > 0. Drawdown and worst day are
    reported alongside.
  - Everything else is exploratory.
- 2026-11-08, **4-year result (replication window 2022-07-01 → 2025-11-06,
  834 tested sessions, $1k max loss per trade, weekly refits). Final exam
  still locked.**
  - Always-short-fly: +$13.6k on ~5,000 trades, Sharpe +0.09, max DD −$64k.
    Essentially breakeven after costs over 3.4 years (realized/implied
    median 0.76).
  - **A, `ml_ridge+cal+ctx`: FAIL by the pre-registered rule.** +$97.4k,
    Sharpe +0.96 [−0.09, +2.09]; vs always-short +$100/day CI [−103, +314].
    Same sign as in the selection year (12-mo +$42k), so it replicates in
    direction but not significance. Forecast IC +0.05.
  - **B, `danger_logit+cal+ctx skip`: FAIL by the pre-registered rule
    (close).** +$45.2k, +$38/day vs always-short CI [−9, +91]; max DD −$46k
    vs −$64k. AUC 0.535, a weak detector. Flip ≈ skip (the 3x threshold
    almost never fires).
  - Exploratory: selling right after events/shocks loses (always-short on
    event rows −$39k), so don't sell into post-event vol. LightGBM+cal
    +$71k. For LightGBM the ctx layer hurt (drop-one +$126/day better
    without it), while ridge gains from it.
  - Money view of ridge+cal+ctx at $1M, quarter-Kelly: +98% over ~3.4 years
    but max DD −59%, and 57% of trades capped by displayed size. Not an
    acceptable risk profile as is.
- 2026-11-08, **Items 1-3 pre-registered (before any result):**
  1. Exits: close when profit >= 50% of max profit or loss >= 50% of max
     loss, checked each minute at the touch (4 more fees); otherwise hold.
  2. A x B: take `ml_ridge+cal+ctx`'s actions, but drop any short that
     `danger_logit+cal+ctx` flags (P > 2x base rate).
  3. No shorts within 60 min after a scheduled event or market shock.
  - All are scored on the same replication window. Because that window has
    now been looked at, these runs are **development**. One final candidate
    is then chosen and scored **once** on the locked exam (2026-09-08 →
    2026-11-06).
- 2026-11-08, **Items 1-3 result (development, replication window):**
  - **Exits hurt.** Closing at +50% / −50% turned always-short from +$13.6k
    into −$112k, and ridge+cal+ctx from +$97k into −$19k. Stops lock in
    losses on paths that would have recovered by the close, and every exit
    crosses four wide 0DTE spreads and pays four more fees. Keep holding to
    expiry.
  - **A x B equals A.** The danger detector never vetoes A's shorts (A
    already avoids those days), so combining adds nothing.
  - **No shorts within 60 min after events helps everywhere:**
    - always-short +$48k vs +$13.6k
    - ridge+cal+ctx +$107k, Sharpe 1.12 [+0.05, +2.23] (first CI above
      zero), max DD −$38.5k vs −$47.5k
    - B skip +$73k, +$72/day vs always-short CI [+3, +143]
  - **Development candidate for the final exam: `ml_ridge+cal+ctx -postevt`,
    held to expiry, 6 fixed times.** Next: test the dealer-gamma layer on it,
    then score the one final version once on the locked exam.
- 2026-11-08, **Lockbox protocol adopted (user's call): data, once seen,
  cannot test a strategy changed because of it.**
  - Spent (development only): SPX 2022-07 → 2026-09.
  - Sealed: SPX 2016 → 2022-05 in half-year blocks (Mon/Wed/Fri 0DTE days),
    NDXP and RUTW (all years), and the SPX exam 2026-09-08 → 2026-11-06.
  - Each block is opened once with `odte/eval/lockbox.py open`, for the
    strategy as committed (the ledger pins the git hash). Models train on
    all data before the block; data after it is dropped. After a block's
    result, the strategy may be revised, but only later sealed blocks can
    score the revision. The ledger (`docs/lockbox_ledger.json`) refuses
    reopening.
  - Strategy entering the first block: `ml_ridge+cal+ctx -postevt`, 6
    fixed times, held to expiry.
- 2026-11-08, **Lockbox candidates fixed before any block is opened.** All use
  6 fixed times, iron fly held to expiry, no shorts within 60 min after
  events, ridge, trained on everything before the block:
  1. `ml_ridge+cal+ctx -postevt` (the development pick)
  2. `ml_ridge+cal+ctx+gex -postevt` (+ dealer gamma from pre-open OI)
  3. `ml_ridge+cal+ctx+es -postevt` (+ ES overnight path: return, realized
     vol, range, volume, last 15 min)
  - Benchmarks: always-short-fly and always-short-fly −postevt.
  - Blocks are opened oldest first: SPX 2016H1 → 2022H1, then NDX and RUT,
    then the SPX exam.
  - A candidate is dropped or revised only between blocks; the ledger shows
    each change.
- 2026-11-08, **TradeFM gate replaced, before any run.** The old gate (beat
  53% next-tick direction) passes on book imbalance and bid-ask bounce,
  which do not pay the spread. New gate (`odte/eval/tick_check.py`): on
  cmbp-1 ticks, trade the model's most confident 1% of price-change events
  as a taker, out after 30 s, after fees. Retrain TradeFM only if
  LightGBM's top-1% P&L has a 30-minute-block 95% CI above $0 AND beats an
  imbalance-only model.
  - The 4 tick "eval" days (2026-09-11 / 09-16 / 10-06 / 10-20) are inside
    the locked exam, so they are excluded.
  - Train: the earliest 12 pre-exam tick days. Test: the latest 4
    (2026-03-31, 04-08, 06-05, 06-22).
  - Synthetic checks: no planted signal → 50% accuracy (no leak); planted
    imbalance signal → found (74%), and correctly NO-GO for the big model,
    since the simple rule matches it.
- 2026-11-08, **TradeFM gate result: NO-GO, so do not retrain.** Sol job,
  14 min CPU, 17.7M train / 6.0M test price-change events.
  - Direction accuracy: LightGBM 51.9%, imbalance-only 50.5%, base rate
    49.6% up. That also fails the old 53% bar.
  - Taker P&L per contract, 30 s hold, after fees, most confident 1%:
    - LightGBM −$208 (CI −$230 to −$187), win rate 19%;
    - imbalance-only −$15.5 (CI −$17.5 to −$13.7).
  - 59% of events had a move that would have paid in hindsight, but the
    models cannot tell which way.
  - Caveat (post hoc, does not change the verdict): the most confident
    calls fall on expensive, wide-spread contracts.
  - Conclusion: at the 30-second horizon, tick data gives no tradeable edge
    for this project. TradeFM stays shelved; no H100/H200 retrain.
- 2026-11-08, **Dev screen for candidates 2 and 3, rule fixed before the run.**
  - Data: seen SPX only, 2022-05-16 → 2026-09-04 (`--data-from`,
    `--holdout-from 2026-09-08`). Report window 2022-07-01 → 2026-09-04.
  - Settings: the same as the lockbox (min 40 train days, weekly refit, sets
    +cal, +cal+ctx, +gex, +es).
  - Keep a candidate for the lockbox only if, in fixed6 with −postevt, both:
    (a) its mean daily P&L minus #1's (paired, same days) is > 0, and
    (b) its Sharpe ≥ #1's.
  - Otherwise it is dropped before any block is opened. This is a screen,
    not proof; the sealed blocks decide.
- 2026-11-08, **Dev screen result (seen SPX 2022-07-14 → 2026-09-04, 1,041
  days, fixed6 −postevt, $1k max loss per trade).**

  | strategy | total | Sharpe [95% CI] | max DD | vs #1 per day [CI] |
  |---|---|---|---|---|
  | #1 ridge+cal+ctx | +$86.7k | 0.77 [−0.22, +1.74] | −$37.3k | — |
  | #2 +gamma | +$63.5k | 0.53 [−0.41, +1.51] | −$37.7k | −$22.2 [−92, +51] |
  | #3 +ES | +$89.9k | 0.79 [−0.19, +1.76] | −$27.5k | +$3.1 [−43, +50] |
  | always-short −postevt | +$32.5k | 0.19 | −$53.3k | |
  | always-short | +$10.6k | 0.06 | −$64.1k | |

  - Rule applied: **#2 dropped** (fails both conditions). **#3 kept** (passes
    both, barely; it differs from #1 on 314 days).
  - Lockbox candidates entering block 1: #1 and #3.
  - Note: on this longer window #1's Sharpe CI includes 0 (the earlier
    2022-07 → 2025-11 run gave 1.12). That is why the sealed blocks matter.
- 2026-11-08, **SPX 2016-01-04 → 2022-05-13 downloaded ($9.60): 963 sessions,
  all with settlements.** Decided before opening anything:
  - SPX 2016H1 is a training-only warm-up (nothing earlier to train on) and
    is never scored.
  - First scored block: 2016H2.
  - Strategies per block: #1 `ml_ridge+cal+ctx -postevt` and #3
    `ml_ridge+cal+ctx+es -postevt`, with always-short as the reference.
  - Each half-year block has roughly 75 sessions (Mon/Wed/Fri expiries before
    2022), so single-block results are noisy. The overall verdict is the
    pooled result across all scored blocks, decided after the last one.
- 2026-11-08, **Scoring rule for the remaining blocks (user approved), fixed
  after 2016H2 and before 2017H1 is opened.**
  - Why: 2016H2 lost at its own touch because bid-ask was 15% of mid then,
    vs about 2% today. Quote-only spreads: 2016 15%, 2018 8%, 2020 5%,
    2021 4%, 2022H1 2.3%.
  - PRIMARY score from 2017H1 on: the same decisions priced at the fly mid
    ∓ 1.69% (`TODAY_COST`: the median entry cost over the seen 2022-07 →
    2026-09 decisions) plus $1.20 per leg, each trade sized to $1k max loss.
  - The real historical touch result is still reported for every block.
  - 2016H2 keeps its touch-only record and is not re-scored.
  - Check on seen data (2025-09..11): actions saved in the decisions file
    reproduce the report's touch totals exactly; today-cost totals there are
    close to touch, as expected when costs match.
- 2026-11-08, **Revision after 2017H1 (user approved): sell-only variants**,
  scored from 2017H2 on. In 2017H1 the models bought the fly 66-95 times
  while options were overpriced. Sell-only turns every model "buy" into
  "sit out".
  - The block strategies are now #1, #3, #1 sell-only and #3 sell-only.
  - The originals are unchanged, so their record stays continuous.
  - The sell-only variants are judged only on blocks from 2017H2 on.
- 2026-11-08, **Iron fly lockbox PAUSED after 2021H1 (user's call).**
  - Ten blocks opened. Pooled at today's cost, the model is about equal to
    always-short; the best variant (#1 sell-only) is −$5.3k over 637 days.
  - SPX 2021H2 and 2022H1 stay sealed, reserved for the next strategy's
    real test.
- 2026-11-08, **New idea (user): buy calls or puts and sell them the same day,
  based on market context and news. Step 1, written before any run:** can
  the S&P's direction 30-60 minutes ahead be called at all?
  (`odte/eval/direction.py`)
  - Development days only: SPX 2016-01 → 2021-06 and 2022-05-16 →
    2026-09-04. Sealed days are refused by the code.
  - Models: logistic regression and LightGBM, walk-forward by day.
  - PASS if, for H = 30 or 60 min, a model beats the best simple rule
    (always-up / momentum / reversal) on both accuracy and average captured
    move, each with a 99% day-resampled CI above 0.
  - An option buyer's break-even move (today's spread plus the hour's decay)
    is reported alongside.
  - FAIL means stop: plan B is a sequence transformer on minute paths, then
    TradeFM.
- 2026-11-08, **Direction step 1 result: FAIL** (1,651 tested days, 27k / 22k
  rows).

  | H | always-up | logistic | LightGBM | captured move vs option break-even |
  |---|---|---|---|---|
  | 30 min | 52.9% | 52.4% | 51.6% | +0.26 bp vs about 3.3 bp |
  | 60 min | 53.3% | 52.2% | 51.6% | +0.27 bp vs about 6.3 bp |

  - Neither model beats always-up. Even the most confident 20% capture
    under 1 bp, about 10x short of what an option buyer needs.
- 2026-11-08, **Plan B (user): sequence transformer on minute paths**
  (`odte/eval/direction_seq.py`), fixed before the run.
  - Inputs: the last 120 minutes plus the same 87 inputs.
  - Two heads (30 and 60 min); refit every 63 days; 2 seeds.
  - Same rows, days and gate as step 1.
  - If it fails too, TradeFM is not retrained for this: there was no signal
    at 30 s in ticks and none at 30-60 min here.
- 2026-11-08, **Direction stage A (user approved $148 of ES order flow): add
  order flow and headline content.** Fixed before the data arrives.
  - Data, development year only: SPX sessions 2025-09-02 → 2026-09-04.
    - ES front-month trades with aggressor side, reduced to per-minute buy
      and sell volume (`odte/data/es_flow.py`).
    - GDELT headlines from about 30 major outlets on finance, Fed,
      inflation and trade policy (`odte/data/gdelt_titles.sql`), scored by
      the same offline sentiment and fear/greed model as the social data.
  - New inputs at t:
    - ES aggressor imbalance over the last 5 / 15 / 30 / 60 min;
    - big-trade (≥20 lots) imbalance over 15 / 60 min;
    - overnight imbalance;
    - volume vs the same minutes on the previous 20 days;
    - headline count, mean sentiment and fear − greed over the last 15 / 60
      min and overnight.
    - Headlines count only from batch + 15 min.
  - Walk-forward inside the year: first 100 days training only, refit every
    21 days.
  - Models: logistic and LightGBM, each with and without the new inputs.
  - Gate: the step 1 gate applied to the WITH models (vs the best simple
    rule, accuracy and captured move, 99% CIs above 0, H = 30 or 60). The
    with-minus-without difference is reported.
  - PASS → stage B: buy the same data for the sealed 2021H2-2022H1 and test
    once, then simulate option trades. FAIL → stop the option-buying idea.
  - About 150 test days, so power is limited.
- 2026-11-08, **Plan B result (sequence transformer, minute paths): FAIL.**

  | H | always-up | transformer | captured move | most confident 20% | break-even |
  |---|---|---|---|---|---|
  | 30 min | 52.9% | 52.8% | +0.11 bp | 54.9%, +0.46 bp | 2.9 bp |
  | 60 min | 53.3% | 53.6% | +0.77 bp | 56.1%, +2.03 bp | 5.5 bp |

  - vs the best rule at 60 min: +0.28% accuracy [−0.62, +1.25], +0.18 bp
    [−0.84, +1.28].
  - It is the best model so far, but not significant, and its most
    confident calls capture under half of an option buyer's break-even.
  - Per the pre-registration, TradeFM is not retrained for direction.
  - Stage A (order flow and headlines) continues.
- 2026-11-08, **Headlines-only run of stage A (free), fixed before the run.**
  - The ES order-flow download was lost to a session restart, so the re-buy
    is on hold pending the user.
  - Same stage A design and gate, with the new inputs = the headline
    features only (`--no-flow`).
- 2026-11-08, **Headlines-only stage A result: FAIL.** Dev year, 155 test days.
  - 30 min: logistic with headlines 52.3% vs 51.5% without (always-up
    51.9%).
    - vs the best rule: +0.35% [−3.9, +5.1].
    - The most confident 20% capture +1.6 bp vs a 3.6 bp break-even.
  - 60 min: nothing (49.9%).
  - Headline content adds a little at 30 min, nowhere near enough.
- 2026-11-08, **Pair-trade idea (user approved): relative value across SPX,
  NDX and RUT 0DTE.**
  - Each decision, sell the iron fly on the index the model thinks is most
    overpriced (lowest predicted realized/implied) and buy it on the least
    overpriced.
  - Both legs are equal-premium sized, the pair scaled to $1k worst case.
  - Crash days hit both legs, so most of the crash risk cancels.
  - **Data split, fixed before any NDX or RUT data was looked at:**
    - Develop: 2022-05-16 → 2024-12-31.
    - Sealed, one test: 2025-01-01 → 2026-09-04.
    - Exam (≥2026-09-08): stays locked.
  - The SPX side of 2025-26 has been seen before, but the pair's result
    depends on the unseen NDX and RUT legs.
  - The final pair rule, benchmarks and gate are written here before the
    sealed period is opened.
- 2026-11-09, **Pair trade, development result (2022-05 → 2024-12): FAIL, so
  the sealed 2025-26 NDX/RUT period is NOT opened for it.**
  - At the touch: model pairs −$162k over 440 days (random pairs −$259k;
    the model beats random by +$104/day [+18, +186]).
  - Passive execution with the strict fill rule (mid only if the market
    traded through it, else cross 5 min later): model −$149k.
  - NDX and RUT fly bid-ask is 7.5% / 9.9% of mid vs 1.9% for SPX. A pair
    pays two legs of that, which swamps the ranking skill.
  - Bug found and fixed before any conclusion: the first passive run priced
    every order at mid. Strict mid-fill rates in the backtest are SPX 26%,
    NDX 8%, RUT 3%.
  - Always-short fly, dev, by share x of the bid-ask gap paid at entry
    (0 = mid, 1 = touch):

    | index | x=0 | x=0.25 | x=0.5 | x=0.75 | x=1 |
    |---|---|---|---|---|---|
    | SPX | +$81k | +$59k | +$38k | +$17k | −$4k |
    | NDX | +$98k | $0 | −$89k | −$171k | −$247k |
    | RUT | +$157k | +$86k | +$21k | −$40k | −$96k |

  - The premium is real, but it sits inside the bid-ask spread. In SPX,
    selling pays if we capture about half of the gap. Whether that is
    achievable is an execution question that only live (paper) trading can
    answer.
- 2026-11-09, **Model improvement round (user asked), fixed before any run**
  (`odte/eval/improve.py`).
  - Development data: all seen SPX fixed6 rows (2016-01 → 2021-06 and
    2022-05-16 → 2026-09-04; 1,901 days). Walk-forward by day: first 250
    days training only, refit every 21 days.
  - All strategies keep the no-short-within-60-min-after-event rule.
  - Scoring: today's cost, entering at HALF the gap (mid × (1 − 0.0085)),
    $1k worst case per trade. The full-gap touch (mid × (1 − 0.0169)) is
    reported too.
  - Baselines:
    - C0: the current bot (ridge on realized/implied, 55 inputs, sell if < 0.85).
    - Always-sell.
  - Candidates:
    - C1 profit target: ridge predicts the half-gap return on risk of
      selling the fly; sell if > 0.
    - C2 danger detector: logistic P(half-gap return on risk ≤ −0.5);
      sell unless P > 2 × the training base rate.
    - C3 small model: C0's rule on 22 inputs chosen by reasoning (the 12
      base, plus ev_fomc, ev_cpi, ev_nfp, events_ahead,
      minutes_to_next_event, vix_prev, vix1d_over_vix, vix_chg,
      spx_ret_prev, spx_rv5, iv_vs_vix1d, megacap_earnings).
  - Selection: the candidate with the highest development Sharpe at
    half-gap that also beats both C0 and always-sell on paired mean daily
    P&L. If none does, C0 stays.
  - The winner then gets ONE test on the sealed SPX 2021H2 + 2022H1, gate
    written before opening. If it passes, it joins the bot's shadow book.
- 2026-11-09, **Improvement round, development result (1,648 tested days,
  half of today's gap):**
  - C1 profit target: +$131k, Sharpe 0.73 [−0.00, +1.54], +$40/day vs C0,
    +$31/day vs always-sell.
    - But −$36k in 2017-21 and +$168k in 2022-26; max DD −$71k.
  - C0: +$65k, Sharpe 0.55.
  - C2: +$108k, Sharpe 0.40.
  - C3: +$59k, Sharpe 0.64, the only one positive in both eras; it fails the
    selection rule vs C0 / always-sell.
  - Always-sell: +$80k, Sharpe 0.29.
  - **Winner by the pre-registered rule: C1.**
- 2026-11-09, **Sealed test of C1, gate fixed before opening** SPX 2021H2 +
  2022H1 (2021-07-01 → 2022-05-13), the last sealed SPX blocks.
  - Same walk-forward (all earlier days), same post-event rule and scoring
    (half of today's gap, $1k worst case per trade).
  - **PASS if:**
    (a) C1's total P&L > 0, AND
    (b) its paired mean daily P&L beats both C0 and always-sell.
  - Touch-cost results are reported as well.
  - About 175 days: a pass is evidence, not proof. It puts C1 in the bot's
    shadow book next to C0 for the live forward test.
  - FAIL means C0 stays and C1 is dropped.
  - Both blocks are recorded in the lockbox ledger as opened.
- 2026-11-09, **Sealed C1 test result: FAIL.** SPX 2021-07-01 → 2022-05-13,
  143 days, half of today's gap:
  - C1: −$36.5k, Sharpe −2.47.
  - C0: −$9.4k, Sharpe −0.99.
  - Always-sell: −$72.9k (the 2022 bear market).
  - C1 beat always-sell by $254/day but lost to C0 by $190/day.
  - C0 stays as the bot's model, and C1 is dropped.
  - No sealed SPX data remains apart from the exam (≥2026-09-08).
  - Across every unseen period, the model's value so far is losing much
    less than always-sell in bad markets, not making money on its own.
- 2026-11-09, **Re-evaluation. Event-day seller, found by slicing seen SPX
  data (hypothesis, not proof).**
  - Always-sell outside the hour after releases, at half of today's gap:
    - CPI/NFP/FOMC days: +$86k, $65/trade, 269 days, mean $320/day
      [CI 23, 601], positive in 8 of 11 years, every event type positive.
    - Other days: +$3k.
  - Also: inverted VIX term structure (VIX/VIX3M > 1) was −$97/trade.
  - Correction: CPI and NFP are 08:30 releases, so those days are sold after
    the news. On FOMC days the measured trades are 10:00-13:00, i.e. before
    the 14:00 statement (14:00 and 15:00 are inside the post-event hour).
- 2026-11-09, **One-shot test of the event-day seller on SEALED NDX and RUT,
  2025-01-01 → 2026-09-04 (never opened). Fixed before the run.**
  - Rule (no fitting):
    - On CPI / NFP / FOMC days (ev_cpi, ev_nfp, ev_fomc, ev_fomc_sep), sell
      the iron fly at the six fixed times;
    - skip a decision within 60 min after an intraday scheduled release;
    - skip the day if yesterday's VIX/VIX3M > 1.0.
  - Comparison group: the same rule on all other days.
  - Each trade sized to a $1k worst case. Priced at mid, at half of the
    index's own bid-ask gap, and at the touch.
  - Built with build_dataset + calendar/context only: no models are fitted
    and no other strategy is scored on this data.
  - **PASS (the pattern is real):** pooling both indices, event-day minus
    other-day mean daily P&L at mid > 0 with a 95% day-resampled CI above 0,
    and each index's difference > 0.
  - Also reported: whether event days are profitable at half of their own
    gap (NDX/RUT gaps are 4-5x SPX's).
  - PASS → the event seller joins the live shadow book. FAIL → drop it.
  - NDX and RUT 2025-01 → 2026-09 are recorded in the ledger as opened.
- 2026-11-09, **Event-day seller on sealed NDX/RUT 2025-01 → 2026-09: PASS.**
  - NDX, 48 event days: event days +$784/day at mid, +$625 at half-gap,
    +$483 at the touch; other days +$89 / −$104 / −$275.
  - RUT, 48 event days: +$419 / +$175 / −$32 per day; other days
    −$47 / −$308 / −$534.
  - Pooled event-minus-other at mid: +$581/day, 95% CI [+53, +1088].
  - Caveats:
    - 48 event days per index.
    - The same calendar days and market as the seen SPX 2025-26, and NDX/RUT
      move with SPX. So this rules out an SPX-only artifact, but it is not
      independent in time.
    - The truly independent tests are the live paper days and the SPX exam
      (≥2026-09-08).
  - Next: the event seller joins the live bot's shadow book.
- 2026-11-09, **Independent review (separate model, read-only, re-ran the
  numbers)** and the changes made in response, all before any live day.
  - Findings accepted:
    - The NDX/RUT "pass" is not an independent test: same 48 days,
      same-day P&L correlation with SPX ~0.75.
    - The pooled-at-mid gate was too lenient: each index's own CI includes
      0, and mid is rarely fillable.
    - RUT rests on 5 days.
    - Only three event types were chosen after looking: PPI and jobless
      claims days were negative, which contradicts the stated mechanism.
    - Negative in 2018, 2022 and 2023.
    - The reviewer puts the chance the edge is real at ~25-30%. Realistic
      P&L at $300 per trade: about +$1k/yr typical (best +$3k, bad −$2k).
  - **Learning loop rule, revised:**
    - Promote only on a paired advantage ≥ 1.5 standard errors over ≥ 60
      shared live days (window 120), ≥ 30 trades, positive total.
    - The automatic on/off switching of model orders is removed; risk is
      handled by the drawdown pause and the kill switch.
    - The old $25/day margin was ~6% of one SE and would have chased noise.
  - **XSP parity:**
    - The bot records XSP's own quotes (nearest-the-money ±8 strikes; SPX
      gets the remaining quote lines).
    - Orders and an extra shadow P&L are priced on XSP's own bid/ask, not
      SPX/10.
  - **Size:** one XSP lot per trade until ≥ 1 year of live data.
  - **Gate before ANY real money (binding):** the event-day seller must
    pass both
    (a) the SPX exam (≥2026-09-08), and
    (b) ≥ 60 live event days.
    Both are scored at the touch on the traded symbol's own quotes, per
    index, not pooled.
  - Open question to answer before trusting the pattern: why PPI and claims
    days lose while CPI/NFP/FOMC days win. Write the expectation down
    before checking.
- 2026-11-09, **Edge search. Hypotheses and gate written BEFORE any fresh data
  is bought or seen.**
  - Fresh, time-independent test data: SPX 0DTE April 2013 → December 2015.
    Only Fridays had a same-day expiry then, about 140 sessions. It is
    before every period used so far, so it is independent in time (the
    gap the review found).
  - Each hypothesis has a mechanism and a predicted sign:
    - **H1 News-day seller.** On CPI / NFP / FOMC days, outside the hour
      after the release, not in stress: selling the iron fly beats ordinary
      days.
      - Mechanism: options carry an event premium that outlives the release.
      - Prediction for PPI / jobless-claims days, stated now: no better than
        ordinary days (smaller, less-feared releases).
      - In 2013-15 mostly NFP Fridays are testable.
    - **H2 Afternoon seller.** Selling at 13:00-14:00 beats 10:00-11:00.
      - Mechanism: intraday volatility is U-shaped, so midday-to-afternoon
        sessions are calmer than a 0DTE price spread evenly over the day
        implies.
    - **H3 Stress filter.** On days when yesterday's VIX > VIX3M, selling
      loses money.
      - Mechanism: inverted term structure = crisis regime with jumps.
    - **H4 Published intraday momentum** (Gao, Han, Li & Zhou, J. Financial
      Economics 2018). The 09:30-10:00 return predicts the 15:30-16:00 return
      with the same sign.
      - Tested first on the underlying; then buy a 0DTE call or put at 15:30,
        held to the close, at the touch.
  - Scoring:
    - Today's cost (1.69% of mid; half the gap) for the fly hypotheses, with
      the touch also reported, $1k worst case per trade.
    - H4 on the index first (does the sign hold?), then the option trade at
      the touch.
  - Gate:
    - A hypothesis PASSES on fresh data only if its effect has the
      predicted sign with a 98.75% day-resampled CI above zero (Bonferroni
      across the four).
    - The development (seen) data is used only to rank and to check the
      code, never as evidence.
  - PASSING hypotheses join the bot's shadow book, then face the SPX exam
    and live days.
- 2026-11-09, **Fresh-data test details, fixed before the 2013-15 data
  arrives** (`odte/eval/fresh_2013.py`).
  - Bought: 118 SPX 0DTE sessions for $0.73.
  - Official BLS (NFP, CPI, PPI) and Fed (FOMC) dates for 2013-15 added
    from bls.gov and federalreserve.gov.
  - No verified 2013-15 early-close list was found, so every possible early
    close (Jul 3, Dec 24, the Friday after Thanksgiving) is dropped,
    preventing the post-bell leak.
  - H1 compares event days with "other" days. The PPI/claims-only days are
    the mechanism check (prediction: no better than other days).
  - H2 is per day: (13:00 + 14:00) − (10:00 + 11:00).
  - H3: the stress-day mean must have a CI entirely below 0.
  - H4 uses the parity forward: 09:31 → 10:00 move, then 15:30 → settlement;
    the option trade is reported at the ask.
  - The block is recorded in the lockbox ledger when opened.
- 2026-11-09, **Fresh-data test, SPX 2013-04 → 2015-12 (113 sessions):
  NONE of H1-H4 passed.**
  - H1 news-day seller:
    - Event days +$123/day vs other days −$40/day (half gap).
    - Difference +$163/day, 98.75% CI [−1229, +1628]: the predicted sign,
      but only 30 event days, far too few to confirm or reject.
    - PPI/claims days minus other days −$207/day: consistent with the
      stated mechanism prediction (no better than ordinary days).
  - H2 afternoon seller: −$128/day [−397, +137]. Wrong sign; fails.
  - H3 stress filter: only 8 stress days, mean +$162. Untestable here.
  - H4 intraday momentum:
    - Index capture −2.3 bp [−7.3, +2.4]. Wrong sign.
    - Buying the 0DTE option at 15:30 lost −$51 per contract [−97, −2].
    - Rejected.
  - The news-day seller has now shown the predicted sign in three samples
    (seen SPX 2016-26, NDX/RUT 2025-26, SPX 2013-15). None of them is
    independently conclusive. It stays a hypothesis for the SPX exam and
    live days.
  - No more hypothesis tests on old data are planned. Further evidence must
    come from live paper days.
- 2026-11-09, **Signal factory (user: "think like a quant"). Protocol fixed
  before any run** (`odte/research/factory.py`).
  - Data: all seen SPX fixed-time decisions, 2016-01 → 2026-09-04 (the exam
    is excluded), 12,197 rows / 2,044 days.
  - **Signals:** about 50, in six families, each computed at decision time
    from information known then:
    - option pricing: IV/RV, IV level vs its own history, IV vs VIX, skew
      and skew vs history, yesterday's realized/implied, VIX9D/VIX,
      VIX/VIX3M, VVIX/VIX, VIX change, quoted spread;
    - price action: gap, move so far, absolute move, range, recent vs
      day-long realized vol, SPX 1d return, SPX RV5/RV20;
    - calendar: every event flag, events/FOMC ahead, day of week, time of
      day, minutes to close;
    - dealer positioning: GEX net / 0DTE, flip distance, max-OI distance
      (signed and absolute), 0DTE put/call OI;
    - cross-market: rates, dollar, crude, gold, HYG, BTC, ES overnight
      return / RV / range / volume, ES last 15 min;
    - regime (idea B): a 4-state k-means of daily market state (VIX, VIX/VIX3M,
      SPX RV20, 20-day SPX return, VVIX), refit each year on earlier days
      only, used as walk-forward target encodings.
  - **Targets** (z = (settle − F) / straddle mid, F = forward at t):
    - T_fly: half-gap return on risk of selling the iron fly (today's cost);
    - T_dir: z (which side the close lands on; idea A);
    - T_size: |z|;
    - T_tail: 1 if |z| > 1.5.
  - **Statistic:** Spearman IC over all rows, with a cluster-robust (by day)
    t-stat.
  - **A signal × target SURVIVES only if both hold:**
    (1) Benjamini-Hochberg FDR q ≤ 0.05 across ALL signal × target tests; AND
    (2) the IC has the same sign in 2016-2021 and 2022-2026.
  - Survivors are combined per target in a walk-forward ridge composite
    (first 250 days training only, refit every 21 days).
  - A composite that improves its target's trading rule on the development
    data becomes a CHALLENGER in the live loop and is judged only on live
    days.
  - Development data never counts as proof.
- 2026-11-09, **Signal factory result (development data only; proof must come
  live).**
  - 2,041 days, 63 signals, 240 tests: 25 pass BH-FDR q ≤ 0.05, and all 25
    keep the same sign in 2016-21 and 2022-26.
  - The dealer-positioning family (gex_net, gex_flip_dist, gex_0dte,
    oi_pcr_0dte, all from pre-open open interest) dominates every target.
    These are correlated, so effectively one signal.
    - Long dealer gamma → smaller moves (IC vs |z| −0.077, t −4.9).
    - Also predicts tails and the side of the close.
  - Composites (walk-forward ridge on each target's survivors):
    - T_fly (ev_month_end + gex_net): OOS IC +0.051 (t 3.0). Selling when
      predicted > 0: +$148k over 1,791 days, Sharpe 0.72 [0.01, 1.49], vs
      always-sell −$16k. But 2016-21 −$10k, 2022-26 +$158k.
    - T_size danger filter (9 signals; skip the predicted-riskiest 20%):
      OOS IC +0.067 (t 3.9), +$122k, positive in both eras (+$6k / +$115k).
    - T_tail: OOS IC +0.062.
    - T_dir (OI put/call, skew, gamma): OOS IC +0.060, side hit rate 53.0%.
      Monetizing it needs vertical spreads (not built).
  - Tension to note: adding GEX to the 55-input ridge earlier (#2) did not
    improve P&L, while on its own it is the strongest family. A focused model
    may use it better than a wide one, or the earlier test was noisy.
  - Next per protocol: the T_fly and T_size composites become live
    CHALLENGERS. They need the pre-open OI each morning
    (`odte/data/fetch_oi.py`, ≈$0.001/day).
- 2026-11-09, **Why GEX "didn't help" the wide model, resolved** (same
  2016-26 rows, walk-forward, half gap):
  - OOS IC vs |z|: 55-input ridge 0.051 → with GEX 0.066 → GEX alone 0.050.
  - Rule "pred < 0.85": +$61k → +$86k with GEX.
  - Rule "skip the predicted-riskiest 20%": +$124k → +$125k.
  - gex_net is the #2 standardized coefficient of 60, and its IC barely
    falls after removing the wide model's prediction (−0.066), so it adds
    information.
  - The earlier "−$22/day" drop of #2 used training data from 2022 only and
    a short test window; its CI [−92, +51] was uninformative.
  - The trading rule matters as much as the inputs: the danger-ranking rule
    roughly doubles P&L versus the overpriced-threshold rule.
  - New live challengers:
    - `ratio_full_gex`: the 55-input ridge plus 5 GEX inputs, ratio rule;
    - `danger_full_gex`: the same model, selling unless the prediction is
      in the riskiest 20% of its training predictions;
    - `gex_fly`: the factory T_fly composite;
    - `size_filter`: the factory T_size composite, rank rule.
  - With 6 challengers, the promotion bar is Bonferroni-adjusted:
    z ≥ max(1.5, Φ⁻¹(1 − 0.10/n_challengers)), about 2.13 for six.

