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

