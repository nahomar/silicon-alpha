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

