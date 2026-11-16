# M: close momentum on short-gamma days, as a partner for B0 (fixed 2026-11-15, before any momentum return was computed)

Owner: "we need another idea on top of it that can help this strategy so we can combine them".

## Why this idea
- B0 sells 0DTE SPX structures. It earns when the index pins and loses when the index runs.
- Dealer gamma decides which of the two happens (odte/data/gamma.py). When dealers are net short gamma
  they hedge in the direction of the move, which pushes the index further the same way into the close.
- Published evidence: Gao, Han, Li & Zhou (2018, JFE) "Market intraday momentum" (the first half hour
  predicts the last half hour); Baltussen, Da, Lammers & Martens (2021, JFE) "Hedging demand and
  market intraday momentum" (the return up to 15:30 predicts the last 30 minutes, stronger when option
  dealers are short gamma).
- So the partner trade earns on the days B0 is most likely to lose. Same input (gamma), opposite bet.
- Evidence against, recorded up front: our 2013-15 test H4 (momentum via options, 118 sessions) had the
  wrong sign. Small sample, different instrument, but it counts.

## The rule (no fitted parameters)
- Days: SPX 0DTE sessions in data/research/library_SPX.parquet with a 15:30 row (full trading days),
  development data only (through 2026-11-06).
- Signal: r = ES return from the previous session's 16:00 ET price (close of the 15:59 bar) to today's
  15:30 (close of the 15:29 bar), same ES contract at both ends (data/futures/es_1m.parquet, ES.c.0).
  Days where the front contract changed in between (rolls) are skipped.
- Filter: gex_net at the 15:30 row < 0 (dealers short gamma). Known at 15:30.
- Trade: 1 MES contract, long if r > 0, short if r < 0. Enter at the open of the 15:30 bar, exit at the
  close of the 15:59 bar. Held to 16:00, no stop.
- Cost: 0.75 index points per round trip (one tick of slippage each way plus IBKR commission). Also
  reported at 0 and 1.5 points.
- P&L in index points; one MES contract = $5 per point.

## Pass rules
- **M1 (stands on its own):** after 0.75-point costs, mean P&L per trade > 0 in BOTH 2016-2021 and
  2022-2026, AND pooled t-stat >= 2.0.
- **M2 (helps B0):** on B0's walk-forward days (reports/stress/stress_spx_daily.parquet,
  2019-06-26 .. 2026-11-06, touch costs; momentum is 0 on days it doesn't trade):
  - correlation of the two daily P&Ls <= 0.10, AND
  - the combination, each side scaled to equal daily volatility measured on 2019-06-26 .. 2021-12-31
    only, has a higher Sharpe than B0 alone AND a lower probability of a losing year than B0 alone
    (same stationary bootstrap as docs/stress_plan.md T3: mean block 20 days, 10,000 one-year paths,
    seed fixed).
- **Verdict:** M1 and M2 pass -> it becomes a paper-only contender next to B0 in the live bot (MES quotes
  need a CME market-data subscription: the owner decides). Any fail is reported as it is, no tuning.

## Reported only (not used for the verdict)
- Same rule on all days (no gamma filter); on gex_net >= 0 days (expected weaker or reversed).
- Gao signal: previous 16:00 -> 10:00 return instead of -> 15:30.
- Entry one minute later (15:31 bar open); entry at 15:00 and 15:45.
- By year; hit rate; average win and loss.

Code: odte/research/gamma_momentum.py -> reports/momentum/.

## Result (2026-11-15): FAIL (reports/momentum/momentum.json)
- **M1 FAIL.** 707 trades (2,066 SPX 0DTE days considered; short-gamma days with no contract roll).
  - After 0.75-point costs: −1.13 points per trade (−$5.66 per MES), t −1.91.
  - 2016-21: +0.05 (t 0.05, 211 trades). 2022-26: −1.63 (t −2.34, 496 trades).
  - Before costs: −0.38 points (t −0.64). There is no momentum into the close to pay for the costs.
  - By year: positive only in 2018, 2020 and 2021; negative every year 2022-26 (2024 −4.2 points).
- **M2 FAIL.** Correlation with B0 −0.06 overall, −0.28 on the 141 days both traded (it does hedge B0 on
  those days), but it loses money: the combination's Sharpe falls from 0.59 to −0.02 and P(losing
  year) rises from 25% to 51%.
- Reported only:
  - Every variant is negative in 2022-26: no gamma filter −1.20 (t −3.18), gex >= 0 −0.86, Gao first-half-hour
    signal −1.14, entry one minute late −1.50, entry 15:45 −1.50, entry 15:00 −0.18.
  - The published close momentum has not survived into the 0DTE era in our data; if anything the last
    30 minutes lean slightly against the day's move.
- Post-hoc observation, not a finding: momentum was mildly positive in 2019-21, the years B0 lost.
- **Verdict:** not added. No tuning.
