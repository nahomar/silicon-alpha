# Execution study: stop overpaying (fixed 2026-11-11, before any of its data was examined)

Owner: "stop hunting for new signals for now, and point the math at execution: where to place
orders, how long to wait, when to use SPX vs XSP."

## Why
- The locked model earns about $27 per $1k risked per trade at half of SPX's gap.
- On 2026-11-11, the XSP fly's half-gap cost $13 per $1k risked vs SPX's $8, and paying the full
  gap would erase the edge.
- IBKR's paper fills are not usable as execution evidence: both day-1 fills came in above mid
  within 6-10 s.

## Data (owner approved $31.08; odte/data/fetch_exec.py)
- **A:** XSP 0DTE near-the-money 1-minute quotes, 2024-11-11 .. 2026-11-10 (500 sessions;
  data/odte_xsp/).
- **B:** SPX and XSP 0DTE near-the-money 1-second quotes, 2026-08-17 .. 2026-11-10 (61 sessions;
  data/odte_1s/).

## E1: SPX vs XSP cost map (data A)
- For the four shapes (fly, condor, put spread, call spread) at every decision time: XSP's own
  relative gap and cost per $1k worst-case risk vs SPX/10, by time of day, year and VIX level.
- **Re-score B0's decisions priced on XSP's own quotes** (same picks, XSP strikes at SPX/10, the
  research payoff code) for 2024-11 .. 2026-11: the edge at the costs this account pays.
- Also: the risk budget per trade at which one SPX contract fits, and what SPX would save there.

## E2: order policies (data B), on the locked model's decisions in those 61 sessions
- **Fill rule:** a resting sell of the combo fills at the first second its touch credit (short
  legs at the bid, long legs at the ask, from the 1-second NBBO) reaches the limit.
  - This is a LOWER bound on fills: real complex orders can trade inside the NBBO through the
    complex order book. Stated with every result.
- **Policies, each for SPX and XSP:**
  - TOUCH: sell at the touch at the decision second.
  - MID(w): limit at mid; if unfilled after w ∈ {10 s, 30 s, 60 s, 2 min, 5 min}, then (a) cross
    at the touch or (b) skip.
  - LADDER: the bot's current rule (mid, then 5 steps 30 s apart down to half the gap).
  - EDGE(k): limit at mid + k × half-gap for k ∈ {0.25, 0.5}, held 5 min, then skip.
- **Per policy:**
  - fill rate;
  - average share of the gap paid;
  - P&L at the close of filled vs unfilled decisions (adverse selection: do easy fills lose
    more?);
  - **expected P&L per decision** (unfilled = $0), with a 90% day-bootstrap CI.
- **Decision rule:** the policy with the highest expected P&L per decision on XSP replaces the
  bot's LADDER in paper orders only if it beats LADDER by ≥ $2 per $1k risk with a CI lower bound
  > 0. Otherwise the ladder stays. It is then checked on live days.

## Noise
- All comparisons are paired: same decisions, same seconds; only the policy differs.

## Additions (fixed 2026-11-11 22:xx, still before any data of this study was examined)
- **E3 optimal stopping** (data B):
  - Fit a fill-time model: the hazard of the combo's touch credit reaching a limit, as a function
    of distance from mid (in half-gaps), seconds elapsed, time of day, and recent 1-minute
    realized movement of the forward. Discrete-time logistic hazard, fitted on the first 41
    sessions.
  - Solve the Bellman equation backwards over a 5-minute horizon (actions each 5 s: hold,
    improve the limit by one tick, cross at touch, abandon). Value = expected P&L per decision
    using the locked model's edge and the measured adverse selection.
  - **Test on the last 20 sessions** against the E2 policies, with the same decision rule
    (≥ $2 per $1k better than LADDER, CI lower bound > 0).
- **N1 delta-hedged edge** (data B; a measuring tool, not a trading rule):
  - For each decision, the P&L of the sold structure delta-hedged each minute with the parity
    forward, using Black-Scholes deltas from each leg's implied vol at its 1-second mid.
  - This removes "where the index lands" and leaves "was the option overpriced vs what then
    happened".
  - Report its standard deviation vs unhedged P&L, and the locked model's selection skill
    (picked vs not picked) measured this way, with a CI.
  - If the noise falls ≥ 3×, it becomes a secondary metric in every later comparison.
- **S1 sequential looks for the forward head-to-head** (docs/round2_plan.md; fixed before its
  first session, 2026-11-12):
  - Interim looks at 11, 22 and 33 sessions. A contender may be declared WIN early only if the
    one-sided z of its daily difference vs B0 exceeds the O'Brien-Fleming boundary (4.05, 2.86,
    2.34 at looks 1-3; 2.02 at 44). No early LOSE.
  - Otherwise the 44-session rules stand unchanged.
- **K1 sizing (before any real money):**
  - Fractional Kelly with parameter uncertainty and a 1-day CVaR(95%) cap, on the effective
    number of independent bets per day, estimated from the correlation of same-day decisions
    across times and indexes.
- **Clarification before running E2 (2026-11-12):**
  - E2 uses the locked model's SPX decisions where it picked the **fly**, the only structure the
    live bot orders. Each is simulated on SPX and on XSP's own fly (the bot's
    `fly.fly_from_quotes`, wings ≥ 1.5 × straddle).
  - Limits are not rounded to ticks.
  - Fees are the shadow book's 4 × $1.20.
  - The decision second is the half-hour mark. Quotes are the last 1-second NBBO at or before
    each second.

## E2 result (2026-11-12): SWITCH (reports/execution/e2.json)
- **Primary** = the locked model's SPX fly decisions with 1-second NBBO, 2026-08-17 .. 2026-11-10:
  145 decisions / 23 days on XSP, 141 on SPX. Per $1k risk per decision, unfilled = $0.
  - **XSP:**
    - MID300_cross (mid for up to 5 min, then take the touch) **+$131.5**, vs LADDER
      **+$62.1 [+18.5, +101.0]**;
    - TOUCH +$125.1 (+$55.7 [+12.2, +93.0]);
    - LADDER +$69.4, fill rate 57%;
    - "skip" variants fill 3-51% and are all worse.
  - **SPX:** the same ordering. MID60_cross +$211, vs LADDER +$58.8 [−13.8, +121.1]
    (CI includes 0).
  - **Adverse selection, measured:** decisions LADDER filled were worth +$129 at mid; those it
    missed were worth +$191. Easy fills are the worse trades, so missing trades costs more than
    paying the gap.
- **Secondary** (all 721 XSP decision times): the same sign, smaller and not significant
  (MID300_cross +$17.3 [−12.0, +44.0]).
- **Rule met** (best XSP policy ≥ $2 better than LADDER with CI lower > 0): **switch the paper
  bot to MID300_cross**.
  - The final step re-prices at the CURRENT touch, as simulated.
  - Checked on live days via the journal (fills, seconds, share of gap paid).
- **Caveats:**
  - This window's edge was unusually high (+$156 per $1k at mid vs about +$47 in development),
    which favours never missing trades.
  - The NBBO fill rule understates real fills inside the spread, which flatters "cross" vs
    LADDER.

## E4: SPY as the traded instrument (added 2026-11-12, before computing)
- Data: SPY same-day options, 1-minute, 2024-11-11 .. 2026-11-10 (data/odte_etf/SPY; owner approved).
- Same measurement as E1, three ways, at every half-hour 10:00-15:30:
  - cost per $1k worst-case risk of the bot's fly (wings ≥ 1.5 × straddle), at half-gap and at the
    touch, for **SPX vs XSP vs SPY**, each on its own quotes;
  - **the locked model's fly picks priced natively on XSP and SPY** (zoo.trade, quoted spreads,
    k_day = 1; SPY settled at its 16:00 close; American exercise, dividends and share delivery
    ignored, so SPY results are an upper bound until those are handled);
  - the same fees model for all three (a caveat: SPX exchange fees are higher, so this flatters SPX).
- Read-out only (no switch rule yet). If SPY is clearly cheaper, a pre-registered live comparison
  follows, including how to close before 16:00 to avoid share delivery.

## E1 + E4 results (2026-11-12; reports/execution/e1_costs.*, e4_spy_itm.parquet)
- **Cost of the bot's fly** per $1k worst-case risk (medians, 2024-11 .. 2026-11, every half-hour):

  | | Half-gap | Touch | Worst case per contract |
  |---|---|---|---|
  | SPX | $7.69 | $15.38 | $1,425 |
  | XSP | $19.87 | $39.74 | $165 |
  | SPY | $6.17 | $12.35 | $177 |

  - XSP is the dearest at every hour; costs rise into the close (XSP $15.8 at 10:30 →
    $31.4 at 15:30).
  - Costs fell each year (XSP $29.9 → $23.4 → $15.5 for 2024 / 25 / 26).
- **The locked model's fly picks priced natively** (1,861 decisions, 262 days, paired, $ per $1k):

  | | Half-gap | Touch |
  |---|---|---|
  | SPX | **+130.1** [109, 151] | +122.3 |
  | XSP | **+72.2** [52, 93] | +48.2 [28, 68] |
  | SPY held to the close | +137.5 | +130.2 (share delivery: not safe for this account) |
  | SPY, all legs closed at 15:55 / 15:58 / 15:59 | +6.9 / +10.5 / +14.9 | −16.4 / −8.8 / −4.5 |
  | **SPY, only ITM legs closed at 15:59 (1 leg in 93%)** | **+73.1** [54, 91] | +56.1 [38, 74] |
  | SPY-ITM − XSP (paired) | +0.9 [−10.3, +11.9] | +7.9 [−3.4, +19.1] |

- **Read-out:**
  - SPY is no better than XSP once share delivery is handled, and it adds last-minute
    execution and pin risk. **Stay on XSP.**
  - **SPX roughly doubles the per-trade edge (+$130 vs +$72) and is cash-settled; it needs about
    $1.4k worst-case risk per contract.** When the risk budget per trade reaches about $1.5k,
    trade SPX.

## N1 clarification (2026-11-12 21:30 PT, before running; odte/research/noise_hedged.py)
- Decisions: the locked model's 732 SPX decision times in the 61 sessions of data B (every half hour
  10:00-15:30), each scored for **all four menu shapes** (fly, condor, put spread, call spread, the
  research legs of zoo.legs_for), entry at mid plus fees, held to the official close, per $1k
  worst-case risk.
- Hedge: each minute from entry to 15:59, hold minus the structure's Black-76 delta in the parity
  forward (each leg's own implied vol from its mid, carried forward when a leg has no quote), last
  interval to the official close. Mids are the 1-minute NBBO snapshots (cbbo-1m, the same quotes as
  the labels) rather than 1-second data sampled at each minute: same prices to within a second, and
  the same code then runs on the whole history as a secondary check.
- **Primary noise ratio** = std(unhedged) / std(hedged) over all decision x shape rows. The
  pre-registered rule applies: >= 3x -> hedged P&L becomes a secondary metric in every later
  comparison.
- **Selection skill** = mean(picked) - mean(not picked) over decision x shape rows (picked = the
  shape the locked model chose), unhedged and hedged, with 90% day-bootstrap CIs; the CI widths are
  the practical noise comparison.

## N1 result (2026-11-12): rule met, hedged P&L becomes a secondary metric (reports/execution/n1_hedged.json)
- 2,928 rows (732 decision times x 4 shapes, 61 sessions). Per $1k worst-case risk, at mid + fees.
- **Noise:** std unhedged $418 vs minute-hedged $122, **ratio 3.42** (fly 3.39, condor 3.14,
  put spread 3.95, call spread 3.38). Rule (>= 3) met.
- **The locked model's selection skill** (picked shape minus the rest):

  | | Estimate | 90% CI | CI width |
  |---|---|---|---|
  | Unhedged | +$77.7 | [+7.7, +150.0] | 142 |
  | **Hedged** | **+$16.5** | **[+2.0, +31.2]** | 29 |
  | Removed part (where the index went) | +$61.2 | [−2.3, +128.5] | 131 |

- **Read-out:**
  - The model picks options that were genuinely overpriced against the movement that followed:
    the hedged skill is positive with a CI above 0, so the edge is not only lucky direction.
  - **Hedging does not speed up verdicts on this model.** The CI is 4.9x narrower, but the measured
    skill is 4.7x smaller too, so the signal-to-noise is about the same (z ~1.9 both ways).
    Most of the unhedged skill sits in the removed part, which is itself too noisy to call.
  - Use: report hedged P&L beside every comparison. It is the faster yardstick for anything whose
    edge is in option pricing (vol selling, execution), not for direction.
