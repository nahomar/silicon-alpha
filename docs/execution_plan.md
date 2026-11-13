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
