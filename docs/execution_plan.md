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
