# Round 2: challengers vs the locked model (fixed 2026-11-11, before any round-2 result)

Owner: "keep finding new alpha... make another out-of-sample final test and see if it beats
the current locked model."

## Data
- The one-time exam (2026-09-08 .. 11-06) has been run, so **that window is now development
  data**.
- Development = everything through **2026-11-06**.
- The only unseen data is the future. The locked model (B0) runs live from 2026-11-12 as the
  lead shadow.

## Baseline
- **B0 = the locked model:** `odte/research/exam.py` spec (calendar + gex + term + flow +
  identity, ridge alpha 1000, sell-only menu, best predicted > 0, never in the post-release
  hour).
- In round-2 development it is re-run with the same walk-forward as every candidate, so the
  comparison is like for like.

## Candidates (each changes one thing vs B0)
- **Constraint:** inputs only from groups the live bot can compute (session, calendar,
  context, gex, term, flow, strategy, identity). No factory or cross-index inputs.
- **C1 daily-expiry era only:** train only on rows from 2022-05-16 (the first day all three
  indexes have daily expiries).
- **C2 recency weighting:** sample weights halve every 1 year.
- **C3 plateau ensemble:** average the predictions of the 10 best sell-only configurations in
  the T2 table (reports/meta/t2_table.csv) that pass its pre-2022 guard and use only
  live-computable groups.
- **C4 confidence sizing:** risk per decision = $1k × clip(predicted best return / the
  training window's median positive prediction, 0, 2), so average exposure stays about $1k.
- **C5 cost-aware:** train on touch labels and decide on predicted touch return > 0.
- **C6 meta-labeling:** per trade, a logistic model of P(profit at half-gap); expected value =
  p × mean win − (1 − p) × mean loss (both from the training window); take the best if > 0.
- **C7 no late decisions:** B0 without the 15:00 and 15:30 decisions.
- **C8 combination:** all candidates that individually beat B0 on the primary metric,
  combined. If more than one sample-weighting choice (C1, C2) qualifies, the better one is
  used.

## Evaluation (development, walk-forward)
- Same as before: expanding window, refit every 63 sessions; half-gap labels; $1k worst case
  per decision; daily P&L pooled over SPX/NDX/RUT.
- **Common window:** 2023-07-03 .. 2026-11-06, where every candidate, including C1, has
  predictions.
- **Primary metric:** Sharpe of daily P&L with the best 1% of days capped (upside only).
- Also reported: touch, Sharpe without the top 10 days, SPX alone, and the two halves
  (2023-07 .. 2024-12, 2025-01 .. 2026-11-06).
- **Challenger rule:**
  - the candidate (C1-C8) with the best primary metric that beats B0 by ≥ 0.10 AND beats B0
    in both halves;
  - deflated Sharpe reported with N = number of candidates;
  - if none qualifies, there is no challenger and B0 stays.

## Forward head-to-head (the second final test)
- The challenger is **frozen** in one commit (spec + model file) **before** the first test
  session. Its hash is written here.
- **Window:** the 44 sessions after the freeze. Both models are scored on the bot's recorded
  quotes with the research code, same days and same costs.
- **SPX is the primary comparison** (recorded exactly); NDX/RUT are reported (recorded
  approximately).
- **Outcome on the daily difference (challenger − B0), half-gap:**
  - **WIN:** mean > 0 and 90% day-bootstrap CI lower bound > 0. The challenger becomes the
    lead shadow.
  - **LEAN:** mean > 0, CI includes 0. Both keep running, and the difference is re-tested
    after another 44 sessions.
  - **LOSE:** otherwise. B0 stays lead.
- Real money still requires docs/go_no_go.md.

## Log
- **2026-11-11, development results** (walk-forward, common window 2023-07-03 .. 2026-11-06;
  reports/round2/candidates.json). Primary = capped Sharpe at half-gap.

  | Candidate | Primary | vs B0 | Both halves better | Touch $/day | SPX Sharpe | Sep-Nov $/day |
  |---|---|---|---|---|---|---|
  | B0 locked | 1.30 | | | +178 | 2.48 | +303 |
  | C1 era only | 1.33 | +0.03 | no | +210 | 2.35 | +302 |
  | C2 recency 1y | 1.33 | +0.03 | no | +240 | 2.26 | +545 |
  | C3 plateau ensemble | 1.24 | −0.06 | no | +150 | 2.46 | +432 |
  | C4 confidence sizing | 1.60 | +0.30 | yes | +513 | 2.40 | +369 |
  | C5 cost-aware (touch) | 1.58 | +0.28 | yes | +300 | 2.63 | +284 |
  | C6 meta-label | 0.81 | −0.49 | no | −41 | 2.14 | +445 |
  | **C7 no 15:00/15:30** | **1.68** | **+0.38** | **yes** | +377 | 2.59 | +584 |
  | C8 = C1+C4+C5+C7 | 1.62 | +0.32 | yes | **+603** | 1.83 | +29 |

  - C8's parts follow the rule: every candidate above B0 on the primary metric. C1 +0.034
    beat C2 +0.028.
  - **Challenger (rule: the best primary that beats B0 by ≥ 0.10 in both halves) = C7.**
    Deflated P(real) 99% with N = 9.
  - Caveat: the late-decision weakness was seen in development before this round (robustness
    by time of day), so C7 is the most data-informed candidate. The forward test is what
    judges it.
- **Challenger frozen** (`odte/research/round2_forward.py`):
  - C7 = the locked model's own file (data/live/model_frozen.pkl, sha256 1e797d05d29b8fd1…) with no
    decisions at 15:00/15:30.
  - Scored from the bot's journals (lead_labels). Window = 44 sessions from **2026-11-12**.
    SPX is primary.
  - Secondary, not deciding: C4, C5, C8 (data/live/round2_models.pkl, sha256 79d9c75aaa2e3771…;
    C5/C8 trained on development through 2026-11-06).
- Challenger freeze commit: e855f4ec4c4701b19e64eb5454df233e52d480c4

## Round 3: a wider trade menu (pre-registered 2026-11-11, before any result)
- **Idea:** B0 chooses among 4 shapes. Give the same model (B0's inputs, ridge alpha 1000, one
  model per trade) 10 more defined-risk shapes, so it can pick the shape that fits the moment.
  - Flies: wing 1.0σ (narrow), 2.0σ (wide).
  - Condors: short strikes ±0.6σ (tight), ±1.4σ (wide); wings 0.5σ.
  - Put credit spreads: short −0.5σ (near), −1.2σ (far). Call credit spreads: short +0.5σ,
    +1.2σ. All with 0.5σ wings.
  - "Win big" pin butterflies (debit, pay several times the cost if the close lands near the
    centre): long put butterfly centred −0.5σ and long call butterfly centred +0.5σ, wings
    0.5σ.
  - σ = the session's implied move, as in the research code.
- **Labels:** zoo.trade with the library's cost scaling, half-gap and touch, $1k worst case
  (`odte/research/menu3.py`). The existing four labels are recomputed in the same pass and must
  equal the matrix's (an exactness check).
- **Candidates:** R3 = B0 with the 14-shape menu; R3L = R3 without 15:00/15:30.
- **Evaluation:** the round-2 walk-forward, window and metrics.
- **Challenger rule:** the better of R3/R3L becomes round 3's challenger if its primary metric
  beats **C7** (round 2's challenger, 1.68) by ≥ 0.10 AND beats C7 in both halves.
  - Reported regardless: each shape's share of picks and of profit, and the pin butterflies'
    hit rate and payoff multiple.
- **Forward test:** frozen like C7, then scored against B0 on the 44 sessions after its own
  freeze. New shapes are priced offline from the bot's saved SPX quotes.
- **2026-11-11, round 3 result: FAIL** (reports/round2/round3.json). Labels built for 48,919 rows;
  the exactness check on the existing four shapes passed (max difference 0).
  - R3 (14 shapes): primary 0.07 (vs C7 1.68, vs B0 1.30); touch −$637/day.
  - R3L (no late decisions): 0.49; touch −$271/day.
  - **Why:** taking the highest of 14 noisy predictions selects whichever shape's prediction
    is most over-estimated (the optimizer's curse). The high-variance shapes get picked too
    often.
  - **"Win big" pin butterflies:** picked 4,695 times; hit rate 30%; average win 1.9× risk;
    633 wins above 2×; total −$228k. Same verdict as T3: no skill at picking which lottery
    tickets pay.
  - **Observation only (seen after the fact, not a result):** the narrow fly (1.0σ wings) was
    the most profitable pick (+$223k, 2,481 picks, +$90 per pick). Any test of it on these
    data is now biased; only a forward test would be clean.
- **2026-11-11, forward contender NF (owner: "yes"), fixed before any session of its window:**
  - **NF = B0's decisions, except that whenever B0 sells the fly, NF sells the narrow fly
    (wings 1.0σ instead of 1.5σ).** Everything else is identical.
  - Priced after the close with zoo.trade on the bot's saved quotes, using the day's logged
    cost scale.
  - Window and rule: the same 44 sessions from 2026-11-12 and the same WIN / LEAN / LOSE rule
    against B0, SPX primary.
  - Multiplicity: two forward contenders (C7, NF), so a WIN needs the CI lower bound > 0 at
    95% (Bonferroni for two), not 90%.
  - Motivated by a post-hoc observation in round 3, which is why only a forward result counts.

## Round 4: one distribution forecast, every trade priced from it (pre-registered 2026-11-11)
- **Why:** round 3 failed from the optimizer's curse, comparing 14 independent noisy
  predictions. A market maker instead forecasts one distribution of the close and prices every
  structure from it, so the errors are shared, not competing.
- **Forecast:**
  - z = (close − F_t) / straddle_t, from the entry minute's forward and ATM straddle.
  - Scale model: ridge alpha 1000 on B0's 41 inputs, target log(|z| + 0.05); s(x) = exp(fit).
  - Shape: the empirical distribution (200 quantiles) of u = z / s(x) on the training rows.
    No direction is forecast (earlier direction tests failed); u keeps the skew and fat tails.
- **Pricing:** for each shape (legs, entry cash at half-gap and touch, risk, fees, exactly as
  zoo.trade), expected value per $1k risk = (entry cash + E[payoff(F_t + straddle_t · s(x) · u)])
  × 100 − fees, divided by risk.
  - Exactness check: the same table must reproduce the realized labels at the actual close.
- **Rule:** take the shape with the highest expected value if > 0; never in the post-release
  hour.
- **Candidates:** D4 (B0's 4 shapes), D14 (all 14), and D4L / D14L (no 15:00 / 15:30).
- **Evaluation:** the round-2 walk-forward, window and metrics.
- **Challenger rule:** the best D beats C7 (1.68) by ≥ 0.10 and in both halves; otherwise it
  is logged and nothing changes.
- **2026-11-11, round 4 result: PASS → challenger D4L** (reports/round2/round4.json).

  | | Primary | Touch $/day | No top 10 | SPX | Halves | Sep-Nov $/day | Take |
  |---|---|---|---|---|---|---|---|
  | D4 | 2.13 | +628 | 1.77 | 2.70 | 2.02 / 2.27 | +647 | 57% |
  | D14 | 1.91 | +382 | 1.36 | 1.34 | 2.26 / 1.58 | +1,053 | 67% |
  | **D4L** | **2.25** | **+662** | **1.90** | **2.72** | **2.04 / 2.50** | **+802** | 49% |
  | D14L | 1.97 | +507 | 1.45 | 1.44 | 2.30 / 1.67 | +1,465 | 57% |

  - D4L beats C7 by +0.58 and in both halves.
  - Scale forecast calibrated: realized / predicted median |z| = 1.02 on average (0.83-1.27
    per refit).
  - Predicted EV of the picks rises with realized profit across quintiles (realized ≈ half of
    predicted). The 14-shape menu's top quintile has absurd EVs from very cheap pin flies.
  - **Checks:**
    - by year vs B0: 2023 +$85k vs +$78k; 2024 +$273k vs +$195k; 2025 +$233k vs +$153k;
      2026 +$188k (Sharpe 2.66) vs +$69k (0.70);
    - by index: SPX 2.72, NDX 1.76, RUT 1.18; 76% of 41 months positive, worst −$25k;
    - deflated P(real) 98.4% with N = 15 round-2..4 candidates;
    - settings ±: alpha 100 / 10,000 → 2.26 / 2.17; offset 0.02 / 0.10 → 2.24 / 2.30;
      100 quantiles → 2.24.
  - Why it should work: it learns the scale of the move (a far less noisy target than each
    trade's P&L) and prices the known payoffs exactly, so it learns only what it must.
- **D4L frozen as a forward contender** (`odte/research/round4.py`, `round2_forward.py`):
  - Forecast trained on all development rows through 2026-11-06 (data/live/round4_model.pkl,
    sha256 4c9b2ae6da875878…).
  - Rule: the highest-EV of fly / condor / put spread / call spread if > 0; never in the
    post-release hour; no 15:00 / 15:30.
  - Computed from the bot's logged inputs (lead_row) and saved quotes (verified against the
    research table: one-decision path exact; live path within 0.002).
  - Window: the same 44 sessions from 2026-11-12, scored on the logged research labels.
  - **WIN = mean (D4L − B0) > 0 with a 95% CI lower bound > 0** (three contenders: C7's 90%
    rule was fixed first; NF and D4L use 95%). SPX primary.
