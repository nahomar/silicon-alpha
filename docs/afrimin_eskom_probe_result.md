# Eskom → PGM transmission probe — result

**Verdict: no transmission lag detected. Do not build the ingestion layer. Do
not buy vendor data.**

This is the gate defined in [`afrimin_track.md`](afrimin_track.md), run to
completion. Total cost: **$0**.

## Question

Does an escalation in South African loadshedding predict subsequent
underperformance in SA PGM miners, beyond what the metal price and the broad
market already explain?

This was chosen as the *decisive* case, not merely the first: South Africa is
~70%+ of mined global platinum, deep-level PGM mining is power-intensive so the
mechanism is physically causal, stage data is daily, and the instruments are
liquid US-listed ADRs. If transmission lag does not appear here, it is unlikely
to appear in thinner cases.

## Data

The stage history was reconstructed at $0 from the **git history of
`manually_specified.yaml`** in the open-source `beyarkay/eskom-calendar` project
(778 revisions, 3,024 unique announcement intervals, each citing an Eskom
announcement). See [`afrimin/data/build_stage_history.py`](../afrimin/data/build_stage_history.py).

The reconstruction independently reproduces known South African history, which
is the main reason to trust it:

| year | % days with loadshedding | mean stage | max |
|---|---|---|---|
| 2022 (from Jul) | 81% | 2.95 | 6 |
| **2023** | **98%** | 4.12 | 6 |
| 2024 | 29% | 0.83 | 6 |
| 2025 (to May) | 10% | 0.26 | 4 |

2023 as the worst year on record, the collapse to **0% from June 2024** (the
~10-month suspension), and a maximum of Stage 6 (Stage 8 was threatened but
never implemented) all match the public record. A faulty git-walk would not
reproduce that pattern.

## Result

Window 2022-07-01 → 2024-05-01 (468 sessions; train 327 / eval 141, time-split,
never shuffled). Miner returns are residualized against platinum (PPLT) and the
broad market (SPY), with factor betas fit **on train only**. t-statistics use
Newey-West HAC errors. Stage is lagged one full session.

| ticker | t (out-of-sample) | beta | t (in-sample) | dir. acc | |
|---|---:|---:|---:|---:|---|
| SBSW | −0.52 | −1.4e-03 | −0.87 | 45.2% | |
| IMPUY | +0.76 | +3.0e-03 | −1.09 | 38.1% | |
| ANGPY | −0.18 | −6.2e-04 | −0.76 | 52.4% | |
| GDX | −1.24 | −2.2e-03 | −1.79 | 54.8% | **placebo** |

All out-of-sample |t| < 2. Directional accuracy is at or below a coin flip.
Notably the **placebo carries the largest magnitude** — gold miners, which have
minimal SA grid exposure — which is the opposite of what a real
power-curtailment mechanism would produce.

## Reading it honestly

- **This is the expected result, and it is informative.** Loadshedding was
  continuous, pre-announced, and the single most-covered story in South African
  media for two years. A supply constraint that everyone can read about in
  advance is a supply constraint that is already in the price. The thesis needed
  African supply information to be *slow to reach* global price discovery;
  loadshedding is the case where it demonstrably was not.
- **It does not disprove the broader thesis.** It tests one mechanism, at one
  horizon, in the most-publicized case. Less-covered events (DRC export policy,
  Guinean political disruption, Zambian smelter outages) plausibly transmit more
  slowly precisely *because* they are less covered. But per the staging rule,
  that is now a hypothesis requiring its own probe — not a licence to build.
- **The value-capture research view (claim (a)) is unaffected.** It was never
  contingent on this.

## Two bugs this run caught

Both produced results that *looked* fine, which is why they matter.

**1. Forward-fill past the end of the record.** `align_to_sessions` used
`reindex(method="ffill")`, which carried the final stage across ~352 sessions
beyond 2025-05-15 — silently asserting "the grid never changed again" and
manufacturing a zero-variance tail. Now NaN outside coverage, with a warning.

**2. A confident null from a test with no power.** The first run reported
"NO TRANSMISSION LAG DETECTED" with every out-of-sample t-statistic *exactly*
0.00. The eval split had landed entirely inside the post-June-2024 dead zone, so
the regressor was constant zero. The probe had tested nothing and said so
confidently — the same failure shape as the corrupted directional target in
[`data_integrity_finding.md`](data_integrity_finding.md). `_require_power` now
refuses to render any verdict when a split holds fewer than 20 non-zero events.

The second is the more dangerous class of bug, and the reason the result above
is quoted only from the windowed run: **a run that completes while measuring
nothing is worse than one that crashes.**

## Reproduce

```bash
PYTHONPATH=. python -m afrimin.data.build_stage_history --out data/eskom_stages.csv
PYTHONPATH=. python -m afrimin.probe.eskom_pgm --stages data/eskom_stages.csv \
    --start 2022-07-01 --end 2024-05-01
```

Network-dependent (clones a public repo, pulls free price data), so intentionally
**not in CI** — matching the precedent in
[`signal_probe_result.md`](signal_probe_result.md). The guards themselves are
offline and *are* tested: `pytest tests/afrimin`.
