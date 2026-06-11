# Signal-presence probe — real-data result

**Question:** does a gradient-boosted tree find next-bar *directional* signal in
real intraday market data, using a leakage-safe pipeline?

**Why this exists:** the OPRA directional question
([`infra/modal/dir_baseline.py`](../infra/modal/dir_baseline.py)) needs paid
OPRA shards on Modal. There is no free source of historical intraday OPRA data
(yfinance only exposes *current* option-chain snapshots, which have no time
axis to compute returns from). So this runs the *same diagnostic methodology* on
free intraday **equity** bars — a proxy for the pipeline, not the OPRA answer.

## Result (7 days × 25 liquid tickers, 1-minute bars)

| metric | value |
|---|---|
| train / eval bars | 54,017 / 13,500 (time-split, 80/20, no shuffle) |
| eval base rate (up) | 49.05% |
| **LightGBM accuracy** | **49.93%  (z = −0.17 vs 50%)** |
| baseline: always-up | 49.05% |
| baseline: momentum | 50.58% |
| **verdict** | **NO extractable signal (~50%)** |

Top features by gain: minute-of-day, r1, r5, r15.

## Reading it honestly

- **This is the expected, correct result.** Liquid equities at a 1-minute
  horizon are near-efficient; a tree should *not* find directional edge, and it
  doesn't (z = −0.17 is indistinguishable from a coin flip). A model that
  *did* claim 55%+ here would be a red flag for look-ahead leakage.
- **It validates the machinery, not a strategy.** What it establishes: the
  corrected per-instrument feature pipeline + a leakage-safe time-split +
  LightGBM probe run end-to-end on real data and return an honest verdict.
- **Even a marginal edge here would not be tradeable** — 1-minute moves of this
  size vanish under spread + fees. Signal-presence ≠ tradeable alpha.

## What this does *not* answer

The 0DTE SPX options thesis. Options microstructure (gamma, pin risk, dealer
hedging flow, the bid/ask structure of a 0DTE chain) is a different and
plausibly less-efficient regime than 1-minute equity bars. Testing it requires
real OPRA data.

## To run the real OPRA test

1. Re-pack OPRA with the corrected pipeline (per-contract returns + retained
   `instrument_id`; see [`data_integrity_finding.md`](data_integrity_finding.md)).
2. Run `dir_baseline.py` (instrument-aware) on the re-packed shards.

Until then: the 0DTE directional thesis is **untested**, and this equities probe
is the honest ceiling of what is freely checkable.

## Reproduce

```bash
PYTHONPATH=. python -m odte.eval.signal_probe
PYTHONPATH=. python -m odte.eval.signal_probe --tickers AAPL MSFT NVDA --days 7
```

(Network-dependent — pulls live yfinance data — so it is intentionally NOT in
CI. Numbers above are from one 7-day window and will shift run-to-run with the
trailing market data; the *verdict* — no signal — is stable.)
