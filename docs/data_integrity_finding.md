# Data-integrity finding: cross-contract returns corrupted the directional target

**Severity: critical — invalidates the "no 0DTE alpha" conclusion and would have
wasted a $20–50k retrain.**

## Summary

The OPRA feature pipeline computed the `ret` (log-return) feature — which is
both a model input **and** the directional prediction target — with a global
`log(mid).diff()` over a stream that interleaves *every* option contract. The
result was not a return but a **cross-contract log-ratio**,
`log(mid of contract A) − log(mid of contract B)`, for whatever two contracts
happened to be adjacent in time. The directional target the 524M transformer
was trained and evaluated against was therefore noise.

## How the data is structured

`odte/data/databento_pack.py` fetches SPX/SPXW under **parent symbology**
(`SPX.OPT`), which Databento expands to every child contract. MBP-1 is a single
**time-ordered stream across all of them**, so consecutive rows are different
strikes/expiries. Rows are written to parquet in exactly that stream order, and
(before this fix) the packed row dropped contract identity entirely
(`expiry=""`, no `instrument_id`).

## The bug

`odte/data/datashop_pack.py::prepare_features` (used by the Databento packer):

```python
df = df.sort_values("ts_ms")
df["ret"] = np.log(df["mid"]).diff().fillna(0.0)   # global diff, no instrument grouping
```

A bare `.diff()` across the interleaved stream subtracts the previous *row's*
mid — a different contract — from the current one.

**The same repo proves this is a bug, not a modeling choice:**
`odte/data/polygon_pack.py` does it correctly —
`sort_values(["ticker", "ts_ns"])` then `groupby("ticker")` — grouping per
instrument before temporal features. The OPRA path simply never did.

## Impact

- **The transformer's "49% held-out return-direction" is uninformative.** You
  cannot predict a cross-contract log-ratio; ~50% is the *expected* result of a
  corrupted target, not evidence about real market signal. The directional
  alpha thesis was never actually tested.
- **The `dir_baseline.py` signal diagnostic** reads these same shards and
  predicts the same corrupted target, so its verdict (signal / no signal) would
  be meaningless on pre-fix data.
- **A directional-head retrain (~$20–50k)** would have optimized against noise.

## The fix

`prepare_features` now computes `ret` and `inter_arrival_ms` **within
instrument groups** when a contract id is present (auto-detected from
`instrument_id` / `symbol` / `ticker`), mirroring `polygon_pack`. When no id is
present it falls back to the global diff **and logs a warning**, so a silent
cross-instrument return can never recur. `databento_pack` now retains
`instrument_id` in each packed row so downstream sequence models and the
diagnostic can separate contracts.

Regression test: `tests/odte/test_feature_prep.py` builds a 2-contract
interleaved frame and asserts the returns are per-contract and that none of the
cross-contract values leak in.

**Caveat (documented, not yet fixed):** features are computed per streamed
chunk, so an instrument's first quote in each chunk gets a 0 return at the chunk
boundary — a second-order effect. Full correctness carries the last mid per
instrument across chunks, or packs per-instrument.

## What has to happen before the alpha question is answerable

1. **Re-pack** the OPRA data with the corrected `prepare_features` (instrument
   id retained, per-contract returns).
2. **Re-run** `dir_baseline.py` (instrument-aware) on the re-packed shards. Only
   then does a GO / NO-GO verdict mean anything.
3. **Re-read** the transformer's directional metric — it was measured against a
   corrupted target and should not be trusted as-is.

Until then, the honest status is: **the 0DTE directional thesis is untested, not
disproven.**
