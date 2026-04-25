# Phase-6 — Alpha Factor Discovery (design spec, no code)

**Status**: design-only. No code in the repo corresponds to this doc.
Phase 6 augments the autoregressive TradeFM (Layer 1) with two
complementary signal-discovery channels: **fundamental-NLP** alpha from
filings/transcripts, and **symbolic-regression** alpha from long-horizon
tick history. Both feed the same neural forecaster as additional
conditioning inputs.

Reference the Silicon Alpha goal in `memory/project_silicon_alpha_goal.md`.

## Why this phase exists

Pure microstructure prediction (Layer 1, TradeFM) captures milliseconds-
to-seconds dynamics of the orderbook. It is silent on:

- **Multi-quarter narratives** (margin expansion vs consensus, capex
  acceleration, guidance revisions)
- **Regime-stable formulaic factors** (the kind discovered by
  symbolic regression — not "black box" patterns that drift, but
  short, interpretable expressions like
  `mean(close, 20d) / std(volume, 5d)` that humans can audit)

Phase 6 builds two extraction pipelines that produce per-instrument,
per-timestamp factor vectors. These get embedded alongside the OPRA
tokens at training time so TradeFM can learn cross-modal conditioning.

## Component A — Fundamental-NLP

### Data
- **Source**: AlphaSense Enterprise OR Refinitiv (LSEG) Filings.
- **Coverage**: 30 years of 10-K, 10-Q filings + earnings call transcripts
  for the SPX universe (~500 tickers × 4 quarters/yr × 30 yr ≈ 60k docs).
- **Volume**: 50k-100k lines of text per ticker.

### Extraction model
- **Long-context LLM** (DeepSeek-V4-Pro 1M-token context, or successor
  with similar context budget at the time of build).
- **"Think Max" mode** — deliberate-reasoning pass over the full ticker
  history to identify "consensus gaps": narratives where sell-side
  consensus expects flat-to-low growth but internal disclosures
  (cost-of-revenue trends, segment-level margin changes, capex
  ramp-up patterns) imply margin expansion ahead.

### Output schema
Per (ticker, quarter), emit a fixed-dimensional factor vector:
```
{
  "consensus_gap_score": float,   # signed: + = bullish gap, - = bearish
  "margin_trend_3q": float,
  "capex_trend_3q": float,
  "guidance_revision_count": int,
  "tone_shift_score": float,      # delta vs prior quarter's tone
  "executive_change_flag": bool,
  ...
}
```
Persist as parquet, indexed by (ticker, quarter, filing_ts).

## Component B — Symbolic Regression (AlphaFormer)

### Data
- **Source**: Refinitiv Tick History via **LSEG Tick History – Query
  (BigQuery)**. Avoids the O(n²) local-storage trap on the 80+ PB tape.
- **Coverage**: L1/L2/L3 quotes + trades for 580+ global venues, 1996-present.
- **Granularity**: Bar / tick / book-snapshot at varying horizons.

### Method
- **AlphaFormer**: a transformer-based symbolic-regression engine that
  searches the space of formulaic alpha expressions. Discovers things
  like:
  - `rank(corr(open, volume, 10d))` (volume-price lead-lag)
  - `mean(close, 20d) / std(volume, 5d)` (trend / dispersion ratio)
  - `delay(rank(close - mean(close, 5d)), 1)` (1-day lag of normalized
    momentum)
- **Search budget**: thousands of candidate expressions evaluated by
  in-sample IC (information coefficient) + out-of-sample stability.
- **Survivor selection**: Pareto front on (IC, turnover, decay,
  capacity).

### Output schema
A library of N (~100-500) survivor formulas, each with:
```
{
  "expression": "mean(close, 20d) / std(volume, 5d)",
  "ic_in_sample": 0.043,
  "ic_oos_2024": 0.029,
  "decay_half_life_days": 4.2,
  "capacity_usd": 50_000_000,
  "regime_dependence": {"low_vol": 0.05, "high_vol": -0.02}
}
```

Per-tick, compute factor values for live universe and persist as
parquet for downstream conditioning.

## How both feed Layer 1 (TradeFM)

The 524M TradeFM gets two new conditioning channels:

1. **Fundamental embedding** — concatenate or cross-attend the
   ticker-level fundamental factor vector at sequence start. Updated
   per quarter (8K-style per-ticker context tokens).
2. **Symbolic factor stream** — at each tick, look up the per-formula
   factor values and inject as additional embedded tokens (modality_id
   = 4, separate from OPRA/ES/Polymarket/Kalshi).

Architecturally: the existing `modality_vocab > 0` scaffold in
`models/config.py` already supports this — modality 4 = fundamental,
modality 5 = symbolic-regression factor. No model-architecture change
needed beyond bumping `modality_vocab`.

## Dependencies — gate before starting

1. Phase-2 524M pretrain done.
2. Phase-2.5 cross-asset fusion validated (proves modality-token
   conditioning works at all).
3. Subscription to AlphaSense **or** LSEG Filings ($30-300k/yr).
4. LSEG Tick History BigQuery seat (typically $250k-$1M+/yr).
5. Cloud budget for the DeepSeek-V4-Pro context-window calls
   (60k docs × 1M tokens per ingestion ≈ tens-of-millions of tokens
   billed; ~$5-30k per full re-ingestion).

None of those exist today. Phase 6 is post-Phase-2-success and
post-revenue, when the trading system is funding its own data
acquisition.

## Cost-benefit at this phase

A formulaic alpha discovered by AlphaFormer with IC=0.03 and
turnover ~5/year on the SPX universe historically translates to ~3-6%
unlevered annual return. Five orthogonal such factors combined =
~15-25% annualized at moderate vol. **That's the order-of-magnitude
prize for a successful Phase 6 — well above the data-acquisition cost
once you have any meaningful AUM.**

Below ~$50M AUM the data costs likely don't amortize. Above $100M they
become a clear win.

## Build order when the time comes

1. **Component B (symbolic regression)** first. Cheaper data
   (BigQuery pay-per-query, not enterprise subscription); more
   immediately usable signal; doesn't require LLM compute.
2. **Component A (fundamental NLP)** second. Requires LLM context
   budget that's getting cheaper over time; data subscription is the
   gating cost.

## Related files

- [`docs/architecture.md`](architecture.md) — diagram with Phase-6
  factor channels feeding into the forecaster
- [`docs/cross_asset_fusion.md`](cross_asset_fusion.md) — Phase-2.5
  spec; same `modality_vocab` mechanism is reused for Phase-6
- [`docs/phase4_strategic_layer.md`](phase4_strategic_layer.md) — HRL
  layer that consumes both microstructure forecasts AND Phase-6 factors
