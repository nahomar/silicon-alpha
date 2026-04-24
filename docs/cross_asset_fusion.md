# Cross-Asset Latent Fusion (Phase-2.5 spec)

**Status**: scaffold only. No data ingestion, no training, no results yet.
Purpose of this doc: pin down the design decisions before any paid compute is
burned on cross-asset experiments.

## Why

The Silicon Alpha goal (see `memory/project_silicon_alpha_goal.md`) claims the
524M TradeFM is a "universal grammar of trade-flow." HRT-class alpha is not
just 0DTE speed — it's the propagation of signals from correlated markets
(Treasuries, ETFs, ES futures) to the options desk within milliseconds. A
single-asset OPRA model leaves that lead-lag edge on the table.

This doc specifies how to add that channel **after** the single-asset SPX
baseline is validated, not before. Order matters: if the base model can't
beat random on SPX alone, adding modalities won't save it — it'll obscure
the debugging signal.

## Gate before starting

Do NOT begin Phase-2.5 data work until:

1. Phase 2 pretrain on single-asset OPRA SPX has completed (524M, 100B tokens
   or reduced scope on free-tier compute).
2. An SPX-only eval loss curve exists and is below-vocab-entropy (meaningful
   learning, not just initialization drift).
3. A downstream signal (e.g. next-tick direction classifier) shows >55%
   accuracy on held-out regimes.

If any of those fail, fix single-asset first.

## Components

### 1. Data modalities (ranked by feasibility)

| ID | Modality | Data source | Rate | Cost | Status |
|---|---|---|---|---|---|
| 0 | **OPRA cmbp-1** (SPX 0DTE options) | Databento `OPRA.PILLAR` | 10k-100k msg/s | ~$7.50/day | live — already ingested |
| 1 | **ES MBP-1** (E-mini S&P 500 futures) | Databento `GLBX.MDP-3` | 1k-50k msg/s | ~$5-20/day | scaffold (`odte/data/cme_es_pack.py`) |
| 2 | **SPY/IVV ETF L1 + NAV-arb spread** | Databento `XNAS.ITCH` or `IEX` | 10k-50k msg/s | ~$3-10/day | not started |
| 3 | **Index-constituent basket imbalance** | aggregate from component L1 | derived | pay per constituent | not started |
| ~~ETF rebalancing flows (daily disclosures)~~ | Stooq or Bloomberg 13F | 1/day | free/paid | **rejected — not real-time** |

Skip daily-disclosure data for this layer. It's post-hoc flow reconstruction,
not a live signal. If you want to model rebalancing, do it as a separate
calendar-aware feature in the downstream executor, not as a model input.

### 2. Token-stream interleaving

Two viable schemes:

**(a) Time-merged stream** — all modalities sorted by timestamp into one
sequence. Each token carries a `modality_id` (0=OPRA, 1=ES, 2=ETF). The
transformer sees the true temporal order; attention learns that ES events
preceding OPRA events predict OPRA movement.

- Pro: most natural; captures true lead-lag.
- Con: one modality's bursts can starve others from the context window.

**(b) Modality-segmented blocks** — fixed-size windows where each window is
padded with the most-recent state of each modality at the time of the last
OPRA event. Like multi-view fusion.

- Pro: guaranteed cross-modality coverage.
- Con: synthetic, loses true tick order.

**Default (start here): scheme (a)**, time-merged. Implement (b) only if (a)
shows capacity pressure at 4096 ctx.

### 3. Model changes (already scaffolded)

See [`odte/transformer_tradefm.py`](../odte/transformer_tradefm.py) — `TradeFM`
now accepts an optional `modality_ids` tensor:

```python
model.forward(tokens, modality_ids=modality_ids)
# or
loss = model.loss(tokens, modality_ids=modality_ids)
```

Behavior:
- `cfg.modality_vocab = 0` (default) → single-modality; `modality_ids` is
  ignored. Current Phase-2 training path unchanged.
- `cfg.modality_vocab > 0` → an `nn.Embedding(modality_vocab, d_model)` is
  summed into `tok_emb(tokens)` before the blocks.

The embedding is *added*, not concatenated, so `d_model` stays fixed and the
persistent CUDA inference kernel (Phase 3) needs zero changes.

### 4. Tokenizer merge

OPRA and ES have overlapping but not identical feature distributions. The
simplest approach: fit separate quantile edges per modality, concatenate
vocab ranges (OPRA tokens 0..V-1, ES tokens V..2V-1). Keeps per-modality
statistics clean at the cost of 2× vocab.

Cheaper alternative: share a single vocab, fit edges on the *mixed* stream.
Likely fine for `ret` and `mid` which are comparable across modalities,
but worse for `bid_sz` / `ask_sz` where ES has orders of magnitude more
size than a single OPRA strike.

**Decision (default): separate vocab ranges per modality, shared model
embedding.** The modality_id channel tells the model which range each token
draws from, so the embedding matrix can specialize per-modality weights.

### 5. Dataset loader changes

`ShardedTokenDataset` will need a `modality_col` argument and return
`(tokens, modality_ids)` tuples instead of just tokens. The collation in the
trainer's DataLoader stays trivial (both are `int64` tensors of the same
shape).

## Rollout plan (when the gate opens)

1. Implement `cme_es_pack.glbx_to_features()` + `iter_es_dbn_chunks()` — mirror
   the OPRA adapter's prepare_features.
2. Pull a small (1-hour) ES sample on Databento, manually sanity-check
   distributions match expectations (ES tick = 0.25, typical BBO size
   hundreds of contracts).
3. Fit a shared-vocab tokenizer on an aligned SPX+ES overlap window.
4. Interleave shards with `modality_id` column; update `ShardedTokenDataset`
   to emit `(tokens, modality_ids)`.
5. Fine-tune the already-pretrained single-asset 524M with
   `modality_vocab=2`, keeping the token embedding warm-start and the
   modality embedding zero-initialized (so cross-asset starts as a no-op and
   only contributes if it helps).
6. Held-out eval on an SPX-only slice: cross-asset model loss must be ≤ the
   single-asset baseline loss on the same data. If it's worse, the fusion
   layer is hurting and should be turned off.

## What this does NOT commit to

- **No timeline**. Cross-asset work starts after Phase 2 validates.
- **No compute budget**. ES data pull cost estimated but not approved.
- **No architecture beyond modality embedding**. If attention-based fusion
  doesn't work, we explicitly reserve the option to try late-fusion (separate
  encoders + cross-attention) or early-fusion (concatenated features per
  token). This doc does not prejudge.

## Related files

- [`odte/data/cme_es_pack.py`](../odte/data/cme_es_pack.py) — ES adapter skeleton (raises NotImplementedError)
- [`odte/transformer_tradefm.py`](../odte/transformer_tradefm.py) — modality embedding added to `TradeFM`
- [`models/config.py`](../models/config.py) — `TradeFMConfig.modality_vocab` field
- `memory/project_silicon_alpha_goal.md` — the north-star goal this ladders up to
