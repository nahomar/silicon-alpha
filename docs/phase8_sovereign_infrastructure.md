# Phase-8 — Sovereign Infrastructure (design spec, no code)

**Status**: design-only. Major infrastructure rewrite that replaces several
foundational components established in Phases 1-3. Adopt only when (a) the
524M TradeFM (Phase 2) is trained and producing meaningful PnL, and (b)
the dense transformer architecture has demonstrably hit a ceiling on real
OPRA tape that warrants the MoE complexity tradeoff.

This phase ports the entire stack to a **High-Flyer / DeepSeek-style
sovereign architecture**: disaggregated RDMA storage, CPU-overlapped
collective communication, DRAM-resident factual memory, and a
Mixture-of-Experts forecaster.

Reference the Silicon Alpha goal in `memory/project_silicon_alpha_goal.md`
and the existing Phase-1/3 specs.

## Why this phase exists

The Phase-1/2 dense 524M transformer assumes:
- All training data fits the bandwidth budget of NVMe + Spark loaders.
- Inference attention costs scale acceptably with context length.
- Static factual knowledge (SEC filings, exchange holiday calendars,
  fundamental refs) is either left out or stuffed into the context window.
- AllReduce overhead is hidden by NVLink / GPUDirect-TCPX bandwidth.

These assumptions all break at the limit cases of (a) training on
multi-trillion-token corpora, (b) running inference on contexts that include
full SEC filing histories, and (c) MoE routing across thousands of experts
where collective-communication stalls dominate. Phase 8 substitutes
infrastructure components that have been **proven in production** at
High-Flyer (the firm behind DeepSeek) to handle these regimes.

## The four components

### 1. Layer -1 — 3FS (Fire-Flyer File System): disaggregated RDMA storage

**Why**: in the trillion-token regime, "starving the GPU" is the dominant
bottleneck. The CPU spends most of its cycles unmarshalling Parquet/Arrow
packets instead of compiling the next batch.

**3FS**:
- Bypasses the kernel via RDMA (similar to GPUDirect Storage).
- Pools thousands of NVMe SSDs into a single logical address space.
- Achieves ~**40 GiB/s random read** per client.
- Decouples storage scaling from compute scaling — add SSDs without
  reprovisioning GPU nodes.

**Replaces**: the current Spark + Modal Volume pipeline for training data.
The Phase-2 corpus (637 OPRA shards on a Modal Volume) becomes a
sub-cluster of a 3FS pool. Inference still pulls features from Sibyl
(Phase 1) — 3FS is purely the training-data substrate.

**Cost**: significant. A meaningful 3FS deployment is 1-3 racks of NVMe
SSDs + RDMA NICs. Estimate $200k-$1M capex + opex. Worth it only at the
training scale where the GPU starvation is documented (e.g., > 16-node
clusters).

### 2. hfreduce: CPU-based AllReduce primitive

**Why**: NCCL AllReduce blocks the GPU SMs while collective communication
runs. For dense ranks this is fine — NVLink is fast enough. For MoE
routing with thousands of experts on lower-bandwidth interconnects, the
GPU spends a meaningful fraction of wall-clock waiting on collectives.

**hfreduce**:
- Moves the AllReduce primitive to the host CPU using DPDK-style userspace
  networking.
- Overlaps with GPU computation: while expert tokens are being routed
  across ranks, the GPU continues forward/backward on the local shard.
- Designed for the **MoE token-routing pattern** specifically — expects
  small, frequent gradient/activation tensors rather than large dense
  tensors.

**When NOT to use it**: dense FSDP training on H100 NVLink + GPUDirect-TCPX
where NCCL is well-optimized and collective overhead is already < 5% of
wall-clock. We default to NCCL for Phase 2 dense training; hfreduce becomes
relevant only if Phase 8 adopts MoE.

### 3. Engram Memory Layer: DRAM-resident O(1) factual lookup

**Why**: forcing a transformer to "remember" SEC filings, fundamental
ratios, holiday calendars, and corporate-action schedules in its hidden
states is wasteful — those facts are static, queryable, and don't deserve
quadratic attention.

**Engram**:
- Hash table in system DRAM, addressed by deterministic keys (ticker +
  filing-type + date, etc.).
- O(1) retrieval at inference time via PCIe DMA to GPU — orders of
  magnitude cheaper than re-attending over context tokens.
- The model learns to **emit a key**, not the value. The value is fetched
  from Engram and concatenated to the residual stream.

**For our use case**: pre-load the Engram with:
- 30 yr of 10-K / 10-Q filings (Phase 6 fundamental data lives here)
- Historical exchange holiday calendars
- Strike-list snapshots per expiry
- Static contract specs (CME ES tick size, OPRA price-display rules)

The MoE forecaster (component 4) emits keys like `<lookup:filing,SPX,Q1-2025>`
and Engram returns the pre-tokenized factor vector.

**Tradeoff**: Engram is brittle to schema drift. If the underlying fact
representation changes (e.g., switching from Bloomberg ticker to FIGI),
every Engram key needs a remap. Build the key namespace deliberately.

### 4. DeepSeek-V4-style MoE forecaster

**Why**: at the trillion-token + multi-modal scale (post Phase 6 + 7),
a dense 524M starts to either underfit (capacity ceiling) or become
prohibitively expensive to scale to dense 70B+. MoE gives 30-100B+
"effective" parameters at the inference cost of a smaller dense model.

**Architecture**:
- Top-k expert routing (k=2 typical) over hundreds-to-thousands of experts.
- Cross-expert load balancing via auxiliary loss.
- Engram lookups emitted as part of the expert output.
- Multi-Head Classifier (MHC) head produces:
  - Next-tick direction (microstructure, like dense TradeFM)
  - Regime label (gamma / vol / event)
  - Per-feature reconstruction (auxiliary)
- Conditional Forecaster Generator (CFG) — a small head that emits the
  control signal consumed by the FPGA actor (Phase 3.5) and QP solver
  (Phase 2 layer).

**Replaces**: dense TradeFM (Phase 1). Both can coexist during transition —
serve the dense model on the live path while shadow-evaluating MoE.

**Cost**: MoE training is harder than dense (load-balance instabilities,
expert-collapse failure modes). Plan for ~3× the engineering effort of
the Phase-2 dense pretrain.

## CANN / Huawei Ascend — explicit non-adoption

CANN is Huawei's CUDA equivalent for the Ascend NPU line. It is real,
production-grade, and a viable alternative to NVIDIA H100 stacks **for
operators who can't access NVIDIA hardware** (US export controls,
sovereign-AI mandates in non-US jurisdictions, etc.).

**We do not adopt CANN as the primary stack.** Reasons:
- Every prior phase targets NVIDIA H100/H200 + SM90a CUDA. Switching
  invalidates `odte/kernels/*.cu`, the FPGA-bridge P2P design, and the
  Phase-2 Modal validation.
- US-based deployment has unrestricted H100 access.
- NVIDIA's tooling ecosystem (PyTorch, TE, FA3, NCCL) is mature; CANN's
  PyTorch interop has historically lagged.

**When to revisit**: if the project ever needs to deploy in a jurisdiction
that prohibits NVIDIA hardware (e.g., China-mainland operations), CANN
becomes the natural port target. Document it as a contingency path; don't
build it speculatively.

## QRAFTI extension to Phase 5

Phase 5 (`agentic_governance.md`) currently spec'd three agents: Research,
Risk, Compliance. **QRAFTI adds a fourth**:

- **Quant Dev Agent** — autonomous code-repair role. When a model
  deployment fails (NaN gradients, distribution shift, broken feature
  pipeline), the Dev Agent ingests the stack trace + Engram-resident
  past-incident reports + git blame, proposes a hotfix PR, and submits it
  to the Risk Agent for KKT validation before merge.

This closes the loop: Research proposes signal mods → Dev Agent translates
into code → Risk validates → Compliance logs → deploy. A self-improving
quant desk with hash-chained accountability.

## Dependencies — gate before starting

1. **Phase 2 trained**, producing meaningful PnL. No infrastructure
   replacement before the existing dense forecaster is proven.
2. Documented evidence the dense 524M has hit a capacity ceiling on real
   OPRA. Without that, MoE is premature optimization.
3. Capex headroom for 3FS deployment ($200k-$1M) and the engineering
   bandwidth (~6-12 person-months) to port training infra to hfreduce.
4. **Strict ordering**: 3FS first (data substrate must be sound before any
   model rewrite), then Engram (orthogonal — works for dense AND MoE),
   then MoE forecaster, then hfreduce (only if MoE routing actually shows
   collective stalls).

## What this does NOT include

- A switch to Huawei Ascend / CANN (see explicit non-adoption above).
- A rewrite of the FPGA-bridge spec (`fpga_bridge.md`). The mmap'd AXI
  register interface is already documented there; Phase 8 doesn't change it.
- The Phase 6/7 data sources. Engram **stores** the Phase-6 fundamental
  factors, but the discovery pipeline (AlphaFormer, fundamental NLP) is
  unchanged.

## Related files

- [`docs/architecture.md`](architecture.md) — diagram with Layer -1
  substrate + Engram-inside-forecaster + QRAFTI Quant Dev agent
- [`docs/cross_asset_fusion.md`](cross_asset_fusion.md) — Phase 2.5
  multi-modal token routing; Engram is the persistence layer for the
  modality-conditioning vectors
- [`docs/phase4_strategic_layer.md`](phase4_strategic_layer.md) — HRL
  layer above the MoE forecaster
- [`docs/agentic_governance.md`](agentic_governance.md) — Phase 5; gets
  the Quant Dev fourth-agent extension
- [`docs/phase6_alpha_factor_discovery.md`](phase6_alpha_factor_discovery.md)
  — produces fundamental factor vectors stored in Engram
