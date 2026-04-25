# silicon-alpha

A µs-latency dual-venue trading engine driven by a 524M decoder-only
transformer (TradeFM) trained on exchange-message grammar from options,
futures, and prediction markets.

Designed to deploy simultaneously across:

- **SPX / NDX 0DTE options** (HRT-tier latency, $166–$800 per unit per day
  target)
- **Polymarket / Kalshi prediction markets** (bundle + cross-venue arb,
  0.1–3.0% per cycle)

## Architecture

End-to-end flow across all 9 phases. Phases 0-2 are live or in-progress;
Phases 2.5-7 are design-only with code scaffolds where appropriate.
HRT-inspired palette: near-black ingestion, warm orange neural + execution
tiers, peach (dashed) for design-only phases, cream for terminal venues.

```mermaid
flowchart TB
    %% ========== LAYER -1: SOVEREIGN INFRASTRUCTURE SUBSTRATE ==========
    subgraph INFRA["Layer -1 - Sovereign Infrastructure (Phase 8, design only)"]
        direction LR
        TFS["3FS RDMA Storage<br/>40 GiB/s random read<br/>kernel-bypass NVMe pool"]
        ENGRAM["Engram Memory<br/>O(1) DRAM hash lookup<br/>SEC filings + statics"]
        HFR["hfreduce<br/>CPU-overlapped AllReduce<br/>(MoE-only)"]
    end

    %% ========== LAYER 0: REAL-TIME MARKET INGESTION ==========
    subgraph DATA["Layer 0 - Real-time Market Ingestion"]
        direction LR
        OPRA["OPRA cmbp-1<br/>SPX/NDX options<br/>(Databento, live)"]
        ES["ES Futures MBP-1<br/>(GLBX, Phase 2.5)"]
        POLY["Polymarket CLOB WSS<br/>(~100ms, Phase 3+)"]
        KAL["Kalshi CLOB WSS<br/>(~100ms, Phase 3+)"]
    end

    %% ========== PHASE 6: ALPHA FACTOR DISCOVERY ==========
    subgraph P6["Phase 6 - Alpha Factor Discovery (design only)"]
        direction LR
        FUND["Fundamental NLP<br/>10-K / 10-Q + earnings<br/>(AlphaSense / LSEG)<br/>DeepSeek 1M-ctx"]
        SYMR["AlphaFormer<br/>symbolic regression<br/>(LSEG Tick BigQuery)<br/>~80 PB tape"]
    end

    %% ========== PHASE 7: ALT DATA ==========
    subgraph P7["Phase 7 - Alternative Data (design only)"]
        direction LR
        PRICE["Bright Data<br/>250+ e-commerce<br/>SKU pricing"]
        TXN["YipitData / Measurable AI<br/>2M+ user receipts<br/>weekly SKU txns"]
        TALENT["PredictLeads<br/>job-posting flow<br/>3-6mo lead signal"]
    end

    %% ========== LAYER 1: NEURAL FORECASTER ==========
    subgraph L1["Layer 1 - Neural Forecaster (H100, 4.6-15.8 us)"]
        direction TB
        RDMA["GPUDirect RDMA<br/>HBM3 ring buffer"]
        KERNEL["Persistent CUDA Kernels<br/>SM90a always-resident"]
        TFM["524M TradeFM<br/>multi-modal token attn"]
        DML["DML Pricer<br/>0DTE Greeks + IV"]
        RDMA --> KERNEL
        KERNEL --> TFM
        KERNEL --> DML
    end

    %% ========== LAYER 2: CALL SHEET ==========
    subgraph L2["Layer 2 - Deterministic Call Sheet"]
        direction TB
        QP["QP Executor<br/>risk-adjusted w*"]
        GAMMA["Gamma / VaR Gates<br/>software kill-switch"]
        ARB["Bundle Arb Detector<br/>YES + NO =/= $1.00"]
        XV["Cross-Venue Arb<br/>Polymarket &lt;-&gt; Kalshi"]
    end

    %% ========== LAYER 1.5: FPGA ACTOR ==========
    subgraph FPGA["Layer 1.5 - FPGA Actor (Phase 3.5, design only)"]
        direction TB
        P2P["PCIe P2P DMA<br/>GPU HBM3 -&gt; FPGA BAR"]
        HWKILL["Hardware Kill-Switch<br/>&lt;100ns dead-man"]
        OUCH["OUCH/FIX Encoder<br/>tick-to-trade &lt; 450ns"]
        P2P --> HWKILL --> OUCH
    end

    %% ========== LAYER 3: STRATEGIC ==========
    subgraph L3["Layer 3 - Strategic (Phase 4, design only)"]
        direction LR
        HLC["HLC PPO<br/>strategy weights<br/>+ child-order slicing"]
        POW["POW-dTS<br/>regime adapter"]
        MORL["MORL reward<br/>[pnl, -IS, -DD,<br/>-gamma, -conc, up]"]
    end

    %% ========== LAYER 4: AGENTIC GOVERNANCE ==========
    subgraph AGENTS["Layer 4 - Agentic Governance (Phase 5 + Phase 8 QRAFTI)"]
        direction LR
        RES["Research Agent<br/>proposes signal mods"]
        DEV["Quant Dev Agent<br/>auto-PR + code repair<br/>(Phase 8)"]
        RISK["Risk Agent<br/>KKT validation<br/>+ hard veto"]
        COMP["Compliance Agent<br/>hash-chained<br/>audit trail"]
    end

    %% ========== VENUES ==========
    subgraph VEN["Execution Venues"]
        direction LR
        SPX["SPX/NDX 0DTE<br/>$166-$800 per unit / day<br/>MM + toxic-flow avoidance"]
        PM["Polymarket<br/>0.1-3.0% per arb cycle"]
        KS["Kalshi<br/>0.1-3.0% per arb cycle"]
    end

    %% ========== EDGES — LIVE / SOLID ==========
    OPRA --> RDMA
    TFM --> QP
    DML --> QP
    QP --> GAMMA
    GAMMA -.software path.-> SPX
    POLY --> ARB
    KAL --> ARB
    TFM --> ARB
    ARB --> PM
    ARB --> KS
    ARB --> XV
    XV --> PM
    XV --> KS

    %% ========== EDGES — PHASE-GATED / DASHED ==========
    TFS -.Phase 8 substrate.-> KERNEL
    TFS -.Phase 8 substrate.-> TFM
    ENGRAM -.Phase 8 lookup.-> TFM
    HFR -.Phase 8 MoE only.-> TFM
    ES -.cross-asset.-> RDMA
    FUND -.Phase 6.-> TFM
    SYMR -.Phase 6.-> TFM
    PRICE -.Phase 7.-> TFM
    TXN -.Phase 7.-> TFM
    TALENT -.Phase 7.-> TFM
    GAMMA -.FPGA path.-> P2P
    OUCH -.wire.-> SPX
    HLC -.Phase 4.-> QP
    POW -.Phase 4.-> HLC
    MORL -.Phase 4.-> HLC
    SPX -.trade logs.-> MORL
    PM -.trade logs.-> MORL
    KS -.trade logs.-> MORL
    RES -.proposes.-> HLC
    RES -.proposes.-> TFM
    RES -.proposes signal mods.-> DEV
    DEV -.draft PR.-> RISK
    RISK -.validates KKT.-> QP
    RISK -.hard veto.-> P2P
    DEV -.logs.-> COMP
    QP -.logs.-> COMP
    P2P -.logs.-> COMP
    SPX -.logs.-> COMP
    PM -.logs.-> COMP
    KS -.logs.-> COMP

    %% ========== HRT-INSPIRED STYLING ==========
    classDef data fill:#1A1A1A,stroke:#1A1A1A,stroke-width:2px,color:#F8F5F0
    classDef layer1 fill:#E85D2E,stroke:#E85D2E,stroke-width:2px,color:#ffffff
    classDef layer2 fill:#D94F23,stroke:#D94F23,stroke-width:2px,color:#ffffff
    classDef phaseDesign fill:#F4A261,stroke:#D94F23,stroke-width:2px,color:#1A1A1A,stroke-dasharray:5 3
    classDef venue fill:#F8F5F0,stroke:#E85D2E,stroke-width:3px,color:#1A1A1A

    class OPRA,ES,POLY,KAL data
    class RDMA,KERNEL,TFM,DML layer1
    class QP,GAMMA,ARB,XV layer2
    class P2P,HWKILL,OUCH phaseDesign
    class HLC,POW,MORL phaseDesign
    class RES,DEV,RISK,COMP phaseDesign
    class FUND,SYMR phaseDesign
    class PRICE,TXN,TALENT phaseDesign
    class TFS,ENGRAM,HFR phaseDesign
    class SPX,PM,KS venue
```

Same diagram with edge-semantics legend, phase-by-phase doc map, and
critical-path tracker also lives at [`docs/architecture.md`](docs/architecture.md).

## Phase roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | 40M TradeFM Colab pretrain | ✅ done ([`notebooks/colab_phase1_tradefm.ipynb`](notebooks/colab_phase1_tradefm.ipynb)) |
| 2 | 524M multi-node H100 pretrain on OPRA | 🔄 pipeline validated on Modal; multi-node compute gated |
| 2.5 | Cross-asset fusion (ES futures modality) | 📝 design ([`docs/cross_asset_fusion.md`](docs/cross_asset_fusion.md)) + opt-in scaffold |
| 3 | Persistent-kernel live inference (4.6–15.8 µs) | 📝 kernels scaffolded, not live |
| 3.5 | FPGA P2P DMA bridge — tick-to-trade <450 ns (OUCH/FIX + hardware kill-switch) | 📝 design only ([`docs/fpga_bridge.md`](docs/fpga_bridge.md)) |
| 4 | Hierarchical RL + MORL (IS reward) + POW-dTS strategic layer | 📝 design only ([`docs/phase4_strategic_layer.md`](docs/phase4_strategic_layer.md)) |
| 5 | Agentic governance (Research + Risk + Compliance agents) | 📝 design only ([`docs/agentic_governance.md`](docs/agentic_governance.md)) |
| 6 | Alpha factor discovery — fundamental NLP (10-K/10-Q + earnings) + AlphaFormer symbolic regression | 📝 design only ([`docs/phase6_alpha_factor_discovery.md`](docs/phase6_alpha_factor_discovery.md)) |
| 7 | Alternative data — e-commerce SKU pricing + receipt panels + talent flow | 📝 design only ([`docs/phase7_alternative_data.md`](docs/phase7_alternative_data.md)) |
| 8 | Sovereign infrastructure — 3FS RDMA storage + Engram DRAM lookup + hfreduce CPU AllReduce + MoE forecaster + QRAFTI Quant Dev agent | 📝 design only ([`docs/phase8_sovereign_infrastructure.md`](docs/phase8_sovereign_infrastructure.md)) |

**Rule**: design docs land in `docs/` for any phase gated by unmet
dependencies; code scaffolds only where a clean opt-in guard keeps them
inert until the dependency chain is satisfied.

## Repo layout

```
configs/              tradefm_40m.yml, tradefm_524m.yml, smoke configs
docs/                 architecture.md, cross_asset_fusion.md, phase4_strategic_layer.md
infra/
  gcp/                phase2_a3mega.sh, launch_torchrun_524m.sh, TCPX env
  modal/              phase2_smoke.py (single-node validation suite)
odte/
  data/               databento_pack.py, polygon_pack.py, streaming_quantiles.py,
                      cme_es_pack.py (Phase 2.5 scaffold)
  kernels/            fused_bin.cu, persistent_decode.cu, rdma_ingest.cu (Phase 3)
  train/              distributed.py (FSDP), checkpoint.py, pretrain_tradefm.py
  transformer_tradefm.py  (TradeFM model; optional modality embedding for 2.5)
models/               config.py (TradeFMConfig)
notebooks/            colab_phase0_dml.ipynb, colab_phase1_tradefm.ipynb
tests/                quantile_parity, etc.
STATE.md              detailed project handoff runbook
BUDGET.md             compute + data cost tracking
```

## Phase 2 validation on Modal ($0.50–$3 per run)

Single-node Modal functions validate the full pipeline before committing to
the $50k multi-node run:

```bash
pip install --user modal && python3 -m modal setup

# 40M Hopper-path smoke (fp8 TE, SDPA, FSDP init, ckpt I/O):
python3 -m modal run infra/modal/phase2_smoke.py

# 524M production-config dry run on 1× H100 (proves 524M instantiates,
# FSDP shards 24 layers, optimizer allocates at scale):
python3 -m modal run infra/modal/phase2_smoke.py::dryrun_524m

# Real OPRA shards from Databento (bypasses batch queue via
# pre-completed job reuse; vectorized pack ~8× faster than naive):
python3 -m modal run infra/modal/phase2_smoke.py::dryrun_databento_reuse

# 40M training on real OPRA tape (vs Markov synthetic):
python3 -m modal run infra/modal/phase2_smoke.py::dryrun_train_real

# 8× H100 single-node NCCL smoke (world_size=8, rank-partitioned ckpts):
python3 -m modal run infra/modal/phase2_smoke.py::dryrun_8gpu
```

## Phase 2 production launch (needs GCP quota + billing + $38–50k)

```bash
export GCP_PROJECT=... GCP_BUCKET=gs://... REPO_URL=https://github.com/nahomar/market-pattern-bot.git
./infra/gcp/phase2_a3mega.sh          # provision 3× A3 Mega (24× H100)
./infra/gcp/launch_torchrun_524m.sh   # launch distributed training
```

See [`STATE.md`](STATE.md) for the full runbook including quota prereqs,
cost estimates, and post-launch monitoring.

## Verified Phase-2 smoke results

| Smoke | Result | Cost |
|---|---|---|
| 40M 200-step on Markov shards | loss 112 → 2.48 (below `log(4096)=8.3` uniform floor) | ~$0.25 |
| Checkpoint resume (step 200 → 400) | loaded clean, no re-init corruption | ~$0.25 |
| 524M production-config dry-run | 500 steps, loss 425 → 1.66, 4.87 GB model ckpt | ~$2.70 |
| Databento batch reuse + tokenizer fit on real OPRA | 636M rows, 7-feature edges fitted | ~$3 |

## Related docs

- [`STATE.md`](STATE.md) — handoff runbook, decisions, session log
- [`BUDGET.md`](BUDGET.md) — compute + data cost tracking
- [`docs/architecture.md`](docs/architecture.md) — full flow diagram with HRT palette
- [`docs/cross_asset_fusion.md`](docs/cross_asset_fusion.md) — Phase 2.5 spec
- [`docs/fpga_bridge.md`](docs/fpga_bridge.md) — Phase 3.5 spec
- [`docs/phase4_strategic_layer.md`](docs/phase4_strategic_layer.md) — Phase 4 spec
- [`docs/agentic_governance.md`](docs/agentic_governance.md) — Phase 5 spec
- [`docs/phase6_alpha_factor_discovery.md`](docs/phase6_alpha_factor_discovery.md) — Phase 6 spec
- [`docs/phase7_alternative_data.md`](docs/phase7_alternative_data.md) — Phase 7 spec
- [`docs/phase8_sovereign_infrastructure.md`](docs/phase8_sovereign_infrastructure.md) — Phase 8 spec
- [`docs/sponsor_email_template.md`](docs/sponsor_email_template.md) — faculty-sponsor email template
