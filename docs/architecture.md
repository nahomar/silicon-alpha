# Silicon Alpha — End-to-End Architecture

Full flow from raw exchange feeds through the neural forecaster, deterministic
call sheet, FPGA actor bridge, agentic governance layer, and dual-venue
execution. The diagram below renders natively on GitHub.

```mermaid
flowchart TB
    %% ========== LAYER 0: INGESTION ==========
    subgraph DATA["Layer 0 - Ingestion"]
        direction LR
        OPRA["OPRA cmbp-1<br/>SPX/NDX options<br/>(Databento, live)"]
        ES["ES Futures MBP-1<br/>(GLBX, Phase 2.5)"]
        POLY["Polymarket CLOB WSS<br/>(~100ms, Phase 3+)"]
        KAL["Kalshi CLOB WSS<br/>(~100ms, Phase 3+)"]
    end

    %% ========== LAYER 1: NEURAL FORECASTER ==========
    subgraph L1["Layer 1 - Neural Forecaster (H100, 4.6-15.8 us)"]
        direction TB
        RDMA["GPUDirect RDMA<br/>HBM3 ring buffer"]
        KERNEL["Persistent CUDA Kernels<br/>SM90a always-resident"]
        TFM["524M TradeFM<br/>next-tick direction"]
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

    %% ========== LAYER 1.5: FPGA ACTOR (Phase 3.5, design only) ==========
    subgraph FPGA["Layer 1.5 - FPGA Actor (Phase 3.5, design only)"]
        direction TB
        P2P["PCIe P2P DMA<br/>GPU HBM3 -&gt; FPGA BAR"]
        HWKILL["Hardware Kill-Switch<br/>&lt;100ns, dead-man"]
        OUCH["OUCH/FIX Encoder<br/>ns-resolution wire frames"]
        P2P --> HWKILL --> OUCH
    end

    %% ========== LAYER 3: STRATEGIC (Phase 4, design only) ==========
    subgraph L3["Layer 3 - Strategic (Phase 4, design only)"]
        direction LR
        HLC["HLC PPO<br/>strategy weights<br/>+ capital allocator"]
        POW["POW-dTS<br/>regime adapter"]
        MORL["MORL reward<br/>Pareto front"]
    end

    %% ========== LAYER 4: AGENTIC GOVERNANCE (Phase 5, design only) ==========
    subgraph AGENTS["Layer 4 - Agentic Governance (Phase 5, design only)"]
        direction LR
        RES["Research Agent<br/>proposes signal mods"]
        RISK["Risk Agent<br/>KKT validation<br/>+ hard veto"]
        COMP["Compliance Agent<br/>hash-chained<br/>audit trail"]
    end

    %% ========== VENUES ==========
    subgraph VEN["Execution Venues"]
        direction LR
        SPX["SPX/NDX 0DTE<br/>$166-$800 per unit per day<br/>MM + toxic-flow avoidance"]
        PM["Polymarket<br/>0.1-3.0% per arb cycle<br/>bundle + cross-venue"]
        KS["Kalshi<br/>0.1-3.0% per arb cycle<br/>market pull on catalysts"]
    end

    %% ========== EDGES ==========
    OPRA --> RDMA
    ES -.Phase 2.5.-> RDMA
    TFM --> QP
    DML --> QP
    QP --> GAMMA
    GAMMA -.software path.-> SPX
    GAMMA -.Phase 3.5 hardware path.-> P2P
    OUCH -.wire.-> SPX
    POLY --> ARB
    KAL --> ARB
    TFM --> ARB
    ARB --> PM
    ARB --> KS
    ARB --> XV
    XV --> PM
    XV --> KS
    HLC -.Phase 4.-> QP
    POW -.Phase 4.-> HLC
    MORL -.Phase 4.-> HLC
    SPX -.trade logs.-> MORL
    PM -.trade logs.-> MORL
    KS -.trade logs.-> MORL
    RES -.proposes.-> HLC
    RES -.proposes.-> TFM
    RISK -.validates KKT.-> QP
    RISK -.hard veto.-> P2P
    QP -.logs.-> COMP
    P2P -.logs.-> COMP
    SPX -.logs.-> COMP
    PM -.logs.-> COMP
    KS -.logs.-> COMP

    %% ========== HRT-INSPIRED STYLING ==========
    %% Near-black foundation, HRT warm orange primary, deeper orange mid,
    %% peach for design-only phases, cream for terminal venues.
    classDef data fill:#1A1A1A,stroke:#1A1A1A,stroke-width:2px,color:#F8F5F0
    classDef layer1 fill:#E85D2E,stroke:#E85D2E,stroke-width:2px,color:#ffffff
    classDef layer2 fill:#D94F23,stroke:#D94F23,stroke-width:2px,color:#ffffff
    classDef layer1p5 fill:#F4A261,stroke:#D94F23,stroke-width:2px,color:#1A1A1A,stroke-dasharray:5 3
    classDef layer3 fill:#F4A261,stroke:#D94F23,stroke-width:2px,color:#1A1A1A,stroke-dasharray:5 3
    classDef layer4 fill:#F4A261,stroke:#D94F23,stroke-width:2px,color:#1A1A1A,stroke-dasharray:5 3
    classDef venue fill:#F8F5F0,stroke:#E85D2E,stroke-width:3px,color:#1A1A1A

    class OPRA,ES,POLY,KAL data
    class RDMA,KERNEL,TFM,DML layer1
    class QP,GAMMA,ARB,XV layer2
    class P2P,HWKILL,OUCH layer1p5
    class HLC,POW,MORL layer3
    class RES,RISK,COMP layer4
    class SPX,PM,KS venue
```

## Legend

HRT-inspired palette: near-black foundation, warm orange for the active
neural + execution tiers, pale peach with dashed borders for design-only
phases (3.5, 4, 5), warm cream for terminal execution venues.

| Color | Layer | Phase | Status |
|---|---|---|---|
| Near-black `#1A1A1A` | Ingestion | 0 | OPRA live; ES scaffold; Poly/Kalshi pending signup |
| HRT orange `#E85D2E` | Neural Forecaster | 1 | Kernels scaffolded; TradeFM untrained (Phase 2 compute-blocked) |
| Deep orange `#D94F23` | Call Sheet | 2 | QP/Gamma not wired live; arb detectors 0% |
| Peach `#F4A261` (dashed) | FPGA Actor | 3.5 | **Design-only** (see `fpga_bridge.md`) |
| Peach `#F4A261` (dashed) | Strategic RL | 4 | **Design-only** (see `phase4_strategic_layer.md`) |
| Peach `#F4A261` (dashed) | Agentic Governance | 5 | **Design-only** (see `agentic_governance.md`) |
| Cream `#F8F5F0` + orange border | Venues | deploy | All pending broker/API integration |

## Edge semantics

- **Solid arrows**: live data/control flow (design intent for Phases 0-2)
- **Dashed arrows**: phase-gated — design spec exists, wiring pending
  upstream completion
- `trade logs -> MORL`: Phase 4 reward signal flows back from venues once live
- `QP / P2P / venues -> COMP`: Phase 5 Compliance Agent passively reads every
  decision and order-emit event into the hash-chained audit trail
- `RISK -> QP (validates KKT)`: Phase 5 Risk Agent is the hard veto on
  every proposed position vector before it reaches the FPGA

## Critical path today

1. ✅ OPRA ingestion (Databento)
2. ✅ Streaming tokenizer fit on real MBP-1 tape
3. 🔄 Real-data 40M training smoke (`smoke_train_real`)
4. ❌ Phase 2 524M multi-node pretrain — blocked on compute
5. ❌ Phase 3 persistent-kernel live inference — blocked on (4)
6. ❌ Broker / exchange order submission — blocked on (5)
7. ❌ Phase 3.5 FPGA bridge — blocked on (6) + $15-30k card + HDL engineer
8. ❌ Phase 4 strategic layer — blocked on (6) + live trade logs
9. ❌ Phase 5 agentic governance — blocked on (6) + QP solver live

## Design docs

- [`cross_asset_fusion.md`](cross_asset_fusion.md) — Phase 2.5 spec (ES fusion)
- [`fpga_bridge.md`](fpga_bridge.md) — Phase 3.5 spec (FPGA P2P DMA actor)
- [`phase4_strategic_layer.md`](phase4_strategic_layer.md) — Phase 4 spec (HRL + MORL + POW-dTS)
- [`agentic_governance.md`](agentic_governance.md) — Phase 5 spec (Research / Risk / Compliance agents)
