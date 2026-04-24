# Silicon Alpha — End-to-End Architecture

Full flow from raw exchange feeds through the neural forecaster and
deterministic call sheet to dual-venue execution. The diagram below
renders natively on GitHub.

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
        GAMMA["Gamma / VaR Gates<br/>kill-switch"]
        ARB["Bundle Arb Detector<br/>YES + NO =/= $1.00"]
        XV["Cross-Venue Arb<br/>Polymarket &lt;-&gt; Kalshi"]
    end

    %% ========== LAYER 3: STRATEGIC (Phase 4, design only) ==========
    subgraph L3["Layer 3 - Strategic (Phase 4, design only)"]
        direction LR
        HLC["HLC PPO<br/>strategy weights<br/>+ capital allocator"]
        POW["POW-dTS<br/>regime adapter"]
        MORL["MORL reward<br/>Pareto front"]
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
    GAMMA --> SPX
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

    %% ========== HRT-INSPIRED STYLING ==========
    %% Near-black foundation, HRT warm orange primary, deeper orange mid,
    %% light peach for design-only layers, warm cream for terminal venues.
    classDef data fill:#1A1A1A,stroke:#1A1A1A,stroke-width:2px,color:#F8F5F0
    classDef layer1 fill:#E85D2E,stroke:#E85D2E,stroke-width:2px,color:#ffffff
    classDef layer2 fill:#D94F23,stroke:#D94F23,stroke-width:2px,color:#ffffff
    classDef layer3 fill:#F4A261,stroke:#D94F23,stroke-width:2px,color:#1A1A1A,stroke-dasharray:5 3
    classDef venue fill:#F8F5F0,stroke:#E85D2E,stroke-width:3px,color:#1A1A1A

    class OPRA,ES,POLY,KAL data
    class RDMA,KERNEL,TFM,DML layer1
    class QP,GAMMA,ARB,XV layer2
    class HLC,POW,MORL layer3
    class SPX,PM,KS venue
```

## Legend

HRT-inspired palette: near-black foundation, warm orange for the active
neural + execution tiers, pale peach for design-only phases, warm cream
for terminal execution venues.

| Color | Layer | Phase | Status |
|---|---|---|---|
| Near-black `#1A1A1A` | Ingestion | 0 | OPRA live; ES scaffold; Poly/Kalshi pending signup |
| HRT orange `#E85D2E` | Neural Forecaster | 1 | Kernels scaffolded; TradeFM untrained (Phase 2 compute-blocked) |
| Deep orange `#D94F23` | Call Sheet | 2 | QP/Gamma not wired live; arb detectors 0% |
| Peach `#F4A261` (dashed) | Strategic | 4 | **Design-only** (see `phase4_strategic_layer.md`) |
| Cream `#F8F5F0` + orange border | Venues | deploy | All pending broker/API integration |

## Edge semantics

- **Solid arrows**: live data/control flow (design intent)
- **Dashed arrows**: Phase-gated — design spec exists, wiring pending upstream completion
- **"trade logs → MORL"**: Phase 4 reward signal flows back from venues once live

## Critical path today

1. ✅ OPRA ingestion (Databento)
2. ✅ Streaming tokenizer fit on real MBP-1 tape
3. 🔄 Real-data 40M training smoke (`smoke_train_real`)
4. ❌ Phase 2 524M multi-node pretrain — blocked on compute
5. ❌ Phase 3 persistent-kernel live inference — blocked on (4)
6. ❌ Broker / exchange order submission — blocked on (5)
7. ❌ Phase 4 strategic layer — blocked on (6) + live trade logs

See also:
- [`cross_asset_fusion.md`](cross_asset_fusion.md) — Phase 2.5 spec
- [`phase4_strategic_layer.md`](phase4_strategic_layer.md) — Phase 4 spec
