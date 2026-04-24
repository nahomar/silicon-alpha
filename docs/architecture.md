# Silicon Alpha — End-to-End Architecture

Full flow from raw exchange feeds → neural forecaster → deterministic call
sheet → dual-venue execution. Mermaid renders natively on GitHub.

```mermaid
flowchart TB
    %% ==================== DATA LAYER ====================
    subgraph DATA["📡 Layer 0 — Ingestion"]
        direction LR
        OPRA["OPRA cmbp-1<br/>SPX/NDX options<br/><i>Databento · live</i>"]
        ES["ES Futures MBP-1<br/><i>GLBX.MDP-3 · Phase 2.5</i>"]
        POLY["Polymarket<br/>CLOB WSS<br/><i>~100 ms · Phase 3+</i>"]
        KAL["Kalshi<br/>CLOB WSS<br/><i>~100 ms · Phase 3+</i>"]
    end

    %% ==================== LAYER 1 — NEURAL FORECASTER ====================
    subgraph L1["🧠 Layer 1 — Neural Forecaster (H100, 4.6-15.8 µs)"]
        direction TB
        RDMA["GPUDirect RDMA<br/>→ HBM3 ring buffer<br/><i>bypasses CPU</i>"]
        KERNEL["Persistent CUDA Kernels<br/>SM90a · always-resident"]
        subgraph MODELS["Forecaster models"]
            direction LR
            TFM["524M TradeFM<br/>next-tick direction<br/><i>zero-shot multi-venue</i>"]
            DML["DML Pricer<br/>0DTE Greeks + IV surface"]
        end
        RDMA --> KERNEL
        KERNEL --> TFM
        KERNEL --> DML
    end

    %% ==================== LAYER 2 — CALL SHEET ====================
    subgraph L2["⚖️ Layer 2 — Deterministic Call Sheet"]
        direction TB
        QP["QP Executor<br/>risk-adjusted position w*<br/><i>closed-form solver</i>"]
        GAMMA["Gamma / VaR Gates<br/>kill-switch"]
        ARB["Bundle Arb Detector<br/>YES+NO≠$1.00"]
        XV["Cross-Venue Arb<br/>Polymarket ↔ Kalshi"]
    end

    %% ==================== LAYER 3 — STRATEGIC (Phase 4) ====================
    subgraph L3["🎯 Layer 3 — Strategic (Phase 4 · design only)"]
        direction LR
        HLC["HLC (PPO)<br/>strategy weights<br/>+ capital allocator"]
        POW["POW-dTS<br/>regime adapter<br/>(±γ, vol, events)"]
        MORL["MORL reward<br/>Pareto front<br/>[pnl, slip, DD, γ, conc, up]"]
    end

    %% ==================== VENUES ====================
    subgraph VEN["💰 Execution Venues"]
        direction LR
        SPX["SPX/NDX 0DTE<br/><b>$166-$800 / unit / day</b><br/><i>MM + toxic-flow avoidance</i>"]
        PM["Polymarket<br/><b>0.1-3.0% / arb cycle</b><br/><i>bundle + cross-venue</i>"]
        KS["Kalshi<br/><b>0.1-3.0% / arb cycle</b><br/><i>market pull on catalysts</i>"]
    end

    %% ==================== FLOW EDGES ====================
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

    %% ==================== HRT-INSPIRED STYLING ====================
    %% Deep-navy primary, steel-blue secondary, cyan accent, white text.
    classDef data fill:#0a1e3f,stroke:#0a1e3f,stroke-width:2px,color:#ffffff
    classDef layer1 fill:#1e3a6f,stroke:#1e3a6f,stroke-width:2px,color:#ffffff
    classDef layer2 fill:#2c5aa0,stroke:#2c5aa0,stroke-width:2px,color:#ffffff
    classDef layer3 fill:#4a7fb8,stroke:#4a7fb8,stroke-width:2px,color:#ffffff,stroke-dasharray:5 3
    classDef venue fill:#22d3ee,stroke:#0a1e3f,stroke-width:2px,color:#0a1e3f
    classDef group fill:#f0f4f8,stroke:#0a1e3f,stroke-width:2px,color:#0a1e3f

    class OPRA,ES,POLY,KAL data
    class RDMA,KERNEL,TFM,DML layer1
    class QP,GAMMA,ARB,XV layer2
    class HLC,POW,MORL layer3
    class SPX,PM,KS venue
    class DATA,L1,L2,L3,VEN,MODELS group
```

## Legend

| Color | Layer | Phase | Status |
|---|---|---|---|
| Deep navy `#0a1e3f` | Ingestion | 0 | OPRA live; ES scaffold; Poly/Kalshi pending signup |
| Steel blue `#1e3a6f` | Neural Forecaster | 1 | Kernels scaffolded; TradeFM untrained (Phase 2 compute-blocked) |
| HRT blue `#2c5aa0` | Call Sheet | 2 | QP/Gamma not wired live; arb detectors 0% |
| Light blue `#4a7fb8` (dashed) | Strategic | 4 | **Design-only** (see `phase4_strategic_layer.md`) |
| Cyan `#22d3ee` | Venues | deploy | All pending broker/API integration |

## Edge semantics

- **Solid arrows**: live data/control flow (design intent)
- **Dashed arrows**: Phase-gated — design spec exists, wiring pending upstream completion
- **"trade logs → MORL"**: Phase 4 reward signal flows back from venues once live

## Critical path legend (where we are today)

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
- `memory/project_silicon_alpha_goal.md` — north-star goal (outside repo)
- `memory/project_phase_roadmap.md` — phase sequencing rules (outside repo)
