# Phase-5 — Agentic Security, Risk & Compliance (design spec, no code)

**Status**: design-only. Zero code in the repo corresponds to this doc yet.
Deploys only **after** the trading engine is live and producing real trade
logs — there's nothing to govern before then.

Reference the Silicon Alpha goal in `memory/project_silicon_alpha_goal.md`.

## Why this phase exists

Once Silicon Alpha is live across SPX 0DTE + Polymarket + Kalshi:

- **Research velocity vs risk trade-off**: signals drift, market regimes
  change, and a human operator can't vet every signal modification in real
  time.
- **Solver correctness** is not self-evident: the QP solver (Layer 2) must
  satisfy KKT conditions for the position to be risk-optimal. A bug or
  edge case can silently produce non-KKT solutions that look reasonable
  but fail under stress.
- **Regulator scrutiny**: 0DTE options and prediction markets are under
  active CFTC / SEC attention. An auditable, timestamped record of every
  decision and risk-check is not optional.

Phase 5 deploys three specialized agents that sit above the trading loop,
each with a narrow, adversarial job.

## The three agents

### 1. Research Agent — signal proposer

**Job**: continuously probe the forecaster + strategy stack for staleness or
drift, propose signal modifications.

Concretely:
- Monitors forecast accuracy drift by regime bucket (gamma regime, vol
  regime, venue).
- Flags features where token-level loss has climbed above baseline by
  more than N sigma over a rolling window.
- Proposes modifications — e.g., "re-weight ES lead-lag feature from 0.3
  to 0.5 for current regime", "retrain tokenizer edges on last 30 days of
  tape" — and submits them to the Risk Agent for validation before
  deployment.

Research Agent **cannot** deploy changes directly. It proposes; Risk gates.

### 2. Risk Agent — hard gatekeeper

**Job**: validate every proposed change against hard constraints before it
touches the live policy. **This is the adversarial role.**

Concretely:
- Validates KKT conditions on every QP solver output (primal feasibility,
  dual feasibility, complementary slackness, stationarity). If the
  solution doesn't satisfy KKT within numerical tolerance, order is
  rejected and a flag is logged.
- Replays the proposed change against historical tape and confirms
  ex-post risk metrics (max drawdown, VaR, concentration) stay within
  the risk-limit envelope.
- Enforces position/strategy/venue limits as hard gates — orders that
  would violate are blocked before reaching the FPGA.
- Kills trading entirely (hard stop) on any of:
  - Realized daily loss > daily VaR × safety_factor.
  - Consecutive signal-validation failures.
  - External "big red button" from the operator.

Risk Agent has a **veto** on every decision. Nothing proposed by Research
or any strategic RL controller goes live without Risk's sign-off.

### 3. Compliance Agent — immutable audit trail

**Job**: maintain a tamper-evident log of every decision the system makes,
with enough context for regulators or internal audit to reconstruct any
trade.

Concretely:
- Every QP output, every order intent, every FPGA kill-switch fire, every
  Research proposal, every Risk decision is hash-chained into an append-only
  log (Merkle tree or blockchain-style; candidates: AWS QLDB, Amazon Managed
  Blockchain, local RocksDB + daily Merkle roots to an external notary).
- Timestamps come from the FPGA's PTP/White Rabbit clock (nanosecond
  precision) so order-of-events can be replayed exactly.
- Produces daily attestations: "on 2026-MM-DD, the engine made N decisions,
  M were kill-switched, P were filled, ∑realized P&L = $X. Log root hash
  = Y." Root hash is published to an external timestamp service so the log
  can't be quietly rewritten.

Compliance Agent has **read access** to everything but **write access only
to the audit log**. It cannot block or modify; it only records.

## How the three agents interact

```
Market event
     │
     ▼
 524M TradeFM + DML Pricer  ──────▶ QP Solver ─────▶ Order Intent
                                       │
                                       │   (Risk Agent validates KKT)
                                       ▼
                                    Risk Gate  ─────▶ FPGA  ─────▶ Wire
                                       │
      ┌─────────────────── Research Agent proposes ──┐
      │                       (out-of-band)          │
      ▼                                              │
  Parameter delta                                    │
      │                                              │
      ▼                                              ▼
  Risk Agent replay + KKT validation          Compliance Agent
      │                                         (reads all events,
      └─────▶ approved? ─────▶ deploy             writes hash-chain)
              rejected? ─────▶ log + notify
```

Key invariants:

- Research proposes changes; Risk vetos; Compliance records both.
- Every order gate (Risk → FPGA) is logged with input state + gate decision
  + output action.
- The Risk Agent's KKT validator runs **in parallel** with the FPGA emit
  path — if KKT fails, an immediate cancel is fired before the fill comes
  back.

## Why not just build one agent?

Single-agent designs conflate roles and create subtle conflicts of interest
(the agent that proposes changes is the same one that evaluates risk). The
three-agent split with adversarial separation mirrors how real trading
firms structure research / risk / compliance as independent functions with
veto authority.

## Dependencies — gate before starting

Phase 5 requires:

1. Phase-3 live inference producing real trades (nothing to govern otherwise).
2. A QP solver implementation whose KKT conditions can be checked (the
   Layer-2 "Call Sheet" from the Silicon Alpha goal — not yet built).
3. Non-zero realized trade volume so Compliance has something to log.
4. A separation of concerns story to describe to regulators — this doc is
   the start of it.

## What prematurely scaffolding looks like (and why we're not)

**Would include**: `odte/agents/` directory with three stub classes, a
fake event bus, simulated trades to test the audit chain on, config
fields for agent weights.

**Why it's a trap**: without a real QP solver to KKT-check, the Risk Agent
is validating nothing. Without real trade logs, Compliance is hashing
placeholders. Research proposes modifications to a forecaster that isn't
trained. Everything is pretend, and pretend governance is worse than no
governance because it creates false confidence.

**What instead**: this doc. When the QP solver ships and trades go live,
we build the Risk Agent first (because it's the veto that keeps us solvent),
then Compliance (because regulators need it), then Research (because it's
the "nice to have" that makes the others more efficient).

## Exit criteria (when to start building)

Begin Phase-5 code only when:

1. QP solver is implemented and producing live orders. (Not yet — see
   Silicon Alpha goal memory.)
2. At least one live trading account is open and producing real trade logs.
3. Realized P&L > operating cost (otherwise agents are just added burn).
4. At least one regulatory touchpoint is on the calendar (broker KYC/onboarding,
   CFTC registration inquiry, LLC formation for trading) — this is what
   actually forces compliance work to get done.

## Build order when the time comes

1. **Risk Agent first**. It's the veto. A broken Research or missing
   Compliance is recoverable; a broken Risk means blown-up positions.
2. **Compliance Agent second**. Regulatory work takes time; start the
   audit trail before the first regulatory touchpoint.
3. **Research Agent third**. The "optimizer" of the three. Useful only
   once the other two are solid.

## Related files

- [`docs/architecture.md`](architecture.md) — shows all three agents
  surrounding the live trading loop
- [`docs/phase4_strategic_layer.md`](phase4_strategic_layer.md) — HRL /
  MORL layer that Research Agent proposes modifications to
- [`docs/fpga_bridge.md`](fpga_bridge.md) — Phase 3.5 FPGA execution layer
  that Risk Agent's kill-switches ultimately route through
- `memory/project_silicon_alpha_goal.md` — north-star
