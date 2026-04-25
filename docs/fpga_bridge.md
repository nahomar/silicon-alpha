# Phase-3.5 — PCIe P2P DMA Bridge to FPGA (design spec, no code)

**Status**: design-only. Zero code in the repo corresponds to this doc yet.
The H100-based Phase-3 runtime achieves 4.6-15.8 µs audited inference, but
0DTE priority races are increasingly won in the **nanosecond regime**. The
OS + user-space still contributes 10-20 µs of jitter that no software-only
system can escape. Phase 3.5 pushes the "acting" half of the loop off the
GPU and onto an FPGA via PCIe Peer-to-Peer DMA.

Reference the Silicon Alpha goal in `memory/project_silicon_alpha_goal.md`.

## Architecture — GPU thinks, FPGA acts

```
 ┌───────────────────────────┐  PCIe P2P DMA (no CPU, no OS)
 │  H100 (Layer 1 — thinker) │ ──────────────────────────────┐
 │  TradeFM + DML + QP       │                               │
 │  writes order intents to  │                               ▼
 │  mapped HBM3 ring buffer  │   ┌────────────────────────────────────┐
 └───────────────────────────┘   │  FPGA (Layer 1.5 — actor)           │
                                 │    - OUCH / FIX protocol encoder    │
                                 │    - hardware kill-switch gates     │
                                 │    - order-rate limiter             │
                                 │    - deterministic ns timestamping  │
                                 │  emits wire frames directly to NIC  │
                                 └────────────────────┬───────────────┘
                                                      │ 10G/25G optical
                                                      ▼
                                              Exchange co-lo switch
```

- GPU writes order intents (target contract, qty, side, price, TIF,
  kill-reason bitmask) into a page-locked HBM region mapped to the FPGA
  via PCIe BAR.
- FPGA polls or receives doorbell writes, validates gates, encodes OUCH/FIX,
  emits wire frames. Zero OS involvement after setup.
- Latency budget: GPU→FPGA ≈ 200-500 ns (PCIe 5.0 P2P), FPGA→wire ≈ 100-300 ns.
  Total order-emit path after GPU decision: **~1 µs end to wire**, vs
  10-20 µs via software TCP stack.
- **Tick-to-trade target: < 450 ns** end-to-end (inbound feed tick to
  outbound order on the wire) for the on-FPGA reactive path —
  achievable when the decision is itself encoded in FPGA logic
  (e.g., quote-pulling when a kill-switch fires, or auto-cancel on
  position-limit breach). Decisions that require the GPU forecaster
  pay an extra 200-500 ns for the P2P round-trip.
- **Memory-mapped AXI**: C++ host process uses `mmap()` to expose the
  FPGA's AXI registers directly into CPU virtual memory. Eliminates
  ioctl/syscall overhead on the (rare) host-side queries; the hot path
  is GPU-direct, never touches the host CPU.

## What the FPGA owns

### 1. OUCH / FIX protocol encoding
Exchange-specific wire protocols are compact binary formats with strict
framing. FPGA can serialize an order in 2-4 clock cycles (~10 ns @ 250 MHz)
vs thousands of CPU cycles through a zerocopy networking stack.

Targets for our venues:
- **OPRA options exchanges** (NYSE Amex, NASDAQ PHLX, CBOE): OUCH 4.2-style
  binary protocols.
- **CME** (for ES hedging): iLink 3 (binary, FAST-like encoding).

### 2. Hardware kill-switches
Immediate (<100 ns) order cancellation on any of:
- Position limit breach (FPGA maintains mirrored position state).
- Gamma / delta exposure over threshold.
- Per-second order-rate > cap (prevents runaway from a buggy GPU write).
- External "big red button" toggle from risk desk (GPIO input).
- Missing heartbeat from GPU (dead-man switch).

The kill-switch is in **hardware logic**, not software — it fires even if the
GPU, CPU, or OS are wedged.

### 3. Deterministic timestamping
FPGA stamps every outbound frame with a PTP/White Rabbit nanosecond-precision
timestamp before emission, and every inbound frame on ingest. Gives us
ground-truth RTT numbers for the audit trail (Phase-5 compliance agent
consumes these).

### 4. Order-book hot state
Maintains a small in-FPGA shadow of the top-of-book for each subscribed
contract, updated from the same NIC's incoming OPRA feed via a separate
PCIe ingest path. Lets the FPGA compute price-limit checks against
real-time book without round-tripping to the GPU.

## What stays on the GPU

- 524M TradeFM forecaster (needs tensor cores)
- DML pricer (needs parallel FMA on Greek tensors)
- QP solver (deterministic but cheap; keep on GPU so it shares HBM with the
  neural outputs — no re-copy).
- Any heavy calibration / training work.

## Dependencies — gate before starting

1. Phase-3 software-only live inference must be running. Otherwise we can't
   tell whether FPGA actually helps vs masks a bug upstream.
2. A co-located FPGA server (Xilinx Alveo U55C or U250, Intel Stratix 10, or
   Achronix Speedster 7t). Budget: ~$15-30k for the card, ~$50-150k/yr for
   exchange co-lo rack + cross-connects.
3. Broker/exchange sponsor agreement allowing direct market access (DMA) —
   typically requires net capital + operational due diligence.
4. Phase-2 524M model trained, QP solver integrated, kill-switch semantics
   defined in Phase-3 software path.

None of those exist today. Phase 3.5 is real, but it's after Phase 3 goes
live and generates enough trade volume to justify the hardware spend.

## Cost-benefit threshold

Order of magnitude: FPGA saves ~15 µs per order. On SPX 0DTE MM, toxicity
events arrive in bursts where first-to-cancel wins. If you're getting
picked off on ~100 orders/day and the FPGA-avoidable loss per pickoff is
$500-2000, the incremental FPGA value is **$50k-$200k/month**. That's the
threshold to cross before buying the card.

If daily volume < ~500 0DTE orders/day, stay software-only.

## Component sketches — not code yet

```text
GPU-side intent buffer (shared HBM region, mapped via nvidia-peermem):
  struct OrderIntent {
      uint64_t  seq;             // monotonic, GPU writes
      uint64_t  timestamp_ns;    // GPU wall-clock at decision
      uint32_t  instrument_id;
      uint32_t  side;            // 0 = buy, 1 = sell
      uint64_t  qty;
      uint64_t  limit_price_q;   // in exchange price units (tick-scaled)
      uint32_t  tif;             // IOC / DAY / FOK
      uint32_t  strategy_tag;
      uint32_t  kill_gate_mask;  // which FPGA gates this order must pass
      uint32_t  crc32;           // integrity
  };

FPGA writes back to a response region:
  struct OrderAck {
      uint64_t  gpu_seq;         // echo of GPU seq
      uint64_t  fpga_timestamp_ns;
      uint32_t  status;          // ACCEPTED / KILLED_* / REJECTED_*
      uint32_t  exchange_seq;    // post-wire exchange ack id (filled later)
  };
```

Both buffers are page-aligned, pinned, mapped into the FPGA's PCIe BAR.

## What prematurely scaffolding looks like (and why we're not)

**Would include**: `odte/fpga/` directory with a `fake_fpga.py` simulator,
config fields for FPGA endpoints, a GPU-side Cython binding stub,
timestamp-mock plumbing through the trainer.

**Why it's a trap**: without real FPGA hardware (or even an ASIC
synthesizer + emulator) everything here is a lie. Simulated FPGA latency
tells you nothing about real PCIe P2P behavior. The binding stubs rot
every time PyTorch / CUDA upgrades. Net: maintenance burden without
validation.

**What instead**: this doc. When the Phase-3 live-trading revenue crosses
the threshold, we buy the card, write the HDL, and delete this note in
favor of `odte/fpga/` with things that actually work.

## Exit criteria (when to start building)

Begin Phase-3.5 hardware/code only when all of:

1. Phase-3 software-only live inference is in production, handling real
   order flow.
2. Trade logs show consistent toxicity-pickoff losses in the 10-20 µs
   latency band (i.e., we're getting beaten by faster players on
   identifiable patterns).
3. Net monthly realized P&L > $200k (the card + co-lo is ~$15k/mo
   amortized; need comfortable headroom).
4. A named HDL engineer is on the team or contracted. This is not a
   "Claude writes Verilog" project.

## Related files

- [`docs/architecture.md`](architecture.md) — shows FPGA bridge in the
  Silicon Alpha diagram
- [`docs/phase4_strategic_layer.md`](phase4_strategic_layer.md) — HRL
  strategic layer (different phase, different concerns)
- [`odte/kernels/`](../odte/kernels/) — GPU-side persistent kernels
  (Phase 3, upstream of this phase)
