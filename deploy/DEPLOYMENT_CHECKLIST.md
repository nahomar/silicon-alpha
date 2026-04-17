# Monday 2026-04-20 — production deployment runbook

Single-page runbook for 0DTE SPX paper-to-live cutover. Every step has
a go / no-go gate. Abort if any **hard** gate fails.

## T-24h — Sunday 2026-04-19

- [ ] **Cluster reserved.** 24× H100 SXM on one of:
  - **Google Cloud A3 Mega** — 3× `a3-megagpu-8g` via
    `./infra/gcp/phase2_a3mega.sh` (HRT's own choice per their 2024 GCP
    partnership). 1600 Gbps GPUDirect-TCPX, use `infra/gcp/tcpx_nccl_env.sh`.
  - **RunPod / Lambda bare-metal** — ConnectX-7 + IB + `nv_peer_memory`
    via `infra/cloud/runpod_h100.sh`. Use `deploy/nccl_env.sh`.
  - Request H100 quota (24 GPUs, us-central1) 48h ahead if first time.
- [ ] **Training run complete** and the 524M checkpoint copied to the
      head node's `checkpoints/tradefm_524m.pt` — for GCP:
      `gcloud storage cp gs://BUCKET/ckpts/.../best/rank_0.pt checkpoints/tradefm_524m.pt`
- [ ] **Checkpoints uploaded.**
      `checkpoints/tradefm_524m.pt`, `checkpoints/dml_pricer.pt`,
      `checkpoints/hybrid_tokenizer.json` synced from S3.
- [ ] **Repo deployed.** `git pull && pip install -r requirements.txt`
      on every node. CUDA build of `odte_kernels_cu` successful.
- [ ] **Databento OPRA key** set as `DATABENTO_API_KEY` in the env file.
- [ ] **Broker margin URL** set as `BROKER_MARGIN_URL` so the intraday
      refresh loop can pull every 5 minutes during RTH.

## T-60m — Monday 08:30 ET

Run **`./deploy/monday_go_live.sh`** on the head node. It does:

| Step | Action | Hard gate |
|---|---|---|
| 1 | Host sanity: GPU count, `ens1` interface, `nv_peer_memory` module | GPU count ≥ 8 |
| 2 | Source `deploy/nccl_env.sh` (pins NCCL to `ens1`) | — |
| 3 | `python setup.py build_ext --inplace` in `odte/kernels/` | import of `odte_kernels_cu` succeeds |
| 4 | `curl $BROKER_MARGIN_URL` → `configs/broker_margin_live.yml` + background refresh loop | file exists, version=1 |
| 5 | `deploy/monday_quantile_refit.py` → `checkpoints/hybrid_tokenizer_monday.json` | ≥ 200 rows of pre-market tape |
| 6 | `deploy/preflight.py --full` | 0 hard fails |
| 7 | Launch `odte_live_paper.py` in headless `screen` | — |

## T-30m — Monday 09:00 ET

Preflight gates that must PASS:

| Gate | Target | Fatal if |
|---|---|---|
| hopper_kernels | import `odte_kernels_cu` | missing |
| rdma (optional) | `ODTE_HAS_RDMA=1` | only if `--require-rdma` |
| dml_greeks | ATM τ=1d Greek error < 2% | > 2% |
| tradefm_ckpt | loads without shape mismatch | missing |
| tokenizer_monday | mtime < 12h | missing / stale |
| margin_table | crisis `k_gamma ≥ 0.002` | missing |
| latency_p99 | ≤ 25 µs (CUDA-persistent backend) | > 25 µs |
| directional_acc | ≥ 53 % on last-1h paper fills | < 53 % |

If a soft warning fires (e.g. NCCL env missing) and you're still running single-node, continue. If a **hard** gate fails, abort — the launch script exits non-zero before starting the screen session.

## T-0 — 09:30 ET opening bell

Runner active inside `screen -S odte_live_YYYYMMDD`. Monitor:

```bash
# tail live log
tail -f reports/monday_live_*/live.log

# live parquet counts
watch -n 5 'ls -l reports/mm_data/$(date -u +%Y-%m-%d)/$(date -u +%H)/'

# attach the session
screen -r odte_live_$(date -u +%Y%m%d)
# detach with Ctrl+a d
```

**Kill switches:**
- `kill $(screen -list | awk -F. '/odte_live/ {print $1}' | tr -d '\t')`
  stops the runner. Open orders on the paper broker are logged but never
  reach a real exchange, so killing is always safe for PAPER mode.
- If this were wired to a live broker (explicitly out of scope here):
  flattening happens through `close_for_pennies_orders` before kill.

## T+360m — 15:30 ET — power hour

- [ ] Broker table refresh loop confirms `tod_mult_last_30` multiplier active
- [ ] RiskGates auto-tightens `gamma_dollar_cap` via `gamma_cap_scale`
      (1.0× at 14:30 → 0.10× at 15:55)
- [ ] pin_score > 70 on any leg → `close_for_pennies_orders()` fires
      automatically; no human intervention required

## T+390m — 16:00 ET — close

Runner ends at `--max-ticks` or at EOD. Post-trade analyzer auto-runs
if `--post-trade` was set (default in `monday_go_live.sh`).

Open the report:
```
reports/post_trade_<ts>.md
```

Expected structure:

| horizon | markout bps | spread bps | edge bps | dir-hit | n |
|---|---|---|---|---|---|
| 1000 ms | ... | ≤ 360 bps (3.6 %) | ... | ≥ 53 % | ... |
| 5000 ms | ... | ... | ... | ... | ... |
| 30000 ms | ... | ... | ... | ... | ... |

**Success criteria** (to justify a real paper→live cutover on a future day):
- **1s edge bps > 0** — signal beats spread friction net of fees
- **dir-hit @ 1s ≥ 53 %** — directional accuracy survives realized spread
- **zero risk-gate breaches** in the session log
- **zero margin-call warnings** from the broker refresh loop

## What this playbook does NOT cover

- **Live-broker order routing.** `odte_live_paper.py` has no code path
  that submits to a real exchange. Wiring one in is an explicitly-reviewed
  future change, NOT something this checklist green-lights.
- **Capital risk.** Even if every gate passes, a 5σ move on 0DTE shorts
  can wipe a book faster than any stop-loss fires. Size accordingly.
- **Regulatory posture.** 0DTE self-directed trading is legal; providing
  quoted liquidity at scale is not without registration.

## If things break

| Symptom | Action |
|---|---|
| `odte_kernels_cu` import fails | confirm CUDA 12.4+, `torch` matches; rebuild with `ODTE_BUILD_RDMA=0` first |
| NCCL init timeout | `ip link show` → set `NCCL_SOCKET_IFNAME` to the actual high-bandwidth NIC |
| p99 latency > 25 µs | profile with `nvprof`; check persistent-kernel is running (not one-shot launches) |
| dir-acc < 53 % pre-open | DO NOT LAUNCH. Roll back to last known-good TradeFM ckpt |
| margin table unreachable | script falls back to `broker_margin_monday_crisis.yml` (5× k_gamma) |
| screen session died | check `reports/monday_live_*/live.log` for Python traceback |
