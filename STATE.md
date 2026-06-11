# STATE.md — 0DTE Alpha Engine: project status & engineering log

Last synced: 2026-04-19. Top-to-bottom status, decisions, and a resume
checklist so I can pick the project back up after time away without
re-deriving context.

## Project Summary

This repo is a 0DTE (zero-days-to-expiry) SPX options alpha engine in active build, spanning a full stack: Differential ML option pricer, a decoder-only transformer (TradeFM) over microstructure tokens, a QP-based deterministic executor with broker-aware margin, paper broker, and post-trade analyzer. Target user is a single operator running the Monday weekly cycle on SPX 0DTE. Current state: Phase 0 (DML pricer) is done on Mac; Phase 1 (40M Mini-TradeFM validation) is actively training on a Colab Pro+ A100 80GB as of minutes ago, ETA ~45-60 min from start. Infrastructure recipes for Phases 2-5 are committed but not yet run. The repo also retains the earlier sentiment-scraper code (this repo's pre-0DTE incarnation) at the top level (`main.py`, `scrapers/`, `ml/sentiment.py`) — that predates the 0DTE pivot and is orthogonal to it.

## Phase Status Table

| Phase | Status | Key Files | Validation Gate | Notes |
|---|---|---|---|---|
| 0 — DML pricer (Mac) | done | `odte/dml_pricer.py`, `odte/train/train_dml.py`, `notebooks/colab_phase0_dml.ipynb` | Greek err ≤ 2% vs analytic BS | Checkpoint used by Phase 1 migration gate when present. |
| 1 — 40M Mini-TradeFM (Colab A100/H100) | active (training NOW) | `odte/transformer_tradefm.py`, `odte/train/pretrain_tradefm.py`, `configs/tradefm_40m.yml`, `notebooks/colab_phase1_tradefm.ipynb` | `odte.train.migration_check` → GO (dir-acc 53-65%, val loss strictly decreasing, DML Greek err ≤2%, no overfit) | Synth data path: `odte/synth_options.py` (diverse multi-strike, fixed in c2955fb). |
| 2 — 524M TradeFM (GCP A3 Mega, 24× H100) | queued | `configs/tradefm_524m.yml`, `infra/gcp/phase2_a3mega.sh`, `infra/gcp/launch_torchrun_524m.sh`, `infra/gcp/tcpx_nccl_env.sh` | Throughput ≥ 210k tok/s per node; loss curve converging | Planned cost ~$50k/run, ~130 GPU-hours. Gated behind Phase 1 GO. |
| 3 — Hopper-native CUDA kernels | queued | `odte/kernels/fused_bin.cu`, `odte/kernels/persistent_decode.cu`, `odte/kernels/rdma_ingest.cu`, `odte/kernels/bindings.cpp`, `odte/kernels/setup.py` | Decode ≥ 2× torch SDPA on H100 | Python fallbacks exist (`*.py` beside each `.cu`). |
| 4 — DML Heston fine-tune + adversarial RL | queued | `odte/rl/agents.py`, `odte/rl/train_world_sim.py`, `odte/world_sim.py`, `models/rl_agent.py`, `models/rl_env.py`, `odte/phase4_smoke.py` | Positive edge after slippage vs RL adversary | Heston path lives inside `dml_pricer` (σ_eff gating). |
| 5 — QP executor + risk gates + paper broker + post-trade | queued (code done, untested live) | `odte/exec/qp.py`, `odte/exec/risk_gates.py`, `odte/exec/broker_margin.py`, `odte/exec/intraday_margin.py`, `odte/exec/paper_broker.py`, `odte/exec/post_trade.py`, `odte/exec/streaming_ofi.py`, `deploy/monday_go_live.sh`, `deploy/preflight.py` | Preflight green, paper broker fills match sim | `configs/broker_margin_example.yml`, `configs/broker_margin_monday_crisis.yml` are the margin scenarios. |

## Current Active Task — Phase 1 Colab Training

Running on Colab Pro+ A100 80GB instance (started minutes before this handoff). Notebook: `notebooks/colab_phase1_tradefm.ipynb`.

| Item | Value |
|---|---|
| Checkpoint dir (Drive) | `/content/drive/MyDrive/tradefm_ckpts/tradefm_40m/` |
| Best snapshot | `best.pt` (updated each time val improves) |
| Final snapshot | `final.pt` |
| Loss history JSON | `train_loss.json`, `val_loss.json` in same dir |
| Runtime target | ~45-60 min |
| Expected dir-acc | 54-65% (NOT 100%; the trivial-synth bug from pre-c2955fb let it hit 100% via memorization) |
| Migration gate | Cell 7 runs `odte.train.migration_check.decide(...)` and prints "GO" or "NO-GO" |

When the run finishes:

1. Open the notebook, jump to Cell 7 output. It writes `reports/migration_decision.md` in the repo checkout.
2. If GO → launch Phase 2 via `infra/gcp/phase2_a3mega.sh` (needs GCP_PROJECT, GCP_BUCKET, REPO_URL env vars set; ~$50k/run budget).
3. If NO-GO → read the `reasons` list in `migration_decision.md`. Common iterations:
   - Val loss not strictly decreasing → more epochs, or lower LR
   - Dir-acc < 53% → synth regime too narrow; tune `odte/synth_options.py`
   - Dir-acc ≥ 99% → still overfitting; add IV-regime jitter or real data
   - DML Greek err > 2% → rerun Phase 0 with more MC paths

## Key Commits (HEAD first)

| Hash | Subject | What it adds |
|---|---|---|
| c2955fb | phase1: diverse multi-strike synth + overfit-aware migration gate | 12 sessions × 3 IV regimes × 21 strikes × {C,P}; relative-slope fix in `_strictly_decreasing`; overfit sentinel (dir-acc ≥0.99 blocks). THIS is what Cell 7 now uses. |
| 0d2a7db | Created using Colab | auto-saved notebook edit |
| d7a7e64 | notebook cell 5: aggressive A100-40G config + diagnostics | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, prints git HEAD so pulls are verifiable, ctx=1024, batch=2, grad_accum=16 on 40G |
| 4d8f750 | transformer: grad_checkpointing + bf16 autocast — fits 40M on A100 40G | `TradeFMConfig.grad_checkpointing` flag, bf16 autocast in pretrain loop |
| 33fbe83 | notebooks: run-all safe — auto VRAM-size, small shards, git pull on reconnect | cell 3 git-pulls, cell 4 shard_rows=8000, cell 5 VRAM auto-detect, cell 8 keepalive gated behind ENABLE=False |
| bcf038c | transformer: use torch SDPA on CUDA by default | cuts peak mem 3-6× vs eager; flash-attn-3 still preferred on H100 when `cfg.use_flash_attn=true` |
| 5659db7 | notebooks: getpass fallback when Colab Secrets times out | |
| 28b58b5 | notebooks: add colab_bootstrap.ipynb escape hatch for private-repo OAuth | |
| 653f3e9 | notebooks: use Colab Secret for private-repo clone | reads `GITHUB_TOKEN` via `google.colab.userdata`, scrubs from `.git/config` after clone |
| a11f99d | Initial commit — 0DTE alpha engine (Phases 0-5 + deploy + Colab) | foundational drop of everything |

HEAD at time of writing: `7f22aa7` (per user). Locally `c2955fb` is tip on main — `7f22aa7` may be a remote-side merge/rebase; verify with `git fetch && git log origin/main --oneline` before doing more work. **Judgment call**: treating `c2955fb` as authoritative; if there's drift, `git pull --rebase`.

## File Layout (top-level)

| Path | Purpose |
|---|---|
| `main.py` | Old sentiment-scraper entrypoint (pre-0DTE pivot; kept for fallback) |
| `mm_*.py`, `mm/` | Market-making stack (microprice, Avellaneda-Stoikov quoter, PRISM orchestrator) |
| `odte_*.py` | Top-level 0DTE smoke/live scripts (`odte_smoke.py`, `odte_live_paper.py`, `odte_phase1_smoke.py`, `odte_phase4_smoke.py`, `odte_budget.py`) |
| `odte/` | 0DTE engine package: pricer, transformer, tokenizer, synth, world sim, executor, plus subdirs |
| `odte/train/` | Pretrain, distributed, eval, LR finder, mem audit, checkpoint, migration_check |
| `odte/exec/` | QP, risk gates, broker margin, intraday margin, paper broker, post-trade, streaming OFI |
| `odte/rl/` | RL agents + world-sim trainer |
| `odte/kernels/` | Hopper-native CUDA (`.cu`) + Python fallbacks |
| `odte/accel/` | Numba JIT kernels |
| `odte/runtime/` | Runtime glue |
| `models/` | TradeFM transformer, tokenizer, DDPM, RL env/agent, unified dataset, configs |
| `ml/` | Sentiment, correlation, patterns (pre-0DTE) |
| `configs/` | YAML configs for `tradefm_smoke`, `tradefm_40m`, `tradefm_524m`, `tradefm_budget`, broker margin scenarios |
| `notebooks/` | Colab notebooks for Phase 0 and Phase 1 + bootstrap helper |
| `infra/gcp/` | GCE provisioning (Phase 1 A100, Phase 2 A3 Mega), NCCL env, torchrun launcher |
| `infra/cloud/` | RunPod + Lambda provisioning scripts |
| `deploy/` | Monday go-live script, preflight, NCCL env, quantile-refit, deployment checklist |
| `feeds/`, `scrapers/`, `data/`, `analysis/` | Upstream data ingestion + reporting (mostly pre-0DTE) |
| `synthetic/` | Synthetic data generators |
| `reports/` | Per-run JSON/MD output. `reports/migration_decision.md` is where the gate writes its verdict. |
| `checkpoints/` | Local (Mac) checkpoints |
| `BUDGET.md`, `README.md` | Cost budget, original sentiment-bot README |

## Provisioning Recipes (one-command launch)

All scripts `set -euo pipefail` and read required env vars.

| Script | Target | Minimum env | Launch |
|---|---|---|---|
| `infra/gcp/phase1_a100.sh` | Single A100 80GB on GCE for Phase 1 | `GCP_PROJECT`, `GCP_BUCKET`, `REPO_URL` | `./infra/gcp/phase1_a100.sh` |
| `infra/gcp/phase2_a3mega.sh` | 3× A3 Mega (24× H100) for Phase 2 | `GCP_PROJECT`, `GCP_BUCKET`, `REPO_URL` | `./infra/gcp/phase2_a3mega.sh` |
| `infra/gcp/launch_torchrun_524m.sh` | Torchrun launcher for 524M distributed | run from inside the provisioned cluster | `./infra/gcp/launch_torchrun_524m.sh` |
| `infra/cloud/runpod_a100_phase1.sh` | Single A100 80GB PCIe on RunPod (~$1.20-1.90/hr) | `RUNPOD_API_KEY`, optional `GH_TOKEN` | `./infra/cloud/runpod_a100_phase1.sh` (UNTRACKED; `git add` before relying on it) |
| `infra/cloud/runpod_h100.sh` | H100 pod on RunPod for Phase 2 | `RUNPOD_API_KEY`, `REPO_URL`, optional `RUNPOD_POD_TYPE` | `./infra/cloud/runpod_h100.sh` |
| `infra/cloud/lambda_a100.sh` | Lambda Cloud A100 | `LAMBDA_INSTANCE_TYPE`, `LAMBDA_REGION`, `LAMBDA_SSH_KEY` | `./infra/cloud/lambda_a100.sh` |
| `notebooks/colab_phase0_dml.ipynb` | Colab DML pricer | Colab Secret `GITHUB_TOKEN` | Open in Colab, Runtime → A100, Run all |
| `notebooks/colab_phase1_tradefm.ipynb` | Colab 40M pretrain | Colab Secret `GITHUB_TOKEN`, Drive mounted | Open in Colab, Runtime → A100 or H100, Run all |
| `deploy/monday_go_live.sh` | Monday live-trading cycle | broker creds via env | `./deploy/monday_go_live.sh` (after Phase 5 validation) |

## Known Issues & Pending Decisions

1. **RunPod pod `i7wndt4y3bjpsq` status unknown.** User provisioned an H100 SXM pod ($2.99/hr) earlier in the session. It may still be billing. **Action**: verify at https://www.runpod.io/console/pods and terminate if not in use. If `runpodctl` is configured: `runpodctl stop pod i7wndt4y3bjpsq` then `runpodctl remove pod i7wndt4y3bjpsq`.
2. **RunPod API key rotation.** The user's API key was pasted in chat earlier; RunPod scans chats and may have auto-revoked it. After pod termination, rotate the key at https://www.runpod.io/console/user/settings. Never commit — it lives only in session env.
3. **`infra/cloud/runpod_a100_phase1.sh` is untracked.** `git status` shows it as untracked in `c2955fb`. If it's the right recipe, add+commit it; otherwise delete.
4. **HEAD divergence.** User reports HEAD on main is `7f22aa7` but local tip is `c2955fb`. Run `git fetch origin && git log origin/main --oneline -5` to reconcile before any new commits.
5. **Migration gate calibration.** As of c2955fb, gate requires dir-acc in [0.53, 0.99). Anything ≥0.99 is treated as memorization and blocks migration. Val-loss check is relative slope ≥ 1e-3 of tail mean (not absolute 1e-4) to avoid false negatives at low loss floors.
6. **Phase 2 prereq: GCP A3 Mega quota.** `a3-megagpu-8g` quota must be pre-approved in the GCP project before `phase2_a3mega.sh` can run. Request via Quotas console; can take days.

## Conventions

- **Private repo** at `github.com/nahomar/silicon-alpha`. Clone with PAT embedded in URL, then immediately scrub from `.git/config` (notebooks already do this).
- **PAT** stored as Colab Secret named `GITHUB_TOKEN`. `google.colab.userdata.get('GITHUB_TOKEN')` reads it. `getpass` fallback for timeout cases.
- **No secrets in commits.** `.env` is gitignored. API keys, PATs, and pod IDs never land in source. `.env.example` shows the shape.
- **Commit messages**: subject line ≤72 chars, blank line, multi-line body explaining the why (not just the what). See c2955fb, d7a7e64, 33fbe83 for the style.
- **Python env**: `python3 -m venv .venv`, `source .venv/bin/activate`, `pip install -r requirements.txt`. User-level default is `python3` (not `python`).
- **Platform**: Apple Silicon Mac (Darwin arm64) for local Phase 0. CUDA is Colab / GCE / RunPod only.

## Resume checklist (when picking the project back up)

1. **`git fetch origin && git log --oneline -10 && git status`** — reconcile
   local vs remote HEAD and confirm the working tree is clean.
2. **Check the latest Colab/Modal run.** Open
   `notebooks/colab_phase1_tradefm.ipynb`, jump to the migration-gate cell, and
   read the GO/NO-GO verdict plus the `reasons` list from
   `reports/migration_decision.md`.
3. **Check any live cloud resources** (RunPod / GCE pods) and tear down or
   confirm-intentional anything still running; rotate API keys if a pod was
   left exposed.

Only after those three is it worth proposing next moves (Phase 2 launch, synth
iteration, or signal-diagnostic run).

## Conversation digest (decisions + rationale)

Dense reference of the "why" behind choices that don't survive in commit messages. Do not restate content above.

### Path chosen vs alternatives

| Decision | Chosen | Rejected | Rationale |
|---|---|---|---|
| Engine tier | Budget Alpha + HRT-scale in parallel | HRT-only | Budget path (`odte_budget.py`, `BUDGET.md`) gives a working signal on Mac while HRT path trains; de-risks total dependency on a $50k cluster run. |
| Phase-1 validator | Colab Pro+ A100 | RunPod A100, GCE A100 | Colab = flat $49.99/mo, zero provisioning, already authenticated; RunPod had a stuck pod and API-key leak risk; GCE overkill for a 40M validation. |
| Phase-2 cluster | GCP A3 Mega (24× H100) | RunPod H100 multi-node, Lambda | HRT-precedent on GCP + TCPX NCCL; H100 availability more predictable; `a3-megagpu-8g` quota path is well-trodden. |
| Colab GPU | A100 80GB (40GB fallback wired) | H100 | H100 unavailable on Pro+ tier at launch; 40M params fits comfortably on A100 with bf16 + grad-ckpt. |
| Repo visibility | Private + Colab Secret | Public repo | PAT-in-URL clone scrubbed post-clone; keeps broker/margin configs and kernel IP out of public search. |
| Attention impl | `torch.nn.functional.scaled_dot_product_attention` | flash-attn-3 | SDPA ships with torch, no build step, no CUDA version drift on Colab; flash-attn-3 kept behind `cfg.use_flash_attn=true` for H100 Phase 2. |
| 40M fit on A100 40GB | grad-ckpt + bf16 + batch=1 + grad_accum=16 | fp32, larger batch | Stacked to survive the 40GB fallback path; batch=2 only works on 80GB. |

### Bug fixes with root causes

| Symptom | Root cause | Fix commit |
|---|---|---|
| Dataloader stuck on Colab | Shards read directly from Google Drive (high-latency FUSE) | 33fbe83 — shard_rows=8000 + local `/content` staging |
| Dir-acc hitting 100% | Single-shard synth → eval shard == train shard → memorization | c2955fb — 12 sessions × 3 regimes × 21 strikes × {C,P} |
| Migration gate false NO-GO on flat val loss | Absolute 1e-4 slope threshold too strict at noise floor | c2955fb — relative slope ≥ 1e-3 of tail mean in `_strictly_decreasing` |
| Overfit masquerading as success | Gate had no upper bound on dir-acc | c2955fb — overfit sentinel blocks migration when dir-acc ≥ 0.99 |
| OOM on 40GB runtime, fine on 80GB | Single config for both VRAM sizes | 33fbe83 + d7a7e64 — runtime auto-detect branch in cell 5 |
| Colab Secret read timing out | "Notebook access toggle OFF" in Secrets UI | 5659db7 — `getpass` fallback after userdata.get() raises |
| 40M OOM'd in fp32 | No activation checkpointing, fp32 default | 4d8f750 — `TradeFMConfig.grad_checkpointing` flag + `torch.utils.checkpoint` in `TradeFM.forward` |
| Private-repo OAuth loop in Colab | Interactive auth unreliable in headless Run-all | 28b58b5 — `colab_bootstrap.ipynb` escape hatch |

### Instrumentation decisions

- **`[diag]` prints in cell 5** (d7a7e64): print `git rev-parse HEAD`, `nvidia-smi` tier, and resolved config before the `pretrain_tradefm` call fires. A 45-60 min train on the wrong commit wastes a Colab session; diag line makes the cost visible at t=0.
- **`log_every=50`, `ckpt_every=500`**: log cadence fast enough to catch divergence in ~30s; checkpoint cadence slow enough that Drive I/O doesn't stall the loop (Drive write ≈ 1-2s per ckpt at 40M fp32).
- **Synth shape `12 × 3 × 21 × 2 = 1512 series`**: deliberately chosen above the 40M model's effective memorization capacity at ctx=1024; prevents the 100% dir-acc artifact that pre-c2955fb produced.
- **Cell 7 writes `reports/migration_decision.md`**: decision is a file, not just stdout, so the GCP Phase-2 launcher can grep it.

### Pending decisions (user-blocking)

| # | Decision | Blocker | Action |
|---|---|---|---|
| 1 | Terminate RunPod pod `i7wndt4y3bjpsq` | Billing at $2.99/hr; status unverified | Visit https://www.runpod.io/console/pods or `runpodctl stop/remove pod i7wndt4y3bjpsq`. Bounded exposure ≤ $6 (≤ 2 hrs). |
| 2 | Rotate RunPod API key | Key pasted in chat; RunPod auto-scans may have revoked | Settings → regenerate at https://www.runpod.io/console/user/settings |
| 3 | If cell 7 → GO | Waiting on training run | Launch `infra/gcp/phase2_a3mega.sh` with `GCP_PROJECT`, `GCP_BUCKET`, `REPO_URL`; confirm `a3-megagpu-8g` quota first. |
| 4 | If cell 7 → NO-GO | Same | Iterate: (a) Hawkes-informed flow in `odte/synth_options.py`; (b) config bump (ctx, layers) in `configs/tradefm_40m.yml`; (c) add real microstructure replay. |

### Cost accounting

| Bucket | Spend / rate | Notes |
|---|---|---|
| RunPod pod `i7wndt4y3bjpsq` | ≤ $6 (≤ 2 hrs × $2.99/hr) | Upper-bounded by assumption; confirm via console |
| Colab Pro+ | $49.99/mo flat | Amortized across Phase 0 + Phase 1 + all iterations |
| Phase 2 (projected) | $40-50k / run | 3× A3 Mega × 24× H100 × ~130 GPU-hours for 524M |
| Phase 0/1 local | $0 | Apple Silicon Mac |

### "Where to look" quick-reference

| Concept | Primary file(s) |
|---|---|
| Tokenizer | `odte/data/datashop_pack.py`, `odte/tokenizer.py` |
| Transformer | `odte/transformer_tradefm.py` |
| DML pricer | `odte/dml_pricer.py`, `odte/train/train_dml.py` |
| Migration gate | `odte/train/migration_check.py` |
| Phase 5 exec | `odte/exec/*.py` |
| Colab notebooks | `notebooks/colab_phase0_dml.ipynb`, `notebooks/colab_phase1_tradefm.ipynb` |
| GCP provisioning | `infra/gcp/*.sh`, `infra/gcp/README.md` |
| Monday deploy | `deploy/DEPLOYMENT_CHECKLIST.md`, `deploy/monday_go_live.sh` |
| Budget path | `BUDGET.md`, `odte_budget.py` |
| Synth data | `odte/synth_options.py` |
| 40M config | `configs/tradefm_40m.yml` |
| 524M config | `configs/tradefm_524m.yml` |

### If training succeeds, do exactly this next

1. Read `reports/migration_decision.md`; confirm verdict `GO` and dir-acc in [0.53, 0.99).
2. Verify GCP `a3-megagpu-8g` quota is approved in target project (Quotas console).
3. Export `GCP_PROJECT`, `GCP_BUCKET`, `REPO_URL` (`github.com/nahomar/silicon-alpha`).
4. Run `./infra/gcp/phase2_a3mega.sh` from ``.
5. Then `./infra/gcp/launch_torchrun_524m.sh` from inside the provisioned head node.

---

## Session log — 2026-04-19 (Phase-2 data-vendor + safety patches)

### Code shipped (three new commits on top of 84a7b23)

- **e3c12dd** — Phase-2 pre-launch safety: `checkpoint.py` hard-fails on world-size mismatch (no silent step-0 restart at Gate 4) and on cfg_hash mismatch on resume. `ShardedTokenDataset` now rank-partitions shards (shared-seed shuffle → rank slice → per-rank within-shard shuffle). `micro_dev` dropped from default `feature_spec` (placeholder until depth feed). `streaming_quantiles` reservoir default bumped 1M → 2M. New `tests/odte/test_quantile_parity.py` passes at 3e-3 max-abs-err gate.
- **640bf19** — `odte/data/databento_pack.py`: new adapter for Databento OPRA.PILLAR feed, pay-per-GB model. Reuses `DataShopPacker` via column translation. Built-in `max_spend_usd=50` cost guard before any bytes transfer.
- **ae89459** — `databento_pack` streaming fix: `iter_dbn_chunks` uses `DBNStore.to_df(count=N)` iterator so memory stays at ~1 GB regardless of file size. Prevents OOM on multi-GB 1-day pulls. Schema constant corrected to `cmbp-1` (OPRA is SIP-consolidated; `mbp-1` is non-consolidated exchange feeds only).

### Vendor decision chain (and the reasoning behind each flip)

1. **DataShop (initial)** — rejected after cost reading: $5-10k/mo institutional bulk license is overkill for one-shot pretrain.
2. **Databento (round 1)** — pay-per-GB better matches one-shot use, but schema matching ambiguity.
3. **Polygon Options Advanced** — $199/mo flat looked cheapest for a multi-month pull. Adapter written (`odte/data/polygon_pack.py`, committed locally but NOT pushed).
4. **Databento (final)** — tipped back by two signals: (a) research surfaced reports of Polygon tick-data inaccuracy (`dev.to` user: "wildly inaccurate ticks that threw off a backtest") which matters MORE for pretraining than backtesting because bad ticks silently poison tokenizer edges; (b) user framed strategy as "we beat HRT algorithmically, not on compute" — in which case data quality dominates cost.

### Databento cost findings (1-day SPX+SPXW, OPRA.PILLAR)

| Schema | $/day | Size | 3yr×2sym est. | What we get |
|---|---|---|---|---|
| `cmbp-1` | $17.78 | 119 GB | $13,439 | Full tick-level NBBO updates + trades ✓ chosen |
| `tcbbo` | $9.85 | 0.05 GB | $7,449 | Trades-only with BBO snapshot (no inter-trade quote dynamics) |
| `trades` | $7.88 | 0.03 GB | $5,959 | Trades only, no BBO |
| `cbbo-1m` | $1.27 | 0.68 GB | $960 | 1-min BBO snapshots (microstructure dead) |

**Key reframe:** 1 day of cmbp-1 already over-feeds 524M by 5-10× (Chinchilla-optimal = 10-20B tokens; 1 day cmbp-1 = ~100B tokens tokenized). **We do NOT need 3 years continuous.** Revised scope: 15 training days + 3 eval days, stratified across 2022-2025 by regime (low-vol / FOMC / OPEX / 0DTE-Monday / tariff-shock / holiday-shortened). Total budget: **~$300** for Phase-2 data, replacing the originally planned $5-10k/mo DataShop.

### Probe results ($0.23, 5-min SPX opening cross 2024-01-03)

- 8.97M rows, file=137 MB compressed
- All adapter column mappings match real cmbp-1 schema (validated live)
- Quality checks: `bid <= ask` ✓, `spread > 0` ✓, `mid ∈ [bid, ask]` ✓
- Streaming memory: bounded ~1 GB regardless of file size
- Adapter's `max_spend_usd` guard validated on the wrong-schema case ($0 spent on `mbp-1` failure)

### Currently running

**$7.59 smoke:** 1-day SPX cmbp-1 download + pack + tokenizer-fit. Background task `b3xs0rwtr`. Expected 30-120 min wall. File: `data/databento_raw/SPX_*.dbn.zst`. When complete, run `python -m odte.data.inspect_smoke --shards reports/odte_shards_real --raw-dbn <path>` for pro-desk quality checklist.

### Strategic framing (to apply at Gate 5 decision point)

User framed the project as: **"we don't have HRT's compute — we beat them by being algorithmically better."** Implications for future decisions:

- **Data quality > quantity** (drove Databento over Polygon)
- **Better features / objectives > bigger models** — at Gate 5, explicitly re-ask whether 524M beat a 40M model with engineered features. If not, redirect Phase 3 (CUDA kernels) and Phase 4 (adversarial RL) budget toward feature engineering and paper-trading feedback loops instead.
- **Uncontested niches > head-to-head.** 0DTE retail decision-making, SPX pin-risk last 30-min dynamics — places HRT hedges but doesn't necessarily model.

### Open admin items (still blockers before Gate 2 $500 smoke)

- [ ] GCP `a3-megagpu-8g` quota request (1-3 day lead — submit today)
- [ ] S3 ckpt bucket + scoped IAM user (Terraform draft pending)
- [ ] W&B workspace + API key
- [ ] OPRA redistribution attestation (usually part of Databento flow — verify at `databento.com/portal`)

### RunPod status

Pod `i7wndt4y3bjpsq` terminated (user confirmed). Old API key rotated.
