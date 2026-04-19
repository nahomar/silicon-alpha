# STATE.md — 0DTE Alpha Engine Project Handoff

Last synced: 2026-04-19. Read top-to-bottom; a fresh Claude Code session should be able to resume without prior context.

## Project Summary

This repo is a 0DTE (zero-days-to-expiry) SPX options alpha engine in active build, spanning a full stack: Differential ML option pricer, a decoder-only transformer (TradeFM) over microstructure tokens, a QP-based deterministic executor with broker-aware margin, paper broker, and post-trade analyzer. Target user is a single operator running the Monday weekly cycle on SPX 0DTE. Current state: Phase 0 (DML pricer) is done on Mac; Phase 1 (40M Mini-TradeFM validation) is actively training on a Colab Pro+ A100 80GB as of minutes ago, ETA ~45-60 min from start. Infrastructure recipes for Phases 2-5 are committed but not yet run. The repo also retains the earlier market-pattern-bot (sentiment scraper) code at the top level (`main.py`, `scrapers/`, `ml/sentiment.py`) — that predates the 0DTE pivot and is orthogonal to it.

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

- **Private repo** at `github.com/nahomar/market-pattern-bot`. Clone with PAT embedded in URL, then immediately scrub from `.git/config` (notebooks already do this).
- **PAT** stored as Colab Secret named `GITHUB_TOKEN`. `google.colab.userdata.get('GITHUB_TOKEN')` reads it. `getpass` fallback for timeout cases.
- **No secrets in commits.** `.env` is gitignored. API keys, PATs, and pod IDs never land in source. `.env.example` shows the shape.
- **Commit messages**: subject line ≤72 chars, blank line, multi-line body explaining the why (not just the what). See c2955fb, d7a7e64, 33fbe83 for the style.
- **Python env**: `python3 -m venv .venv`, `source .venv/bin/activate`, `pip install -r requirements.txt`. User-level default is `python3` (not `python`).
- **Platform**: Apple Silicon Mac (Darwin arm64) for local Phase 0. CUDA is Colab / GCE / RunPod only.

## If I'm a Fresh Claude Code Session Reading This, My First Action Is:

1. **`cd /Users/nahom/market-pattern-bot && git fetch origin && git log --oneline -10 && git status`** — reconcile local `c2955fb` vs reported remote `7f22aa7`, see if there's an untracked `infra/cloud/runpod_a100_phase1.sh` still pending, and confirm working tree is clean.
2. **Check the Colab run.** Ask the user to open `notebooks/colab_phase1_tradefm.ipynb`, jump to Cell 7, and share the GO/NO-GO verdict plus the `reasons` list from `reports/migration_decision.md`. If the run hasn't finished yet, wait or poll.
3. **Check the RunPod pod.** Ask the user to visit https://www.runpod.io/console/pods and confirm pod `i7wndt4y3bjpsq` is either terminated or intentionally still running. If terminated, remind them to rotate the RunPod API key.

Only after those three should a fresh session propose next moves (Phase 2 launch, synth iteration, or pod teardown).
