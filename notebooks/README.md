# Colab Pro+ playbook

You are here because you bought Colab Pro+ ($49.99/mo) to start training
the 0DTE alpha engine. This is the **right call** for Phase 0 and Phase 1.
It is **not** the right platform for Phase 2 (524M / trillion-token / live).

## Notebooks (run in this order)

| # | Notebook | What it trains | Hardware | Time | Gate |
|---|---|---|---|---|---|
| 0 | [`colab_phase0_dml.ipynb`](colab_phase0_dml.ipynb) | DML pricer Greeks | T4+ | ~10 min | ATM τ=1d Greek err ≤ 2 % |
| 1 | [`colab_phase1_tradefm.ipynb`](colab_phase1_tradefm.ipynb) | 40M Mini-TradeFM | A100 40GB+ | 12-48 h | val-loss strictly decreasing + dir-acc ≥ 53 % |

Only after Notebook 1 emits `✅ GO` should you spin up the $40-50k
H100 cluster (`infra/gcp/phase2_a3mega.sh`).

## Colab Pro+ runtime cheat-sheet

| Task | Hardware setting | Notes |
|---|---|---|
| Notebook 0 (DML) | T4 or L4 OK | tiny model, 4M params |
| Notebook 1 Mini-TradeFM | **A100 40GB or 80GB** | preferred; H100 is faster but scarce |
| H100 Colab ($1.86/hr in 2026) | select if offered | best $/FLOP at this tier |

Colab Pro+ **background execution**: close the browser, the runtime
persists up to 24 h. Checkpoints must land in **Drive** (not `/content`,
which wipes on disconnect).

## File layout this playbook creates

```
Google Drive/MyDrive/tradefm_ckpts/
  dml_pricer.pt                  # Notebook 0 output
  phase1_shards/opra_*.parquet   # packed synth training data
  tradefm_40m/                   # Notebook 1 output
    best_*.pt   final_*.pt   ckpt_*.pt
  migration_decision.md          # ✅ GO / ❌ NO-GO report
  keepalive.log                  # optional BG-exec heartbeat
```

## Workflow

### 0. One-time prep

Repo is already at **`https://github.com/nahomar/market-pattern-bot`** (private).
Colab needs a PAT to clone it. Set it up once:

1. GitHub → *Settings → Developer settings → Personal access tokens → Tokens (classic)
   → Generate new token (classic)*. Scope = `repo`. Expiry 30d is fine.
2. In Colab → left-side **🔑 key icon** → *+ Add new secret*:
   - Name: **`GITHUB_TOKEN`**
   - Value: the PAT
   - Toggle *Notebook access* ON
3. In Colab: *Runtime → Change runtime type → A100*.

The notebooks' cell 3 reads the secret via `google.colab.userdata`, clones with
the token embedded in the URL, then immediately scrubs the token from
`.git/config` so it never lands in Drive. If you ever leak the token, revoke
it in GitHub settings and generate a new one.

### 1. Run Notebook 0
- Pass gate (≤ 2 % Greek error) before proceeding.
- If it fails, inspect the delta error heat-map cell and tune
  `DMLConfig.pretrain_steps` or `MaturityGate.sigma_floor`.

### 2. Run Notebook 1
- Start with the default 5 000 steps; loss should drop from ~7 to ~3.
- If the migration cell says `❌ NO-GO`, re-run with more steps or
  richer synth (`N_DAYS = 20` in cell 4).
- Once `✅ GO` appears, copy the winning `best_*.pt` out of Drive to
  your laptop and push to the GCS bucket the H100 cluster will use.

### 3. Migrate
Stop here. On your laptop:

```bash
gcloud storage cp best_*.pt gs://my-tradefm/seed_ckpts/
./infra/gcp/phase2_a3mega.sh
./infra/gcp/launch_torchrun_524m.sh
```

The distributed trainer's `--resume` flag will pick up the Colab-trained
weights as a warm-start, saving ~20 % of the H100 token budget.

## Known Colab gotchas

1. **Drive I/O throttles** with > 10 000 shards — pre-pack fewer, larger
   shards (our `DataShopPacker` defaults to 1M rows/shard; keep it).
2. **Session dies during training** — always resume via the checkpoint
   in Drive. `odte/train/pretrain_tradefm.py` loads the latest ckpt
   automatically if the `ckpt_dir` is non-empty.
3. **A100 40GB OOM** with `ctx_len=2048, batch=64` — drop to
   `batch=32 grad_accum=2` or `ctx_len=1024`.
4. **"Pro+ quota exceeded"** — happens if you keep the runtime pinned.
   Disconnect when a run completes; quota resets hourly.
5. **Drive permissions** — Colab sometimes fails to write `best_*.pt`
   due to quota. Watch the log; if it silently stops, switch `ckpt_dir`
   to `/content/ckpts_tmp` and rsync to Drive at the end.

## Scaling wall — the ONLY reason to leave Colab

| Limit | Colab ceiling | H100 cluster |
|---|---|---|
| # of GPUs | 1 | 24+ |
| Fabric | shared network | NCCL on TCPX or IB |
| Dataset size | few GB in Drive | many TB in GCS |
| Live trading | paper only | RDMA ingest possible |
| Model size | ≤ 60M params | 524M+ |
| Wall clock per run | 24 h | days-to-weeks |

When the migration gate passes, do not try to "scale up in Colab". It
cannot get you to 524M. That's what the A3 Mega playbook is for.
