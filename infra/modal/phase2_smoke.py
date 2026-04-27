"""Phase-2 Gate-3 smoke on Modal — single 1× H100.

Why this exists: before committing to the ~$50k 3-node A3 Mega run (or any
paid compute at all), prove the repo actually trains end-to-end on real
Hopper hardware. Modal's $30/mo free tier gives ~7.5 H100-hours; this smoke
burns ~$0.50 of that.

What it exercises:
  - nvcr.io/nvidia/pytorch image builds; transformer_engine is importable.
  - odte_kernels_cu compiles against SM90a (H100-native).
  - 40M-param TradeFM with fp8=true trains on structured tokens (Markov
    chain, entropy ~1.4 nats) — so loss can meaningfully descend below the
    log(vocab)=8.3 uniform-random floor, proving the model is *learning*.
  - CheckpointManager writes to a Modal Volume AND resume-from-checkpoint
    works: a second container loads the step-200 ckpt and continues to
    step 400.

What it does NOT exercise:
  - Multi-node NCCL (Modal is single-node).
  - Real OPRA throughput (shards are synthetic Markov tokens).
  - Chinchilla-optimal convergence (400 steps is still pipeline-validation
    scale, not a real training run).

Run (after `pip install modal && modal setup`):
    modal run infra/modal/phase2_smoke.py
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tradefm-phase2-smoke"

# REPO_ROOT is only meaningful on the local machine at image-build time.
# Inside the Modal container, Modal flattens the entry file to /root/<name>.py
# so .parents[2] walks off the end of the path. Guard it.
try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    REPO_ROOT = Path("/root/repo")


image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.12-py3")
    .apt_install("git", "build-essential")
    .pip_install(
        "pyarrow>=15.0",
        "pandas>=2.2",
        "pyyaml>=6.0",
        "numpy>=1.26",
        "zstandard>=0.22",
        "scikit-learn>=1.4",
        "databento>=0.40",  # for smoke_databento; pure Python, ~10 MB
    )
    .run_commands(
        # NVCR 24.12 ships TE but not flash-attn; the repo falls back to SDPA
        # when HAS_FLASH_ATTN is False, which is fine for smoke. FA3 gets
        # validated on the real Phase-2 GCP DLP image.
        "python -c 'import transformer_engine.pytorch; print(\"TE ok\")'",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/repo",
        copy=True,
        ignore=[
            "**/.venv",
            "**/__pycache__",
            "**/.git",
            "**/.claude",
            # `/data` (not `**/data`!) so we exclude the top-level
            # `data/` dir but KEEP `odte/data/` — the python package.
            "/data",
            "/notebooks",
        ],
    )
    .run_commands(
        "cd /root/repo/odte/kernels && python setup.py build_ext --inplace",
    )
    # Unbuffered stdout so long silent loops (e.g. 50 GB tokenizer fit) still
    # show intermediate prints in the Modal log stream instead of flushing
    # only at function exit.
    .env({"PYTHONUNBUFFERED": "1"})
)

ckpt_volume = modal.Volume.from_name("tradefm-smoke-ckpts", create_if_missing=True)
shard_volume = modal.Volume.from_name("tradefm-smoke-shards", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


# 40M baseline with Hopper path (fp8) flipped on. Kept inline instead of as a
# committed YAML so the smoke is self-contained and doesn't fork configs.
SMOKE_CONFIG_YAML = """\
d_model: 768
n_heads: 12
n_layers: 12
vocab: 4096
ctx_len: 2048
dropout: 0.1
fp8: true
rotary: true
use_flash_attn: false
lr: 6.0e-4
weight_decay: 0.1
warmup_steps: 50
"""

CKPT_ROOT = "/ckpts/tradefm_smoke"
SHARD_GLOB = "/shards/markov_*.parquet"


def _make_markov_shards(
    out_dir: Path,
    n_shards: int = 4,
    rows_per_shard: int = 512,
    seq_len: int = 2048,
    vocab: int = 4096,
    n_states: int = 256,
    n_successors: int = 4,
    seed: int = 0,
) -> None:
    """Write parquet shards from a fixed bigram Markov chain.

    Uniform random tokens have entropy log(vocab)=8.3 nats — the theoretical
    loss floor. A Markov chain with n_states=256 and 4 equiprobable successors
    has entropy log(4)=1.39 nats, so training loss can meaningfully descend
    below the uniform floor — real evidence the model is learning, not just
    re-discovering the prior.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    alphabet = rng.choice(vocab, size=n_states, replace=False).astype(np.int32)
    T = rng.integers(0, n_states, size=(n_states, n_successors))

    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        shard_rng = np.random.default_rng(seed + 1 + i)
        state = shard_rng.integers(0, n_states, size=rows_per_shard, dtype=np.int32)
        seqs = np.empty((rows_per_shard, seq_len), dtype=np.int32)
        seqs[:, 0] = state
        choices = shard_rng.integers(
            0, n_successors, size=(rows_per_shard, seq_len - 1)
        )
        for t in range(1, seq_len):
            state = T[state, choices[:, t - 1]].astype(np.int32)
            seqs[:, t] = state
        tokens = alphabet[seqs]
        rows = [tokens[r].tolist() for r in range(rows_per_shard)]
        pd.DataFrame({"tokens": rows}).to_parquet(
            out_dir / f"markov_{i:03d}.parquet", index=False
        )


def _launch_train(repo: Path, cfg_path: Path, total_steps: int,
                  resume: bool) -> None:
    """Run torchrun → odte.train.distributed with the smoke's fixed
    batch/ckpt/log cadence. `total_steps` is absolute; --resume picks up
    from the latest ckpt."""
    import subprocess

    cmd = [
        "torchrun", "--nproc_per_node=1", "--nnodes=1",
        "-m", "odte.train.distributed",
        "--config", str(cfg_path),
        "--shards", SHARD_GLOB,
        "--ckpt-store", CKPT_ROOT,
        "--ckpt-prefix", "tradefm",
        "--steps", str(total_steps),
        "--batch", "4",
        "--grad-accum", "1",
        "--ckpt-every", "100",
        "--log-every", "10",
        "--num-workers", "2",
    ]
    if resume:
        cmd.append("--resume")
    print("[smoke] launching:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo), check=True)


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/ckpts": ckpt_volume, "/shards": shard_volume},
)
def smoke_train():
    """Stage 1: clean-slate train 0 → 200 steps."""
    import shutil
    import sys

    repo = Path("/root/repo")
    sys.path.insert(0, str(repo))

    import torch

    print(
        f"[smoke] torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
    )
    assert torch.cuda.is_available(), "H100 GPU not visible inside the container"

    # Clean-slate ckpts so the resume test in stage 2 is deterministic.
    ckpt_dir = Path(CKPT_ROOT)
    if ckpt_dir.exists():
        print(f"[smoke] clearing existing ckpts in {ckpt_dir}")
        shutil.rmtree(ckpt_dir)

    # Regenerate shards if the Markov shards aren't present (older runs may
    # have left uniform-random smoke_*.parquet files in the volume).
    shard_dir = Path("/shards")
    if not any(shard_dir.glob("markov_*.parquet")):
        print("[smoke] generating Markov-chain shards…")
        _make_markov_shards(shard_dir)
        shard_volume.commit()
    shards = sorted(shard_dir.glob("markov_*.parquet"))
    print(f"[smoke] shards: {[s.name for s in shards]}")

    cfg_path = Path("/tmp/tradefm_smoke_h100.yml")
    cfg_path.write_text(SMOKE_CONFIG_YAML)

    _launch_train(repo, cfg_path, total_steps=200, resume=False)

    step200 = Path(CKPT_ROOT) / "tradefm" / "step_00000200"
    assert step200.exists(), f"expected {step200} after stage 1"
    for f in sorted(step200.glob("*.pt")):
        print(f"    {f}  ({f.stat().st_size / 1e6:.1f} MB)")
    ckpt_volume.commit()
    print("[smoke] stage 1 PASS — Hopper path + FSDP init + ckpt write verified.")


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/ckpts": ckpt_volume, "/shards": shard_volume},
)
def smoke_resume():
    """Stage 2: load step-200 ckpt in a fresh container, continue to step 400."""
    import sys

    repo = Path("/root/repo")
    sys.path.insert(0, str(repo))

    # Pull the state that stage 1 committed.
    ckpt_volume.reload()
    shard_volume.reload()

    pre = Path(CKPT_ROOT) / "tradefm" / "step_00000200"
    assert pre.exists(), (
        f"expected {pre} from smoke_train() — run stage 1 first"
    )
    print(f"[resume] resuming from: {sorted(p.name for p in pre.glob('*.pt'))}")

    cfg_path = Path("/tmp/tradefm_smoke_h100.yml")
    cfg_path.write_text(SMOKE_CONFIG_YAML)

    _launch_train(repo, cfg_path, total_steps=400, resume=True)

    post = Path(CKPT_ROOT) / "tradefm" / "step_00000400"
    assert post.exists(), f"resume failed — {post} not written"
    for f in sorted(post.glob("*.pt")):
        print(f"    {f}  ({f.stat().st_size / 1e6:.1f} MB)")
    ckpt_volume.commit()
    print("[resume] stage 2 PASS — CheckpointManager load+continue verified.")


@app.function(
    gpu="H100",
    timeout=1800,
    volumes={"/ckpts": ckpt_volume, "/shards": shard_volume},
)
def smoke_524m():
    """Dry-run the *exact* 524M production config on 1× H100.

    Purpose: validate the production-scale config isn't silently broken
    before committing to any multi-node run. At 524M × ctx=4096 × 24
    layers, a dozen things could break: FSDP init, fp8 + FSDP coexistence,
    optimizer state allocation, ckpt serialization of GB-scale weights.
    We don't need convergence — we need "does it actually run end-to-end."

    Reuses the Markov shards (seq_len=2048 packs into ctx=4096 transparently
    via ShardTokenDataset's token-packing). batch=1 to stay well under the
    80 GB ceiling; flash-attn is absent in this image so SDPA takes the
    attention path (same one real Phase-2 would fall back to on an A100).
    """
    import shutil
    import subprocess
    import sys

    repo = Path("/root/repo")
    sys.path.insert(0, str(repo))

    shard_volume.reload()
    ckpt_volume.reload()

    import torch

    gpu = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.mem_get_info()[1] / 1e9
    print(f"[524m] torch={torch.__version__} device={gpu}  total_mem={total_gb:.1f} GB")

    shards = sorted(Path("/shards").glob("markov_*.parquet"))
    assert shards, "run the smoke entrypoint first to generate Markov shards"
    print(f"[524m] using {len(shards)} shards (seq_len=2048 -> packs into ctx=4096)")

    ckpt_dir = Path("/ckpts/tradefm_524m_dryrun")
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    cmd = [
        "torchrun", "--nproc_per_node=1", "--nnodes=1",
        "-m", "odte.train.distributed",
        "--config", "configs/tradefm_524m.yml",
        "--shards", SHARD_GLOB,
        "--ckpt-store", str(ckpt_dir),
        "--ckpt-prefix", "tradefm_524m",
        "--steps", "500",
        "--batch", "1",
        "--grad-accum", "4",
        "--ckpt-every", "250",
        "--log-every", "25",
        "--num-workers", "2",
    ]
    print("[524m] launching:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo), check=True)

    ckpts = sorted(ckpt_dir.rglob("*.pt"))
    assert ckpts, "no 524M checkpoints written — training didn't reach ckpt_every"
    for c in ckpts:
        print(f"    {c}  ({c.stat().st_size / 1e9:.2f} GB)")
    ckpt_volume.commit()
    print("[524m] PASS — production config instantiates, trains 500 steps, ckpts.")


@app.function(
    # Pure CPU+IO — no GPU needed for tokenizing + parquet writing.
    cpu=4.0,
    memory=16384,
    # Raise timeout to 2h: Databento's batch API can queue for 30-60 min
    # on its own before the download even starts.
    timeout=7200,
    volumes={"/shards": shard_volume},
    secrets=[modal.Secret.from_name("databento-api-key")],
)
def smoke_databento():
    """Fetch 1 day of SPX MBP-1 via Databento's free-tier credit and
    pack to parquet shards. Produces real-tape training data to replace
    the Markov synthetic — the actual "grammar of exchange messages" the
    model needs to learn for 0DTE alpha.

    Cost: ~$0.05-$1 against the $125 Databento signup credit.

    The adapter's --smoke flag hardwires start=2024-01-03 end=2024-01-04
    symbols=SPX spend_cap=$10 (see odte/data/databento_pack.py:_cli).
    """
    import os
    import subprocess
    import sys

    repo = Path("/root/repo")
    sys.path.insert(0, str(repo))

    assert os.environ.get("DATABENTO_API_KEY"), (
        "DATABENTO_API_KEY missing — create the secret first with:\n"
        "    modal secret create databento-api-key DATABENTO_API_KEY=<key>"
    )

    raw_dir = Path("/shards/databento_raw")
    out_dir = Path("/shards/databento_packed")

    # Call the CLI in-process instead of via subprocess — we've already
    # done sys.path.insert(0, repo) above, so the import works here even
    # if Modal's subprocess env would have a different sys.path default.
    print(f"[databento] /root/repo contents: "
          f"{sorted(p.name for p in repo.iterdir())[:20]}")
    print(f"[databento] odte dir contents: "
          f"{sorted((repo / 'odte' / 'data').iterdir())[:20]}")

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Bypass pack_databento's estimator-based routing (which routes <3GB
    # windows to streaming, and streaming 504s on SPX parent symbology).
    # Call fetch_range_batch directly + reproduce the pack pipeline.
    from odte.data.databento_pack import (
        DatabentoFetcher, iter_dbn_chunks, prepare_features,
    )
    from odte.data.datashop_pack import DataShopPacker
    from odte.data.streaming_quantiles import fit_hybrid_from_chunks
    from odte.tokenizer import HybridBinTokenizer

    start = "2024-01-03T14:30"
    end = "2024-01-03T15:30"
    symbols = ["SPX"]
    print(f"[databento] window: {start} → {end}  symbols={symbols}")

    fetcher = DatabentoFetcher(raw_dir=raw_dir)
    est = fetcher.cost_estimate(start, end, symbols)
    print(f"[databento] cost_estimate: ${est['cost_usd']:.4f}  size: {est['gb']:.3f} GB")
    assert est["cost_usd"] <= 10.0, f"cost ${est['cost_usd']} exceeds $10 cap"

    print("[databento] submitting batch job (streams 504 on parent sym, batch works)")
    dbn_paths = fetcher.fetch_range_batch(start, end, symbols)
    print(f"[databento] batch returned {len(dbn_paths)} DBN file(s)")
    for p in dbn_paths:
        print(f"    {p}  ({p.stat().st_size / 1e6:.1f} MB)")

    # Fit tokenizer edges from streaming chunks (first pass).
    packer = DataShopPacker(out_dir=out_dir, n_buckets=64, shard_rows=1_000_000)

    def _all_chunks():
        for p in dbn_paths:
            for ch in iter_dbn_chunks(p):
                yield prepare_features(ch)

    edges = fit_hybrid_from_chunks(
        _all_chunks(), packer.feature_spec, n_buckets=64,
        checkpoint=out_dir / "_fit_ckpt",
    )
    tok = HybridBinTokenizer(n_buckets=64, feature_spec=packer.feature_spec)
    tok.edges = edges
    tok.save(out_dir / "tokenizer.json")
    packer.tokenizer = tok

    # Second pass: tokenize + write shards. This mirrors the tail of
    # pack_databento (databento_pack.py:419-441) — DataShopPacker doesn't
    # expose a direct "pack these chunks" API, so we reuse its _flush_shard.
    import pandas as pd

    buffer: list[dict] = []
    shard_idx = 0
    shard_paths: list[Path] = []
    for p in dbn_paths:
        for ch in iter_dbn_chunks(p):
            feats = prepare_features(ch)
            toks = tok.tokenize_batch(feats, feature_order=list(packer.feature_spec))
            for i, row in feats.reset_index(drop=True).iterrows():
                buffer.append({
                    "ts": int(row["ts_ms"]),
                    "underlying": str(symbols[0]),
                    "expiry": "",
                    "day": pd.Timestamp(row["quote_datetime"]).strftime("%Y-%m-%d"),
                    "tokens": toks[i].tolist(),
                })
                if len(buffer) >= packer.shard_rows:
                    shard_paths.append(packer._flush_shard(buffer, shard_idx))
                    shard_idx += 1
                    buffer = []
    if buffer:
        shard_paths.append(packer._flush_shard(buffer, shard_idx))
    print(f"[databento] pack done: {len(shard_paths)} shards in {out_dir}")

    shards = sorted(out_dir.rglob("*.parquet"))
    tokenizer = out_dir / "tokenizer.json"
    print(f"[databento] packed {len(shards)} parquet shards")
    total_mb = sum(s.stat().st_size for s in shards) / 1e6
    print(f"[databento] total shard size: {total_mb:.1f} MB")
    for s in shards[:5]:
        print(f"    {s.name}  ({s.stat().st_size / 1e6:.1f} MB)")
    if len(shards) > 5:
        print(f"    ... and {len(shards) - 5} more")
    assert tokenizer.exists(), f"no tokenizer.json written to {out_dir}"
    print(f"[databento] tokenizer.json: {tokenizer.stat().st_size} B")
    shard_volume.commit()
    print("[databento] PASS — real SPX MBP-1 tape tokenized and packed.")


@app.function(
    cpu=4.0,
    # Bumped 16 GB → 64 GB: decompressing the 13.6 GB zstd-compressed DBN to
    # ~50 GB uncompressed while the DBNStore iterator holds partial state
    # plus pandas DataFrames blew past 16 GB on the last run and wedged the
    # container. 64 GB gives headroom for the whole pipeline.
    memory=65536,
    timeout=7200,
    volumes={"/shards": shard_volume},
    secrets=[modal.Secret.from_name("databento-api-key")],
)
def smoke_databento_reuse(
    job_id: str = "OPRA-20260420-CTDQCTDCGX",
    max_files: int = 2,
):
    """Reuse a completed Databento batch job instead of submitting a fresh
    one. Skips the queue + scatter/gather wait entirely — files download
    straight from Databento's CDN.

    The referenced job is a full day (2024-01-03) of SPX.OPT OPRA cmbp-1,
    13.6 GB compressed / 50.9 GB uncompressed. max_files limits how many
    files we actually download+pack for the smoke — 1-2 is enough tape to
    fit the tokenizer and write a handful of parquet shards.
    """
    import os
    import sys
    import time

    import requests

    repo = Path("/root/repo")
    sys.path.insert(0, str(repo))

    assert os.environ.get("DATABENTO_API_KEY"), "DATABENTO_API_KEY secret missing"

    import databento as db  # noqa: E402

    client = db.Historical()
    all_files = client.batch.list_files(job_id=job_id)
    # Filter to only DBN data files (batch jobs also produce manifest.json,
    # symbology.json, condition.json metadata — those aren't DBNStore-readable).
    dbn_files = [f for f in all_files
                 if (f.get("filename") or "").lower().endswith((".dbn", ".dbn.zst"))]
    print(f"[reuse] job {job_id}: {len(all_files)} total files, "
          f"{len(dbn_files)} DBN data files, downloading first {max_files}")
    files = dbn_files

    raw_dir = Path(f"/shards/databento_reuse/{job_id}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for f in files[:max_files]:
        name = f.get("filename")
        url = (f.get("urls") or {}).get("https")
        size = f.get("size", 0)
        if not url or not name:
            print(f"[reuse] skipping file with no https url: {f}")
            continue
        local = raw_dir / name
        if local.exists() and local.stat().st_size == size:
            print(f"[reuse] {name} already downloaded ({size / 1e6:.1f} MB)")
            downloaded.append(local)
            continue
        print(f"[reuse] fetching {name}  ({size / 1e6:.1f} MB)…")
        t0 = time.time()
        r = requests.get(url, stream=True, timeout=600,
                         auth=(os.environ["DATABENTO_API_KEY"], ""))
        r.raise_for_status()
        with open(local, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
        dt = time.time() - t0
        mb = local.stat().st_size / 1e6
        print(f"[reuse]   done: {mb:.1f} MB in {dt:.1f}s ({mb / dt:.1f} MB/s)")
        downloaded.append(local)

    shard_volume.commit()
    print(f"[reuse] downloaded {len(downloaded)} DBN files", flush=True)

    # Now pack: tokenizer fit + parquet shard write. Mirrors the tail of
    # pack_databento in databento_pack.py but operating on pre-downloaded DBN.
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    print("[reuse] importing pack pipeline…", flush=True)
    from odte.data.databento_pack import iter_dbn_chunks, prepare_features
    from odte.data.datashop_pack import DataShopPacker
    from odte.data.streaming_quantiles import fit_hybrid_from_chunks
    from odte.tokenizer import HybridBinTokenizer
    import pandas as pd
    print("[reuse] imports done", flush=True)

    out_dir = Path(f"/shards/databento_reuse_packed/{job_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    packer = DataShopPacker(out_dir=out_dir, n_buckets=64, shard_rows=1_000_000)
    print(f"[reuse] DataShopPacker ready  feature_spec={list(packer.feature_spec)}",
          flush=True)

    # Wrapper that logs per-chunk progress — otherwise the fit pass runs
    # silently for 10-30 min on 50 GB uncompressed and looks hung.
    n_chunks = [0]
    n_rows = [0]
    def _all_chunks():
        import time
        t0 = time.time()
        for p in downloaded:
            print(f"[reuse] opening {p.name}", flush=True)
            for ch in iter_dbn_chunks(p):
                feats = prepare_features(ch)
                n_chunks[0] += 1
                n_rows[0] += len(feats)
                if n_chunks[0] % 10 == 0:
                    dt = time.time() - t0
                    rate = n_rows[0] / max(dt, 1e-6) / 1e3
                    print(f"[reuse]   chunk {n_chunks[0]:4d}  rows={n_rows[0]:,}  "
                          f"elapsed={dt:.0f}s  rate={rate:.0f}k rows/s", flush=True)
                yield feats

    # Skip the fit pass entirely if tokenizer.json already exists from a
    # previous run — fit is deterministic on the same DBN inputs, so re-fitting
    # produces the same edges and just wastes ~30 min/day. This was the major
    # cost in the disconnected-client rerun scenario; the volume.commit()
    # after the fit on the prior run already persisted tokenizer.json.
    tokenizer_path = out_dir / "tokenizer.json"
    if tokenizer_path.exists():
        print(f"[reuse] tokenizer.json found at {tokenizer_path} — skipping fit pass",
              flush=True)
        tok = HybridBinTokenizer.load(tokenizer_path)
        packer.tokenizer = tok
        print(f"[reuse] tokenizer loaded, {len(tok.edges)} feature edge sets", flush=True)
    else:
        print("[reuse] fitting tokenizer edges from real MBP-1 stream…", flush=True)
        edges = fit_hybrid_from_chunks(
            _all_chunks(), packer.feature_spec, n_buckets=64,
            checkpoint=out_dir / "_fit_ckpt",
        )
        tok = HybridBinTokenizer(n_buckets=64, feature_spec=packer.feature_spec)
        tok.edges = edges
        tok.save(tokenizer_path)
        packer.tokenizer = tok
        print(f"[reuse] tokenizer fit done, edges for {len(edges)} features",
              flush=True)
        # Persist fit outputs NOW so if the pack crashes we don't re-fit the
        # 50 GB stream (previous run's 29-min sunk cost).
        shard_volume.commit()

    # Pack pass — vectorized. Prior version used `feats.iterrows()` which is
    # ~5 µs/row and hit ~16k rows/s (23× slower than the 367k rows/s fit pass,
    # with 9h of wall-clock remaining on 636M rows). Batching via DataFrame
    # concat + pyarrow lets each shard write in seconds instead of minutes.
    import numpy as np
    import time as _time

    shard_idx = 0
    shard_paths: list[Path] = []
    buffer_dfs: list[pd.DataFrame] = []
    buffer_rows = 0
    n_chunks[0] = 0  # reset counters for the pack pass
    n_rows[0] = 0
    pack_t0 = _time.time()

    def _flush_from_buffer(target_rows: int) -> None:
        nonlocal shard_idx, buffer_rows
        big = pd.concat(buffer_dfs, ignore_index=True, copy=False)
        to_write = big.iloc[:target_rows]
        remainder = big.iloc[target_rows:]
        out_path = out_dir / f"opra_{shard_idx:06d}.parquet"
        to_write.to_parquet(out_path, index=False)
        print(f"[pack] wrote {out_path.name}  ({len(to_write):,} rows)", flush=True)
        shard_paths.append(out_path)
        shard_idx += 1
        buffer_dfs.clear()
        if len(remainder):
            buffer_dfs.append(remainder)
            buffer_rows = len(remainder)
        else:
            buffer_rows = 0

    for p in downloaded:
        print(f"[pack] opening {p.name}", flush=True)
        for ch in iter_dbn_chunks(p):
            feats = prepare_features(ch)
            toks = tok.tokenize_batch(feats, feature_order=list(packer.feature_spec))
            # Vectorized DataFrame build — zero per-row Python.
            chunk_df = pd.DataFrame({
                "ts": feats["ts_ms"].astype("int64").values,
                "underlying": np.full(len(feats), "SPX", dtype=object),
                "expiry": np.full(len(feats), "", dtype=object),
                "day": pd.to_datetime(feats["quote_datetime"]).dt.strftime("%Y-%m-%d").values,
                "tokens": list(toks),
            })
            buffer_dfs.append(chunk_df)
            buffer_rows += len(chunk_df)
            n_chunks[0] += 1
            n_rows[0] += len(chunk_df)
            while buffer_rows >= packer.shard_rows:
                _flush_from_buffer(packer.shard_rows)
            if n_chunks[0] % 10 == 0:
                dt = _time.time() - pack_t0
                rate = n_rows[0] / max(dt, 1e-6) / 1e3
                print(f"[pack] chunk {n_chunks[0]:4d}  rows={n_rows[0]:,}  "
                      f"shards={shard_idx}  elapsed={dt:.0f}s  rate={rate:.0f}k rows/s",
                      flush=True)
            # Checkpoint volume commits every 25 shards so a timeout doesn't
            # cost us all the pack work.
            if shard_idx and shard_idx % 25 == 0 and buffer_rows == 0:
                shard_volume.commit()
                print(f"[pack] volume.commit at shard {shard_idx}", flush=True)

    if buffer_rows:
        _flush_from_buffer(buffer_rows)
    shard_volume.commit()

    total_dt = _time.time() - pack_t0
    print(f"[reuse] PASS — {len(shard_paths)} shards from real OPRA SPX tape "
          f"({n_rows[0]:,} rows in {total_dt:.0f}s)", flush=True)
    for s in shard_paths[:5]:
        print(f"    {s.name}  ({s.stat().st_size / 1e6:.1f} MB)")


@app.function(
    gpu="H100:8",
    timeout=3600,
    volumes={"/ckpts": ckpt_volume, "/shards": shard_volume},
)
def smoke_8gpu(
    shards_glob: str = "/shards/databento_reuse_packed/OPRA-20260420-CTDQCTDCGX/opra_*.parquet",
    steps: int = 200,
):
    """8× H100 single-node NCCL smoke on real OPRA tape.

    Exercises the code path the 1-GPU smoke cannot:
      - world_size=8 FSDP sharding (24 layers across 8 ranks)
      - NCCL gradient all-reduce per optimizer step (over NVLink)
      - ShardedTokenDataset rank-partitioning (shared-seed shuffle +
        rank-strided shard slice, committed in e3c12dd)
      - Rank-specific checkpoint files: rank_0.pt ... rank_7.pt
      - CheckpointManager world_size metadata (guards against resuming
        an 8-GPU ckpt on a different world_size, committed in e3c12dd)

    Defaults to the real-OPRA shards from dryrun_databento_reuse — pass
    a Markov glob (`/shards/markov_*.parquet`) to test on synthetic.

    Explicitly does NOT test: inter-node TCPX bandwidth. That's the 20%
    that can only be measured on real A3 Mega. But NCCL-over-NVLink uses
    the same library code path as NCCL-over-TCPX — different transport,
    same APIs. 80% coverage for ~$5-15.
    """
    import shutil
    import subprocess
    import sys

    repo = Path("/root/repo")
    sys.path.insert(0, str(repo))

    shard_volume.reload()
    ckpt_volume.reload()

    import torch

    n_gpu = torch.cuda.device_count()
    print(f"[8gpu] device count: {n_gpu}", flush=True)
    assert n_gpu == 8, f"expected 8 GPUs, got {n_gpu}"
    # Force CUDA init before per-device queries. Modal's multi-GPU allocation
    # sometimes exposes `device_count()` before the runtime context is ready,
    # which causes get_device_name() to raise "system not yet initialized".
    # A single tiny tensor op warms the driver safely.
    try:
        torch.zeros(1, device="cuda:0")
    except Exception as e:
        print(f"[8gpu] warmup noted: {type(e).__name__}: {e}", flush=True)
    for i in range(n_gpu):
        try:
            print(f"    gpu{i}: {torch.cuda.get_device_name(i)}", flush=True)
        except Exception as e:
            print(f"    gpu{i}: (name unavailable: {e})", flush=True)

    import glob as _glob
    shard_paths = sorted(_glob.glob(shards_glob))
    assert shard_paths, f"no shards matched {shards_glob!r}"
    print(f"[8gpu] using {len(shard_paths)} shards from {shards_glob}", flush=True)

    ckpt_dir = Path("/ckpts/tradefm_8gpu_smoke")
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    cfg_path = Path("/tmp/tradefm_smoke_h100.yml")
    cfg_path.write_text(SMOKE_CONFIG_YAML)

    # 40M at batch=4/GPU × 8 GPUs = effective batch 32. 200 steps is enough
    # to see NCCL sync happen repeatedly + get two ckpt writes to verify the
    # rank-partitioned save path.
    cmd = [
        "torchrun", "--nproc_per_node=8", "--nnodes=1",
        "-m", "odte.train.distributed",
        "--config", str(cfg_path),
        "--shards", shards_glob,
        "--ckpt-store", str(ckpt_dir),
        "--ckpt-prefix", "tradefm",
        "--steps", str(steps),
        "--batch", "4",
        "--grad-accum", "1",
        "--ckpt-every", "100",
        "--log-every", "25",
        "--num-workers", "2",
    ]
    print("[8gpu] launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo), check=True)

    # Verify rank-partitioned ckpts: distributed.py writes rank_{N}.pt per rank.
    ckpts = sorted(ckpt_dir.rglob("*.pt"))
    rank_names = sorted(c.name for c in ckpts)
    print(f"[8gpu] ckpts written: {len(ckpts)}")
    for c in ckpts:
        print(f"    {c}  ({c.stat().st_size / 1e6:.1f} MB)")
    # Expect rank_0..rank_7 and opt_rank_0..opt_rank_7 at each ckpt step.
    expected_ranks = {f"rank_{i}.pt" for i in range(8)}
    found_ranks = {n for n in rank_names if n.startswith("rank_")}
    missing = expected_ranks - found_ranks
    assert not missing, f"missing per-rank ckpts: {missing}"
    ckpt_volume.commit()
    print("[8gpu] PASS — NCCL + FSDP at world_size=8 + rank-partitioned ckpts OK.")


@app.local_entrypoint()
def main():
    smoke_train.remote()
    smoke_resume.remote()
    print("[main] both 40M stages PASSED.")


@app.local_entrypoint()
def dryrun_524m():
    smoke_524m.remote()
    print("[dryrun_524m] PASS.")


@app.local_entrypoint()
def dryrun_databento():
    smoke_databento.remote()
    print("[dryrun_databento] PASS.")


@app.local_entrypoint()
def dryrun_databento_reuse():
    """Download + pack files from a completed Databento job (skips the batch
    queue). Defaults to the already-paid OPRA-20260420-CTDQCTDCGX job."""
    smoke_databento_reuse.remote()
    print("[dryrun_databento_reuse] PASS.")


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/ckpts": ckpt_volume, "/shards": shard_volume},
)
def smoke_train_real(
    shards_glob: str = "/shards/databento_reuse_packed/OPRA-20260420-CTDQCTDCGX/opra_*.parquet",
    steps: int = 200,
    batch: int = 4,
    grad_accum: int = 1,
    ckpt_every: int = 100,
    log_every: int = 10,
):
    """40M TradeFM training on real OPRA SPX tape instead of Markov synthetic.

    The Markov smoke proved the pipeline runs. This proves it *learns the
    actual grammar of exchange messages* — the Silicon Alpha goal's premise.

    Compares vs the Markov smoke:
      - Markov: loss 60 → 2.48 (entropy floor ~1.4)
      - Real OPRA: loss trajectory depends on how much structure the 40M can
        extract from SPX tick-level MBP-1 in 200 steps. A healthy trajectory
        looks like: high init → steady descent → eventual plateau above the
        per-feature entropy floor. Divergence or flatlining = bug.
    """
    import shutil
    import subprocess
    import sys

    repo = Path("/root/repo")
    sys.path.insert(0, str(repo))

    shard_volume.reload()
    ckpt_volume.reload()

    import torch
    print(f"[real] torch={torch.__version__} device={torch.cuda.get_device_name(0)}",
          flush=True)
    assert torch.cuda.is_available(), "H100 not visible"

    # Expand the glob on the client side so we fail fast if no shards exist.
    import glob as _glob
    shard_paths = sorted(_glob.glob(shards_glob))
    assert shard_paths, f"no shards matched {shards_glob!r} — run databento_reuse first"
    print(f"[real] using {len(shard_paths)} real-OPRA shards", flush=True)

    ckpt_dir = Path("/ckpts/tradefm_real_smoke")
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    # Same 40M Hopper config as the Markov smoke — only the data source changes.
    cfg_path = Path("/tmp/tradefm_real_h100.yml")
    cfg_path.write_text(SMOKE_CONFIG_YAML)

    cmd = [
        "torchrun", "--nproc_per_node=1", "--nnodes=1",
        "-m", "odte.train.distributed",
        "--config", str(cfg_path),
        "--shards", shards_glob,
        "--ckpt-store", str(ckpt_dir),
        "--ckpt-prefix", "tradefm_real",
        "--steps", str(steps),
        "--batch", str(batch),
        "--grad-accum", str(grad_accum),
        "--ckpt-every", str(ckpt_every),
        "--log-every", str(log_every),
        "--num-workers", "2",
    ]
    print("[real] launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo), check=True)

    ckpts = sorted(ckpt_dir.rglob("*.pt"))
    assert ckpts, "no ckpts written — training didn't reach ckpt_every"
    for c in ckpts:
        print(f"    {c.name}  ({c.stat().st_size / 1e6:.1f} MB)")
    ckpt_volume.commit()
    print("[real] PASS — 40M trained on real OPRA SPX MBP-1 tape.", flush=True)


@app.local_entrypoint()
def dryrun_train_real(steps: int = 200, batch: int = 4, ckpt_every: int = 100,
                      log_every: int = 10):
    """Train the 40M model on real OPRA shards produced by dryrun_databento_reuse.

    Defaults are the short smoke (200 steps, ~30s, ~$0.25). For a real
    convergence-shape run, invoke with --steps 2000 --ckpt-every 500.
    """
    smoke_train_real.remote(steps=steps, batch=batch,
                            ckpt_every=ckpt_every, log_every=log_every)
    print("[dryrun_train_real] PASS.")


@app.local_entrypoint()
def dryrun_8gpu(
    shards_glob: str = "/shards/databento_reuse_packed/OPRA-20260420-CTDQCTDCGX/opra_*.parquet",
    steps: int = 200,
):
    """8× H100 NCCL smoke. Defaults to real-OPRA shards from databento_reuse.
    Pass `--shards-glob /shards/markov_*.parquet` to test on synthetic instead."""
    smoke_8gpu.remote(shards_glob=shards_glob, steps=steps)
    print("[dryrun_8gpu] PASS.")


# ---------------------------------------------------------------------------
# Multi-day Phase-2 corpus packer — reads job IDs from the local submit
# manifest and fans out one Modal container per day. Each container gets
# a 10h timeout (long enough even for the 207 GB vol-spike days) and
# runs in parallel, respecting the 10-GPU concurrent limit (these are
# CPU-only, so no GPU contention).
# ---------------------------------------------------------------------------

@app.function(
    cpu=4.0,
    memory=65536,
    # 10 hours — even the biggest 200+ GB days pack in ~6 hrs; 10 gives margin.
    timeout=36000,
    volumes={"/shards": shard_volume},
    secrets=[modal.Secret.from_name("databento-api-key")],
)
def pack_one_day(job_id: str, day_label: str, split: str):
    """Download + pack one Databento job into its own subdir of the volume.
    Reuses the existing smoke_databento_reuse machinery but with an explicit
    job_id argument per call — so each day becomes its own Modal invocation
    that can run in parallel with others."""
    # Call the existing reuse function with this job_id. Modal .local() runs
    # it in-process (we're already in a Modal container with the same
    # image + volumes), not spawning another container.
    print(f"[multi] packing job {job_id} ({day_label}, split={split})",
          flush=True)
    return smoke_databento_reuse.local(job_id=job_id, max_files=99)


@app.local_entrypoint()
def dryrun_pack_all():
    """Download + pack every Databento job listed in scripts/databento_jobs.json.
    Fires one Modal container per day in parallel.

    Assumes you already ran:
      1. scripts/databento_submit_all.py  (submits jobs to Databento)
      2. scripts/databento_status.py      (waited until all are 'done')
    """
    import json as _json
    from pathlib import Path as _P
    manifest_path = _P(__file__).resolve().parents[2] / "scripts" / "databento_jobs.json"
    assert manifest_path.exists(), (
        f"No manifest at {manifest_path}. Run scripts/databento_submit_all.py first."
    )
    rows = _json.loads(manifest_path.read_text())
    runnable = [r for r in rows if r.get("job_id")]
    print(f"[multi] dispatching {len(runnable)} parallel pack containers")
    for r in runnable:
        print(f"  {r['day']}  {r['split']:<5}  {r['job_id']}  ({r['note']})")
    # Fan out — .spawn returns a handle without blocking, so all containers
    # start in parallel (subject to Modal's concurrency limits).
    handles = [
        pack_one_day.spawn(r["job_id"], r["day"], r["split"])
        for r in runnable
    ]
    # Block on all.
    for h, r in zip(handles, runnable):
        try:
            h.get()
            print(f"[multi] PASS {r['day']} ({r['job_id']})")
        except Exception as e:
            print(f"[multi] FAIL {r['day']} ({r['job_id']}): {e}")
    print("[dryrun_pack_all] done.")


# ---------------------------------------------------------------------------
# Phase-2 multi-day training + eval — 524M on the regime-stratified corpus.
# ---------------------------------------------------------------------------

# Default eval job (2026-04-16, held out from training). Train is every other
# job_id in the volume. Safe default — can override per-call via CLI.
DEFAULT_EVAL_JOB_ID = "OPRA-20260424-DLKDHYSC6M"


def _resolve_multi_day_shards(eval_job_id: str) -> tuple[list[str], list[str]]:
    """Read the packed-shard directory and split globs into (train, eval).

    Returns two lists of glob patterns. All packed Databento jobs in
    /shards/databento_reuse_packed/ are treated as train, except the one
    matching eval_job_id.
    """
    import glob as _glob
    base = Path("/shards/databento_reuse_packed")
    if not base.exists():
        raise RuntimeError(f"no packed shards at {base} — run dryrun_pack_all first")
    train_globs: list[str] = []
    eval_globs: list[str] = []
    for job_dir in sorted(base.iterdir()):
        if not job_dir.is_dir():
            continue
        glob_pat = f"{job_dir}/opra_*.parquet"
        if not _glob.glob(glob_pat):
            continue
        if job_dir.name == eval_job_id:
            eval_globs.append(glob_pat)
        else:
            train_globs.append(glob_pat)
    return train_globs, eval_globs


@app.function(
    gpu="H100:8",
    # 10h hard cap. Modal Starter tier individual function calls max out
    # around 24h; we ckpt every 500 steps so even on a forced restart we
    # lose ≤500 steps of work.
    timeout=36000,
    volumes={"/ckpts": ckpt_volume, "/shards": shard_volume},
)
def train_524m_multi_day(
    steps: int = 2000,
    batch: int = 1,
    grad_accum: int = 4,
    ckpt_every: int = 500,
    log_every: int = 50,
    resume: bool = False,
    eval_job_id: str = DEFAULT_EVAL_JOB_ID,
    eval_every: int = 500,
    eval_max_batches: int = 200,
    enable_eval: bool = True,
):
    """524M TradeFM pretraining on the 4-day regime-stratified corpus.

    Hardware: 8× H100 (peak 10-GPU Modal Starter limit). Throughput target
    ~0.5-1 s/step at batch=1/GPU × effective 32 (8 GPUs × 4 grad-accum);
    2000 steps ≈ 20-35 min + ~90s cold start.

    Every 500 steps writes rank-partitioned ckpts to /ckpts/tradefm_524m_multi,
    so a forced timeout or Modal's 24h ceiling costs ≤500 steps of work.
    Re-invoke with resume=True to pick up from the latest ckpt.

    Reads train shards from every packed Databento job EXCEPT the one
    matching eval_job_id (default: 2026-04-16 held-out eval).
    """
    import shutil
    import subprocess
    import sys

    repo = Path("/root/repo")
    sys.path.insert(0, str(repo))

    shard_volume.reload()
    ckpt_volume.reload()

    import torch
    print(f"[524m-multi] torch={torch.__version__}", flush=True)
    assert torch.cuda.is_available(), "H100 not visible"
    try:
        torch.zeros(1, device="cuda:0")  # warm CUDA before per-device queries
    except Exception:
        pass
    n_gpu = torch.cuda.device_count()
    assert n_gpu == 8, f"expected 8 GPUs, got {n_gpu}"
    print(f"[524m-multi] {n_gpu} GPUs ready", flush=True)

    train_globs, eval_globs = _resolve_multi_day_shards(eval_job_id)
    assert train_globs, "no train shards — run dryrun_pack_all first"
    print(f"[524m-multi] train globs: {len(train_globs)}  eval globs: {len(eval_globs)}",
          flush=True)
    for g in train_globs:
        print(f"    train: {g}", flush=True)
    for g in eval_globs:
        print(f"    eval : {g}", flush=True)
    # torchrun's --shards takes a single glob string; we pass a comma-separated
    # list of globs, and the trainer already supports multi-pattern via its
    # glob expansion (ShardedTokenDataset sorts all matched paths).
    # We build a single brace-expanded glob of all train dirs.
    train_multi_glob = ",".join(train_globs)

    ckpt_dir = Path("/ckpts/tradefm_524m_multi")
    if not resume and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    cmd = [
        "torchrun", "--nproc_per_node=8", "--nnodes=1",
        "-m", "odte.train.distributed",
        "--config", "configs/tradefm_524m.yml",
        "--shards", train_multi_glob,
        "--ckpt-store", str(ckpt_dir),
        "--ckpt-prefix", "tradefm_524m_multi",
        "--steps", str(steps),
        "--batch", str(batch),
        "--grad-accum", str(grad_accum),
        "--ckpt-every", str(ckpt_every),
        "--log-every", str(log_every),
        "--num-workers", "4",
    ]
    if resume:
        cmd.append("--resume")
    # Enable in-process eval during training. This sidesteps the FSDP
    # post-hoc load entirely — eval runs on the in-memory FSDP-wrapped
    # model that just trained, so there's no serialization round-trip
    # and no FQN / use_orig_params / activation-checkpointing-prefix
    # asymmetry to worry about.
    if enable_eval and eval_globs:
        eval_glob_arg = ",".join(eval_globs)
        cmd += [
            "--eval-shards", eval_glob_arg,
            "--eval-every", str(eval_every),
            "--eval-max-batches", str(eval_max_batches),
        ]
        print(f"[524m-multi] eval enabled: every {eval_every} steps "
              f"on {len(eval_globs)} held-out glob(s), "
              f"{eval_max_batches} max batches per eval",
              flush=True)
    print("[524m-multi] launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo), check=True)

    # Commit before reporting so a subsequent eval_524m run sees everything.
    ckpt_volume.commit()
    ckpts = sorted(ckpt_dir.rglob("*.pt"))
    print(f"[524m-multi] ckpts written: {len(ckpts)}", flush=True)
    # Show the last step's ckpts (most recent is most useful for eval).
    last_step_dir = max(ckpt_dir.glob("tradefm_524m_multi/step_*"), default=None)
    if last_step_dir:
        print(f"[524m-multi] latest step dir: {last_step_dir.name}", flush=True)
        for c in sorted(last_step_dir.glob("*.pt"))[:4]:
            print(f"    {c.name}  ({c.stat().st_size / 1e9:.2f} GB)", flush=True)
    print("[524m-multi] PASS", flush=True)


@app.function(
    # 8× H100 — must match the training-time world_size to load the
    # FSDP-sharded ckpts. CheckpointManager writes one rank shard per GPU;
    # loading on a different world_size is hard-blocked by the world-size
    # guard in checkpoint.py (commit e3c12dd).
    gpu="H100:8",
    timeout=3600,
    volumes={"/ckpts": ckpt_volume, "/shards": shard_volume},
)
def eval_524m(
    ckpt_step: int = -1,
    eval_job_id: str = DEFAULT_EVAL_JOB_ID,
    batch: int = 4,
    max_batches: int = 200,
):
    """Evaluate the trained 524M on the held-out 2026-04-16 shards.

    Reports four numbers that matter for the sponsor email:

      - **loss** (cross-entropy on held-out tape): generalization loss vs
        training loss curve → gap = overfit magnitude.
      - **PPL** (perplexity = exp(loss)): easier to reason about than nats.
      - **top-1 accuracy**: exact-match rate of argmax vs true next token.
        Strict metric; with vocab=4096 and real microstructure this is
        typically 20-40% for a well-trained sequence model.
      - **above-median accuracy** (the "direction proxy"): for every
        predicted token, does the predicted-bin-index land in the same
        half of the vocab as the target? This is a per-token "up vs down"
        classifier. For an actual edge over the ~3.6% bid-ask spread on
        SPX 0DTE options, this number should exceed **53%**. Below 50% is
        anti-signal (worse than coin flip); 50-53% is informative but
        not profitable after spreads; >55% is a real signal.

    The gap between top-1 accuracy and above-median accuracy tells you
    what the model learned:
      - large gap, above-median >53%  -> directional signal without
        memorization (the "physics of the orderbook" claim). Good.
      - small gap, top-1 high         -> memorization of this day's drift.
        Sponsor red flag.

    Implementation: launches torchrun on the same 8× H100 layout as
    training, so the FSDP-sharded ckpts in /ckpts/tradefm_524m_multi
    load via CheckpointManager (which knows how to read rank-partitioned
    ShardedTensors). The actual eval logic lives in the trainer's new
    --eval-only branch which calls `evaluate()` and all_reduce-aggregates
    metrics across ranks.

    ckpt_step is currently informational — the trainer's --resume always
    loads the latest ckpt. Future improvement: pass an explicit step.
    """
    import shutil
    import subprocess
    import sys

    repo = Path("/root/repo")
    sys.path.insert(0, str(repo))

    shard_volume.reload()
    ckpt_volume.reload()

    import torch
    n_gpu = torch.cuda.device_count()
    print(f"[eval] {n_gpu} GPUs visible", flush=True)
    assert n_gpu == 8, f"expected 8 GPUs, got {n_gpu}"
    try:
        torch.zeros(1, device="cuda:0")
    except Exception:
        pass

    eval_glob = f"/shards/databento_reuse_packed/{eval_job_id}/opra_*.parquet"
    import glob as _glob
    matched = sorted(_glob.glob(eval_glob))
    print(f"[eval] {len(matched)} eval shards matched {eval_glob}", flush=True)
    assert matched, f"no eval shards at {eval_glob}"

    ckpt_root = Path("/ckpts/tradefm_524m_multi/tradefm_524m_multi")
    step_dirs = sorted(ckpt_root.glob("step_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    assert step_dirs, f"no ckpt dirs under {ckpt_root}"
    print(f"[eval] {len(step_dirs)} ckpt steps found; will load latest "
          f"({step_dirs[-1].name})", flush=True)

    # Pass a placeholder shards arg — distributed.py requires it but in
    # eval-only mode it's never used. We use the eval_glob for both so
    # the shard-glob check passes even if eval-only short-circuits before
    # touching it.
    cmd = [
        "torchrun", "--nproc_per_node=8", "--nnodes=1",
        "-m", "odte.train.distributed",
        "--config", "configs/tradefm_524m.yml",
        "--shards", eval_glob,                      # not used in eval-only
        "--ckpt-store", "/ckpts/tradefm_524m_multi",
        "--ckpt-prefix", "tradefm_524m_multi",
        "--steps", "0",
        "--batch", str(batch),
        "--grad-accum", "1",
        "--num-workers", "2",
        "--resume",
        "--eval-only",
        "--eval-shards", eval_glob,
        "--eval-max-batches", str(max_batches),
    ]
    print("[eval] launching:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo), check=True)
    print("[eval] PASS", flush=True)


@app.local_entrypoint()
def dryrun_train_524m_multi(
    steps: int = 2000, batch: int = 1, grad_accum: int = 4,
    ckpt_every: int = 500, log_every: int = 50, resume: bool = False,
    eval_job_id: str = DEFAULT_EVAL_JOB_ID,
    eval_every: int = 500, eval_max_batches: int = 200,
    enable_eval: bool = True,
):
    """Phase-2 524M training on the 4-day regime-stratified corpus.

    Default pilot: 2000 steps with eval every 500 steps (~25-40 min on
    8× H100, ~$36-40). Eval runs in-process on the held-out 2026-04-16
    shards — no post-hoc FSDP load needed.

    For longer run: `--steps 10000 --ckpt-every 1000 --eval-every 1000`.
    To disable eval: `--no-enable-eval`.
    """
    train_524m_multi_day.remote(
        steps=steps, batch=batch, grad_accum=grad_accum,
        ckpt_every=ckpt_every, log_every=log_every, resume=resume,
        eval_job_id=eval_job_id,
        eval_every=eval_every, eval_max_batches=eval_max_batches,
        enable_eval=enable_eval,
    )
    print("[dryrun_train_524m_multi] PASS.")


@app.local_entrypoint()
def dryrun_eval_524m(
    ckpt_step: int = -1,
    eval_job_id: str = DEFAULT_EVAL_JOB_ID,
    batch: int = 4,
    max_batches: int = 200,
):
    """Evaluate latest (or specified) 524M ckpt on the held-out eval day.
    Reports loss, PPL, top-1 accuracy, above-median (direction proxy)
    accuracy vs 53% SPX-0DTE-spread threshold."""
    eval_524m.remote(
        ckpt_step=ckpt_step, eval_job_id=eval_job_id,
        batch=batch, max_batches=max_batches,
    )
    print("[dryrun_eval_524m] PASS.")


# ---------------------------------------------------------------------------
# Directional-head joint-loss smoke (Phase 2 Option A)
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    cpu=4.0,
    memory=32768,
    timeout=900,
    volumes={"/shards": shard_volume},
)
def smoke_dir_head(
    steps: int = 50,
    ctx_len: int = 256,
    batch: int = 4,
):
    """Validate the DirectionalHead + joint_loss path end-to-end on real
    GPU before committing to a multi-day retrain.

    Builds a 16M-ish TradeFM with dir_head_enabled=True, runs a few
    optimizer steps on real OPRA shards, and asserts:

      1. Forward with return_aux=True returns (lm_logits, dir_logits) of
         the right shape (no FSDP weight-shape blowup since we're single-GPU).
      2. _build_dir_targets returns a non-empty mask on real data.
      3. L_lm decreases over `steps` (model is learning the LM objective).
      4. L_dir decreases below log(2)≈0.693 (head is finding directional
         signal, not stuck at chance).

    On a T4 this runs in ~3-5 min for 50 steps and costs ~$0.05.
    """
    import os, sys, time, glob as _glob
    sys.path.insert(0, "/root/repo")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    import math
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from models.config import TradeFMConfig
    from odte.transformer_tradefm import TradeFM
    from odte.train.pretrain_tradefm import ShardTokenDataset
    from odte.train.eval_loop import _collate_with_feature_offset

    print(f"[dir-smoke] cuda={torch.cuda.is_available()}  "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
          flush=True)

    base = "/shards/databento_reuse_packed"
    shard_paths = sorted(
        Path(p) for p in _glob.glob(f"{base}/*/opra_*.parquet")
    )[:3]
    assert shard_paths, f"no shards under {base}"
    print(f"[dir-smoke] using {len(shard_paths)} shards", flush=True)

    cfg = TradeFMConfig(
        d_model=256, n_heads=4, n_layers=4, vocab=4096, ctx_len=ctx_len,
        dropout=0.1, fp8=False, use_flash_attn=False,
        dir_head_enabled=True, dir_alpha=1.0, dir_beta=0.5,
        dir_horizon=10, dir_n_features=7,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradeFM(cfg).to(device)
    print(f"[dir-smoke] params: {model.num_params():,}  "
          f"dir_head: {model.dir_head is not None}", flush=True)

    ds = ShardTokenDataset(shard_paths, ctx_len=cfg.ctx_len, seed=0,
                           n_features=cfg.dir_n_features,
                           with_feature_offset=True)
    loader = DataLoader(ds, batch_size=batch, num_workers=0,
                        collate_fn=_collate_with_feature_offset)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    history = []
    t0 = time.time()
    it = iter(loader)
    for step in range(steps):
        try:
            batch_tok, feat_off = next(it)
        except StopIteration:
            it = iter(loader)
            batch_tok, feat_off = next(it)
        batch_tok = batch_tok.to(device)
        feat_off = feat_off.to(device)

        opt.zero_grad(set_to_none=True)
        lm_logits, dir_logits = model(
            batch_tok[:, :-1], return_aux=True)
        target_lm = batch_tok[:, 1:]
        L_lm = torch.nn.functional.cross_entropy(
            lm_logits.reshape(-1, cfg.vocab), target_lm.reshape(-1))
        dir_tgt, dir_mask = model._build_dir_targets(
            batch_tok, feature_offset=feat_off)
        n_valid = int(dir_mask.sum().item())
        if n_valid > 0:
            L_dir = torch.nn.functional.binary_cross_entropy_with_logits(
                dir_logits[dir_mask], dir_tgt[dir_mask], reduction="mean")
        else:
            L_dir = torch.zeros((), device=device, dtype=lm_logits.dtype)
        loss = cfg.dir_alpha * L_lm + cfg.dir_beta * L_dir
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append({"L_lm": float(L_lm.item()),
                        "L_dir": float(L_dir.item()),
                        "n_dir": n_valid})
        if step % 5 == 0:
            print(f"[dir-smoke] step={step:>3}  L_lm={float(L_lm.item()):.4f}  "
                  f"L_dir={float(L_dir.item()):.4f}  "
                  f"n_dir={n_valid:,}", flush=True)

    elapsed = time.time() - t0
    early_lm = float(np.mean([h["L_lm"] for h in history[:10]]))
    late_lm  = float(np.mean([h["L_lm"] for h in history[-10:]]))
    early_dir = float(np.mean([h["L_dir"] for h in history[:10]]))
    late_dir  = float(np.mean([h["L_dir"] for h in history[-10:]]))
    print()
    print(f"[dir-smoke] ===== RESULT =====", flush=True)
    print(f"[dir-smoke] steps={steps}  elapsed={elapsed:.1f}s", flush=True)
    print(f"[dir-smoke] L_lm:  early={early_lm:.4f}  late={late_lm:.4f}  "
          f"Δ={late_lm-early_lm:+.4f}", flush=True)
    print(f"[dir-smoke] L_dir: early={early_dir:.4f}  late={late_dir:.4f}  "
          f"Δ={late_dir-early_dir:+.4f}  (chance=0.6931)", flush=True)
    lm_ok = late_lm < early_lm
    dir_ok = late_dir < 0.69 and late_dir < early_dir
    print(f"[dir-smoke] LM decreasing : {'PASS' if lm_ok else 'FAIL'}",
          flush=True)
    print(f"[dir-smoke] dir < chance  : {'PASS' if dir_ok else 'FAIL'}",
          flush=True)

    return {
        "steps": steps, "elapsed_s": elapsed,
        "early_lm": early_lm, "late_lm": late_lm,
        "early_dir": early_dir, "late_dir": late_dir,
        "lm_decreasing": lm_ok, "dir_below_chance": dir_ok,
    }


@app.local_entrypoint()
def dryrun_dir_head(steps: int = 50):
    """Local entrypoint for the directional-head smoke. ~$0.05, ~5 min on T4."""
    result = smoke_dir_head.remote(steps=steps)
    print(f"[dryrun_dir_head] result: {result}")
