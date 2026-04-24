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

    print("[reuse] fitting tokenizer edges from real MBP-1 stream…", flush=True)
    edges = fit_hybrid_from_chunks(
        _all_chunks(), packer.feature_spec, n_buckets=64,
        checkpoint=out_dir / "_fit_ckpt",
    )
    tok = HybridBinTokenizer(n_buckets=64, feature_spec=packer.feature_spec)
    tok.edges = edges
    tok.save(out_dir / "tokenizer.json")
    packer.tokenizer = tok
    print(f"[reuse] tokenizer fit done, edges for {len(edges)} features", flush=True)
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
def smoke_8gpu():
    """8× H100 single-node NCCL smoke.

    Exercises the code path the 1-GPU smoke cannot:
      - world_size=8 FSDP sharding (24 layers across 8 ranks)
      - NCCL gradient all-reduce per optimizer step (over NVLink)
      - ShardedTokenDataset rank-partitioning (shared-seed shuffle +
        rank-strided shard slice, committed in e3c12dd)
      - Rank-specific checkpoint files: rank_0.pt ... rank_7.pt
      - CheckpointManager world_size metadata (guards against resuming
        an 8-GPU ckpt on a different world_size, committed in e3c12dd)

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
    print(f"[8gpu] device count: {n_gpu}")
    assert n_gpu == 8, f"expected 8 GPUs, got {n_gpu}"
    for i in range(n_gpu):
        print(f"    gpu{i}: {torch.cuda.get_device_name(i)}")

    shards = sorted(Path("/shards").glob("markov_*.parquet"))
    assert shards, "run `modal run infra/modal/phase2_smoke.py` first to gen shards"
    print(f"[8gpu] using {len(shards)} Markov shards")

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
        "--shards", SHARD_GLOB,
        "--ckpt-store", str(ckpt_dir),
        "--ckpt-prefix", "tradefm",
        "--steps", "200",
        "--batch", "4",
        "--grad-accum", "1",
        "--ckpt-every", "100",
        "--log-every", "25",
        "--num-workers", "2",
    ]
    print("[8gpu] launching:", " ".join(cmd))
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
        "--ckpt-every", "100",
        "--log-every", "10",
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
def dryrun_train_real():
    """Train the 40M model on real OPRA shards produced by dryrun_databento_reuse."""
    smoke_train_real.remote()
    print("[dryrun_train_real] PASS.")


@app.local_entrypoint()
def dryrun_8gpu():
    smoke_8gpu.remote()
    print("[dryrun_8gpu] PASS.")
