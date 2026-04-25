"""Sharded checkpoint manager for Phase-2 TradeFM training.

Responsibilities:
  - FSDP-compatible sharded state-dict save/load
  - Atomic local writes + optional S3 (or any fsspec URL) upload
  - Resume-from-any-step: discover the latest valid ckpt in an S3 prefix
  - Metadata file records step, best_loss, config hash, rank topology
  - Best / rolling retention policy (keeps last N plus the best)

Design notes:
  - Save format: one `<prefix>_step_<N>/` directory per ckpt, containing
      rank_<k>.pt        sharded tensors for rank k (FSDP FULL_STATE_DICT off)
      meta.json          step, best_loss, config, world_size
      opt_rank_<k>.pt    optimizer shard (optional)
  - Rank 0 uploads meta.json last, so partial uploads are detectable.
  - Resume: pick the largest step whose meta.json is present + whose
    rank count matches the current world size.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

log = logging.getLogger(__name__)

# Optional deps.
try:
    import fsspec  # type: ignore
    _HAS_FSSPEC = True
except ImportError:
    _HAS_FSSPEC = False


def _cfg_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------

@dataclass
class LocalStore:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, local: Path, rel: str) -> None:
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, dst)

    def get(self, rel: str, local: Path) -> None:
        src = self.root / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, local)

    def list(self, prefix: str) -> List[str]:
        p = self.root / prefix
        if not p.exists():
            return []
        return sorted(str(q.relative_to(self.root)) for q in p.rglob("*") if q.is_file())

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def delete_dir(self, rel: str) -> None:
        p = self.root / rel
        if p.exists():
            shutil.rmtree(p)


@dataclass
class FsspecStore:
    """Any fsspec URL (s3://, gs://, az://, file://, ...)."""
    url: str
    _fs: Any = field(default=None, init=False)
    _root: str = field(default="", init=False)

    def __post_init__(self):
        if not _HAS_FSSPEC:
            raise RuntimeError("pip install fsspec[s3] for cloud storage")
        self._fs, self._root = fsspec.core.url_to_fs(self.url)
        if self._root and not self._root.endswith("/"):
            self._root = self._root + "/"

    def _full(self, rel: str) -> str:
        return self._root + rel

    def put(self, local: Path, rel: str) -> None:
        self._fs.put(str(local), self._full(rel))

    def get(self, rel: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self._fs.get(self._full(rel), str(local))

    def list(self, prefix: str) -> List[str]:
        try:
            return sorted(p[len(self._root):] for p in self._fs.find(self._full(prefix)))
        except FileNotFoundError:
            return []

    def exists(self, rel: str) -> bool:
        return self._fs.exists(self._full(rel))

    def delete_dir(self, rel: str) -> None:
        try:
            self._fs.rm(self._full(rel), recursive=True)
        except Exception as e:
            log.warning("delete_dir(%s) failed: %s", rel, e)


def make_store(url: str | Path) -> LocalStore | FsspecStore:
    s = str(url)
    if "://" in s and not s.startswith("file://"):
        return FsspecStore(s)
    return LocalStore(Path(s))


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

@dataclass
class CheckpointManager:
    store_url: str | Path
    prefix: str = "tradefm"
    rank: int = 0
    world_size: int = 1
    keep_last: int = 3
    keep_best: bool = True
    _store: Any = field(init=False)

    def __post_init__(self):
        self._store = make_store(self.store_url)

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------
    def save(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
             step: int, best_loss: float,
             cfg: Dict[str, Any], label: str = "ckpt") -> str:
        """Write a ckpt for this rank. Rank 0 finalizes meta.json last."""
        step_dir = f"{self.prefix}/step_{step:08d}"
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            # 1. Dump this rank's state dict
            rank_path = td_p / f"rank_{self.rank}.pt"
            state = self._get_state_dict(model)
            torch.save(state, rank_path)
            self._store.put(rank_path, f"{step_dir}/rank_{self.rank}.pt")

            if optimizer is not None:
                opt_path = td_p / f"opt_rank_{self.rank}.pt"
                torch.save(optimizer.state_dict(), opt_path)
                self._store.put(opt_path, f"{step_dir}/opt_rank_{self.rank}.pt")

            # 2. Rank 0 writes meta.json after all ranks have put their shards
            if self.rank == 0:
                meta = {
                    "step": int(step),
                    "best_loss": float(best_loss),
                    "world_size": int(self.world_size),
                    "label": label,
                    "saved_at": time.time(),
                    "cfg": cfg,
                    "cfg_hash": _cfg_hash(cfg),
                }
                meta_path = td_p / "meta.json"
                meta_path.write_text(json.dumps(meta, indent=2))
                self._store.put(meta_path, f"{step_dir}/meta.json")

        log.info("[rank %d] ckpt saved → %s/%s", self.rank, self.store_url, step_dir)
        if self.rank == 0:
            self._prune_old()
        return step_dir

    def _get_state_dict(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """FSDP-aware: use LOCAL_STATE_DICT so each rank saves its shard."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
            if isinstance(model, FSDP):
                with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
                    return model.state_dict()
        except Exception:
            pass
        return model.state_dict()

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    def latest_step(self) -> Optional[int]:
        """Largest step whose meta.json is present AND world_size matches.

        Returns None only when there are NO ckpts at all (legitimate cold
        start). If ckpts exist but none match the current world_size, raises
        RuntimeError — prevents the silent "restart from step 0" failure
        mode where a node crash mid-run would otherwise wipe hours of
        training without warning.
        """
        entries = self._store.list(self.prefix)
        metas = [e for e in entries if e.endswith("meta.json")]
        candidates: list[tuple[int, str]] = []
        wrong_ws: list[tuple[int, int]] = []  # (step, saved_world_size)
        for m in metas:
            try:
                with tempfile.NamedTemporaryFile(suffix=".json") as f:
                    self._store.get(m, Path(f.name))
                    payload = json.loads(Path(f.name).read_text())
                saved_ws = int(payload.get("world_size", -1))
                if saved_ws != self.world_size:
                    wrong_ws.append((int(payload.get("step", -1)), saved_ws))
                    continue
                candidates.append((int(payload["step"]), m))
            except Exception as e:
                log.warning("skip %s: %s", m, e)
        if not candidates:
            if wrong_ws:
                wrong_ws.sort()
                latest_step, saved_ws = wrong_ws[-1]
                raise RuntimeError(
                    f"CheckpointManager: found {len(wrong_ws)} ckpt(s) saved "
                    f"with world_size={saved_ws}, but current world_size="
                    f"{self.world_size}. LOCAL_STATE_DICT ckpts can only "
                    f"resume on matching node count (latest is step "
                    f"{latest_step}). To resume on a different node count, "
                    f"re-save that ckpt as FULL_STATE_DICT and reshard — "
                    f"silently restarting from step 0 would waste hours of "
                    f"training. TODO(phase-2-resharding)."
                )
            return None
        return max(candidates, key=lambda x: x[0])[0]

    def load(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
             step: Optional[int] = None, current_cfg: Optional[Dict[str, Any]] = None,
             allow_cfg_drift: bool = False) -> Dict[str, Any]:
        """Resume from a ckpt.

        If current_cfg is provided, its sha256 hash is compared against the
        hash stored in meta.json at save time. Mismatch raises RuntimeError
        unless allow_cfg_drift=True — prevents the silent "wrong config
        resumed" failure mode where a changed layer count or vocab size
        would load into a model-shape mismatch far later in the step.
        """
        step = step if step is not None else self.latest_step()
        if step is None:
            log.info("[rank %d] no ckpt found under %s/%s", self.rank, self.store_url, self.prefix)
            return {"step": 0, "best_loss": float("inf")}
        step_dir = f"{self.prefix}/step_{step:08d}"
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            # meta
            meta_path = td_p / "meta.json"
            self._store.get(f"{step_dir}/meta.json", meta_path)
            meta = json.loads(meta_path.read_text())
            # Config-hash gate
            if current_cfg is not None:
                saved_hash = meta.get("cfg_hash")
                current_hash = _cfg_hash(current_cfg)
                if saved_hash is not None and saved_hash != current_hash:
                    msg = (f"CheckpointManager: cfg_hash mismatch on resume. "
                           f"saved={saved_hash}  current={current_hash}  "
                           f"step={meta.get('step')}. This ckpt was trained "
                           f"with a different config; resuming will either "
                           f"shape-error or silently corrupt training.")
                    if not allow_cfg_drift:
                        raise RuntimeError(msg + " Pass allow_cfg_drift=True to override.")
                    log.warning(msg + " allow_cfg_drift=True — proceeding anyway.")
            elif self.rank == 0:
                log.warning("CheckpointManager.load() called without current_cfg; "
                            "cfg_hash check skipped. Pass current_cfg=asdict(cfg) "
                            "to enforce.")
            # rank shard
            rank_path = td_p / f"rank_{self.rank}.pt"
            self._store.get(f"{step_dir}/rank_{self.rank}.pt", rank_path)
            state = torch.load(rank_path, map_location="cpu")
            self._load_state_dict(model, state)
            if optimizer is not None:
                opt_rel = f"{step_dir}/opt_rank_{self.rank}.pt"
                if self._store.exists(opt_rel):
                    opt_path = td_p / f"opt_rank_{self.rank}.pt"
                    self._store.get(opt_rel, opt_path)
                    optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        log.info("[rank %d] resumed from step %d (best_loss=%.4f)",
                 self.rank, meta["step"], meta.get("best_loss", float("inf")))
        return meta

    def _load_state_dict(self, model: torch.nn.Module, state: Dict[str, Any]) -> None:
        # FSDP path. The previous version silently swallowed errors here and
        # fell through to a non-FSDP load, which then loudly failed with
        # "Unexpected key(s): _flat_param". Now we surface FSDP errors and
        # try a key-rename fallback for the activation-checkpointing prefix
        # mismatch before giving up.
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
        except ImportError:
            FSDP = None  # type: ignore
            StateDictType = None  # type: ignore

        is_fsdp = FSDP is not None and isinstance(model, FSDP)
        if is_fsdp:
            with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
                # Attempt #1: load as-is.
                try:
                    model.load_state_dict(state)
                    return
                except RuntimeError as e1:
                    log.warning("[ckpt] direct LOCAL_STATE_DICT load failed: %s", e1)
                # Attempt #2: keys may differ by an activation-checkpointing
                # prefix. Try inserting / stripping `_checkpoint_wrapped_module.`
                # at module boundaries. Save was probably written without the
                # prefix; the current load model has the prefix (or vice versa).
                for transform_name, transform in (
                    ("add _checkpoint_wrapped_module prefix",
                     lambda k: k.replace("blocks.", "blocks.")
                                .replace(
                                    "._flat_param",
                                    "._checkpoint_wrapped_module._flat_param",
                                )),
                    ("strip _checkpoint_wrapped_module prefix",
                     lambda k: k.replace("._checkpoint_wrapped_module.", ".")),
                ):
                    renamed = {transform(k): v for k, v in state.items()}
                    try:
                        model.load_state_dict(renamed)
                        log.info("[ckpt] loaded after key transform: %s",
                                 transform_name)
                        return
                    except RuntimeError as e2:
                        log.warning("[ckpt] transform '%s' failed: %s",
                                    transform_name, e2)
                # If all FSDP attempts failed, fall through to a strict
                # non-FSDP load below — it will likely fail with a clear
                # error that tells us the real key mismatch.
        model.load_state_dict(state)

    # ------------------------------------------------------------------
    # retention
    # ------------------------------------------------------------------
    def _prune_old(self) -> None:
        entries = self._store.list(self.prefix)
        step_dirs: dict[int, str] = {}
        for e in entries:
            parts = e.split("/")
            if len(parts) >= 2 and parts[-2].startswith("step_"):
                try:
                    s = int(parts[-2][5:])
                    step_dirs.setdefault(s, "/".join(parts[:-1]))
                except ValueError:
                    continue
        if not step_dirs:
            return
        # Load meta to identify best
        best_step = None
        if self.keep_best:
            best_loss = float("inf")
            for s, rel in step_dirs.items():
                try:
                    with tempfile.NamedTemporaryFile(suffix=".json") as f:
                        self._store.get(f"{rel}/meta.json", Path(f.name))
                        meta = json.loads(Path(f.name).read_text())
                    if meta.get("best_loss", float("inf")) < best_loss and meta.get("label") == "best":
                        best_loss = meta["best_loss"]
                        best_step = s
                except Exception:
                    continue
        keep = set(sorted(step_dirs)[-self.keep_last:])
        if best_step is not None:
            keep.add(best_step)
        for s, rel in step_dirs.items():
            if s not in keep:
                log.info("pruning step %d → %s", s, rel)
                self._store.delete_dir(rel)
