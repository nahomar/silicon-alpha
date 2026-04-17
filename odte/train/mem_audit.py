"""Memory headroom audit for TradeFM configurations.

Before burning cloud GPU hours on a run, sweep (batch, ctx_len) to find the
biggest viable training footprint on a single GPU. Reports:
  - dry_alloc_gb     parameters + optimizer state only
  - fwd_peak_gb      after one forward
  - fwd_bwd_peak_gb  after forward+backward (the real ceiling)
  - headroom_gb      device_total - fwd_bwd_peak_gb

For CUDA: uses `torch.cuda.reset_peak_memory_stats` + `max_memory_allocated`.
For MPS/CPU: uses `torch.mps.current_allocated_memory` when available; on
CPU reports process RSS via resource.getrusage as a best-effort number.

Run locally OR on a fresh pod as:
    python -m odte.train.mem_audit --config configs/tradefm_524m.yml \
        --batch 1 2 4 --ctx 1024 2048 4096

Stops early on OOM for a given (batch, ctx) row.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import resource
from dataclasses import asdict
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from odte.transformer_tradefm import TradeFM
from odte.train.pretrain_tradefm import load_config

log = logging.getLogger(__name__)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


def _peak_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    if device.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / (1024 ** 3)
        except AttributeError:
            return float("nan")
    # CPU: RSS via getrusage (maxrss is kB on Linux, bytes on macOS)
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if r > 1e9:
        return r / (1024 ** 3)       # bytes → GB (macOS)
    return r / (1024 ** 2)           # kB → GB (Linux)


def _total_gb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    if device.type == "mps":
        try:
            return torch.mps.driver_allocated_memory() / (1024 ** 3)
        except AttributeError:
            pass
    return None


def audit(cfg, batch_sizes: List[int], ctx_lens: List[int],
          device: torch.device, accum_steps: int = 1) -> list[dict]:
    rows: list[dict] = []
    device_total = _total_gb(device)
    for ctx in ctx_lens:
        for b in batch_sizes:
            cfg_run = type(cfg)(**{**asdict(cfg), "ctx_len": ctx})
            row = {"batch": b, "ctx": ctx, "status": "ok",
                   "dry_alloc_gb": float("nan"),
                   "fwd_peak_gb": float("nan"),
                   "fwd_bwd_peak_gb": float("nan"),
                   "headroom_gb": float("nan") if device_total is None else device_total}
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            try:
                model = TradeFM(cfg_run).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                _reset_peak(device)
                # dry alloc
                row["dry_alloc_gb"] = _peak_gb(device)

                toks = torch.randint(0, cfg_run.vocab, (b, ctx + 1),
                                      device=device, dtype=torch.long)
                _reset_peak(device)
                logits = model(toks[:, :-1])
                _ = logits.sum()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                row["fwd_peak_gb"] = _peak_gb(device)

                _reset_peak(device)
                for _ in range(accum_steps):
                    loss = F.cross_entropy(model(toks[:, :-1]).reshape(-1, cfg_run.vocab),
                                            toks[:, 1:].reshape(-1))
                    (loss / accum_steps).backward()
                opt.step(); opt.zero_grad()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                row["fwd_bwd_peak_gb"] = _peak_gb(device)
                if device_total is not None:
                    row["headroom_gb"] = device_total - row["fwd_bwd_peak_gb"]
            except RuntimeError as e:
                msg = str(e)
                row["status"] = "oom" if ("out of memory" in msg.lower()
                                          or "MPS backend out of memory" in msg) else "err"
                row["error"] = msg.splitlines()[0][:240]
            finally:
                try:
                    del model, opt, toks, logits, loss
                except Exception:
                    pass
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            rows.append(row)
            log.info("audit: batch=%d ctx=%d status=%s fwd=%.2fGB fwd+bwd=%.2fGB",
                     b, ctx, row["status"], row.get("fwd_peak_gb", float("nan")),
                     row.get("fwd_bwd_peak_gb", float("nan")))
            if row["status"] == "oom":
                break   # larger batches at same ctx will also OOM
    return rows


def _device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and hasattr(torch.backends, "mps") \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cli():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ctx", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    dev = _device(a.device)
    log.info("device=%s  total=%s GB", dev, _total_gb(dev))
    rows = audit(cfg, a.batch, a.ctx, dev, accum_steps=a.accum_steps)
    payload = {"device": str(dev), "total_gb": _total_gb(dev), "rows": rows}
    out = a.out or f"reports/mem_audit_{a.config.split('/')[-1].replace('.', '_')}.json"
    from pathlib import Path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2))
    log.info("written → %s", out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    _cli()
