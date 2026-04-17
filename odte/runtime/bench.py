"""Runtime latency benchmark.

Reports p50/p90/p99 next-token latency for a checkpoint on the current
device. On Mac the "backend" line will say cpu-fallback; on an H100 box
with the built extension it will say cuda-persistent and should hit
the 4.6–15.8 µs target.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from odte.kernels import HAS_CUDA_PERSISTENT_DECODE, PersistentDecoder

log = logging.getLogger(__name__)


@dataclass
class RuntimeBench:
    ckpt_path: Path
    device: str = "cpu"
    ctx: int = 512
    vocab: int = 64
    n_iters: int = 1000

    def run(self) -> dict:
        pd = PersistentDecoder(ckpt_path=self.ckpt_path, device=self.device,
                               ctx_len=self.ctx)
        stats = pd.bench(n_iters=self.n_iters, ctx=self.ctx, vocab=self.vocab)
        stats["hopper_native"] = HAS_CUDA_PERSISTENT_DECODE
        return stats


def _cli():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--out", default="reports/runtime_bench.json")
    a = ap.parse_args()
    bench = RuntimeBench(ckpt_path=Path(a.ckpt), device=a.device,
                         ctx=a.ctx, vocab=a.vocab, n_iters=a.iters)
    stats = bench.run()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _cli()
