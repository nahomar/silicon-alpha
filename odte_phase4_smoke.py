"""Phase-4 smoke: DML BS→Heston fine-tune + adversarial RL loop.

Runs quickly on Mac MPS:
  1. Pretrains DMLPricer on BS (2000 steps).
  2. Fine-tunes on Heston MC payoffs (100 steps, 500 paths each).
  3. Trains 2 informed + 4 uninformed PPO agents vs. an A-S market maker
     inside WorldSim-lite for 2048 env steps.
  4. Reports before/after Greek error, MM PnL distribution, policy-loss
     trend per agent group.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.config import DMLConfig
from odte.dml_pricer import DMLPricer, greek_error_on_atm
from odte.train.train_dml import pretrain_bs, finetune_heston
from odte.rl import adversarial_train

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("odte_p4_smoke")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--bs-steps", type=int, default=1200)
    ap.add_argument("--heston-steps", type=int, default=60)
    ap.add_argument("--heston-paths", type=int, default=300)
    ap.add_argument("--rl-steps", type=int, default=2048)
    args = ap.parse_args()

    log.info("=== DML pretrain on BS ===")
    pricer = DMLPricer(DMLConfig())
    pretrain_bs(pricer, steps=args.bs_steps, batch=512, device=args.device)
    err_bs = greek_error_on_atm(pricer, device=args.device)
    log.info("after BS pretrain: ATM τ=1d Greek error = %s", err_bs)

    log.info("=== DML fine-tune on Heston MC ===")
    finetune_heston(pricer, steps=args.heston_steps, batch=32,
                    device=args.device, n_mc_paths=args.heston_paths,
                    n_mc_steps=24, lr=1e-4)
    err_heston = greek_error_on_atm(pricer, device=args.device)
    log.info("after Heston fine-tune: ATM τ=1d Greek error = %s", err_heston)

    log.info("=== Adversarial RL (informed vs uninformed) ===")
    rl_result = adversarial_train(
        n_informed=2, n_uninformed=4,
        steps_per_epoch=args.rl_steps, epochs=1,
        device=args.device, seed=7,
    )
    # Summarize per-agent policy loss trend (last quartile)
    per = {"informed": [], "uninformed": []}
    for r in rl_result["agent_updates"]:
        per[r["agent"]].append(r["policy"])
    summary = {k: {"n_updates": len(v),
                   "last_policy": float(np.mean(v[-max(1, len(v) // 4):])) if v else None}
               for k, v in per.items()}
    log.info("agent policy-loss trailing mean: %s", summary)

    out = {
        "dml_greek_err_after_bs": err_bs,
        "dml_greek_err_after_heston": err_heston,
        "rl_mm_terminal_pnl": rl_result["mm_terminal_pnl"],
        "rl_mm_pnl_mean": rl_result["mm_pnl_mean"],
        "rl_mm_pnl_std": rl_result["mm_pnl_std"],
        "rl_mm_inv_end": rl_result["mm_inv"],
        "rl_n_steps": rl_result["total_steps"],
        "rl_agent_summary": summary,
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    p = Path("reports") / f"odte_phase4_smoke_{time.strftime('%Y%m%dT%H%M%S')}.json"
    p.write_text(json.dumps(out, indent=2))
    log.info("smoke summary → %s", p)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
