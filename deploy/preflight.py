"""Preflight go/no-go — Monday 0DTE morning.

Runs every gate that must pass BEFORE the 9:30 bell:

  1. Hopper kernels importable + persistent decode responds
  2. DML checkpoint loads, Greeks within tolerance on ATM τ=1d
  3. TradeFM checkpoint loads, vocab matches tokenizer
  4. HybridBinTokenizer refit output exists and has current-day mtime
  5. Broker margin table loads, crisis multiplier present
  6. P99 end-to-end latency ≤ 25 µs on the current device
  7. Directional accuracy ≥ 53 % on last 30 min of pre-market tape
  8. NCCL env vars set (warn if missing; fatal only if torchrun needed)

Exit codes:
  0 → all green, safe to launch
  1 → one or more soft gates failed (warnings logged)
  2+→ hard failure; launch script should abort
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("preflight")


@dataclass
class GateCheck:
    name: str
    passed: bool
    fatal: bool
    details: str = ""

    def __str__(self):
        mark = "✅" if self.passed else ("❌" if self.fatal else "⚠ ")
        return f"{mark} [{self.name}] {self.details}"


@dataclass
class Preflight:
    require_kernels: bool = True
    require_rdma: bool = False
    require_latency_us: float = 25.0
    require_dir_acc: float = 0.53
    tokenizer_path: Path = ROOT / "checkpoints" / "hybrid_tokenizer_monday.json"
    dml_ckpt: Path = ROOT / "checkpoints" / "dml_pricer.pt"
    tradefm_ckpt: Path = ROOT / "checkpoints" / "tradefm_524m.pt"
    mini_tradefm_ckpt: Path = ROOT / "checkpoints" / "mini_tradefm.pt"
    margin_table: Path = ROOT / "configs" / "broker_margin_live.yml"
    results: List[GateCheck] = field(default_factory=list)

    # -------- checks ----------------------------------------------------
    def _check_kernels(self) -> GateCheck:
        try:
            importlib.import_module("odte_kernels_cu")
            return GateCheck("hopper_kernels", True,
                             fatal=self.require_kernels,
                             details="odte_kernels_cu imported")
        except Exception as e:
            return GateCheck("hopper_kernels", False,
                             fatal=self.require_kernels,
                             details=f"import failed: {e}")

    def _check_rdma(self) -> GateCheck:
        rdma = os.environ.get("ODTE_HAS_RDMA") == "1"
        return GateCheck("rdma", rdma, fatal=self.require_rdma,
                         details="ODTE_HAS_RDMA=1" if rdma else "RDMA disabled")

    def _check_nccl_env(self) -> GateCheck:
        need = ["NCCL_SOCKET_IFNAME", "NCCL_DEBUG"]
        missing = [k for k in need if not os.environ.get(k)]
        return GateCheck("nccl_env", not missing, fatal=False,
                         details=f"missing={missing}" if missing
                         else "NCCL_SOCKET_IFNAME + NCCL_DEBUG set")

    def _check_dml(self) -> GateCheck:
        from odte.dml_pricer import DMLPricer, greek_error_on_atm
        from models.config import DMLConfig
        try:
            import torch
            if not self.dml_ckpt.exists():
                return GateCheck("dml_ckpt", False, fatal=True,
                                 details=f"missing {self.dml_ckpt}")
            device = "cuda" if torch.cuda.is_available() \
                else ("mps" if getattr(torch.backends, "mps", None)
                       and torch.backends.mps.is_available() else "cpu")
            p = DMLPricer(DMLConfig()).to(device)
            blob = torch.load(self.dml_ckpt, map_location=device)
            p.load_state_dict(blob.get("state", blob))
            err = greek_error_on_atm(p, device=device)
            worst = max(err.values())
            ok = worst < 2.0      # 2% Greek-error gate
            return GateCheck("dml_greeks", ok, fatal=True,
                             details=f"max Greek err ATM τ=1d = {worst:.3f}%  {err}")
        except Exception as e:
            return GateCheck("dml_greeks", False, fatal=True, details=str(e))

    def _check_tradefm(self) -> GateCheck:
        try:
            import torch
            from odte.transformer_tradefm import TradeFM
            from models.config import TradeFMConfig
            ck = self.tradefm_ckpt if self.tradefm_ckpt.exists() \
                else self.mini_tradefm_ckpt
            if not ck.exists():
                return GateCheck("tradefm_ckpt", False, fatal=True,
                                 details="no TradeFM ckpt (524M or mini)")
            blob = torch.load(ck, map_location="cpu")
            cfg = TradeFMConfig(**blob["cfg"]) if "cfg" in blob else TradeFMConfig.mini()
            m = TradeFM(cfg)
            m.load_state_dict(blob.get("state", blob))
            return GateCheck("tradefm_ckpt", True, fatal=True,
                             details=f"loaded {ck.name}  {m.num_params():,} params")
        except Exception as e:
            return GateCheck("tradefm_ckpt", False, fatal=True, details=str(e))

    def _check_tokenizer_monday(self) -> GateCheck:
        if not self.tokenizer_path.exists():
            return GateCheck("tokenizer_monday", False, fatal=True,
                             details=f"missing {self.tokenizer_path}")
        mtime = self.tokenizer_path.stat().st_mtime
        age_h = (time.time() - mtime) / 3600
        ok = age_h < 12
        return GateCheck("tokenizer_monday", ok, fatal=ok is False,
                         details=f"age={age_h:.1f}h  (refit must be < 12h)")

    def _check_margin(self) -> GateCheck:
        try:
            from odte.exec.broker_margin import BrokerMarginTable, TableLookup
            if not self.margin_table.exists():
                return GateCheck("margin_table", False, fatal=True,
                                 details=f"missing {self.margin_table}")
            t = BrokerMarginTable(path=self.margin_table)
            q = TableLookup(underlying="SPX", maturity_bucket="0dte",
                            iv_regime="crisis", minute_of_day=360)
            c = t.resolve(q)
            # crisis multiplier must scale k_gamma above baseline
            ok = c.k_gamma >= 0.002
            return GateCheck("margin_table", ok, fatal=True,
                             details=f"SPX 0dte crisis k_gamma={c.k_gamma:.4f}  (≥0.002 req)")
        except Exception as e:
            return GateCheck("margin_table", False, fatal=True, details=str(e))

    def _check_latency(self) -> GateCheck:
        try:
            import torch
            from odte.kernels import PersistentDecoder
            ck = self.mini_tradefm_ckpt if self.mini_tradefm_ckpt.exists() \
                else self.tradefm_ckpt
            if not ck.exists():
                return GateCheck("latency_p99", False, fatal=True,
                                 details="no TradeFM ckpt for bench")
            dev = "cuda" if torch.cuda.is_available() \
                else ("mps" if getattr(torch.backends, "mps", None)
                       and torch.backends.mps.is_available() else "cpu")
            pd = PersistentDecoder(ckpt_path=ck, device=dev, ctx_len=128)
            stats = pd.bench(n_iters=300, ctx=128, vocab=64)
            p99 = stats["p99_us"]
            ok = p99 <= self.require_latency_us
            fatal = True
            return GateCheck("latency_p99", ok, fatal=fatal,
                             details=f"{stats['backend']}  p99={p99:.1f}µs  "
                                     f"target ≤{self.require_latency_us}µs")
        except Exception as e:
            return GateCheck("latency_p99", False, fatal=True, details=str(e))

    def _check_directional_accuracy(self) -> GateCheck:
        """Live directional accuracy on last 30m of pre-market tape.

        Reads recent odte_fills parquet (populated by a pre-market warmup
        run or prior paper sessions) and asks the post-trade analyzer for
        the 1s-horizon hit rate. Fails closed if no data.
        """
        try:
            from odte.exec.post_trade import PostTradeAnalyzer
            summary = PostTradeAnalyzer(hours=1).run()
            if not summary.get("n_fills"):
                return GateCheck("directional_acc", False, fatal=True,
                                 details="no fills in last 1h — run warmup session first")
            hzn = summary.get("per_horizon", [])
            hit = next((r["directional_hit_rate"] for r in hzn
                        if r.get("horizon_ms") == 1000), None)
            if hit is None:
                return GateCheck("directional_acc", False, fatal=True,
                                 details="no 1s horizon data")
            ok = hit >= self.require_dir_acc
            return GateCheck("directional_acc", ok, fatal=True,
                             details=f"1s hit-rate={hit*100:.1f}%  target ≥{self.require_dir_acc*100:.0f}%")
        except Exception as e:
            return GateCheck("directional_acc", False, fatal=True, details=str(e))

    # -------- runner ---------------------------------------------------
    def run(self, full: bool = True) -> int:
        checks: List[Callable[[], GateCheck]] = [
            self._check_nccl_env,
            self._check_rdma,
            self._check_kernels,
            self._check_dml,
            self._check_tradefm,
            self._check_tokenizer_monday,
            self._check_margin,
            self._check_latency,
        ]
        if full:
            checks.append(self._check_directional_accuracy)
        for c in checks:
            r = c()
            self.results.append(r)
            log.info("%s", r)

        hard = [r for r in self.results if not r.passed and r.fatal]
        soft = [r for r in self.results if not r.passed and not r.fatal]

        log.info("=" * 60)
        log.info("Preflight: %d hard fails, %d soft warnings, %d passed",
                 len(hard), len(soft),
                 sum(1 for r in self.results if r.passed))
        if hard:
            log.error("HARD FAILURES — abort launch:")
            for r in hard:
                log.error("  %s", r)
            return 2
        if soft:
            log.warning("Soft warnings — review before launch")
            return 1
        log.info("ALL GREEN — safe to launch")
        return 0


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also run directional-accuracy check (needs warmup fills)")
    ap.add_argument("--require-rdma", action="store_true")
    ap.add_argument("--latency-us", type=float, default=25.0)
    ap.add_argument("--dir-acc", type=float, default=0.53)
    a = ap.parse_args()
    pf = Preflight(require_rdma=a.require_rdma,
                   require_latency_us=a.latency_us,
                   require_dir_acc=a.dir_acc)
    rc = pf.run(full=a.full)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    out = ROOT / "reports" / f"preflight_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps([{"name": r.name, "passed": r.passed,
                                 "fatal": r.fatal, "details": r.details}
                                for r in pf.results], indent=2))
    sys.exit(rc)


if __name__ == "__main__":
    _cli()
