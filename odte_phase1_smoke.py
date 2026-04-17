"""Phase-1 smoke: synth DataShop-format CSV → pack → pretrain TradeFM.

End-to-end test for the odte/data + odte/train modules. Runs locally on
Mac CPU/MPS; on cloud A100 the same commands scale up via
`python -m odte.train.pretrain_tradefm --config configs/tradefm_40m.yml ...`.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from odte.data import DataShopPacker
from odte.train import pretrain
from odte.train.pretrain_tradefm import TrainArgs

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("odte_p1_smoke")

ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "reports" / "fake_datashop"
SHARD_DIR = ROOT / "reports" / "odte_shards"
CKPT_DIR = ROOT / "checkpoints" / "tradefm_smoke"
CONFIG = ROOT / "configs" / "tradefm_40m.yml"
SMOKE_CFG = ROOT / "configs" / "tradefm_smoke.yml"


def make_fake_datashop(n_days: int = 2, rows_per_day: int = 20000,
                       seed: int = 7) -> list[Path]:
    """Write DataShop-shaped CSVs of a mock SPX 0DTE tape."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    base = pd.Timestamp("2026-04-15 09:30")
    for d in range(n_days):
        day = base + pd.Timedelta(days=d)
        ts = day + pd.to_timedelta(np.arange(rows_per_day) * 180, unit="ms")
        S = 5500 + np.cumsum(rng.normal(0, 1.2, rows_per_day))
        spread = np.abs(rng.normal(0.1, 0.05, rows_per_day)).clip(min=0.05)
        bid = S - spread / 2
        ask = S + spread / 2
        sz = rng.integers(1, 200, rows_per_day)
        strikes = 5500 + rng.integers(-100, 100, rows_per_day)
        df = pd.DataFrame({
            "underlying_symbol": "SPX",
            "quote_datetime": ts,
            "root": "SPX",
            "expiration": day.strftime("%Y-%m-%d"),
            "strike": strikes,
            "option_type": rng.choice(["C", "P"], rows_per_day),
            "open": S, "high": S, "low": S, "close": S,
            "trade_volume": rng.integers(0, 50, rows_per_day),
            "bid_size": sz, "bid": bid,
            "ask_size": sz, "ask": ask,
            "underlying_bid": S - 0.25, "underlying_ask": S + 0.25,
            "implied_volatility": rng.uniform(0.1, 0.4, rows_per_day),
            "delta": rng.uniform(0, 1, rows_per_day),
            "gamma": rng.uniform(0, 0.05, rows_per_day),
            "theta": rng.uniform(-5, 0, rows_per_day),
            "vega": rng.uniform(0, 2, rows_per_day),
            "rho": rng.uniform(-1, 1, rows_per_day),
            "open_interest": rng.integers(0, 1000, rows_per_day),
        })
        p = CSV_DIR / f"spx_{day.strftime('%Y%m%d')}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
        log.info("wrote %s (%d rows)", p, len(df))
    return paths


def write_smoke_config():
    """A tiny TradeFM config so the smoke test runs in seconds on MPS."""
    SMOKE_CFG.write_text("""d_model: 128
n_heads: 4
n_layers: 2
vocab: 64
ctx_len: 128
dropout: 0.1
fp8: false
rotary: true
use_flash_attn: false
lr: 6.0e-4
weight_decay: 0.1
warmup_steps: 10
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()

    log.info("=== generate fake DataShop CSVs ===")
    csv_paths = make_fake_datashop(n_days=args.days)

    log.info("=== pack (fit tokenizer + tokenize) ===")
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    packer = DataShopPacker(out_dir=SHARD_DIR, n_buckets=64, shard_rows=5000)
    packer.fit_tokenizer(csv_paths)
    shards = packer.pack(csv_paths)
    log.info("shards: %s", shards)

    log.info("=== pretrain smoke TradeFM ===")
    write_smoke_config()
    stats = pretrain(TrainArgs(
        shard_glob=str(SHARD_DIR / "opra_*.parquet"),
        ckpt_dir=str(CKPT_DIR),
        config_path=str(SMOKE_CFG),
        steps=args.steps, batch=8, grad_accum=1,
        ckpt_every=max(args.steps // 2, 1), log_every=max(args.steps // 10, 1),
        device=args.device, seed=0, max_shards=None,
    ))
    log.info("pretrain stats: %s", stats)


if __name__ == "__main__":
    main()
