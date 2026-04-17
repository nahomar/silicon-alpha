"""End-to-end 0DTE smoke test.

Pipeline:
  1. synth_options.generate_session → underlying + trade tape + chain
  2. HybridBinTokenizer fit + encode (quantile price, log volume/dt)
  3. MiniTradeFM one-epoch LM-loss training on the token stream
  4. DMLPricer pretrain on Black-Scholes; report Greek error at ATM τ=1d
  5. WorldSim 1k-step rollout with informed/uninformed Hawkes flow
  6. DeterministicExecutor solves a tiny portfolio over a fake chain
  7. Print PnL/Sharpe/turnover for the toy "quote and re-hedge" strategy

Runs on Mac MPS / CPU in ~minutes. This is a smoke test for wiring,
NOT a backtest for alpha.
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

import torch

from odte import best_device
from odte.synth_options import SessionSpec, generate_session
from odte.tokenizer import HybridBinTokenizer
from odte.transformer_tradefm import MiniTradeFM, TradeFM, wrap_fp8_autocast
from odte.dml_pricer import DMLPricer, train_dml_bs, greek_error_on_atm
from odte.world_sim import WorldSim
from odte.executor import DeterministicExecutor, RiskGates
from models.config import DMLConfig, WorldSimConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("odte_smoke")

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
REPORT_DIR = ROOT / "reports"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _build_stream_df(under: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Join underlying + trades into a unified event stream for tokenization."""
    df = trades.copy()
    # last_px, last_sz are in trades already
    df["mid"] = df["last_px"]
    # ret = log-return vs previous trade price
    df["ret"] = np.log(df["last_px"]).diff().fillna(0.0)
    df["micro_dev"] = 0.0  # no micro-vs-mid in the synth; leave zero for quantile path
    df["spread"] = np.abs(df["last_px"]) * 1e-4    # ~1 bp placeholder
    df["bid_sz"] = df["last_sz"]
    df["ask_sz"] = df["last_sz"]
    df["inter_arrival_ms"] = (df["ts_sec"].diff().fillna(0.001) * 1000).clip(lower=1e-3)
    return df


def _train_tradefm(tokens: np.ndarray, device: str, steps: int = 200,
                   batch: int = 16, ctx: int = 128):
    model = MiniTradeFM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=model.cfg.lr,
                            weight_decay=model.cfg.weight_decay)
    # Build rolling context windows from the flat token stream.
    # tokens shape: (T, F) -> flatten per feature into one sequence per channel.
    flat = tokens.reshape(-1)
    n = (len(flat) // ctx) * ctx
    arr = flat[:n].reshape(-1, ctx).astype(np.int64)
    n_win = arr.shape[0]
    if n_win < 2:
        log.warning("not enough tokens for TradeFM training")
        return model, []
    losses = []
    log.info("TradeFM: %d params, %d windows of ctx=%d", model.num_params(), n_win, ctx)
    for step in range(steps):
        idx = np.random.randint(0, n_win, size=batch)
        batch_tok = torch.tensor(arr[idx], dtype=torch.long, device=device)
        with wrap_fp8_autocast():
            loss = model.loss(batch_tok)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 10) == 0:
            log.info("tradefm step %d  loss=%.4f", step, float(loss.item()))
        losses.append(float(loss.item()))
    return model, losses


def _strategy_pnl(world_df: pd.DataFrame, executor: DeterministicExecutor,
                  n_opts: int = 21) -> dict:
    """Toy: each simulation tick, solve executor with random μ from toxicity proxy."""
    rng = np.random.default_rng(0)
    rets = []
    turnovers = []
    prev_w = np.zeros(n_opts)
    # Crude "predicted return" per option = underlying return times a per-option
    # delta proxy (linear in strike rank).
    deltas = np.linspace(0.9, -0.9, n_opts)
    for i in range(len(world_df) - 1):
        underlying_ret = (world_df["mid"].iloc[i + 1] - world_df["mid"].iloc[i]) \
            / max(world_df["mid"].iloc[i], 1e-9)
        mu = deltas * underlying_ret + 1e-4 * rng.standard_normal(n_opts)
        # Covariance: diagonal scaled by |delta|
        Sigma = np.diag(np.abs(deltas) * 1e-4) + 1e-6 * np.eye(n_opts)
        w = executor.solve(mu, Sigma, B=1.0, lam=5.0)
        # Realized per-unit return (toy): mu.dot(w)
        rets.append(float(mu @ w))
        turnovers.append(float(np.abs(w - prev_w).sum()))
        prev_w = w
    rets = np.array(rets)
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252 * 390) if len(rets) > 1 else 0.0
    return {"pnl": float(rets.sum()), "sharpe": float(sharpe),
            "turnover": float(np.mean(turnovers)), "n_steps": int(len(rets))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=best_device())
    ap.add_argument("--synth-days", type=int, default=1,
                    help="number of simulated sessions")
    ap.add_argument("--mini", action="store_true", help="smaller sim for speed")
    ap.add_argument("--tradefm-steps", type=int, default=100)
    ap.add_argument("--dml-steps", type=int, default=400)
    args = ap.parse_args()

    log.info("device=%s mini=%s", args.device, args.mini)
    spec = SessionSpec(n_steps=1200 if args.mini else 3600, dt_seconds=6.0, seed=7)

    all_trades = []
    for d in range(args.synth_days):
        spec.seed = 7 + d
        under, trades, chain = generate_session(spec, write=True)
        all_trades.append(trades)
    trades = pd.concat(all_trades, ignore_index=True)
    log.info("synth trades=%d", len(trades))

    stream = _build_stream_df(under, trades)
    tok = HybridBinTokenizer(n_buckets=64, feature_spec={
        "ret": "quantile", "mid": "quantile", "micro_dev": "quantile",
        "spread": "log", "bid_sz": "log", "ask_sz": "log",
        "inter_arrival_ms": "log",
    })
    tok.fit(stream)
    tok.save(CKPT_DIR / "hybrid_tokenizer.json")
    tokens = tok.tokenize_batch(stream, feature_order=["ret", "spread", "inter_arrival_ms"])
    log.info("tokens shape=%s  vocab=%d", tokens.shape, tok.n_buckets)

    model, tx_losses = _train_tradefm(tokens, args.device, steps=args.tradefm_steps)
    if tx_losses:
        torch.save(model.state_dict(), CKPT_DIR / "mini_tradefm.pt")

    log.info("training DML pricer (%d steps)…", args.dml_steps)
    pricer = DMLPricer(DMLConfig())
    train_dml_bs(pricer, steps=args.dml_steps, batch=512, device=args.device,
                 S_ref=float(stream["mid"].median()))
    torch.save(pricer.state_dict(), CKPT_DIR / "dml_pricer.pt")
    greek_err = greek_error_on_atm(pricer, S=float(stream["mid"].median()),
                                   tau_days=1.0, device=args.device)
    log.info("DML Greek error (ATM τ=1d): %s", greek_err)

    log.info("world sim rollout…")
    sim = WorldSim(cfg=WorldSimConfig(n_ticks=1000 if args.mini else 3000), seed=11)
    world_df = sim.run()
    log.info("world sim rows=%d", len(world_df))

    log.info("executor smoke…")
    executor = DeterministicExecutor()
    strat = _strategy_pnl(world_df, executor, n_opts=21)
    log.info("strategy: %s", strat)

    risk = RiskGates()
    # Fake greeks/strikes for the smoke risk check:
    fake_greeks = {"gamma": np.linspace(0.01, 0.05, 21),
                   "vega":  np.linspace(0.5, 2.0, 21)}
    fake_strikes = np.linspace(5390, 5610, 21)
    margins = np.ones(21) * 100.0
    fake_w = np.zeros(21); fake_w[10] = 0.5
    gates = risk.check(fake_w, fake_greeks, fake_strikes,
                       spot=float(stream["mid"].iloc[-1]),
                       margins=margins)
    log.info("risk gates: %s", gates)

    report = REPORT_DIR / f"odte_smoke_{time.strftime('%Y%m%dT%H%M%S')}.md"
    report.write_text(f"""# 0DTE smoke test — {time.strftime('%Y-%m-%d %H:%M')}

- device: `{args.device}`
- trades simulated: {len(trades)}
- tokens shape: {tokens.shape}
- TradeFM params: {model.num_params():,} (Mini)
- TradeFM final loss: {tx_losses[-1] if tx_losses else 'n/a'}
- DML Greek error (ATM τ=1d): {greek_err}
- world sim rows: {len(world_df)}
- strategy pnl: {strat['pnl']:+.4f}  sharpe: {strat['sharpe']:+.2f}  turnover: {strat['turnover']:.3f}
- risk gates: {gates}

Artifacts:
- checkpoints/hybrid_tokenizer.json
- checkpoints/mini_tradefm.pt
- checkpoints/dml_pricer.pt

This is a wiring check, NOT a backtest for alpha. See the plan at
~/.claude/plans/playful-sparking-sloth.md for the Phase 1+ path to real data.
""")
    log.info("report → %s", report)


if __name__ == "__main__":
    main()
