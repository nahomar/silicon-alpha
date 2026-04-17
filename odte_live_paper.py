"""odte_live_paper.py — the live 0DTE paper runner.

Spec'd Phase-5 pipeline:

    Ingest      feed (synthetic | fmp | polygon | databento_opra | cboe_replay)
        ↓
    Predict     HybridBinTokenizer  →  TradeFM (next-token distribution)
                DMLPricer (fair value + Δ/Γ/𝒱 in one pass)
                StreamingFeatures (per-tick Numba OFI + microprice)
        ↓
    Execute     QPExecutor (broker-margin-aware μᵀw − ½λwᵀΣw, ‖w‖₁ ≤ B)
                RiskGates (time-decaying gamma + pin-score + SPAN margin)
                PaperBroker (spread-aware market fills; partial fills via
                             volume-at-strike; realized spread cost tagged)
        ↓
    Log         RollingParquetWriter (odte_orders / odte_fills); end-of-run
                post_trade analyzer runs automatically.

LIVE_TRADING is intentionally NOT wired anywhere — the only submission
path is PaperBroker. Any decision to place live orders has to go through
a separate, explicitly-reviewed PR.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from odte.accel.numba_kernels import warmup as numba_warmup
from odte.exec import (
    BrokerMarginTable, InstrumentGreeks, PaperBroker, QPExecutor,
    RiskGates, StreamingFeatures, TableLookup, write_sample_table,
)
from odte.exec.risk_gates import close_for_pennies_orders
from odte.exec.post_trade import PostTradeAnalyzer
from odte.dml_pricer import DMLPricer
from odte.transformer_tradefm import TradeFM
from odte.tokenizer import HybridBinTokenizer
from models.config import DMLConfig, TradeFMConfig

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("odte_live_paper")

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Feed factory
# ---------------------------------------------------------------------------

async def _synthetic_stream(underlying: str = "SPX",
                            ticks: int = 10_000,
                            tick_ms: int = 10) -> AsyncIterator[dict]:
    """Replays a multi-strike synthetic 0DTE chain.

    Emits 5 strikes (ATM ± 2 steps) per second so pin-risk and OI weighting
    actually have structure to test against.
    """
    from odte.synth_options import SessionSpec, generate_session
    under, trades, _chain = generate_session(SessionSpec(n_steps=max(ticks // 5, 120),
                                                          dt_seconds=1.0, seed=11),
                                              write=False)
    # 5-strike ladder around ATM
    for i, row in under.iterrows():
        S = float(row["S"])
        base_K = round(S / 5) * 5
        for dk in (-10, -5, 0, 5, 10):
            K = base_K + dk
            # toy price: intrinsic + small time value falling off with |K-S|
            tv = max(2.0 - abs(K - S) * 0.05, 0.05)
            mid = max(S - K, 0.0) + tv
            spread = max(0.05, mid * 0.036)        # 3.6% avg spread
            bid = mid - spread / 2
            ask = mid + spread / 2
            yield {
                "kind": "tob",
                "ts": int(time.time() * 1000) + i * tick_ms,
                "underlying": underlying,
                "strike": float(K),
                "expiry": "0DTE",
                "cp_flag": "C",
                "bid_px": max(bid, 0.01), "bid_sz": 40,
                "ask_px": max(ask, 0.02), "ask_sz": 40,
                "last_sz": 5.0,
                "iv": 0.2, "delta": 0.5, "gamma": 0.002, "vega": 0.8,
                "underlying_spot": S,
                "open_interest": 1000 + abs(dk) * 200,
                "minute_of_day": (i * tick_ms // 60000) % 390,
                "minutes_to_expiry": max(5.0, 60.0 - (i * tick_ms / 60000)),
            }
        await asyncio.sleep(tick_ms / 1000 / 10)
        if i * 5 >= ticks:
            break


def _make_feed(name: str, underlying: str, args) -> AsyncIterator[dict]:
    if name == "synthetic":
        return _synthetic_stream(underlying=underlying, ticks=args.max_ticks,
                                  tick_ms=10)
    if name == "fmp":
        from feeds.fmp import FMPFeed
        return FMPFeed().live_iter(underlying=underlying)
    if name == "polygon":
        from feeds.polygon_feed import PolygonFeed
        return PolygonFeed().options_live_iter(underlying=underlying)
    if name == "databento_opra":
        from feeds.databento_opra import DatabentoOPRAFeed
        return DatabentoOPRAFeed().live_iter([underlying])
    raise ValueError(f"unknown feed: {name}")


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

class Latency:
    def __init__(self, window: int = 2000):
        self.buf: deque = deque(maxlen=window)
    def record(self, ns: int) -> None:
        self.buf.append(ns)
    def snapshot(self) -> dict:
        if not self.buf:
            return {}
        arr = np.array(self.buf, dtype=np.float64) / 1000.0   # ns → µs
        return {"p50_us": float(np.median(arr)),
                "p90_us": float(np.quantile(arr, 0.90)),
                "p99_us": float(np.quantile(arr, 0.99)),
                "n": len(arr)}


# ---------------------------------------------------------------------------
# Inference stack
# ---------------------------------------------------------------------------

class InferenceStack:
    """TradeFM + DMLPricer + HybridBinTokenizer wrapper."""

    def __init__(self, dml_ckpt: Optional[Path], tradefm_ckpt: Optional[Path],
                 tokenizer_path: Optional[Path], device: str):
        self.device = device
        self.dml = DMLPricer(DMLConfig()).to(device)
        if dml_ckpt and Path(dml_ckpt).exists():
            blob = torch.load(Path(dml_ckpt), map_location=device)
            self.dml.load_state_dict(blob.get("state", blob))
            log.info("loaded DML ckpt %s", dml_ckpt)
        self.dml.eval()

        self.tradefm = None
        if tradefm_ckpt and Path(tradefm_ckpt).exists():
            blob = torch.load(Path(tradefm_ckpt), map_location=device)
            cfg = TradeFMConfig(**blob["cfg"]) if "cfg" in blob else TradeFMConfig.mini()
            self.tradefm = TradeFM(cfg).to(device)
            self.tradefm.load_state_dict(blob.get("state", blob))
            self.tradefm.eval()
            log.info("loaded TradeFM ckpt %s  (%d params)",
                     tradefm_ckpt, self.tradefm.num_params())

        self.tok = None
        if tokenizer_path and Path(tokenizer_path).exists():
            self.tok = HybridBinTokenizer.load(Path(tokenizer_path))
            log.info("loaded tokenizer %s  (features=%s)",
                     tokenizer_path, list(self.tok.feature_spec))

    def predict(self, feat_row: dict,
                 tok_ctx: Optional[list] = None) -> dict:
        """Return {mu_next_token, dml_price, delta, gamma, vega}.

        TradeFM runs under torch.no_grad() for speed; DMLPricer requires
        grad-enabled tensors for the AAD Greek computation.
        """
        out = {}
        if self.tradefm is not None and tok_ctx is not None and len(tok_ctx) > 1:
            with torch.no_grad():
                toks = torch.tensor([tok_ctx], dtype=torch.long, device=self.device)
                logits = self.tradefm(toks)[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                out["argmax_token"] = int(probs.argmax(dim=-1).item())
                vocab = probs.shape[-1]
                idx = torch.arange(vocab, device=self.device, dtype=torch.float32)
                mean_tok = float((probs * idx).sum().item())
                out["mu_next"] = (mean_tok - vocab / 2) / (vocab / 2)
        else:
            out["mu_next"] = 0.0

        if self.dml is not None and "spot" in feat_row and "strike" in feat_row:
            # DML needs grad for AAD — deliberately NOT under no_grad().
            S = torch.tensor([feat_row["spot"]], dtype=torch.float32, device=self.device)
            K = torch.tensor([feat_row["strike"]], dtype=torch.float32, device=self.device)
            tau_y = max(feat_row.get("minutes_to_expiry", 30) * 60.0 / (365 * 24 * 3600), 1e-9)
            tau = torch.tensor([tau_y], dtype=torch.float32, device=self.device)
            r = torch.tensor([0.0], dtype=torch.float32, device=self.device)
            sig = torch.tensor([feat_row.get("iv", 0.2)], dtype=torch.float32, device=self.device)
            with torch.enable_grad():
                p, d, g, v = self.dml(S, K, tau, r, sig)
            out.update(dml_price=float(p.item()), delta=float(d.item()),
                       gamma=float(g.item()), vega=float(v.item()))
        return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run(args) -> dict:
    numba_warmup()

    table_path = Path(args.margin_table)
    if not table_path.exists():
        log.info("no broker table at %s — writing sample", table_path)
        write_sample_table(table_path)
    broker_table = BrokerMarginTable(path=table_path)

    qp = QPExecutor(lam=args.risk_aversion, B=args.gross_budget,
                    broker=broker_table)
    gates = RiskGates(
        pin_dist_cap=args.pin_dist_cap,
        orders_per_sec_cap=args.orders_per_sec_cap,
        notional_velocity_cap=args.notional_velocity_cap,
        gamma_dollar_cap=args.gamma_dollar_cap,
        equity_buffer_pct=args.equity_buffer_pct,
    )
    paper = PaperBroker(fill_rate_per_tick=args.fill_rate_per_tick,
                        penny_threshold=args.penny_threshold)
    feats = StreamingFeatures()
    inf = InferenceStack(
        dml_ckpt=Path(args.dml_ckpt) if args.dml_ckpt else None,
        tradefm_ckpt=Path(args.tradefm_ckpt) if args.tradefm_ckpt else None,
        tokenizer_path=Path(args.tokenizer) if args.tokenizer else None,
        device=args.device,
    )

    latency = Latency()
    equity = float(args.equity)
    t_start = time.time()
    ticks = orders_submitted = orders_filled = orders_vetoed = penny_flatten = 0

    # Position book + latest bid/ask per symbol for close_for_pennies
    positions: dict = {}
    last_bid: dict = {}; last_ask: dict = {}

    # Rolling token context (one per underlying)
    tok_ctx: list = []

    feed = _make_feed(args.feed, args.underlying, args)
    async for ev in feed:
        t0 = time.perf_counter_ns()
        if ev.get("kind") != "tob" or ev.get("bid_px") is None:
            continue

        key = f"{ev['underlying']}/{int(ev.get('strike', 0))}{ev.get('cp_flag','')}"
        feat = feats.update(key, float(ev["bid_px"]), float(ev["bid_sz"]),
                             float(ev["ask_px"]), float(ev["ask_sz"]))
        last_bid[key] = float(ev["bid_px"]); last_ask[key] = float(ev["ask_px"])

        # token context update
        if inf.tok is not None:
            try:
                tok = feats.encode_token(
                    feat["ofi"],
                    inf.tok.edges.get("ret") if "ret" in inf.tok.edges
                    else next(iter(inf.tok.edges.values())),
                )
                tok_ctx.append(int(tok))
                if len(tok_ctx) > 256:
                    tok_ctx = tok_ctx[-256:]
            except Exception:
                pass

        pred = inf.predict({
            "spot": float(ev.get("underlying_spot",
                                  (ev["bid_px"] + ev["ask_px"]) / 2)),
            "strike": float(ev["strike"]),
            "iv": float(ev.get("iv", 0.2)),
            "minutes_to_expiry": float(ev.get("minutes_to_expiry", 60)),
        }, tok_ctx=tok_ctx)

        spot = float(ev.get("underlying_spot",
                            (ev["bid_px"] + ev["ask_px"]) / 2))
        g = InstrumentGreeks(
            spot=np.array([spot]),
            delta=np.array([pred.get("delta", ev.get("delta", 0.5))]),
            gamma=np.array([pred.get("gamma", ev.get("gamma", 0.002))]),
            vega=np.array([pred.get("vega", ev.get("vega", 0.5))]),
            multiplier=np.array([100.0]),
        )
        # Combine per-tick OFI and TradeFM token drift into expected-return.
        mu_ofi = feat["ofi"] * 1e-4
        mu_tx = pred.get("mu_next", 0.0) * 1e-3
        mu = np.array([mu_ofi + mu_tx])
        Sigma = np.array([[max(feat["micro"], 1e-6) * 1e-6]])

        lookup = TableLookup(
            underlying=ev["underlying"],
            maturity_bucket="0dte",
            iv_regime="normal",
            minute_of_day=int(ev.get("minute_of_day", 0)),
        )
        res = qp.solve(mu, Sigma, g, equity=equity, lookup=lookup)

        gate = gates.check(
            res.w, g, res.required_margin, equity,
            strikes=np.array([ev["strike"]]),
            spot=spot,
            minute_of_day=int(ev.get("minute_of_day", 0)),
            minutes_to_expiry=float(ev.get("minutes_to_expiry", 9999)),
            open_interest=np.array([ev.get("open_interest", 0)]),
            symbols=[key],
        )
        latency.record(time.perf_counter_ns() - t0)

        if not gate.ok:
            orders_vetoed += 1
            # Pin-score veto → auto flatten via close_for_pennies
            if "pin_score_block" in gate.failed and gate.recommend_close_for_pennies:
                for spec in close_for_pennies_orders(
                    gate.recommend_close_for_pennies, positions,
                    last_bid, last_ask, penny_threshold=args.penny_threshold):
                    sym = spec["symbol"]
                    o = paper.submit_market(
                        sym, spec["side"], spec["qty"],
                        current_bid=last_bid.get(sym, 0.0),
                        current_ask=last_ask.get(sym, 0.0),
                        volume_at_strike=float(ev.get("bid_sz", 40)),
                        reason=spec["reason"],
                    )
                    penny_flatten += 1
                    positions[sym] = positions.get(sym, 0) \
                        + (o.filled_qty if spec["side"] == "buy"
                            else -o.filled_qty)
        elif abs(res.w[0]) > 1e-3:
            side = "buy" if res.w[0] > 0 else "sell"
            o = paper.submit_market(
                key, side, abs(res.w[0]),
                current_bid=float(ev["bid_px"]),
                current_ask=float(ev["ask_px"]),
                volume_at_strike=float(ev["ask_sz"] if side == "buy" else ev["bid_sz"]),
                reason="signal",
            )
            orders_submitted += 1
            if o.status in ("filled", "partial"):
                positions[key] = positions.get(key, 0.0) \
                    + (o.filled_qty if side == "buy" else -o.filled_qty)
                if o.status == "filled":
                    orders_filled += 1

        # Top up any partial fills triggered by the new tick
        paper.on_tob(key, float(ev["bid_px"]), float(ev["ask_px"]),
                     ev["ts"], volume_at_strike=float(ev.get("bid_sz", 40)))

        ticks += 1
        if ticks % args.log_every == 0:
            lat = latency.snapshot()
            log.info(
                "tick=%d  lat p50=%.1fµs p99=%.1fµs  "
                "orders=%d filled=%d vetoed=%d penny=%d  margin=$%.0f",
                ticks, lat.get("p50_us", 0), lat.get("p99_us", 0),
                orders_submitted, orders_filled, orders_vetoed,
                penny_flatten, res.required_margin,
            )
        if ticks >= args.max_ticks:
            break

    paper.close()
    elapsed = time.time() - t_start
    summary = {
        "feed": args.feed, "underlying": args.underlying,
        "ticks": ticks, "orders_submitted": orders_submitted,
        "orders_filled": orders_filled, "orders_vetoed": orders_vetoed,
        "penny_flatten": penny_flatten,
        "latency_us": latency.snapshot(),
        "elapsed_s": elapsed, "ticks_per_s": ticks / max(elapsed, 1e-9),
        "broker_table": str(table_path),
        "positions_end": positions,
    }
    out = ROOT / "reports" / f"odte_live_paper_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    log.info("summary → %s", out)

    # Post-trade analyzer runs automatically
    if args.post_trade:
        try:
            a = PostTradeAnalyzer(hours=1)
            pt = a.run()
            rpt = a.write_report(pt)
            log.info("post-trade report → %s", rpt)
            summary["post_trade"] = {k: pt[k] for k in
                                      ("n_fills", "n_orders", "mean_spread_bps",
                                       "per_horizon") if k in pt}
        except Exception as e:
            log.warning("post-trade failed: %s", e)

    print(json.dumps(summary, indent=2, default=str))
    return summary


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", choices=["synthetic", "fmp", "polygon",
                                        "databento_opra"],
                    default="synthetic")
    ap.add_argument("--underlying", default="SPX")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dml-ckpt", default="checkpoints/dml_pricer.pt")
    ap.add_argument("--tradefm-ckpt", default="checkpoints/mini_tradefm.pt")
    ap.add_argument("--tokenizer", default="checkpoints/hybrid_tokenizer.json")
    ap.add_argument("--margin-table",
                    default="configs/broker_margin_example.yml")
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--risk-aversion", type=float, default=5.0)
    ap.add_argument("--gross-budget", type=float, default=1.0)
    ap.add_argument("--max-ticks", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--post-trade", action="store_true",
                    help="run the post-trade analyzer after the session")
    # Risk gate tuning
    ap.add_argument("--pin-dist-cap", type=float, default=0.001)
    ap.add_argument("--orders-per-sec-cap", type=int, default=20)
    ap.add_argument("--notional-velocity-cap", type=float, default=2e6)
    ap.add_argument("--gamma-dollar-cap", type=float, default=5e7)
    ap.add_argument("--equity-buffer-pct", type=float, default=0.95)
    # Broker tuning
    ap.add_argument("--fill-rate-per-tick", type=float, default=0.10)
    ap.add_argument("--penny-threshold", type=float, default=0.05)
    a = ap.parse_args()
    asyncio.run(run(a))


if __name__ == "__main__":
    _cli()
