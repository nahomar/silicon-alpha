"""End-to-end MM stack smoke test on synthetic LOB data.

Runs the full loop: synth book → features → toxicity → predictor → PRISM
quoter → fills sim → markouts → summary.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mm import (
    simulate_book, microprice_features, vpin, realized_variance, variance_ratio,
    ShortHorizonPredictor, PRISM, MarketMakingEnv, compute_markouts,
)
from mm.avellaneda_stoikov import ASParams
from mm.prism import PRISMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("mm_smoke")


def main():
    log.info("Simulating 20,000-tick synthetic book…")
    book = simulate_book(n_steps=20_000, seed=7)

    log.info("Computing features…")
    feats = microprice_features(book)
    trades = book.dropna(subset=["last_px"]).copy()
    tox = pd.DataFrame({
        "vpin": vpin(trades).reindex(feats.index).ffill().fillna(0.0),
        "rv":   realized_variance(feats["mid"]).fillna(0.0),
        "vr":   variance_ratio(feats["mid"]).fillna(1.0),
    }, index=feats.index)

    log.info("Training short-horizon predictor…")
    pred = ShortHorizonPredictor(horizon_steps=5)
    X = pred.build_features(feats, tox)
    y = pred.build_target(feats["mid"])
    split = int(0.7 * len(X))
    pred.fit(X.iloc[:split], y.iloc[:split])

    log.info("Running PRISM on held-out segment…")
    prism = PRISM(
        predictor=pred,
        cfg=PRISMConfig(
            vpin_widen=0.35, vpin_pull=0.65, sigma_halt=5e-2,
            # kappa high → tight quotes; gamma small → low risk-aversion.
            # Sigma in price units (synth book stdev of mid increments ≈ 0.02).
            as_params=ASParams(gamma=0.01, sigma=0.02, kappa=200.0,
                               horizon=1.0, inv_limit=50, tick_size=0.01),
        ),
    )
    window = 32
    fills = []
    test_start = split
    for i in range(test_start + window, len(book) - 1):
        tob_window = book.iloc[i - window: i][["bid_px", "bid_sz", "ask_px", "ask_sz"]]
        trades_window = book.iloc[i - 2000 if i > 2000 else 0: i].dropna(subset=["last_px"])
        q = prism.decide(tob_window, trades_window)
        if "pulled" in q:
            continue
        # fill simulation vs next trade
        nxt = book.iloc[i + 1]
        if not np.isnan(nxt["last_px"]):
            if nxt["last_side"] < 0 and nxt["last_px"] <= q["bid"] and prism.state.inv < 50:
                prism.on_fill("buy", 1, q["bid"])
                fills.append({"ts": nxt.name, "side": "buy", "fill_px": q["bid"],
                              "mu": q["mu"], "vpin": q["vpin"], "inv": prism.state.inv})
            if nxt["last_side"] > 0 and nxt["last_px"] >= q["ask"] and prism.state.inv > -50:
                prism.on_fill("sell", 1, q["ask"])
                fills.append({"ts": nxt.name, "side": "sell", "fill_px": q["ask"],
                              "mu": q["mu"], "vpin": q["vpin"], "inv": prism.state.inv})

    fills_df = pd.DataFrame(fills)
    log.info("Fills: %d", len(fills_df))

    if len(fills_df):
        mk = compute_markouts(fills_df, feats[["mid"]], horizons=(1, 5, 25, 100, 500))
        print("\n=== Mean markouts by horizon (ticks = mkout / tick_size) ===")
        print(mk[[c for c in mk.columns if c.startswith("mkout_")]].mean())
        # signal-conditioned edge curve
        from mm.markout import edge_curve
        print("\n=== Edge curve by predicted-return decile ===")
        print(edge_curve(mk, "mu", horizons=(1, 5, 25, 100, 500), n_buckets=3))
        final_pnl = prism.state.cash + prism.state.inv * feats["mid"].iloc[-1]
        print(f"\nFinal inventory: {prism.state.inv}")
        print(f"Final MTM P&L (synthetic, no fees): {final_pnl:+.4f}")
    else:
        print("No fills — try loosening spread or lengthening run.")


if __name__ == "__main__":
    main()
