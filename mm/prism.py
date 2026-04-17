"""PRISM — agentic orchestrator for the MM stack.

Inspired by mid-frequency market-making orchestration: no single model makes
the call; a supervisor coordinates signals and risk. Rules here are simple
and explicit so behavior is auditable.

Responsibilities:
  1. Fetch TOB + trades from the feed (abstract).
  2. Compute features (microprice, OFI, VPIN, RV, VR).
  3. Ask the ST predictor for (μ, σ).
  4. Ask the fill-prob model P(fill | quote, state).
  5. Run Avellaneda-Stoikov with signal overlay + toxicity widening.
  6. Apply agentic risk overlay:
       - if VPIN > vpin_pull_threshold → PULL
       - if inventory |q| > inv_warn → skew quotes harder
       - if predicted_sigma > sigma_halt → halt quoting
  7. Emit quotes and log everything for markout analysis.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .avellaneda_stoikov import ASParams, as_quotes
from .microprice import microprice_features
from .toxicity import vpin, realized_variance, variance_ratio

log = logging.getLogger(__name__)


@dataclass
class PRISMState:
    inv: int = 0
    cash: float = 0.0
    halted: bool = False
    pulled_reason: str = ""


@dataclass
class PRISMConfig:
    vpin_widen: float = 0.4        # widen spread aggressively above this
    vpin_pull: float = 0.7         # pull quotes completely above this
    sigma_halt: float = 5e-3       # halt if predicted sigma spikes
    inv_warn_ratio: float = 0.7    # |q|/max_inv above this → extra skew
    as_params: ASParams = field(default_factory=ASParams)
    # Optional strategy overlay. Receives (feats, tox_row, predictor_out)
    # and returns a dict that is merged on top of the A-S quote. Use this
    # to plug in DeterministicExecutor (0DTE) or any other strategy without
    # forking prism.py.
    strategy: Optional[Callable[[Any, Any, Any], dict]] = None


@dataclass
class PRISM:
    predictor: Any
    fill_model: Any | None = None
    cfg: PRISMConfig = field(default_factory=PRISMConfig)
    state: PRISMState = field(default_factory=PRISMState)
    quote_log: list = field(default_factory=list)

    def decide(self, tob_window: pd.DataFrame, trades_window: pd.DataFrame) -> dict:
        """Run one tick. Returns quote dict or {'pulled': reason}."""
        feats = microprice_features(tob_window)
        rv = realized_variance(feats["mid"])
        vr = variance_ratio(feats["mid"])
        vp = vpin(trades_window) if len(trades_window) > 10 else pd.Series([0.0])

        tox_row = pd.DataFrame({"vpin": [float(vp.iloc[-1])],
                                "rv": [float(rv.iloc[-1])],
                                "vr": [float(vr.iloc[-1])]})
        x = self.predictor.build_features(feats.iloc[[-1]], tox_row)
        mu, sigma = self.predictor.predict(x)
        mu, sigma = float(mu[0]), float(sigma[0])

        vpin_now = float(vp.iloc[-1])
        # Agentic risk gates
        if sigma > self.cfg.sigma_halt:
            self.state.halted = True
            self.state.pulled_reason = f"sigma={sigma:.2e}"
            return {"pulled": self.state.pulled_reason, "sigma": sigma}
        if vpin_now > self.cfg.vpin_pull:
            self.state.pulled_reason = f"vpin={vpin_now:.2f}"
            return {"pulled": self.state.pulled_reason, "vpin": vpin_now}

        # Widening: map vpin [vpin_widen, vpin_pull] → toxicity [0, 1]
        tox = max(0.0, min(1.0,
                           (vpin_now - self.cfg.vpin_widen) /
                           max(self.cfg.vpin_pull - self.cfg.vpin_widen, 1e-9)))

        S = float(feats["micro"].iloc[-1])
        drift = S * mu    # price units
        bid, ask, info = as_quotes(
            s=S, q=self.state.inv, t=0.0,
            params=self.cfg.as_params, predicted_drift=drift, toxicity=tox,
        )
        # Extra inventory skew
        if abs(self.state.inv) > self.cfg.inv_warn_ratio * self.cfg.as_params.inv_limit:
            skew = 0.25 * self.cfg.as_params.tick_size * np.sign(self.state.inv)
            bid -= skew; ask -= skew

        # Queue-ahead calculation: if our quote is at/inside the touch we join
        # the queue, size ahead = current resting size at that price level.
        # If we improve the price (quote inside the spread) we'd be alone.
        last_tob = tob_window.iloc[-1]
        bid_queue_ahead = float(last_tob["bid_sz"]) if bid <= last_tob["bid_px"] else 0.0
        ask_queue_ahead = float(last_tob["ask_sz"]) if ask >= last_tob["ask_px"] else 0.0

        quote = {"ts": tob_window.index[-1], "bid": bid, "ask": ask,
                 "micro": S, "mu": mu, "sigma": sigma, "vpin": vpin_now,
                 "inv": self.state.inv,
                 "bid_queue_ahead": bid_queue_ahead,
                 "ask_queue_ahead": ask_queue_ahead,
                 **info}
        # Optional strategy overlay (e.g. 0DTE DeterministicExecutor).
        strategy = getattr(self.cfg, "strategy", None)
        if strategy is not None:
            try:
                overlay = strategy(feats, tox_row, {"mu": mu, "sigma": sigma})
                if isinstance(overlay, dict):
                    quote.update(overlay)
            except Exception as e:
                log.warning("strategy overlay failed: %s", e)
        self.quote_log.append(quote)
        return quote

    def on_fill(self, side: str, qty: int, px: float) -> None:
        if side == "buy":
            self.state.cash -= qty * px
            self.state.inv += qty
        else:
            self.state.cash += qty * px
            self.state.inv -= qty

    def export_quotes(self) -> pd.DataFrame:
        return pd.DataFrame(self.quote_log)
