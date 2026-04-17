"""Dynamic intraday-margin engine (2026 SEC regime).

The 2026 SEC rule replaces the fixed $25,000 Pattern-Day-Trader wealth
barrier with a *dynamic* intraday-margin requirement: an account may
engage in intraday 0DTE trading at any size, provided live equity meets
a real-time exposure test based on:

  • gross notional       sum of |position × spot|
  • delta-dollar exposure sum of |position × delta × spot|
  • vega exposure         sum of |position × vega|
  • gamma dollars         sum of |position × gamma × spot²|

Our implementation follows the SEC description as reported: the required
intraday margin R(t) is a piecewise-linear combination of the four
exposures above, with regime multipliers that scale up in high-volatility
conditions. We expose the coefficients as Config so they can be updated
if the rule text evolves.

This is an *estimator*, not legal advice. Broker-of-record margin checks
are authoritative. Use this module to:
  1. Pre-check orders before submission
  2. Size positions under a self-imposed cap
  3. Decide when to flatten before margin call

Integrated with odte.executor.RiskGates through compute_account_exposure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position model
# ---------------------------------------------------------------------------

@dataclass
class OptionPosition:
    contract: str                   # OCC symbol
    qty: float                      # +long / -short
    spot: float                     # underlying reference price
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    contract_multiplier: int = 100
    # Cash-settled index options (SPX, NDX) are settled in cash, so spot
    # exposure = abs(qty × spot × multiplier). For equity options same form.


@dataclass
class IntradayMarginState:
    gross_notional: float
    delta_dollars: float
    vega: float
    gamma_dollars: float
    required_equity: float


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

@dataclass
class DynamicIntradayMargin:
    """Piecewise-linear margin estimator.

    Defaults are a STARTING POINT. Every broker publishes its own
    intraday haircut table, and the SEC rule leaves the exact coefficients
    to the SRO / broker. These land a small delta-hedged 0DTE book between
    Reg T initial and maintenance margin. Override with your broker's
    actual numbers before sizing real orders.

    Gamma coefficient is small because `gamma·S²` is already on the scale
    of notional exposure; ~0.1% of gamma-dollars aligns with CME SPAN-style
    haircuts.
    """
    k_gross: float = 0.05             # 5% of gross notional
    k_delta: float = 0.15             # 15% of delta-dollar exposure
    k_vega: float = 1.2               # $1.2 per vega-dollar unit
    k_gamma: float = 0.001            # 0.1% of gamma-dollars (SPAN-like)
    vol_regime_mult: float = 1.0      # scale up during high-VIX days
    floor_dollars: float = 500.0      # absolute minimum required equity

    def required(self, positions: Iterable[OptionPosition]) -> IntradayMarginState:
        gross = 0.0; dd = 0.0; vg = 0.0; gd = 0.0
        for p in positions:
            notional = abs(p.qty) * p.spot * p.contract_multiplier
            gross += notional
            dd += abs(p.qty) * abs(p.delta) * p.spot * p.contract_multiplier
            vg += abs(p.qty) * abs(p.vega) * p.contract_multiplier
            gd += abs(p.qty) * abs(p.gamma) * p.spot * p.spot * p.contract_multiplier
        req = self.vol_regime_mult * (
            self.k_gross * gross
            + self.k_delta * dd
            + self.k_vega * vg
            + self.k_gamma * gd
        )
        req = max(req, self.floor_dollars)
        return IntradayMarginState(
            gross_notional=gross, delta_dollars=dd, vega=vg,
            gamma_dollars=gd, required_equity=req,
        )

    def can_place_order(self, current_positions: Iterable[OptionPosition],
                        candidate: OptionPosition, equity: float) -> tuple[bool, IntradayMarginState]:
        combined = list(current_positions) + [candidate]
        state = self.required(combined)
        return (equity >= state.required_equity, state)


# ---------------------------------------------------------------------------
# Convenience: hook into odte.executor.RiskGates
# ---------------------------------------------------------------------------

def compute_account_exposure(
    positions: Iterable[OptionPosition],
    vol_regime_mult: float = 1.0,
) -> dict:
    """One-shot exposure summary — feed to the risk gates."""
    engine = DynamicIntradayMargin(vol_regime_mult=vol_regime_mult)
    state = engine.required(positions)
    return {
        "gross_notional": state.gross_notional,
        "delta_dollars": state.delta_dollars,
        "vega": state.vega,
        "gamma_dollars": state.gamma_dollars,
        "required_equity": state.required_equity,
    }
