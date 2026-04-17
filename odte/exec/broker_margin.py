"""Broker-specific intraday margin tables.

The margin k-coefficients defined in odte.exec.intraday_margin are a
STARTING POINT. In production each broker publishes its own intraday
haircut schedule that varies by:
  - underlying (SPX, NDX, SPY, etc.)
  - maturity bucket (<=1d = 0DTE gets the highest gamma coefficient)
  - IV regime (calm / normal / elevated / crisis)
  - time-of-day (gamma explosion factor in last 30 min of trading)

This module loads a broker table from YAML/JSON and hands the correct
coefficients to the QP executor on every solve. Supports:

  • on-disk reload (watches mtime) — catches intraday updates
  • per-symbol, per-maturity, per-regime lookup
  • inheritance (missing keys fall back to defaults)
  • validator that refuses to load tables that would trivially under-require

Sample table (schema v1) at `configs/broker_margin_example.yml`.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MarginCoefficients:
    k_gross: float
    k_delta: float
    k_vega: float
    k_gamma: float
    floor_dollars: float = 500.0

    @classmethod
    def defaults(cls) -> "MarginCoefficients":
        return cls(k_gross=0.05, k_delta=0.15, k_vega=1.2, k_gamma=0.001)


@dataclass
class TableLookup:
    underlying: str
    maturity_bucket: str       # "0dte", "short", "medium", "long"
    iv_regime: str             # "calm", "normal", "elevated", "crisis"
    minute_of_day: int         # 0..390 (NYSE minutes)


# ---------------------------------------------------------------------------
# BrokerMarginTable
# ---------------------------------------------------------------------------

@dataclass
class BrokerMarginTable:
    """Loadable broker table with cached coefficients + mtime watch."""

    path: Optional[Path] = None
    _data: Dict[str, Any] = field(default_factory=dict)
    _mtime: float = 0.0

    def __post_init__(self):
        if self.path is not None:
            self.load(self.path)

    # -------- load / refresh -------------------------------------------
    def load(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.suffix in (".yml", ".yaml"):
            import yaml
            self._data = yaml.safe_load(self.path.read_text())
        else:
            self._data = json.loads(self.path.read_text())
        self._mtime = self.path.stat().st_mtime
        self._validate(self._data)
        log.info("BrokerMarginTable loaded: %s  (entries=%d)",
                 self.path, len(self._data.get("symbols", {})))

    def maybe_reload(self) -> bool:
        """Returns True if the file changed on disk and we reloaded."""
        if self.path is None:
            return False
        try:
            m = self.path.stat().st_mtime
        except FileNotFoundError:
            return False
        if m <= self._mtime:
            return False
        self.load(self.path)
        return True

    # -------- lookup ----------------------------------------------------
    def resolve(self, q: TableLookup) -> MarginCoefficients:
        """Apply inheritance: underlying → maturity bucket → IV regime → TOD mult."""
        if not self._data:
            return MarginCoefficients.defaults()

        base = self._data.get("defaults", {})
        sym = self._data.get("symbols", {}).get(q.underlying, {})

        def _pull(d, key):
            v = d.get(key)
            return v if v is not None else base.get(key)

        def _merge(parent: dict, child: dict) -> dict:
            return {**(parent or {}), **(child or {})}

        node = _merge(base, sym)
        for layer in (q.maturity_bucket, q.iv_regime):
            node = _merge(node, node.get(layer, {}))

        # Time-of-day multiplier (gamma explosion in last 30 min).
        tod_mult = 1.0
        if q.minute_of_day >= 360:      # last 30 min
            tod_mult = node.get("tod_mult_last_30", 1.5)
        elif q.minute_of_day >= 330:    # previous 30 min
            tod_mult = node.get("tod_mult_power_hour", 1.2)

        mc = MarginCoefficients(
            k_gross=float(node.get("k_gross", 0.05)) * tod_mult,
            k_delta=float(node.get("k_delta", 0.15)) * tod_mult,
            k_vega=float(node.get("k_vega", 1.2)) * tod_mult,
            k_gamma=float(node.get("k_gamma", 0.001)) * tod_mult,
            floor_dollars=float(node.get("floor_dollars", 500.0)),
        )
        return mc

    # -------- validation -----------------------------------------------
    def _validate(self, d: dict) -> None:
        if "version" not in d:
            raise ValueError("margin table missing 'version'")
        if d.get("version") != 1:
            raise ValueError(f"unsupported margin table version: {d['version']}")
        defaults = d.get("defaults", {})
        for k in ("k_gross", "k_delta", "k_vega", "k_gamma"):
            v = defaults.get(k, 0.0)
            if v < 0:
                raise ValueError(f"negative {k} in defaults")
        # sanity: if gross ≤ 0 and delta ≤ 0, the table would bypass all
        # margin — refuse
        if defaults.get("k_gross", 0.0) < 0.001 \
                and defaults.get("k_delta", 0.0) < 0.001:
            raise ValueError("k_gross and k_delta both near-zero — "
                             "table would trivially under-require margin")


# ---------------------------------------------------------------------------
# Helper: write a sample table
# ---------------------------------------------------------------------------

SAMPLE_TABLE_YAML = """\
version: 1

# Defaults if the specific symbol/regime/bucket isn't found.
defaults:
  k_gross: 0.05
  k_delta: 0.15
  k_vega: 1.2
  k_gamma: 0.001
  floor_dollars: 500.0
  tod_mult_power_hour: 1.2
  tod_mult_last_30: 1.5

symbols:
  SPX:
    0dte:
      k_gamma: 0.0025            # 2.5× default — 0DTE gamma is extreme
      calm:    { k_vega: 0.8 }
      normal:  {}
      elevated: { k_gross: 0.08, k_vega: 1.5 }
      crisis:   { k_gross: 0.12, k_delta: 0.25, k_vega: 2.5, k_gamma: 0.005 }
    short:   # 1-7 DTE
      k_gamma: 0.0015
    medium:  # 8-30 DTE
      k_gamma: 0.0008
    long:    # 31+ DTE
      k_gamma: 0.0004

  NDX:
    0dte:
      k_gamma: 0.003             # NDX gamma slightly higher per-unit-spot
      elevated: { k_gross: 0.09, k_vega: 1.6 }
"""


def write_sample_table(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SAMPLE_TABLE_YAML)
    return path
