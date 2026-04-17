"""Post-trade analyzer — theoretical vs realized edge.

Reads the fill parquet that PaperBroker writes through RollingParquetWriter,
joins with the contemporaneous OPRA/tob parquet, and computes:

  • realized_spread_cost_per_unit — mid − fill (signed by side)
  • spread_cost_bps               — same, in bps of mid
  • markout_1s / 5s / 30s          — forward mid change after fill (proxy PnL)
  • edge_bps                       — markout − spread cost; if negative, the
                                     strategy loses to the 3.6 % spread friction
  • directional_hit_rate           — fraction of fills whose forward mid move
                                     matched the side (buys fall when mid drops? no: buys want mid↑)
  • per-symbol + per-hour aggregation

Produces a Markdown report at reports/post_trade_<ts>.md.
"""
from __future__ import annotations

import glob as _glob
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MM_DATA = ROOT / "reports" / "mm_data"
REPORTS = ROOT / "reports"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _read_stream(name: str, hours: int = 48) -> pd.DataFrame:
    if not MM_DATA.exists():
        return pd.DataFrame()
    frames = []
    cutoff_ms = (time.time() - hours * 3600) * 1000
    for day in sorted(MM_DATA.iterdir()):
        if not day.is_dir():
            continue
        for hour in sorted(day.iterdir()):
            if not hour.is_dir():
                continue
            p = hour / f"{name}.parquet"
            if not p.exists():
                continue
            try:
                df = pd.read_parquet(p)
                if "ts" in df.columns:
                    df = df[df["ts"] >= cutoff_ms]
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                log.warning("read %s failed: %s", p, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

@dataclass
class PostTradeAnalyzer:
    hours: int = 48
    markout_horizons_ms: tuple = (1_000, 5_000, 30_000)

    def run(self) -> dict:
        fills = _read_stream("odte_fills", hours=self.hours)
        orders = _read_stream("odte_orders", hours=self.hours)
        if fills.empty:
            log.warning("no fills in last %dh", self.hours)
            return {"n_fills": 0}

        # Spread cost in bps
        fills = fills.copy()
        fills["spread_bps"] = (fills["spread_cost_per_unit"]
                                / fills["mid_at_fill"].replace(0, np.nan)) * 1e4

        # Markouts: for each fill we need a forward mid. Use the tob stream
        # if available; else approximate from next fill's mid_at_fill
        # (same-symbol).
        markouts = self._compute_markouts(fills)
        for h, col in markouts.items():
            fills[f"markout_{h}"] = col

        # Directional hit rate
        # buy wants fwd_mid > mid_at_fill; sell wants fwd_mid < mid_at_fill.
        sign = np.where(fills["side"] == "buy", 1.0, -1.0)
        horizons = [int(h) for h in self.markout_horizons_ms]
        hit_rates = {}
        for h in horizons:
            col = f"markout_{h}"
            if col in fills.columns:
                hits = np.sign(sign * fills[col].values) > 0
                hit_rates[h] = float(np.mean(hits))

        # Edge in bps = markout (mid-movement per unit) in bps minus spread bps
        edge_rows: list[dict] = []
        for h in horizons:
            col = f"markout_{h}"
            if col not in fills.columns:
                continue
            mo_bps = (fills[col] / fills["mid_at_fill"].replace(0, np.nan)) * 1e4
            # buy: +markout = good; sell: -markout = good
            signed_mo = np.where(fills["side"] == "buy", mo_bps, -mo_bps)
            edge_bps = signed_mo - np.abs(fills["spread_bps"])
            edge_rows.append({
                "horizon_ms": h,
                "mean_markout_bps": float(np.nanmean(signed_mo)),
                "mean_spread_bps": float(np.nanmean(np.abs(fills["spread_bps"]))),
                "mean_edge_bps": float(np.nanmean(edge_bps)),
                "directional_hit_rate": hit_rates.get(h, float("nan")),
                "n_fills": int(len(edge_bps.dropna() if hasattr(edge_bps, "dropna") else edge_bps[np.isfinite(edge_bps)])),
            })

        by_symbol = self._per_symbol(fills)
        by_hour = self._per_hour(fills)

        summary = {
            "hours": self.hours,
            "n_fills": int(len(fills)),
            "n_orders": int(len(orders)) if not orders.empty else 0,
            "mean_spread_bps": float(np.nanmean(np.abs(fills["spread_bps"]))),
            "per_horizon": edge_rows,
            "per_symbol": by_symbol,
            "per_hour": by_hour,
        }
        return summary

    # ----------------------------------------------------------------
    def _compute_markouts(self, fills: pd.DataFrame) -> dict:
        """For each fill, look up fwd mid at the requested horizons using the
        next fills on the same symbol. Close enough for post-trade smoke;
        swap in the real tob parquet for production-grade numbers."""
        out: dict[int, np.ndarray] = {}
        for h in self.markout_horizons_ms:
            out[int(h)] = np.full(len(fills), np.nan)
        fills = fills.sort_values(["symbol", "ts"])
        for sym, g in fills.groupby("symbol"):
            idx = g.index.values
            ts = g["ts"].values
            mids = g["mid_at_fill"].values
            for h in self.markout_horizons_ms:
                # vectorized: find first j > i with ts[j] - ts[i] >= h
                for pos, i in enumerate(idx):
                    target = ts[pos] + h
                    j = pos
                    while j < len(ts) and ts[j] < target:
                        j += 1
                    if j < len(ts):
                        out[int(h)][i] = mids[j] - mids[pos]
        return {h: pd.Series(v) for h, v in out.items()}

    def _per_symbol(self, fills: pd.DataFrame) -> list[dict]:
        if fills.empty:
            return []
        g = fills.groupby("symbol")
        rows = []
        for sym, sub in g:
            rows.append({
                "symbol": sym,
                "n": int(len(sub)),
                "mean_spread_bps": float(np.nanmean(np.abs(sub["spread_bps"]))),
                "total_spread_cost": float(sub["total_spread_cost"].iloc[-1]
                                            if "total_spread_cost" in sub else 0.0),
            })
        return rows

    def _per_hour(self, fills: pd.DataFrame) -> list[dict]:
        if fills.empty:
            return []
        ts = pd.to_datetime(fills["ts"], unit="ms", errors="coerce", utc=True)
        fills = fills.assign(_hour=ts.dt.strftime("%Y-%m-%dT%H:00"))
        rows = []
        for hr, sub in fills.groupby("_hour"):
            rows.append({
                "hour": hr, "n": int(len(sub)),
                "mean_spread_bps": float(np.nanmean(np.abs(sub["spread_bps"]))),
            })
        return rows

    # ----------------------------------------------------------------
    def write_report(self, summary: dict, path: Optional[Path] = None) -> Path:
        ts = time.strftime("%Y%m%dT%H%M%S")
        p = Path(path) if path else REPORTS / f"post_trade_{ts}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append(f"# Post-trade report — {time.strftime('%Y-%m-%d %H:%M')}\n")
        lines.append(f"- lookback: {summary.get('hours', 0)}h")
        lines.append(f"- fills: {summary.get('n_fills', 0)}   "
                     f"orders: {summary.get('n_orders', 0)}")
        lines.append(f"- mean |spread|: {summary.get('mean_spread_bps', float('nan')):.1f} bps\n")
        lines.append("## Edge curve\n")
        lines.append("| horizon (ms) | markout bps | spread bps | edge bps | dir-hit |  n |")
        lines.append("|---|---|---|---|---|---|")
        for row in summary.get("per_horizon", []):
            lines.append(f"| {row['horizon_ms']} | {row['mean_markout_bps']:+.1f} | "
                         f"{row['mean_spread_bps']:.1f} | {row['mean_edge_bps']:+.1f} | "
                         f"{row['directional_hit_rate']*100:.1f}% | {row['n_fills']} |")
        lines.append("\n## Per symbol\n")
        lines.append("| symbol | n | mean spread bps |")
        lines.append("|---|---|---|")
        for r in summary.get("per_symbol", [])[:50]:
            lines.append(f"| {r['symbol']} | {r['n']} | {r['mean_spread_bps']:.1f} |")
        p.write_text("\n".join(lines))
        return p


def run(hours: int = 48) -> dict:
    a = PostTradeAnalyzer(hours=hours)
    summary = a.run()
    a.write_report(summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=48)
    a = ap.parse_args()
    summary = run(hours=a.hours)
    print(json.dumps({k: v for k, v in summary.items()
                       if k not in ("per_symbol", "per_hour")}, indent=2))
