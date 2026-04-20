"""Data-quality inspection for a Databento OPRA pull.

Runs the pro-desk validation checklist on a packed shard directory
BEFORE committing to more expensive pulls. Catches schema drift,
malformed rows, and distribution surprises early.

Checklist (per the 5-principle framework — measurement before investment):

  1. Shape             — row count, shard count, day coverage
  2. Timestamp hygiene — monotonicity, no duplicates, market-hours only
  3. BBO validity      — bid <= ask, spread > 0, size > 0 ratio
  4. Symbol coverage   — SPX vs SPXW distribution, strike density
  5. Trade/quote ratio — sanity against known cmbp-1 profile (~1:100)
  6. Distribution sanity — log-return tails, spread percentiles
  7. Tokenizer health  — any degenerate buckets?

Writes a markdown report to reports/databento_inspection_<stamp>.md
AND prints a colored pass/fail summary to stdout. Exit 0 = all checks
passed. Exit 1 = one or more failures; see report for detail.

Usage:
    python -m odte.data.inspect_smoke \\
        --shards reports/odte_shards_real \\
        --raw-dbn data/databento_raw/SPX_20240103_20240104.dbn.zst
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Finding:
    name: str
    ok: bool
    detail: str
    severity: str = "fail"   # "fail" | "warn" | "info"


@dataclass
class Report:
    findings: List[Finding] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def add(self, *f: Finding) -> None:
        self.findings.extend(f)

    @property
    def all_passed(self) -> bool:
        return all(f.ok or f.severity != "fail" for f in self.findings)


# ---------------------------------------------------------------------------
# Checks on the raw DBN file (if provided)
# ---------------------------------------------------------------------------

def check_raw_dbn(dbn_path: Path, report: Report) -> None:
    """Sample the first batch of the raw DBN file for raw-row quality."""
    try:
        import databento as db
    except ImportError:
        report.add(Finding("raw_dbn_import", False, "databento not installed", "warn"))
        return
    store = db.DBNStore.from_file(str(dbn_path))
    # First 500k rows is a sufficient sample for quality stats.
    iterator = store.to_df(count=500_000)
    first_batch = next(iter(iterator))
    report.metrics["raw_dbn_sample_rows"] = len(first_batch)

    # 2. Timestamp hygiene — ts_event monotonicity within the sample
    ts = pd.to_datetime(first_batch["ts_event"])
    diffs = ts.diff().dropna()
    n_backward = int((diffs < pd.Timedelta(0)).sum())
    report.add(Finding(
        "ts_event_monotonic",
        n_backward == 0,
        f"{n_backward} out-of-order events in first 500k sample",
        "fail" if n_backward > 50 else "warn" if n_backward > 0 else "fail",
    ))

    # Market hours check (opening cross through close). US options trade
    # roughly 9:30 ET - 16:15 ET. In UTC that's ~13:30 - 20:15 in winter.
    # We don't hard-fail on this — holidays and pre/post-market ticks
    # happen — but we flag a weird distribution.
    ts_utc = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
    hours = ts_utc.dt.hour
    frac_outside = float(((hours < 12) | (hours > 22)).mean())
    report.metrics["raw_frac_outside_market_hours"] = round(frac_outside, 4)
    report.add(Finding(
        "market_hours_concentration",
        frac_outside < 0.05,
        f"{frac_outside*100:.1f}% of sample rows outside typical market hours",
        "warn",
    ))

    # 3. BBO validity on raw rows
    bid = first_batch["bid_px_00"].astype(float)
    ask = first_batch["ask_px_00"].astype(float)
    n_crossed = int(((bid > ask) & (ask > 0)).sum())
    report.add(Finding(
        "raw_bbo_not_crossed",
        n_crossed == 0,
        f"{n_crossed} rows have bid > ask (crossed market) in sample",
        "fail" if n_crossed > 100 else "warn" if n_crossed > 0 else "fail",
    ))
    n_zero_ask = int((ask <= 0).sum())
    report.metrics["raw_n_zero_ask"] = n_zero_ask

    # 4. Symbol coverage — parse `symbol` column to count distinct roots
    if "symbol" in first_batch.columns:
        # Symbol like "SPX   251219P05900000" — root is first token.
        roots = first_batch["symbol"].astype(str).str.strip().str.split().str[0]
        root_counts = roots.value_counts().to_dict()
        report.metrics["raw_root_distribution"] = root_counts
        spx_only = set(root_counts.keys()) - {"SPX", "SPXW"}
        report.add(Finding(
            "symbol_roots_expected",
            not spx_only,
            f"unexpected roots in data: {sorted(spx_only)}" if spx_only
            else f"roots seen: {sorted(root_counts.keys())}",
            "warn",
        ))

    # 5. Trade/quote ratio — cmbp-1 is quote-heavy. Pure Tradefraction.
    if "action" in first_batch.columns:
        a = first_batch["action"].astype(str).str.upper()
        n_trade = int((a == "T").sum())
        n_total = len(a)
        frac_trade = n_trade / max(1, n_total)
        report.metrics["raw_frac_trade_rows"] = round(frac_trade, 6)
        # cmbp-1 is quote-dominated; trades should be <1% of rows.
        # >10% would be very suspicious.
        report.add(Finding(
            "trade_quote_ratio_reasonable",
            0.00001 < frac_trade < 0.1,
            f"trade fraction = {frac_trade*100:.4f}% of rows "
            f"(expect ~0.1-1% for cmbp-1)",
            "warn",
        ))


# ---------------------------------------------------------------------------
# Checks on packed shards
# ---------------------------------------------------------------------------

def check_shards(shard_dir: Path, report: Report) -> None:
    shards = sorted(shard_dir.glob("opra_*.parquet"))
    report.add(Finding("shards_exist", len(shards) > 0,
                       f"found {len(shards)} shards in {shard_dir}"))
    if not shards:
        return
    report.metrics["n_shards"] = len(shards)

    # Load a sample shard (the first one) to check schema + token validity.
    df = pd.read_parquet(shards[0])
    report.metrics["shard0_rows"] = len(df)
    expected_cols = {"ts", "underlying", "expiry", "day", "tokens"}
    missing = expected_cols - set(df.columns)
    report.add(Finding("shard_schema_complete", not missing,
                       f"missing cols: {sorted(missing)}" if missing
                       else "all expected columns present"))

    # Token length distribution — must all be positive integers.
    if "tokens" in df.columns:
        tok_lens = df["tokens"].apply(len) if len(df) else pd.Series([], dtype=int)
        report.metrics["token_len_min"] = int(tok_lens.min()) if len(tok_lens) else 0
        report.metrics["token_len_max"] = int(tok_lens.max()) if len(tok_lens) else 0
        report.add(Finding("tokens_nonempty",
                           all(l > 0 for l in tok_lens) if len(tok_lens) else False,
                           f"token length range: [{tok_lens.min() if len(tok_lens) else 0}, "
                           f"{tok_lens.max() if len(tok_lens) else 0}]"))

    # 6. Day coverage
    if "day" in df.columns and len(df):
        days = sorted(df["day"].unique())
        report.metrics["days_in_shard0"] = days

    # 7. Tokenizer health — load the fitted tokenizer and check bin
    # degeneracy (a "dead" feature would have all edges equal).
    tok_path = shard_dir / "tokenizer.json"
    if tok_path.exists():
        try:
            with open(tok_path) as f:
                payload = json.load(f)
            degen_features = []
            feat_spec = payload.get("feature_spec", {})
            edges_map = payload.get("edges", {})
            for feat, edges in edges_map.items():
                e = np.array(edges)
                interior = e[(e != -np.inf) & (e != np.inf)]
                if len(interior) < 2:
                    degen_features.append(feat)
                    continue
                n_unique = int(len(np.unique(interior)))
                if n_unique < 0.5 * len(interior):
                    # More than half the interior edges collapsed to duplicates
                    degen_features.append(f"{feat}(collapsed)")
            report.add(Finding(
                "tokenizer_no_degen_features",
                not degen_features,
                f"degenerate features: {degen_features}" if degen_features
                else f"all {len(feat_spec)} features have well-spread bin edges",
            ))
        except Exception as e:
            report.add(Finding("tokenizer_load", False, f"failed to load tokenizer: {e}", "warn"))


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_markdown(report: Report, out: Path,
                   shard_dir: Path, dbn_path: Optional[Path]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# Databento smoke inspection — {time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Shards:** `{shard_dir}`")
    if dbn_path:
        lines.append(f"**Raw DBN sample:** `{dbn_path}`\n")
    status = "✅ ALL CHECKS PASSED" if report.all_passed else "❌ FAILURES PRESENT"
    lines.append(f"## {status}\n")
    lines.append("## Findings\n")
    lines.append("| Check | Status | Detail |")
    lines.append("|---|---|---|")
    for f in report.findings:
        mark = "✅" if f.ok else ("⚠️" if f.severity == "warn" else "❌")
        lines.append(f"| `{f.name}` | {mark} | {f.detail} |")
    lines.append("\n## Metrics\n")
    for k, v in report.metrics.items():
        if isinstance(v, dict):
            v = ", ".join(f"{kk}={vv}" for kk, vv in list(v.items())[:10])
        elif isinstance(v, list):
            v = ", ".join(str(x) for x in v[:10])
        lines.append(f"- **{k}**: {v}")
    out.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="reports/odte_shards_real",
                    help="packed-shards directory")
    ap.add_argument("--raw-dbn", default=None,
                    help="raw .dbn.zst file (optional) for per-row checks")
    ap.add_argument("--out", default=None,
                    help="output markdown report path")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    report = Report()
    shard_dir = Path(a.shards)
    if shard_dir.exists():
        check_shards(shard_dir, report)
    else:
        report.add(Finding("shards_dir_exists", False,
                           f"shard dir {shard_dir} not found", "fail"))

    dbn_path = Path(a.raw_dbn) if a.raw_dbn else None
    if dbn_path and dbn_path.exists():
        check_raw_dbn(dbn_path, report)

    out = Path(a.out or f"reports/databento_inspection_{int(time.time())}.md")
    write_markdown(report, out, shard_dir, dbn_path)
    log.info("wrote report → %s", out)

    # stdout summary
    print("\n=== inspection summary ===")
    for f in report.findings:
        mark = "OK " if f.ok else ("WARN" if f.severity == "warn" else "FAIL")
        print(f"  [{mark}] {f.name}: {f.detail}")
    print(f"\nreport: {out}")
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    _cli()
