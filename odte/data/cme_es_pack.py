"""CME E-mini S&P 500 (ES) futures -> sharded tokenized parquet.

Scaffold for cross-asset fusion Phase-2.5: adds ES MBP-1 as a lead signal
for the SPX 0DTE options forecast (ES typically leads OPRA by 1-10 ms
during high-volatility moments — exactly the lead-lag edge HRT-class firms
extract).

Design notes (not yet executed — see docs/cross_asset_fusion.md):
  - Dataset: Databento GLBX.MDP-3 (CME Globex market-by-price level 1).
  - Symbol: ES parent symbology — front-month auto-rolls. For backtest
    alignment with a specific OPRA window, may also pin to the explicit
    contract (e.g. ESZ3 for Dec 2023).
  - Schema: MBP-1 (same as OPRA adapter). MBP-10 would give more depth
    but 10× the bandwidth for marginal signal.
  - Feature spec mirrors databento_pack but omits options-specific fields.
  - Tokens written with modality_id=1 (OPRA=0). Tokenizer vocab is shared
    with OPRA via edge-merge at fit time (see _fit_cross_asset_edges).

Budget:
  - Databento GLBX MBP-1 for ES: ~$5-20/day (vs OPRA SPX ~$7.50/day).
  - A full regime-stratified 15-day corpus: ~$75-300.
  - Our remaining Databento credit (~$117 of $125 after the 4/20 job) covers
    ~6-20 days depending on daily volume. Plan per-day before committing.

Not yet functional — the pack() function is a stub that raises until a
corresponding fit-time tokenizer merge is implemented in streaming_quantiles.
Filling it in is the first task of Phase-2.5 data work.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ES_DATASET = "GLBX.MDP-3"
ES_SCHEMA = "mbp-1"
ES_STYPE_IN = "parent"
ES_DEFAULT_SYMBOLS = ["ES"]

# Feature spec for ES MBP-1. Subset of the OPRA spec — no spread (ES is a
# single instrument with tight BBO), no size-weighted flow (LOB depth at
# L1 is just bid/ask size). Kept as dict[str, str] for consistency with
# DataShopPacker.feature_spec; values are the tokenizer strategy.
ES_FEATURE_SPEC: dict[str, str] = {
    "ret": "quantile",
    "mid": "quantile",
    "bid_sz": "log",
    "ask_sz": "log",
    "last_sz": "log",
    "inter_arrival_ms": "log",
}


# ---------------------------------------------------------------------------
# Schema translation — ES GLBX.MDP -> DataShop-style feature DataFrame
# ---------------------------------------------------------------------------

def glbx_to_features(batch: pd.DataFrame) -> pd.DataFrame:
    """Translate a GLBX MBP-1 batch DataFrame into the feature schema that
    our tokenizer + trainer consume. Mirrors databento_pack.prepare_features
    but adapted to ES's single-instrument flow.

    Expected input columns (Databento MBP-1 standard):
        ts_event, instrument_id, action, side, price, size, bid_px_00,
        ask_px_00, bid_sz_00, ask_sz_00
    Output columns (matching ES_FEATURE_SPEC + ts_ms for ordering):
        ts_ms, quote_datetime, ret, mid, bid_sz, ask_sz, last_sz,
        inter_arrival_ms
    """
    raise NotImplementedError(
        "Cross-asset fusion is scaffold-only. When ready to train with ES "
        "data: (1) implement glbx_to_features() mirroring the OPRA "
        "databento_to_datashop_schema helper, (2) verify feature "
        "distributions are sane via a small smoke pull, (3) merge tokenizer "
        "edges via streaming_quantiles. See docs/cross_asset_fusion.md."
    )


def iter_es_dbn_chunks(path: Path, chunk_rows: int = 200_000
                       ) -> Iterable[pd.DataFrame]:
    """Stream a GLBX MBP-1 DBN file in chunks (reuses databento_pack's
    iter_dbn_chunks pattern to keep memory constant)."""
    raise NotImplementedError("see glbx_to_features() TODO")


# ---------------------------------------------------------------------------
# High-level: fetch + pack
# ---------------------------------------------------------------------------

def pack_es(start: str, end: str, symbols: Optional[List[str]] = None,
            raw_dir: Optional[Path] = None,
            out_dir: Optional[Path] = None,
            max_spend_usd: float = 20.0,
            modality_id: int = 1) -> list[Path]:
    """Fetch a time window of ES MBP-1 from Databento and pack into shards
    compatible with the OPRA trainer (same parquet schema + modality_id
    field so ShardedTokenDataset can interleave them).

    Not yet implemented — see module docstring for why and docs/cross_asset_fusion.md
    for the rollout plan.
    """
    raise NotImplementedError(
        "pack_es() is scaffold. Before filling this in, confirm: (a) "
        "single-asset 524M pretrain has validated on SPX OPRA alone "
        "(Phase 2 gate), and (b) the token-interleaving schema in "
        "docs/cross_asset_fusion.md has been prototyped on a small "
        "SPX+ES overlap window."
    )


if __name__ == "__main__":
    # Keep a CLI stub so `python -m odte.data.cme_es_pack` gives a clear
    # "not ready yet" signal instead of a cryptic traceback.
    import sys
    sys.stderr.write(
        "odte.data.cme_es_pack is scaffold-only. "
        "See docs/cross_asset_fusion.md and the module docstring.\n"
    )
    sys.exit(1)
