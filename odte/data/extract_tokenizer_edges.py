"""Re-fit the streaming-quantile tokenizer on raw DBN files and save the
edges as a JSON sidecar. Needed because the original packer (cme_es_pack
in earlier commits) tokenized but didn't persist the tokenizer state, so
inverse-decoding bin index → log-return wasn't possible.

This script regenerates the same edges that the original pack produced,
since `streaming_quantiles.fit_hybrid_from_chunks` is deterministic given
the same data + same n_buckets + same feature_spec.

Usage:
    python -m odte.data.extract_tokenizer_edges \\
        --dbn-dir /scratch/$USER/data/ES/GLBX-20260505-3CHPNJTXX5 \\
        --out-json /scratch/$USER/data/packed/es/tokenizer.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .databento_pack import iter_dbn_chunks
from .datashop_pack import default_feature_spec_v1, prepare_features
from .streaming_quantiles import fit_hybrid_from_chunks

log = logging.getLogger(__name__)


def extract_edges(dbn_dir: Path, out_json: Path,
                  n_buckets: int = 64,
                  feature_spec: dict | None = None,
                  file_glob: str = "*.dbn.zst") -> dict:
    dbn_dir = Path(dbn_dir).expanduser()
    out_json = Path(out_json).expanduser()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    spec = feature_spec or default_feature_spec_v1()
    files = sorted(dbn_dir.glob(file_glob))
    if not files:
        raise RuntimeError(f"no {file_glob} files in {dbn_dir}")
    log.info("re-fitting tokenizer on %d DBN files (%d features)",
             len(files), len(spec))

    def _chunks():
        for f in files:
            log.info("scanning %s", f.name)
            for ch in iter_dbn_chunks(f):
                yield prepare_features(ch)

    edges = fit_hybrid_from_chunks(_chunks(), spec, n_buckets=n_buckets)

    payload = {
        "n_buckets": n_buckets,
        "feature_spec": spec,
        "edges": {col: e.tolist() for col, e in edges.items()},
        "feature_order": list(spec.keys()),
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("saved tokenizer state → %s (%d features)",
             out_json, len(edges))
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbn-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--n-buckets", type=int, default=64)
    args = ap.parse_args()
    extract_edges(args.dbn_dir, args.out_json, n_buckets=args.n_buckets)
