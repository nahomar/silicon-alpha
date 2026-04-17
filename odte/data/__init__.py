"""Phase 1 data plumbing: CBOE DataShop → sharded training set.

Submodules:
  datashop_pack        CSV/CSV.zst → tokenized parquet shards
  streaming_quantiles  online quantile + log-edge estimator over ≥1T tokens
"""
from .streaming_quantiles import StreamingQuantileFitter, StreamingLogEdgeFitter
from .datashop_pack import DataShopPacker, pack_folder
from .mixer import (
    QualityWeights, WeightedShardTokenDataset, compute_row_weights,
    describe_weighting,
)
