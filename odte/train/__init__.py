"""TradeFM training harness (Phases 1+).

Submodules:
  pretrain_tradefm   single-node training loop for 40M/524M TradeFM on
                     packed parquet shards produced by odte.data.DataShopPacker.
"""
from .pretrain_tradefm import pretrain, load_config
from .checkpoint import CheckpointManager, LocalStore, FsspecStore, make_store
from .distributed import train as train_distributed
