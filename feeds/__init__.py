"""Real order-book feed adapters.

All feeds produce the same pandas DataFrame schema so the MM stack is
feed-agnostic:

Required TOB columns:
    bid_px, bid_sz, ask_px, ask_sz
Required trade columns (may be zero/NaN on non-trade ticks):
    last_px, last_sz, last_side  (+1=buy-initiated, -1=sell-initiated)

Feeds:
  coinbase_feed : FREE, real-time L2 via WebSocket (crypto). Works today.
  binance_feed  : FREE, real-time L2 via WebSocket (crypto).
  databento     : Paid. Equities/futures historical + live MBO/MBP-10.
  polygon       : Paid. US equity/options NBBO + L2 snapshots.
  iex_deep      : Free historical PCAP samples; paid real-time.
  alpaca        : Free L1, paid IEX L2.
"""
from .coinbase_feed import coinbase_l2_stream, coinbase_multi_stream, coinbase_rest_snapshot
from .databento_feed import DatabentoFeed
from .polygon_feed import PolygonFeed
from .databento_opra import DatabentoOPRAFeed
from .cboe_datashop import CBOEDataShopReplay
from .fmp import FMPFeed
