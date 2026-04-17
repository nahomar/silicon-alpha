"""Market-making / microstructure stack.

Pipeline:
    feed → book features → {ST predictor, fill-prob, toxicity, markouts}
         → quote decision (Avellaneda-Stoikov + signal overlay)
         → PRISM orchestrator (agentic risk / param manager)

Every module is feed-agnostic: pass either a synthetic_book stream or a real
LOB replay (Databento / Polygon.io / IEX DEEP) with identical columns.

Required LOB columns for real data:
    ts, bid_px, bid_sz, ask_px, ask_sz, last_px, last_sz, last_side
Optional: full book depth levels 1..N and trade print side flag.
"""
from .microprice import microprice, microprice_features, book_imbalance, ofi
from .toxicity import vpin, kyle_lambda, realized_variance, variance_ratio
from .fill_model import FillProbabilityModel
from .markout import compute_markouts
from .predictor import ShortHorizonPredictor
from .avellaneda_stoikov import as_quotes
from .synthetic_book import simulate_book
from .mm_env import MarketMakingEnv
from .prism import PRISM
