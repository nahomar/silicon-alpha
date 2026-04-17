"""Execution / risk layer (Phase 5 scaffolding).

Includes the 2026 dynamic intraday-margin calculator so small-account 0DTE
works under the new SEC rule that replaced the $25k PDT wealth barrier.
"""
from .intraday_margin import (
    DynamicIntradayMargin, IntradayMarginState, OptionPosition,
    compute_account_exposure,
)
from .broker_margin import (
    BrokerMarginTable, MarginCoefficients, TableLookup, write_sample_table,
)
from .qp import QPExecutor, InstrumentGreeks, QPResult
from .risk_gates import RiskGates, GateResult
from .streaming_ofi import StreamingFeatures, TobState
from .paper_broker import PaperBroker, PaperOrder
from .risk_gates import close_for_pennies_orders, gamma_cap_scale, pin_score
from .post_trade import PostTradeAnalyzer, run as run_post_trade
