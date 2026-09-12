"""Free daily price bars — the price side of the transmission probe.

Deliberately thin. This is not an ingestion layer; it is just enough to
falsify a hypothesis with $0 of data. Per `docs/chokepoint_track.md`, a general
ingestion layer is gated on the probe returning signal, not the other way round.

Free price history is revised, and delisted names silently vanish from the
provider (survivorship). That is acceptable for falsification and is NOT
acceptable for sizing a position — a distinction `docs/signal_probe_result.md`
already draws for the equities probe.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:  # pragma: no cover - optional dep
    yf = None  # type: ignore
    _HAS_YF = False


# The PGM complex, split by role in the factor model.
#
# Miners are the dependent variable; metal and market are the factors we
# neutralize against, because "miners move with platinum" is not alpha.
PGM_MINERS = {
    "SBSW": "Sibanye Stillwater (SA PGM + gold, NYSE ADR)",
    "IMPUY": "Impala Platinum (ADR)",
    "ANGPY": "Anglo American Platinum (ADR)",
}
PGM_METAL = {
    "PPLT": "abrdn Physical Platinum ETF",
    "PALL": "abrdn Physical Palladium ETF",
}
FACTORS = {
    "SPY": "US broad market",
    "ZAR=X": "USD/ZAR — the FX transmission channel",
}
# Gold miners with minimal SA grid exposure; a stage effect that shows up here
# just as strongly is a macro artifact, not a power-curtailment mechanism.
CONTROLS = {
    "GDX": "global gold miners (placebo)",
}


class PriceDataUnavailable(RuntimeError):
    """Raised when price history cannot be fetched. Never substituted."""


@dataclass(frozen=True)
class PricePanel:
    """Aligned daily close prices and log returns for a set of tickers."""

    close: pd.DataFrame
    returns: pd.DataFrame

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return self.returns.index  # type: ignore[return-value]

    def describe(self) -> str:
        lo, hi = self.returns.index[0].date(), self.returns.index[-1].date()
        return (
            f"{len(self.returns)} sessions {lo} → {hi}, "
            f"{len(self.returns.columns)} series: {list(self.returns.columns)}"
        )


def fetch(tickers: list[str], start: str, end: str | None = None,
          min_coverage: float = 0.80) -> PricePanel:
    """Fetch daily closes and log returns.

    Tickers whose coverage over the window falls below `min_coverage` are
    dropped with a warning rather than forward-filled across long gaps — a
    stale price produces a fake zero return, which looks like real data.
    """
    if not _HAS_YF:
        raise PriceDataUnavailable(
            "yfinance is not installed. `pip install yfinance` — it is the free "
            "price source used by the existing equities probe as well."
        )

    log.info("fetching %d tickers from %s", len(tickers), start)
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, group_by="column",
    )
    if raw is None or raw.empty:
        raise PriceDataUnavailable(
            f"no data returned for {tickers} from {start}. Check connectivity "
            f"and ticker validity."
        )

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.to_frame() if isinstance(close, pd.Series) else close
    close = close.dropna(how="all")

    coverage = close.notna().mean()
    weak = coverage[coverage < min_coverage]
    if len(weak):
        log.warning(
            "dropping %d ticker(s) below %.0f%% coverage: %s. Forward-filling "
            "these would manufacture zero returns on days with no real trade.",
            len(weak), 100 * min_coverage,
            {k: f"{v:.0%}" for k, v in weak.items()},
        )
        close = close.drop(columns=list(weak.index))
    if close.empty:
        raise PriceDataUnavailable("all tickers fell below the coverage floor")

    # Short gaps only: a 1-2 day hole is a holiday mismatch across venues; a
    # long one is a listing problem and was dropped above.
    close = close.ffill(limit=2).dropna(how="any")
    returns = np.log(close).diff().dropna(how="any")

    panel = PricePanel(close=close, returns=returns)
    log.info("price panel: %s", panel.describe())
    return panel
