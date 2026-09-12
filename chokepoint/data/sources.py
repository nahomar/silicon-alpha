"""Registry of data sources, keyed by what they can physically support.

This module exists because of a specific, recorded failure mode. From
`docs/signal_probe_result.md`: the OPRA directional probe stalled because the
free data source exposed *current snapshots*, which "have no time axis to
compute returns from."

Most African mineral data has the same defect — USGS, BGS, EITI and Comtrade are
annual or monthly with multi-month publication lag. They are excellent for
measuring supply concentration and value capture, and structurally incapable of
supporting a days-horizon trading signal.

So a source here does not just carry a URL. It must declare its `frequency` and
`publication_lag`, from which `min_signal_horizon` follows mechanically. Asking a
source for a signal shorter than it can support raises. The guard is code, not a
convention, because conventions are what fail at 2am six months from now.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Iterator

log = logging.getLogger(__name__)


class Frequency(Enum):
    """Nominal spacing between observations."""

    INTRADAY = timedelta(minutes=1)
    DAILY = timedelta(days=1)
    WEEKLY = timedelta(weeks=1)
    MONTHLY = timedelta(days=30)
    QUARTERLY = timedelta(days=91)
    ANNUAL = timedelta(days=365)
    SNAPSHOT = timedelta(0)  # current-state only; NO time axis at all

    @property
    def period(self) -> timedelta:
        return self.value


class Cost(Enum):
    FREE = "free"
    TOKEN = "token"  # free tier, requires registration
    PAID = "paid"


class SourceUnusableError(RuntimeError):
    """Raised when a source is asked for something it cannot physically give."""


# The outer bound of any horizon this track would actually trade. Anything
# slower is a research input, not a signal input.
#
# This constant is what makes `is_signal_capable` mean something. Without it,
# "can this support a signal?" is trivially true for every non-snapshot source
# — annual data "supports" a two-year-horizon signal, which is true and useless.
TRADEABLE_HORIZON = timedelta(days=30)


@dataclass(frozen=True)
class Source:
    """A data source annotated with what it can and cannot answer.

    `publication_lag` is the delay between an event occurring and the data
    describing it becoming available — not the API's response time. USGS annual
    production figures describe a year that ended up to 18 months prior.
    """

    key: str
    name: str
    url: str
    cost: Cost
    frequency: Frequency
    publication_lag: timedelta
    coverage: str
    supports_claim: str  # "(a) research", "(c) signal", or both
    notes: str = ""

    # ------------------------------------------------------------------
    @property
    def min_signal_horizon(self) -> timedelta:
        """Shortest alpha horizon this source could possibly support.

        You need at least one fresh observation plus the time it takes that
        observation to arrive. Anything shorter is predicting the past.
        """
        return self.frequency.period + self.publication_lag

    @property
    def has_time_axis(self) -> bool:
        """SNAPSHOT sources have nothing to difference — no returns, no events."""
        return self.frequency is not Frequency.SNAPSHOT

    @property
    def is_signal_capable(self) -> bool:
        """Can this source support a signal at a horizon we would trade?

        Note this is stricter than `has_time_axis`. USGS has a time axis (one
        point per year) but cannot inform anything inside 30 days.
        """
        return (self.has_time_axis
                and self.min_signal_horizon <= TRADEABLE_HORIZON)

    def require_horizon(self, horizon: timedelta) -> None:
        """Assert this source can support a signal at `horizon`, else raise.

        Call this at the top of any code path that turns a source into a
        predictive feature.
        """
        if not self.has_time_axis:
            raise SourceUnusableError(
                f"{self.key}: snapshot-only source has no time axis; it cannot "
                f"support any signal horizon. Usable for research claim (a) only."
            )
        if horizon < self.min_signal_horizon:
            raise SourceUnusableError(
                f"{self.key}: requested horizon {horizon} is shorter than the "
                f"minimum this source can support ({self.min_signal_horizon} = "
                f"{self.frequency.name.lower()} spacing + {self.publication_lag} "
                f"publication lag). Predicting at this horizon would be "
                f"predicting the past. Use it for research, or find a faster source."
            )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
# Ordered roughly fastest -> slowest. Note how few are signal-capable: that
# scarcity IS the finding, and it is why the track gates on one probe rather
# than building a general ingestion layer first.

SOURCES: dict[str, Source] = {
    s.key: s
    for s in [
        Source(
            key="eskom_stages",
            name="Eskom loadshedding stage",
            url="https://eskomsepush.gumroad.com/l/api (free tier token)",
            cost=Cost.TOKEN,
            frequency=Frequency.DAILY,
            publication_lag=timedelta(0),
            coverage="South Africa, national grid stage 0-8",
            supports_claim="(c) signal",
            notes=(
                "The only genuinely high-frequency, causally-linked African "
                "supply signal identified so far, and the reason the decisive "
                "probe is built on it. Free tier is oriented to CURRENT and "
                "near-term schedules; deep history is the unmet dependency and "
                "must be supplied as CSV. See chokepoint/data/eskom.py."
            ),
        ),
        Source(
            key="prices",
            name="Commodity + equity daily bars (yfinance)",
            url="https://finance.yahoo.com",
            cost=Cost.FREE,
            frequency=Frequency.DAILY,
            publication_lag=timedelta(0),
            coverage="PGM/metal ETFs, miner ADRs, FX, broad market",
            supports_claim="(c) signal",
            notes=(
                "The price side, not the event side. Free history is revised "
                "and survivorship-biased: sufficient to falsify a thesis, not "
                "to size a position."
            ),
        ),
        Source(
            key="comtrade",
            name="UN Comtrade trade flows",
            url="https://comtradeplus.un.org",
            cost=Cost.FREE,
            frequency=Frequency.MONTHLY,
            publication_lag=timedelta(days=60),
            coverage="bilateral commodity trade, most countries, HS codes",
            supports_claim="(a) research",
            notes=(
                "Monthly with ~2mo lag, and African reporting is frequently "
                "late or absent — mirror-statistics (partner-reported) are "
                "often more complete than direct reports. Good for value-capture "
                "analysis; too slow and too revised for signal."
            ),
        ),
        Source(
            key="usgs_mcs",
            name="USGS Mineral Commodity Summaries",
            url="https://www.usgs.gov/centers/national-minerals-information-center",
            cost=Cost.FREE,
            frequency=Frequency.ANNUAL,
            publication_lag=timedelta(days=365),
            coverage="global production + reserves by country and commodity",
            supports_claim="(a) research",
            notes=(
                "The canonical source for the '~30% of world reserves' class of "
                "claim. Annual, heavily revised, reserves are estimates that "
                "move with price as much as with geology. Research only."
            ),
        ),
        Source(
            key="bgs_wms",
            name="BGS World Mineral Statistics",
            url="https://www2.bgs.ac.uk/mineralsuk/statistics/worldStatistics.html",
            cost=Cost.FREE,
            frequency=Frequency.ANNUAL,
            publication_lag=timedelta(days=548),
            coverage="production by country, long historical series",
            supports_claim="(a) research",
            notes="Deeper history than USGS; correspondingly longer lag.",
        ),
        Source(
            key="eiti",
            name="Extractive Industries Transparency Initiative",
            url="https://eiti.org/countries",
            cost=Cost.FREE,
            frequency=Frequency.ANNUAL,
            publication_lag=timedelta(days=548),
            coverage="government revenue, company payments, member countries",
            supports_claim="(a) research",
            notes=(
                "The best free source for claim (a) specifically — it is about "
                "who captures the money, which production data cannot show. "
                "Coverage is voluntary, so absence is not zero."
            ),
        ),
        Source(
            key="worldbank_wdi",
            name="World Bank World Development Indicators",
            url="https://databank.worldbank.org/source/world-development-indicators",
            cost=Cost.FREE,
            frequency=Frequency.ANNUAL,
            publication_lag=timedelta(days=365),
            coverage="resource rents as % GDP, all African countries",
            supports_claim="(a) research",
        ),
    ]
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get(key: str) -> Source:
    try:
        return SOURCES[key]
    except KeyError:
        raise KeyError(
            f"unknown source {key!r}; known: {sorted(SOURCES)}"
        ) from None


def signal_capable(max_horizon: timedelta = TRADEABLE_HORIZON) -> Iterator[Source]:
    """Yield sources that can support a signal at or below `max_horizon`."""
    for s in SOURCES.values():
        if s.has_time_axis and s.min_signal_horizon <= max_horizon:
            yield s


def _check_registry_invariant() -> None:
    """The declared `supports_claim` must match the mechanical capability.

    Catches the drift where someone adds a slow source, labels it "(c) signal"
    out of optimism, and builds a feature on it. The arithmetic wins over the
    label, and the mismatch is loud at import time.
    """
    for s in SOURCES.values():
        declares_signal = "(c)" in s.supports_claim
        if declares_signal and not s.is_signal_capable:
            raise AssertionError(
                f"registry inconsistent: {s.key} declares '{s.supports_claim}' "
                f"but its minimum horizon is {s.min_signal_horizon.days}d, "
                f"beyond TRADEABLE_HORIZON ({TRADEABLE_HORIZON.days}d)."
            )
        if s.is_signal_capable and not declares_signal:
            raise AssertionError(
                f"registry inconsistent: {s.key} is fast enough for a signal "
                f"({s.min_signal_horizon.days}d) but declares only "
                f"'{s.supports_claim}'."
            )


_check_registry_invariant()


def summary() -> str:
    """Human-readable table. `python -m chokepoint.data.sources`."""
    rows = [
        (
            s.key,
            s.cost.value,
            s.frequency.name.lower(),
            f"{s.publication_lag.days}d",
            f"{s.min_signal_horizon.days}d" if s.has_time_axis else "n/a",
            s.supports_claim,
        )
        for s in SOURCES.values()
    ]
    head = ("source", "cost", "freq", "lag", "min horizon", "supports")
    widths = [
        max(len(str(r[i])) for r in (*rows, head)) for i in range(len(head))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*head), fmt.format(*("-" * w for w in widths))]
    out += [fmt.format(*r) for r in rows]

    sig = sorted(s.key for s in signal_capable())
    out.append("")
    out.append(
        f"{len(sig)} of {len(SOURCES)} sources can support a signal at the "
        f"{TRADEABLE_HORIZON.days}d horizon this track would trade: {sig}"
    )
    out.append(
        "  The other "
        f"{len(SOURCES) - len(sig)} are research-only (claim (a)). Annual "
        "reserve data cannot"
    )
    out.append(
        "  answer a days-horizon question, no matter how many countries it "
        "covers — and"
    )
    out.append(
        "  the event side of the thesis rests on exactly one source. That "
        "scarcity is the"
    )
    out.append(
        "  finding: it is why the track gates on one probe instead of building "
        "ingestion."
    )
    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover
    print(summary())
