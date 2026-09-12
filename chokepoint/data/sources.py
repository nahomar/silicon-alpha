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
#
# Raised 30d -> 90d when the track pivoted. The original value was set for the
# Eskom test, which asked whether a grid event moved PGM miners the NEXT DAY;
# that question died with an honest null (chokepoint_eskom_probe_result.md).
# The live question is now whether a supply dislocation is mispriced over weeks
# to months, so monthly customs data published four days after month-end is a
# legitimate signal input for it where it was not for the daily test.
#
# This is a real loosening of a standard and is recorded as such: it is
# justified by the horizon of the strategy, NOT by wanting more sources to
# qualify. If the horizon tightens again, this must come back down.
TRADEABLE_HORIZON = timedelta(days=90)


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
            key="lme_stocks",
            name="LME warehouse stocks + stock movements",
            url="https://www.lme.com/en/market-data/reports-and-data",
            cost=Cost.FREE,
            frequency=Frequency.DAILY,
            publication_lag=timedelta(days=1),
            coverage="global: copper, nickel, aluminium, zinc, lead, tin, cobalt",
            supports_claim="(c) signal",
            notes=(
                "Exchange inventory is the fastest honest read on physical "
                "tightness: metal leaving warehouses is a supply shock being "
                "absorbed, visible daily and long before customs statistics. "
                "It is NOT country-attributable — you see the world balance "
                "tightening, not which government caused it — so it pairs with "
                "a country source rather than replacing one.\n"
                "ACCESS: direct requests return HTTP 403 (Akamai); this is not "
                "fetchable from here. Copper inventories ARE obtainable monthly "
                "via `cochilco_cl`, which republishes LME/COMEX/SHFE stocks. "
                "For daily granularity or non-copper metals, this needs either "
                "a browser session or a paid LME licence."
            ),
        ),
        Source(
            key="shfe_stocks",
            name="Shanghai Futures Exchange inventory",
            url="https://www.shfe.com.cn",
            cost=Cost.FREE,
            frequency=Frequency.DAILY,
            publication_lag=timedelta(days=1),
            coverage="China: copper, nickel, zinc, lead, tin, aluminium",
            supports_claim="(c) signal",
            notes=(
                "Published after the Shanghai close (07:00 UTC). The closest "
                "thing to a daily window into Chinese metal balances, which "
                "matters because China is the largest chokepoint on this map "
                "(9 commodities) and is otherwise invisible. Weekly stock "
                "reports are the more-cited series; daily is available."
            ),
        ),
        Source(
            key="cme_cobalt",
            name="CME cobalt futures settlements (Fastmarkets-settled)",
            url="https://www.cmegroup.com/markets/metals/battery-metals/cobalt-metal-fastmarkets.settlements.html",
            cost=Cost.FREE,
            frequency=Frequency.DAILY,
            publication_lag=timedelta(days=1),
            coverage="global cobalt (DR Congo ~70% of mine supply)",
            supports_claim="(c) signal",
            notes=(
                "CME lists BOTH Cobalt Metal and Cobalt Hydroxide CIF China "
                "futures, settling against daily Fastmarkets assessments. "
                "Settlement prices are published free.\n"
                "This corrects an earlier claim in this track that cobalt has "
                "no tradeable instrument because nobody mines it on purpose. "
                "That is true of EQUITIES and false of futures: the exposure is "
                "directly expressible. Caveat is liquidity, not existence — "
                "these contracts are thin, so check volume and open interest "
                "before assuming a position can be entered or exited.\n"
                "Like LME stocks this is a world price, not a DRC statistic; "
                "but at ~70% mine share the two are close to the same thing."
            ),
        ),
        Source(
            key="esdm_hpm_id",
            name="Indonesia ESDM benchmark mineral price (HPM)",
            url="https://www.esdm.go.id",
            cost=Cost.FREE,
            frequency=Frequency.MONTHLY,
            publication_lag=timedelta(days=5),
            coverage="Indonesia: nickel ore benchmark price by grade",
            supports_claim="(c) signal",
            notes=(
                "Official government benchmark price published monthly, used "
                "as the tax basis for nickel ore. It is a POLICY price, not a "
                "market-clearing one — which makes it unusually informative "
                "here, because changes to it are deliberate government acts "
                "that reprice the whole Indonesian cost curve. The April 2026 "
                "formula revision more than doubled the effective benchmark on "
                "1.6% ore.\n"
                "Pairs with RKAB annual production quotas (also ESDM), which "
                "are the quantity lever to HPM's price lever."
            ),
        ),
        Source(
            key="comexstat_br",
            name="Brazil ComexStat foreign-trade statistics",
            url="https://api-comexstat.mdic.gov.br/docs",
            cost=Cost.FREE,
            frequency=Frequency.MONTHLY,
            publication_lag=timedelta(days=5),
            coverage="Brazil: exports/imports by HS code, value and tonnage",
            supports_claim="(c) signal",
            notes=(
                "Official government API, no key, no registration. VERIFIED "
                "LIVE: on 2026-09-12 it reported data updated 2026-09-04 "
                "covering month 08 — a four-day lag on monthly customs data, "
                "not the year-plus that USGS carries. Brazil is 90% of world "
                "niobium, so this is direct visibility into the single most "
                "concentrated chokepoint on the map."
            ),
        ),
        Source(
            key="cochilco_cl",
            name="Cochilco (Chilean Copper Commission) monthly bulletin",
            url="https://boletin.cochilco.cl/estadisticas/boletin.asp",
            cost=Cost.FREE,
            frequency=Frequency.MONTHLY,
            publication_lag=timedelta(days=30),
            coverage=(
                "Chile: copper production, prices, exports; AND republished "
                "LME/COMEX/SHFE copper inventories"
            ),
            supports_claim="(c) signal",
            notes=(
                "LOADER BUILT: chokepoint/data/cochilco.py. Plain HTML at a "
                "predictable URL, no bot protection.\n"
                "Its most valuable table is not Chilean at all: table 4_1 "
                "republishes exchange copper inventories for LME, COMEX and "
                "SHFE with month-on-month changes. That is the series the "
                "exchanges themselves refuse us (403/404), available from a "
                "government agency that does not block. Generalisable lesson: "
                "when a primary source is bot-walled, a national statistics "
                "agency consuming the same data is often open — at the cost of "
                "latency and of reading their transcription rather than the "
                "exchange's own print.\n"
                "CORRECTION: this entry previously claimed the bulletin carries "
                "production by individual company, which would have let us map "
                "national output onto listed equities. It does not — that lives "
                "in Cochilco's separate Excel database. The claim came from a "
                "search summary and was not verified against the bulletin."
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
            supports_claim="(a) research + (c) signal, marginally",
            notes=(
                "Sits exactly on the 90d boundary (monthly + 60d lag), so it "
                "qualifies as a signal input by arithmetic rather than by "
                "merit. Reclassified from research-only when TRADEABLE_HORIZON "
                "rose to 90d — the invariant check forced the relabel, which is "
                "what it is for.\n"
                "Treat it as the weakest signal source here: reporting from "
                "several African producers is late or absent, and mirror "
                "statistics (partner-reported) are often more complete than "
                "direct reports. Where a national source exists — ComexStat for "
                "Brazil, Cochilco for Chile — prefer it; those are both faster "
                "and more reliable."
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
        f"  The other {len(SOURCES) - len(sig)} are research-only (claim (a)): "
        f"annual reserve data cannot"
    )
    out.append(
        "  answer a weeks-horizon question, no matter how many countries it "
        "covers."
    )
    out.append("")
    out.append(
        "  NOTE: an earlier version of this summary asserted that the event "
        "side of the"
    )
    out.append(
        "  thesis rested on a single source and called that scarcity the "
        "finding. That"
    )
    out.append(
        "  was true when only Eskom was registered. It stopped being true "
        "after a"
    )
    out.append(
        "  deliberate search turned up exchange inventories, national customs "
        "APIs and"
    )
    out.append(
        "  government benchmark prices. The scarcity was in the looking, not "
        "in the world."
    )
    return "\n".join(out)


if __name__ == "__main__":  # pragma: no cover
    print(summary())
