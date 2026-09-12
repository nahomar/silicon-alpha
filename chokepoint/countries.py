"""Part 1 — the country dependency map: who does the world depend on, and can we see them?

Country, not commodity, is the right unit. South Africa is not a platinum
story: when Eskom fails it constrains platinum, chromium, manganese and gold at
the same time. One national event, four markets. Indexing by commodity splits a
single event into four and hides that they share a cause.

So this flips the production table onto its country axis and asks two questions
per country:

  1. **How much leverage does it have?** How much of world supply could it
     disrupt, across everything it produces.
  2. **Can we see inside it?** What is the fastest data we have on it — from
     `chokepoint.data.sources`.

The second question is the one that decides whether any of this is tradeable,
and it is the reason the source registry records frequency and publication lag.
Leverage without visibility is not an opportunity; it is a country you will read
about in the news at the same time as everyone else.

## Three numbers, reported separately

There is no single "importance" index here, deliberately — collapsing these
into one score would hide exactly the distinction that matters.

- **chokehold** — the largest share it holds in any one commodity. Its single
  strongest grip. DR Congo's 70% of cobalt.
- **breadth** — how many commodities it holds a material share of. A country
  with one chokehold is a single-commodity risk; one with six is a systemic one.
- **supply_at_risk** — summed share across the commodities it materially
  controls. A rough ceiling on how much world supply one national disruption
  could touch.

A country ranks as a chokepoint on `chokehold` alone. Breadth is what makes it
dangerous.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from chokepoint.data import sources
from chokepoint.research import concentration

log = logging.getLogger(__name__)

# A country needs at least this share of a commodity for a disruption there to
# plausibly move the world price. Below it, other producers absorb the shock.
MATERIAL_SHARE = 0.20

# At or above this, the country is a single point of failure for that commodity.
CHOKEHOLD_SHARE = 0.40

# Fastest data that still counts as "we could see it coming". Tied to the
# registry's own definition so the two cannot drift apart — the horizon the
# track trades is the horizon that decides whether a country is watchable.
WATCHABLE_HORIZON = sources.TRADEABLE_HORIZON


@dataclass
class CountryProfile:
    country: str
    chokehold: float                      # 0-1, largest single-commodity share
    chokehold_commodity: str
    breadth: int                          # commodities held >= MATERIAL_SHARE
    supply_at_risk: float                 # summed share over those
    commodities: list[tuple[str, float]] = field(default_factory=list)
    fastest_source: sources.Source | None = None

    @property
    def is_chokepoint(self) -> bool:
        return self.chokehold >= CHOKEHOLD_SHARE

    @property
    def is_watchable(self) -> bool:
        """Do we have any data on this country faster than ~monthly?"""
        s = self.fastest_source
        return s is not None and s.min_signal_horizon <= WATCHABLE_HORIZON

    @property
    def actionable(self) -> bool:
        """Leverage AND visibility. Only these are worth building against."""
        return self.is_chokepoint and self.is_watchable

    @property
    def visibility(self) -> str:
        s = self.fastest_source
        if s is None:
            return "none"
        return f"{s.frequency.name.lower()} (+{s.publication_lag.days}d)"


# Country names as they appear in the share table, matched against the free-text
# `coverage` field of each registered source. Listed explicitly rather than
# inferred so that adding a source without wiring it to a country is a visible
# omission rather than a silent one.
_COUNTRY_NAMES = (
    "south africa", "brazil", "chile", "china", "indonesia", "dr congo",
    "russia", "australia", "kazakhstan", "peru", "guinea", "zambia",
    "zimbabwe", "canada", "morocco", "gabon",
)


def _country_sources() -> dict[str, sources.Source]:
    """Fastest registered source per country.

    Sources record coverage as free text, so this matches country names against
    it. Anything unmatched falls back to the best *global* source — USGS annual
    — which is the honest default: for most producing countries, annual
    production data published a year in arrears is genuinely all we hold.
    """
    out: dict[str, sources.Source] = {}
    for s in sources.SOURCES.values():
        cov = s.coverage.lower()
        for name in _COUNTRY_NAMES:
            if name not in cov:
                continue
            key = "DR Congo" if name == "dr congo" else name.title()
            prev = out.get(key)
            if prev is None or s.min_signal_horizon < prev.min_signal_horizon:
                out[key] = s
    return out


def build(shares_path: str | None = None) -> list[CountryProfile]:
    """Load the share table and build the country map."""
    return build_from(concentration.load_shares(shares_path))


def build_from(df: pd.DataFrame) -> list[CountryProfile]:
    """Build the country map from an in-memory share table.

    Split out from `build` so the ranking logic is testable without a file on
    disk — the seed table is approximate and should not be what the tests pin.
    """
    per_country = _country_sources()
    global_default = sources.get("usgs_mcs")

    profiles: list[CountryProfile] = []
    for country, grp in df.groupby("country"):
        shares = (grp.set_index("commodity")["share_pct"].astype(float) / 100.0)
        material = shares[shares >= MATERIAL_SHARE].sort_values(ascending=False)
        top_commodity = shares.idxmax()

        profiles.append(CountryProfile(
            country=str(country),
            chokehold=float(shares.max()),
            chokehold_commodity=str(top_commodity),
            breadth=int(len(material)),
            supply_at_risk=float(material.sum()),
            commodities=[(str(c), float(v)) for c, v in material.items()],
            fastest_source=per_country.get(str(country), global_default),
        ))

    return sorted(profiles, key=lambda p: (p.chokehold, p.breadth), reverse=True)


def report(profiles: list[CountryProfile]) -> None:
    chokes = [p for p in profiles if p.is_chokepoint]

    print(f"\n{'country':<18} {'chokehold':>10}  {'on':<13} {'breadth':>7} "
          f"{'at risk':>9}  visibility")
    print("-" * 78)
    for p in chokes:
        flag = "  <<< ACTIONABLE" if p.actionable else ""
        # supply_at_risk is a SUM of shares across different commodities, so it
        # exceeds 1.0 for multi-commodity countries. Rendering it as a percent
        # would read as a broken calculation; "4.6x" means four and a half
        # commodities' worth of world supply under one government.
        print(f"{p.country:<18} {p.chokehold:>9.0%}  {p.chokehold_commodity:<13} "
              f"{p.breadth:>7} {p.supply_at_risk:>7.1f}x  {p.visibility}{flag}")

    print(f"\n{len(chokes)} of {len(profiles)} countries hold >= "
          f"{CHOKEHOLD_SHARE:.0%} of at least one commodity.")

    print("\nMulti-commodity countries (one national event hits several markets):")
    for p in sorted(chokes, key=lambda p: p.breadth, reverse=True)[:5]:
        if p.breadth < 2:
            continue
        items = ", ".join(f"{c} {v:.0%}" for c, v in p.commodities)
        print(f"  {p.country}: {items}")

    actionable = [p for p in chokes if p.actionable]
    blind = [p for p in chokes if not p.is_watchable]

    print("\n" + "=" * 78)
    print(f"ACTIONABLE (leverage AND data inside the "
          f"{WATCHABLE_HORIZON.days}d horizon): {len(actionable)}")
    for p in actionable:
        print(f"  {p.country} — {p.visibility}")
    print(f"\nBLIND (real leverage, nothing inside the "
          f"{WATCHABLE_HORIZON.days}d horizon): {len(blind)}")
    print("  " + ", ".join(p.country for p in blind))
    print("=" * 78)

    if len(actionable) <= 1:
        print(
            "\nVERDICT: leverage is not the scarce resource here — visibility is.\n"
            "  This was the pre-declared stop condition, and it is the finding:\n"
            "  every blind country above has world-moving supply power while we\n"
            "  hold nothing on it faster than annual production statistics\n"
            "  published a year in arrears.\n\n"
            "  READ THIS HONESTLY. The column reflects the CURRENT CONTENTS of\n"
            "  chokepoint.data.sources, not the state of the world. Exactly one\n"
            "  country-specific fast source has ever been registered (Eskom grid\n"
            "  stages), so 'only South Africa is watchable' partly restates that\n"
            "  we have only ever gone looking once. Fast data may well exist for\n"
            "  Chile, Indonesia or Brazil — port throughput, export licences,\n"
            "  utility load — and has simply not been searched for.\n\n"
            "  So the next step is NOT a model. It is a deliberate hunt for fast\n"
            "  country-level data on the seven blind countries, priced as a\n"
            "  data-acquisition project. Only if that hunt fails is the real\n"
            "  conclusion 'these countries are unwatchable'."
        )
    else:
        print(f"\nVERDICT: {len(actionable)} countries carry both leverage and\n"
              f"  visibility. Those are the only ones worth instrumenting.\n"
              f"  Note this reflects the current source registry, not the limit\n"
              f"  of what data exists — unregistered fast sources may exist for\n"
              f"  the blind countries above.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shares", default=None)
    args = ap.parse_args()
    report(build(args.shares))


if __name__ == "__main__":  # pragma: no cover
    main()
