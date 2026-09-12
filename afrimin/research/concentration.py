"""Claim (a): African supply share and global production concentration.

This is the *research* half of the track and makes no trading claim. It answers
"how concentrated is global supply of commodity X, and how much of it is
African?" — which is the quantitative form of the motivating observation, and is
appropriately served by annual data (see `afrimin.data.sources`: USGS and BGS
are research-capable and explicitly NOT signal-capable).

Two measures:

- **Africa share** — fraction of global mined production from African countries.
  The headline number people quote.
- **HHI** (Herfindahl-Hirschman Index) over producer countries, 0-10,000. This
  is the more informative one: a commodity can be 30% African and unconcentrated
  (many producers, substitutable) or 70% African and a single-country chokepoint.
  Only the second creates the supply-shock transmission that claim (c) tests.
  Conventional reading: >2,500 is highly concentrated.

  Computed over listed producers only, so it is a **lower bound** on true HHI —
  see the note in `profile()` for why a residual bucket would be worse than
  omitting the tail.

**On the input data.** Production shares are loaded from CSV. A small seed file
ships with the module so the computation is runnable, carrying widely-cited
approximate figures — they are adequate for structure and ordering and are NOT
research-grade. Verify against the current USGS Mineral Commodity Summaries
before citing any number from this module in anything that leaves your machine.
Reserve figures in particular are estimates that move with price as much as with
geology.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

SEED_CSV = Path(__file__).with_name("production_shares_seed.csv")

# Sub-Saharan + North Africa. Used to compute the aggregate share.
AFRICAN_COUNTRIES = {
    "South Africa", "DR Congo", "Zimbabwe", "Zambia", "Guinea", "Ghana",
    "Mali", "Tanzania", "Namibia", "Botswana", "Morocco", "Egypt", "Nigeria",
    "Burkina Faso", "Madagascar", "Mozambique", "Gabon", "Ethiopia", "Sudan",
    "Ivory Coast", "Sierra Leone", "Liberia", "Niger", "Angola", "Algeria",
}

HHI_CONCENTRATED = 2500


@dataclass(frozen=True)
class CommodityProfile:
    commodity: str
    africa_share: float          # 0-1
    hhi: float                   # 0-10000
    top_producer: str
    top_share: float             # 0-1
    n_producers: int

    @property
    def is_concentrated(self) -> bool:
        return self.hhi >= HHI_CONCENTRATED

    @property
    def is_african_chokepoint(self) -> bool:
        """Concentrated AND the dominant producer is African.

        This is the subset where a supply disruption could plausibly move a
        global price — i.e. the only subset where claim (c) is even testable.
        """
        return self.is_concentrated and self.top_producer in AFRICAN_COUNTRIES


def load_shares(path: str | Path | None = None) -> pd.DataFrame:
    """Load a commodity x country production-share table.

    Schema: commodity,country,share_pct  (share_pct = % of global production)
    """
    p = Path(path) if path else SEED_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"no production share data at {p}. Expected CSV with columns "
            f"commodity,country,share_pct (share_pct = percent of global "
            f"mined production). Source: USGS Mineral Commodity Summaries."
        )
    df = pd.read_csv(p, comment="#")
    missing = {"commodity", "country", "share_pct"} - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing column(s): {sorted(missing)}")

    if path is None:
        log.warning(
            "using the SEED share table — approximate, widely-cited figures "
            "adequate for structure but not research-grade. Verify against the "
            "current USGS Mineral Commodity Summaries before citing."
        )
    return df


def profile(df: pd.DataFrame) -> list[CommodityProfile]:
    out: list[CommodityProfile] = []
    for commodity, grp in df.groupby("commodity"):
        shares = grp.set_index("country")["share_pct"].astype(float)
        total = shares.sum()
        if total > 100.5:
            log.warning("%s: shares sum to %.1f%% (>100)", commodity, total)

        frac = shares / 100.0
        africa = float(frac[frac.index.isin(AFRICAN_COUNTRIES)].sum())

        # HHI over LISTED producers only, with no residual bucket.
        #
        # Lumping the unlisted tail into one synthetic "rest of world" producer
        # would square it as though it were a single firm, which massively
        # overstates concentration for long-tailed commodities — gold, listed
        # to 48%, would score as "concentrated" on the strength of a 52% lump
        # that is in reality dozens of small producers.
        #
        # Summing only the listed squares instead yields a strict LOWER BOUND
        # on true HHI (the omitted tail can only add a small positive amount),
        # and the bound is tight precisely when the tail is fragmented — which
        # is the case whenever it matters. Understating concentration is also
        # the safe direction here: it can only make us *less* likely to claim a
        # chokepoint exists.
        hhi = float((frac**2).sum() * 10_000)
        top = frac.idxmax()
        out.append(CommodityProfile(
            commodity=str(commodity),
            africa_share=float(africa),
            hhi=hhi,
            top_producer=str(top),
            top_share=float(frac[top]),
            n_producers=int(len(grp)),
        ))
    return sorted(out, key=lambda c: c.africa_share, reverse=True)


def report(profiles: list[CommodityProfile]) -> None:
    print(f"\n{'commodity':<14} {'Africa %':>9} {'HHI':>7} {'top producer':<14} "
          f"{'top %':>7}  flags")
    print("-" * 70)
    for c in profiles:
        flags = []
        if c.is_african_chokepoint:
            flags.append("AFRICAN CHOKEPOINT")
        elif c.is_concentrated:
            flags.append("concentrated")
        print(f"{c.commodity:<14} {c.africa_share:>8.1%} {c.hhi:>7.0f} "
              f"{c.top_producer:<14} {c.top_share:>6.1%}  {', '.join(flags)}")

    chokepoints = [c for c in profiles if c.is_african_chokepoint]
    print(f"\n{len(chokepoints)} of {len(profiles)} commodities are African "
          f"chokepoints (HHI >= {HHI_CONCENTRATED} with an African top producer):")
    print(f"  {[c.commodity for c in chokepoints]}")
    print(
        "\nThis subset — not the aggregate '30% of world minerals' figure — is\n"
        "where a supply disruption could plausibly move a global price, and so\n"
        "is the only place the transmission-lag thesis is even testable.\n"
        "An aggregate reserve share says nothing about price formation: broad,\n"
        "substitutable supply absorbs shocks that a single-country chokepoint\n"
        "transmits. Concentration is the mechanism; share is just the headline."
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shares", default=None,
                    help="CSV: commodity,country,share_pct (default: seed table)")
    args = ap.parse_args()
    report(profile(load_shares(args.shares)))


if __name__ == "__main__":  # pragma: no cover
    main()
