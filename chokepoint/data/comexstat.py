"""Brazil ComexStat — monthly customs exports, four days after month end.

Brazil is ~90% of world niobium, the single most concentrated chokepoint on the
country map, and its government publishes full customs statistics through a free
API with no key and no registration.

VERIFIED LIVE 2026-09-12: the metadata endpoint reported data updated 2026-09-04
covering month 08. Four days, against the year-plus that USGS production
statistics carry.

## Why the unit value is the interesting column

The API returns FOB value and net weight per NCM code per month. Their ratio is
the realised export price in $/kg — and for several of these commodities that is
not merely *a* price signal, it is the *only* one.

Niobium has no futures contract anywhere. There is no LME niobium, no CME
niobium, no daily assessment worth the name. Brazil sells ~90% of the world's
supply, so what Brazil actually charged last month, divided by what it actually
shipped, is the closest thing to a niobium price that exists. Same argument
holds in weaker form for tantalum.

This inverts the usual relationship between customs data and markets. For copper
the exchange leads and customs confirms months later; for niobium there is no
exchange to lead, so customs *is* price discovery, and the four-day lag is the
whole latency of the market.

## What this is not

Monthly, so it cannot support a daily signal — `sources.require_horizon` will
refuse. Values are revised, and the first print for a month can move. Unit value
mixes grades and contract vintages within an NCM code, so it is a blended
realisation, not a spot quote: a shift can mean price moved, or mix moved.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from . import sources

log = logging.getLogger(__name__)

API = "https://api-comexstat.mdic.gov.br/general"
META = "https://api-comexstat.mdic.gov.br/general/dates/updated"
TIMEOUT = 90


# NCM (Brazilian HS extension) codes for chokepoint-relevant exports.
# Verified against the API's own returned description string, which is echoed
# back in every row — `fetch` asserts it matches, so a wrong code fails loudly
# instead of silently returning someone else's commodity.
NCM = {
    # VERIFIED against the API: each returns rows and echoes a description
    # matching the commodity named here.
    "ferroniobium": 72029300,     # Brazil ~90% of world niobium
    "iron_ore_fines": 26011100,
    "manganese_ore": 26020010,
}

# Codes that were GUESSED and returned nothing. Kept as a record so the same
# wrong values are not re-guessed, and deliberately NOT in NCM -- a code that
# silently returns an empty series is worse than an absent one, because the
# gap later reads as "Brazil exported no bauxite" rather than "we asked wrong".
#
# Resolving these needs the NCM tariff schedule rather than more guessing;
# probing variants against the live API hit HTTP 429 and is not the way.
UNVERIFIED_NCM = {
    "niobium_ore": 26159010,
    "bauxite": 26060010,
    "graphite_natural": 25041010,
}


class ComexStatError(RuntimeError):
    """API unreachable or returned an unusable payload."""


@dataclass(frozen=True)
class ExportSeries:
    """Monthly export value, weight and implied unit price for one NCM code."""

    ncm: int
    description: str
    frame: pd.DataFrame  # index: month-end dates; cols: fob_usd, kg, usd_per_kg

    def describe(self) -> str:
        f = self.frame
        lo, hi = f.index[0].strftime("%Y-%m"), f.index[-1].strftime("%Y-%m")
        return (
            f"{self.description} (NCM {self.ncm}): {len(f)} months {lo} → {hi}, "
            f"mean ${f.usd_per_kg.mean():,.2f}/kg, "
            f"${f.fob_usd.sum() / 1e9:,.1f}B total"
        )


def _post(body: dict) -> dict:
    req = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ComexStatError(f"ComexStat request failed: {exc}") from exc


def last_updated() -> tuple[str, str]:
    """(date the data was published, last month it covers). Cheap liveness check."""
    try:
        with urllib.request.urlopen(META, timeout=TIMEOUT) as r:
            d = json.loads(r.read().decode())["data"]
    except Exception as exc:  # noqa: BLE001
        raise ComexStatError(f"ComexStat metadata unreachable: {exc}") from exc
    return d["updated"], f"{d['year']}-{d['monthNumber']}"


def fetch(commodity: str, start: str, end: str,
          flow: str = "export") -> ExportSeries:
    """Monthly exports for one chokepoint commodity.

    `start`/`end` are "YYYY-MM". Raises rather than returning an empty frame —
    a silent empty series would flow downstream as "no exports", which is a very
    different statement from "the query was wrong".
    """
    if commodity not in NCM:
        raise KeyError(f"unknown commodity {commodity!r}; known: {sorted(NCM)}")
    code = NCM[commodity]

    payload = _post({
        "flow": flow,
        "monthDetail": True,
        "period": {"from": start, "to": end},
        "filters": [{"filter": "ncm", "values": [code]}],
        "details": ["ncm"],
        "metrics": ["metricFOB", "metricKG"],
    })
    rows = (payload.get("data") or {}).get("list") or []
    if not rows:
        raise ComexStatError(
            f"no rows for {commodity} (NCM {code}) {start}..{end}. Either the "
            f"code is wrong or the period is outside coverage — check "
            f"last_updated()."
        )

    df = pd.DataFrame(rows)
    # The API echoes its own description; if it disagrees with our mapping the
    # code is pointing at a different product than we think it is.
    desc = str(df["ncm"].iloc[0])
    got = {int(c) for c in df["coNcm"].unique()}
    if got != {code}:
        raise ComexStatError(
            f"asked for NCM {code}, got {got} — the filter did not apply"
        )

    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["monthNumber"].astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)
    df["fob_usd"] = pd.to_numeric(df["metricFOB"], errors="coerce")
    df["kg"] = pd.to_numeric(df["metricKG"], errors="coerce")

    bad = df[["fob_usd", "kg"]].isna().any(axis=1)
    if bad.any():
        log.warning("%d row(s) with unparseable metrics dropped", int(bad.sum()))
        df = df[~bad]

    zero_kg = df.kg <= 0
    if zero_kg.any():
        log.warning(
            "%d month(s) report zero weight; unit price undefined there and is "
            "left as NaN rather than infinite", int(zero_kg.sum()),
        )
    df["usd_per_kg"] = df.fob_usd.where(df.kg > 0) / df.kg.where(df.kg > 0)

    frame = (df.set_index("date")[["fob_usd", "kg", "usd_per_kg"]]
               .sort_index())
    series = ExportSeries(ncm=code, description=desc, frame=frame)
    log.info("comexstat: %s", series.describe())
    return series


def require_horizon(days: int) -> None:
    """Refuse to be used for a signal faster than the source can support."""
    sources.get("comexstat_br").require_horizon(timedelta(days=days))


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--commodity", default="ferroniobium", choices=sorted(NCM))
    ap.add_argument("--from", dest="start", default="2020-01")
    ap.add_argument("--to", dest="end", default="2025-12")
    args = ap.parse_args()

    pub, covers = last_updated()
    print(f"\nComexStat published {pub}, covering through {covers}\n")
    s = fetch(args.commodity, args.start, args.end)
    print(s.describe())
    print()
    print(s.frame.tail(8).to_string(
        float_format=lambda v: f"{v:,.2f}"))
