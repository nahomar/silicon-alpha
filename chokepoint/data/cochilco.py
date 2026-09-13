"""Cochilco monthly bulletin — Chilean copper, and a way past the exchange paywall.

Cochilco is the Chilean Copper Commission, a government agency. Its monthly
bulletin is plain HTML at a predictable URL and is not behind bot protection.

## Why this matters beyond Chile

Direct requests to LME and CME return HTTP 403 (Akamai), and the SHFE daily
files 404. Those were recorded as blocked sources.

Cochilco republishes **exchange copper inventories for LME, COMEX and SHFE** in
table 4_1 of every bulletin, with month-on-month changes. So the inventory
series that could not be fetched from the exchanges directly is available from
a government agency that does not block us.

That is worth stating plainly because it generalises: when a primary source is
bot-walled, a national statistics agency that consumes the same data is often
open. The cost is latency (monthly, not daily) and trust (we are reading their
transcription, not the exchange's own print).

## Table map (verified 2025-06 bulletin)

    tabla1     copper price, nominal and real, annual + monthly
    tabla3     copper futures curve out to 27 months
    tabla4_1   EXCHANGE INVENTORIES — LME / COMEX / SHFE, totals + changes
    tabla7_1   gold prices
    tabla9     molybdenum prices
    tabla10    platinum, palladium, selenium
    tabla18_1  Chilean mining export shipment values
    tabla21    Chilean copper production by product

Note what is NOT here: production by individual company. That lives in
Cochilco's separate Excel database, not the bulletin — a correction to this
track's earlier note, which claimed the bulletin carried company-level output
and would have had us map national statistics onto listed equities from a table
that does not exist.

## HTML shape

Each `<td>` packs several periods separated by `<br>`, aligned positionally
across columns:

    <td>2021<br>2022<br>2023</td><td>88.950<br>88.925<br>167.300</td>

so a cell is a column fragment, not a value. Numbers are European-formatted:
`88.950` is eighty-eight thousand, `1.414,4` is one thousand four hundred.
"""
from __future__ import annotations

import logging
import re
import urllib.request
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

BASE = "https://boletin.cochilco.cl/productos/boletin.asp"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")
TIMEOUT = 60

TABLES = {
    "copper_price": "tabla1",
    "copper_futures": "tabla3",
    "exchange_inventories": "tabla4_1",
    "gold_price": "tabla7_1",
    "moly_price": "tabla9",
    "pgm_price": "tabla10",
    "export_values": "tabla18_1",
    "copper_production": "tabla21",
}

_MONTHS = {
    "ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DIC": 12,
}


class CochilcoError(RuntimeError):
    """Bulletin unreachable or in an unexpected shape."""


def _euro_number(s: str) -> float | None:
    """Parse Chilean/European numerics: '88.950' -> 88950, '1.414,4' -> 1414.4.

    Returns None rather than 0.0 for blanks and placeholders. A missing
    observation read as zero inventory would be a severe and silent error —
    zero stock is a market event, absence of data is not.
    """
    s = (s or "").strip().replace("\xa0", " ")
    if not s or s in {"-", "--", "n/d", "nd", "s/i"}:
        return None
    s = re.sub(r"[^\d,.\-]", "", s)
    if not s or s in {"-", ".", ","}:
        return None
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_period(s: str) -> pd.Timestamp | None:
    """'2021' -> year end; 'ENE/JAN 2024' -> month end."""
    s = (s or "").strip()
    if re.fullmatch(r"(19|20)\d{2}", s):
        return pd.Timestamp(int(s), 12, 31)
    m = re.match(r"([A-Z]{3})\s*/?\s*[A-Z]*\s*((19|20)\d{2})", s.upper())
    if m and m.group(1) in _MONTHS:
        return (pd.Timestamp(int(m.group(2)), _MONTHS[m.group(1)], 1)
                + pd.offsets.MonthEnd(0))
    return None


def _parse_period_column(fragments: list[str]) -> list[pd.Timestamp | None]:
    """Parse a period column, carrying the year forward across bare months.

    The bulletin writes a monthly row as:

        ENE/JAN 2022 | FEB | MAR | ABR/APR | MAY | ... | DIC/DEC

    The year is stated once, on January, and every later month is bare. Parsing
    each fragment independently therefore recovers ONE month in twelve and
    silently discards the rest — which is exactly what happened here: table 4_1
    yielded 8 sparse periods instead of four years of monthly observations, with
    no error raised, and very nearly produced a 'not enough history to test'
    conclusion out of a parsing defect.

    So a bare month inherits the year of the last dated fragment in its own row.
    Rows that never state a year yield None throughout rather than guessing.
    """
    out: list[pd.Timestamp | None] = []
    year: int | None = None
    for frag in fragments:
        dated = _parse_period(frag)
        if dated is not None:
            year = dated.year
            out.append(dated)
            continue
        m = re.match(r"([A-Z]{3})", (frag or "").strip().upper())
        if m and m.group(1) in _MONTHS and year is not None:
            out.append(pd.Timestamp(year, _MONTHS[m.group(1)], 1)
                       + pd.offsets.MonthEnd(0))
        else:
            out.append(None)
    return out


def _fetch_html(year: int, month: int, table: str) -> str:
    url = f"{BASE}?anio={year}&mes={month:02d}&tabla={table}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read().decode("latin-1", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        raise CochilcoError(f"cochilco unreachable ({url}): {exc}") from exc


def _rows_from_html(html: str) -> list[list[list[str]]]:
    """Every <tr> as a list of cells, each cell split on <br> into fragments."""
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I):
        cells = []
        for td in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I):
            parts = re.split(r"<br\s*/?>", td, flags=re.I)
            cells.append([
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", p)).strip()
                for p in parts
            ])
        if cells:
            out.append(cells)
    return out


@dataclass(frozen=True)
class Bulletin:
    year: int
    month: int
    table: str
    frame: pd.DataFrame

    def describe(self) -> str:
        f = self.frame
        if f.empty:
            return f"{self.table} {self.year}-{self.month:02d}: EMPTY"
        return (f"{self.table} {self.year}-{self.month:02d}: {len(f)} periods "
                f"{f.index[0].date()} → {f.index[-1].date()}, "
                f"cols {list(f.columns)}")


def fetch(kind: str, year: int, month: int) -> Bulletin:
    """Fetch and parse one bulletin table into a period-indexed frame.

    Column names are positional (`c0`, `c1`, ...) because the bulletin's headers
    are multi-row and merged; naming them here would bake in a header layout
    that differs across tables and vintages. Callers that need semantics should
    map positions explicitly and assert on them — see `exchange_inventories`.
    """
    if kind not in TABLES:
        raise KeyError(f"unknown table {kind!r}; known: {sorted(TABLES)}")
    html = _fetch_html(year, month, TABLES[kind])
    rows = _rows_from_html(html)

    records: dict[pd.Timestamp, dict[str, float | None]] = {}
    for cells in rows:
        if len(cells) < 2:
            continue
        periods = _parse_period_column(cells[0])
        if not any(p is not None for p in periods):
            continue
        for i, period in enumerate(periods):
            if period is None:
                continue
            rec = records.setdefault(period, {})
            for col_idx, frag in enumerate(cells[1:], start=1):
                if i < len(frag):
                    val = _euro_number(frag[i])
                    if val is not None:
                        rec.setdefault(f"c{col_idx}", val)

    if not records:
        raise CochilcoError(
            f"parsed no periods from {kind} {year}-{month:02d}. The bulletin "
            f"layout may have changed; inspect the raw HTML before trusting "
            f"any downstream series."
        )

    frame = pd.DataFrame.from_dict(records, orient="index").sort_index()
    frame.index.name = "period"
    b = Bulletin(year=year, month=month, table=TABLES[kind], frame=frame)
    log.info("cochilco: %s", b.describe())
    return b


def exchange_inventories(year: int, month: int) -> pd.DataFrame:
    """LME / COMEX / SHFE copper inventories (tonnes), with m/m changes.

    Column order in table 4_1 is:
        c1 LME total, c2 LME change, c3 COMEX total, c4 COMEX change,
        c5 SHFE total, c6 SHFE change, c7 combined total, c8 combined change

    This is the workaround for LME and CME returning 403 to direct requests.
    """
    frame = fetch("exchange_inventories", year, month).frame
    names = {
        "c1": "lme", "c2": "lme_chg", "c3": "comex", "c4": "comex_chg",
        "c5": "shfe", "c6": "shfe_chg", "c7": "total", "c8": "total_chg",
    }
    present = {k: v for k, v in names.items() if k in frame.columns}
    out = frame[list(present)].rename(columns=present)

    # The combined column must equal the three venues, or the positional
    # mapping above has drifted with a layout change. Checked rather than
    # trusted: a silent column shift would mislabel SHFE stock as LME stock.
    if {"lme", "comex", "shfe", "total"} <= set(out.columns):
        parts = out[["lme", "comex", "shfe"]].sum(axis=1)
        rel = ((parts - out["total"]).abs() / out["total"].replace(0, pd.NA))
        bad = rel[rel > 0.02].dropna()
        if len(bad):
            log.warning(
                "%d period(s) where LME+COMEX+SHFE disagrees with the printed "
                "total by >2%% (worst %.1f%%). The positional column mapping "
                "may have drifted — verify against the bulletin before use.",
                len(bad), 100 * bad.max(),
            )
    return out


def inventory_history(bulletins: list[tuple[int, int]] | None = None
                      ) -> pd.DataFrame:
    """Stitch several bulletins into one long monthly inventory series.

    Each bulletin's table 4_1 carries roughly four years of monthly history, so
    a handful of well-spaced issues covers a decade. Where issues overlap the
    EARLIER bulletin wins: figures are revised, and using the first print keeps
    the series closer to what was actually observable at the time. That matters
    for a backtest — a revised number is information you did not have.
    """
    if bulletins is None:
        bulletins = [(2013, 12), (2017, 12), (2021, 12), (2025, 6)]

    frames = []
    for year, month in bulletins:
        try:
            frames.append(exchange_inventories(year, month))
        except CochilcoError as exc:
            log.warning("bulletin %d-%02d unusable, skipped: %s",
                        year, month, exc)
    if not frames:
        raise CochilcoError("no bulletin could be parsed")

    # Oldest first, keep='first' -> earliest print survives on overlap.
    combined = pd.concat(frames).sort_index()
    out = combined[~combined.index.duplicated(keep="first")]
    log.info("inventory history: %d months %s → %s from %d bulletin(s)",
             len(out), out.index[0].date(), out.index[-1].date(), len(frames))
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", default="exchange_inventories",
                    choices=sorted(TABLES))
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--month", type=int, default=6)
    args = ap.parse_args()

    if args.table == "exchange_inventories":
        df = exchange_inventories(args.year, args.month)
        print("\nCopper inventories on exchanges (tonnes)\n")
        print(df.tail(14).to_string(float_format=lambda v: f"{v:,.0f}"))
    else:
        print(fetch(args.table, args.year, args.month).frame.tail(12).to_string())
