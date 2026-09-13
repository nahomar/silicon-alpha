"""Record the supply side, preserving vintages.

The price recorder captures what the market thinks. This captures what the
producing countries report — and, crucially, *when they first reported it*.

## Why vintages, not values

Official statistics are revised. Brazil publishes niobium exports about four
days after month end and restates them later; Cochilco's inventory table is the
same. If we only ever store the current value, then six months from now every
figure in the store is the revised one, and there is no way to reconstruct what
was actually knowable at the time.

That makes backtests quietly optimistic: the model gets a number nobody had.

So the store is keyed on **(series, period, captured_at)** rather than
(series, period). A row is appended only when the value for a period is new or
has *changed* — polling an unchanged series writes nothing. The result is a
revision history: the first row for a period is what was first published, and
any later rows are restatements with the date we learned them.

This is the piece that cannot be backfilled. Prices can be re-downloaded from
any vendor at any time; a first print exists only if someone wrote it down
before it was revised.

## What is polled

    comexstat   Brazil customs, monthly, ~4 days after month end
    cochilco    LME/COMEX/SHFE copper inventories, monthly

Eskom stage history is deliberately NOT polled here. It is reconstructed by
walking 778 git revisions of an upstream file, which takes minutes and produces
a series that only changes when South Africa is actually shedding load. Refresh
it on demand with `chokepoint.data.build_stage_history`.
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

STORE = Path("data/supply")
COLUMNS = ["source", "series", "period", "value", "captured_at"]

# Relative change below which a restatement is treated as noise rather than a
# revision. Guards against float round-trips through parquet and through the
# vendor's own formatting appending a row on every poll forever.
REVISION_EPS = 1e-9

# ComexStat rate-limits and answers 429 rather than throttling. An unpaced loop
# over commodities trips it, and because each failure is caught per-commodity
# the run would "succeed" having recorded nothing. Pace the calls.
API_PAUSE = 2.0


@dataclass
class PollReport:
    source: str
    periods_seen: int = 0
    first_prints: int = 0
    revisions: int = 0
    unchanged: int = 0
    error: str = ""
    revised_periods: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.error:
            return f"{self.source}: FAILED — {self.error}"
        s = (f"{self.source}: {self.periods_seen} periods seen, "
             f"{self.first_prints} new, {self.revisions} revised, "
             f"{self.unchanged} unchanged")
        if self.revised_periods:
            s += f"\n    revised: {', '.join(self.revised_periods[:6])}"
        return s


def _path(source: str) -> Path:
    return STORE / f"{source}.parquet"


def load(source: str) -> pd.DataFrame:
    p = _path(source)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_parquet(p)


def append_observations(source: str, obs: pd.DataFrame) -> PollReport:
    """Append only what is new or changed. Never overwrites a prior vintage.

    `obs` needs columns: series, period, value.
    """
    rep = PollReport(source=source)
    if obs.empty:
        rep.error = "poll returned no observations"
        return rep

    obs = obs.dropna(subset=["value"]).copy()
    obs["period"] = pd.to_datetime(obs["period"])
    rep.periods_seen = int(obs.groupby(["series", "period"]).ngroups)

    existing = load(source)
    captured_at = pd.Timestamp.utcnow()

    if existing.empty:
        latest: dict[tuple[str, pd.Timestamp], float] = {}
    else:
        e = existing.copy()
        e["period"] = pd.to_datetime(e["period"])
        # Most recent vintage per (series, period) is what we currently believe.
        e = e.sort_values("captured_at").drop_duplicates(
            subset=["series", "period"], keep="last")
        latest = {(r.series, r.period): r.value for r in e.itertuples()}

    new_rows = []
    for r in obs.itertuples():
        key = (r.series, r.period)
        prev = latest.get(key)
        if prev is None:
            rep.first_prints += 1
        else:
            denom = max(abs(prev), 1e-12)
            if abs(r.value - prev) / denom <= REVISION_EPS:
                rep.unchanged += 1
                continue
            rep.revisions += 1
            rep.revised_periods.append(
                f"{r.series}@{r.period:%Y-%m} {prev:,.0f}->{r.value:,.0f}")
        new_rows.append({
            "source": source, "series": r.series, "period": r.period,
            "value": float(r.value), "captured_at": captured_at,
        })

    if new_rows:
        fresh = pd.DataFrame(new_rows)
        # Concatenating onto an all-NA empty frame warns and may change dtypes.
        out = fresh if existing.empty else pd.concat(
            [existing, fresh], ignore_index=True)
        STORE.mkdir(parents=True, exist_ok=True)
        out.to_parquet(_path(source), index=False)
    return rep


# ---------------------------------------------------------------------------
# Pollers
# ---------------------------------------------------------------------------

def poll_comexstat(start: str = "2019-01") -> PollReport:
    """Brazil customs exports for every mapped chokepoint commodity."""
    from chokepoint.data import comexstat

    try:
        _, covers = comexstat.last_updated()
        end = covers  # only ask for what the vendor says exists
    except Exception as exc:  # noqa: BLE001
        return PollReport(source="comexstat", error=f"metadata: {exc}")

    frames = []
    for commodity in comexstat.NCM:
        try:
            s = comexstat.fetch(commodity, start, end)
        except Exception as exc:  # noqa: BLE001
            # One bad NCM code must not lose the others.
            log.warning("comexstat %s skipped: %s", commodity, exc)
            continue
        finally:
            time.sleep(API_PAUSE)
        f = s.frame.reset_index()
        for col in ("fob_usd", "kg", "usd_per_kg"):
            frames.append(pd.DataFrame({
                "series": f"{commodity}.{col}",
                "period": f["date"],
                "value": f[col],
            }))
    if not frames:
        return PollReport(source="comexstat", error="no commodity returned data")
    return append_observations("comexstat", pd.concat(frames, ignore_index=True))


def poll_cochilco(year: int | None = None, month: int | None = None
                  ) -> PollReport:
    """Exchange copper inventories from the most recent readable bulletin."""
    from chokepoint.data import cochilco

    now = pd.Timestamp.utcnow()
    # Bulletins lag; walk back until one parses rather than guessing the lag.
    candidates = []
    if year and month:
        candidates = [(year, month)]
    else:
        for back in range(1, 7):
            d = (now - pd.DateOffset(months=back))
            candidates.append((int(d.year), int(d.month)))

    for y, m in candidates:
        try:
            df = cochilco.exchange_inventories(y, m)
        except Exception as exc:  # noqa: BLE001
            log.debug("cochilco %d-%02d unusable: %s", y, m, exc)
            continue
        f = df.reset_index()
        frames = [
            pd.DataFrame({"series": f"copper_inventory.{col}",
                          "period": f["period"], "value": f[col]})
            for col in df.columns
        ]
        rep = append_observations("cochilco", pd.concat(frames, ignore_index=True))
        log.info("cochilco: used bulletin %d-%02d", y, m)
        return rep

    return PollReport(source="cochilco",
                      error=f"no readable bulletin in last {len(candidates)} months")


def poll_all() -> list[PollReport]:
    return [poll_comexstat(), poll_cochilco()]


def status() -> str:
    if not STORE.exists():
        return "no supply store yet"
    lines = []
    for p in sorted(STORE.glob("*.parquet")):
        d = pd.read_parquet(p)
        d["period"] = pd.to_datetime(d["period"])
        vintages = d.groupby(["series", "period"]).size()
        revised = int((vintages > 1).sum())
        lines.append(
            f"{p.stem}: {len(d):,} rows, {d.series.nunique()} series, "
            f"{d.period.min():%Y-%m} → {d.period.max():%Y-%m}, "
            f"{revised} period(s) revised since first print"
        )
    return "\n".join(lines) if lines else "supply store is empty"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["poll", "status"])
    args = ap.parse_args()

    if args.mode == "status":
        print(status())
        return

    reports = poll_all()
    print()
    for r in reports:
        print(r.summary())
    if any(r.error for r in reports):
        # Non-fatal: a scheduled job should record what it can and say what it
        # could not, rather than failing the whole run for one dead endpoint.
        print("\n(one or more sources failed; the rest were recorded)")


if __name__ == "__main__":  # pragma: no cover
    main()
