"""Eskom loadshedding stage history — the event side of the decisive probe.

South Africa is ~70%+ of mined global platinum. Deep-level PGM mining is
power-intensive (hoisting, ventilation, refrigeration), so national grid
rationing is a physically causal supply constraint, published *daily*. That
combination — dominant supply share, causal mechanism, daily time axis, liquid
instruments — is why this is the case the whole track gates on.

## The unmet dependency, stated plainly

There is no free official API for *historical* loadshedding stages. EskomSePush
offers a free-tier token, but it is oriented to current and near-term schedules,
not deep history. So history must be supplied as a CSV.

## Why this module raises instead of falling back

It would be trivial to synthesize a plausible stage series and let the probe run
green. That would be the most expensive line of code in this repo.

`docs/data_integrity_finding.md` records what already happened once here: a
silently-wrong target (cross-contract returns) made a 524M transformer look like
it had been tested when it had not, and nearly justified a $20-50k retrain
against noise. The lesson taken from that was not "be careful" — it was to make
the failure *loud*. `prepare_features` now logs a warning on its fallback path
specifically so a silent corruption cannot recur.

Same principle here: **no synthetic stage data is generated anywhere in this
module.** Absent data raises `MissingStageData` with instructions. A probe that
cannot run is a fact; a probe that runs on invented data is a lie that costs
money later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Eskom's published scale. Stage 0 = no loadshedding; each stage ≈ 1000 MW shed.
MIN_STAGE, MAX_STAGE = 0, 8

# Below this, mines largely absorb the cut with backup/scheduling. At and above
# it, curtailment of hoisting and smelting becomes material. Used only as a
# reporting convenience — the probe uses the continuous stage.
MATERIAL_STAGE = 4


class MissingStageData(RuntimeError):
    """Raised when stage history is unavailable. Never silently substituted."""


_HOWTO = """
Historical Eskom loadshedding stages are not available from a free official API,
so they must be supplied as a CSV:

    date,stage
    2023-01-03,4
    2023-01-04,6
    2023-01-05,6

  date   ISO-8601 (YYYY-MM-DD), one row per calendar day, no gaps preferred
  stage  integer 0-8; 0 means no loadshedding that day. If a day saw multiple
         stages, use the daily MAX (the binding constraint on mine operations),
         and keep that choice consistent across the whole file.

Where to get it:
  - EskomSePush API (free tier token) for going-forward collection:
    https://eskomsepush.gumroad.com/l/api
  - Community-maintained historical CSVs exist on GitHub; verify the stage
    convention (daily max vs modal vs hours-weighted) before trusting one.
  - Eskom media statements / SA press archives for manual reconstruction.

Whichever you use, record the provenance in the file header as a comment line
starting with '#'. A stage series whose convention you cannot state is not
usable as a research input.
""".strip()


@dataclass(frozen=True)
class StageHistory:
    """Daily national loadshedding stage, indexed by date."""

    stages: pd.Series  # index: DatetimeIndex (naive, SAST calendar days)
    provenance: str

    def __post_init__(self) -> None:
        s = self.stages
        if not isinstance(s.index, pd.DatetimeIndex):
            raise ValueError("stages must be indexed by DatetimeIndex")
        if not s.index.is_monotonic_increasing:
            raise ValueError("stage index must be sorted ascending")
        if s.index.has_duplicates:
            dupes = s.index[s.index.duplicated()].unique()[:5]
            raise ValueError(f"duplicate dates in stage history: {list(dupes)}")
        bad = s[(s < MIN_STAGE) | (s > MAX_STAGE)]
        if len(bad):
            raise ValueError(
                f"{len(bad)} stage values outside [{MIN_STAGE},{MAX_STAGE}]: "
                f"{bad.head().to_dict()}"
            )

    # ------------------------------------------------------------------
    @property
    def span(self) -> tuple[date, date]:
        return self.stages.index[0].date(), self.stages.index[-1].date()

    def describe(self) -> str:
        s = self.stages
        lo, hi = self.span
        shed = s[s > 0]
        material = s[s >= MATERIAL_STAGE]
        return (
            f"{len(s)} days {lo} → {hi}  |  "
            f"loadshedding on {len(shed)} days ({len(shed) / len(s):.0%}), "
            f"stage ≥{MATERIAL_STAGE} on {len(material)} days "
            f"({len(material) / len(s):.0%})  |  "
            f"mean {s.mean():.2f}, max {int(s.max())}  |  src: {self.provenance}"
        )

    def daily_change(self) -> pd.Series:
        """Day-over-day change in stage — the 'news' relative to yesterday.

        A market that already knows the country is at Stage 4 should have priced
        Stage 4. What is potentially unpriced is the *escalation*.
        """
        return self.stages.diff()


# ---------------------------------------------------------------------------

def load_stages(path: str | Path) -> StageHistory:
    """Load stage history from CSV. Raises rather than substituting anything."""
    p = Path(path)
    if not p.exists():
        raise MissingStageData(
            f"no loadshedding stage history at {p}\n\n{_HOWTO}"
        )

    provenance = f"{p.name}"
    try:
        # '#' lines carry provenance; keep the first as the recorded source.
        with p.open() as fh:
            for line in fh:
                if line.startswith("#"):
                    provenance = line.lstrip("# ").strip()
                    break
        df = pd.read_csv(p, comment="#")
    except Exception as exc:  # noqa: BLE001 - surface the real parse error
        raise MissingStageData(f"could not parse {p}: {exc}\n\n{_HOWTO}") from exc

    missing = {"date", "stage"} - set(df.columns)
    if missing:
        raise MissingStageData(
            f"{p} is missing required column(s): {sorted(missing)}; "
            f"found {list(df.columns)}\n\n{_HOWTO}"
        )
    if df.empty:
        raise MissingStageData(f"{p} contains no rows\n\n{_HOWTO}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        n = int(df["date"].isna().sum())
        raise MissingStageData(
            f"{p}: {n} unparseable date(s); expected ISO-8601 YYYY-MM-DD"
        )
    df["stage"] = pd.to_numeric(df["stage"], errors="coerce")
    if df["stage"].isna().any():
        n = int(df["stage"].isna().sum())
        raise MissingStageData(f"{p}: {n} non-numeric stage value(s)")

    s = (
        df.set_index("date")["stage"]
        .sort_index()
        .astype(float)
        .rename("stage")
    )

    hist = StageHistory(stages=s, provenance=provenance)
    _warn_on_gaps(hist)
    log.info("loaded stage history: %s", hist.describe())
    return hist


def _warn_on_gaps(hist: StageHistory) -> None:
    """Loudly flag calendar gaps.

    A gap is ambiguous: it can mean 'no loadshedding' or 'nobody recorded it'.
    Those are very different inputs, and conflating them biases the series
    toward calm. We do not fill them — we say so.
    """
    idx = hist.stages.index
    full = pd.date_range(idx[0], idx[-1], freq="D")
    missing = full.difference(idx)
    if len(missing):
        pct = len(missing) / len(full)
        log.warning(
            "stage history has %d missing calendar day(s) (%.1f%% of span, "
            "e.g. %s). These are NOT filled: a gap may mean 'no loadshedding' "
            "or 'unrecorded', and assuming the former biases the series calm. "
            "Days absent from the series are dropped from the probe, not "
            "treated as stage 0.",
            len(missing), 100 * pct, [d.date() for d in missing[:3]],
        )


def align_to_sessions(hist: StageHistory, sessions: pd.DatetimeIndex,
                      lag_days: int = 1) -> pd.Series:
    """Align stages onto trading sessions, lagged to prevent look-ahead.

    Stages are announced and revised intraday, so same-day stage information is
    not cleanly available before the close. `lag_days=1` means a prediction for
    session t uses only stage information from session t-1 or earlier.

    Weekend and holiday stages are carried forward onto the next session: a
    Saturday escalation is real news for Monday's open.
    """
    if lag_days < 1:
        raise ValueError(
            f"lag_days must be >= 1 to avoid look-ahead on intraday-revised "
            f"stage announcements; got {lag_days}"
        )
    first, last = hist.stages.index[0], hist.stages.index[-1]
    daily = hist.stages.reindex(pd.date_range(first, last, freq="D")).ffill()

    norm = sessions.normalize()
    on_sessions = daily.reindex(norm, method="ffill")
    on_sessions.index = sessions

    # Do NOT extrapolate beyond the recorded span. reindex(method="ffill")
    # carries the final stage forward forever, silently asserting "the grid
    # never changed again" for every session after the record ends. That
    # manufactures a long tail of zero-variance observations which looks like
    # data and reads like evidence. Outside coverage is NaN; the caller drops it.
    outside = (norm < first) | (norm > last)
    if outside.any():
        on_sessions[outside] = np.nan
        log.warning(
            "%d of %d sessions fall outside the stage record (%s → %s) and are "
            "dropped, not forward-filled. Extrapolating there would fabricate "
            "zero-variance observations.",
            int(outside.sum()), len(sessions), first.date(), last.date(),
        )
    return on_sessions.shift(lag_days).rename(f"stage_lag{lag_days}")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stages", required=True, help="path to stage CSV")
    args = ap.parse_args()
    try:
        print(load_stages(args.stages).describe())
    except MissingStageData as exc:
        raise SystemExit(f"\n{exc}\n")
