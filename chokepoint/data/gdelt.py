"""GDELT 2.0 → Africa event store (CHOKEPOINT stage 2, docs/africa_realtime_plan.md).

GDELT publishes every 15 minutes: an EVENTS table (who did what to whom, where; CAMEO codes,
tone, number of sources) and a GKG table (themes, locations, tone of each article), from news in
100+ languages. DATEADDED is the 15-minute batch in which GDELT first saw an event: the "first
trace" timestamp the lag probes need.

Kept (append-only, one parquet per 15-minute batch):
  events  every event whose action, actor 1 or actor 2 location is in one of 54 African countries
          (+ Western Sahara)  -> data/africa/gdelt/events/YYYY-MM-DD/HHMMSS.parquet
  gkg     (live only) articles with an African location AND a theme matching THEMES (economy,
          resources, energy, conflict, disaster, trade, unrest)   -> data/africa/gdelt/gkg/...

  python3 -m chokepoint.data.gdelt live              # catch up to the latest batch (launchd, 15 min)
  python3 -m chokepoint.data.gdelt backfill 2015-02-19 2026-11-11 [--workers 8]
          events only, one file per day: data/africa/gdelt/events_daily/YYYY-MM-DD.parquet
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "africa" / "gdelt"
BASE = "https://data.gdeltproject.org/gdeltv2/"
UA = {"User-Agent": "silicon-alpha-research/1.0 (personal research; low rate)"}

# FIPS 10-4 country codes (GDELT's geo codes) for the 54 African states + Western Sahara
AFRICA = {"AG": "Algeria", "AO": "Angola", "BN": "Benin", "BC": "Botswana", "UV": "Burkina Faso", "BY": "Burundi",
          "CV": "Cabo Verde", "CM": "Cameroon", "CT": "Central African Republic", "CD": "Chad", "CN": "Comoros",
          "CG": "DR Congo", "CF": "Republic of the Congo", "IV": "Cote d'Ivoire", "DJ": "Djibouti", "EG": "Egypt",
          "EK": "Equatorial Guinea", "ER": "Eritrea", "WZ": "Eswatini", "ET": "Ethiopia", "GB": "Gabon", "GA": "Gambia",
          "GH": "Ghana", "GV": "Guinea", "PU": "Guinea-Bissau", "KE": "Kenya", "LT": "Lesotho", "LI": "Liberia",
          "LY": "Libya", "MA": "Madagascar", "MI": "Malawi", "ML": "Mali", "MR": "Mauritania", "MP": "Mauritius",
          "MO": "Morocco", "MZ": "Mozambique", "WA": "Namibia", "NG": "Niger", "NI": "Nigeria", "RW": "Rwanda",
          "TP": "Sao Tome and Principe", "SG": "Senegal", "SE": "Seychelles", "SL": "Sierra Leone", "SO": "Somalia",
          "SF": "South Africa", "OD": "South Sudan", "SU": "Sudan", "TZ": "Tanzania", "TO": "Togo", "TS": "Tunisia",
          "UG": "Uganda", "ZA": "Zambia", "ZI": "Zimbabwe", "WI": "Western Sahara"}

EVENT_COLS = ["GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate", "Actor1Code", "Actor1Name",
              "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode", "Actor1Religion1Code",
              "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code", "Actor2Code",
              "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode", "Actor2EthnicCode", "Actor2Religion1Code",
              "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code", "IsRootEvent",
              "EventCode", "EventBaseCode", "EventRootCode", "QuadClass", "GoldsteinScale", "NumMentions",
              "NumSources", "NumArticles", "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName",
              "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat",
              "Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName",
              "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat",
              "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type", "ActionGeo_FullName",
              "ActionGeo_CountryCode", "ActionGeo_ADM1Code", "ActionGeo_ADM2Code", "ActionGeo_Lat",
              "ActionGeo_Long", "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL"]
GKG_COLS = ["GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName", "DocumentIdentifier",
            "Counts", "V2Counts", "Themes", "V2Themes", "Locations", "V2Locations", "Persons", "V2Persons",
            "Organizations", "V2Organizations", "V2Tone", "Dates", "GCAM", "SharingImage", "RelatedImages",
            "SocialImageEmbeds", "SocialVideoEmbeds", "Quotations", "AllNames", "Amounts", "TranslationInfo",
            "Extras"]
THEMES = re.compile(r"(ENV_|ECON_|STRIKE|COUP|BLOCKADE|SANCTION|NATURAL_DISASTER|MANMADE_DISASTER|FOOD_|AGRI|"
                    r"MINING|MINE|OIL|GAS|ELECTRIC|POWER|PORT|SHIP|FUEL|EXPORT|IMPORT|TARIFF|NATIONALIZ|PROTEST|"
                    r"UNREST|REBEL|INSURGEN|ARMEDCONFLICT|TERROR|DROUGHT|FLOOD|CROP|PIPELINE|REFINER|SMUGGL|"
                    r"CORRUPTION|ELECTION|REGIME|MILITARY)")
AFR_LOC = re.compile(r"#(" + "|".join(AFRICA) + r")#")


def _get(url: str, tries: int = 4) -> bytes | None:
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            time.sleep(5 * (i + 1))
    return None


def _read_zip(blob: bytes, cols: list) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    return pd.read_csv(io.BytesIO(raw), sep="\t", header=None, names=cols, dtype=str, quoting=3,
                       on_bad_lines="skip", encoding="latin-1", engine="python" if len(raw) < 1 else "c")


def africa_events(df: pd.DataFrame) -> pd.DataFrame:
    m = (df["ActionGeo_CountryCode"].isin(AFRICA) | df["Actor1Geo_CountryCode"].isin(AFRICA)
         | df["Actor2Geo_CountryCode"].isin(AFRICA))
    return df[m]


def africa_gkg(df: pd.DataFrame) -> pd.DataFrame:
    loc = df["V2Locations"].fillna("").str.contains(AFR_LOC)
    th = df["V2Themes"].fillna("").str.contains(THEMES)
    keep = ["GKGRECORDID", "DATE", "SourceCommonName", "DocumentIdentifier", "V2Themes", "V2Locations", "V2Tone",
            "V2Counts", "Amounts", "TranslationInfo"]
    return df.loc[loc & th, keep]


def _save(frame: pd.DataFrame, kind: str, stamp: str) -> Path:
    day = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
    f = OUT / kind / day / f"{stamp[8:]}.parquet"
    f.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(f.with_suffix(".tmp"), index=False)
    f.with_suffix(".tmp").replace(f)
    return f


def batch(stamp: str, gkg: bool) -> dict:
    """Fetch, filter and store one 15-minute batch. Returns counts (or {} if not published)."""
    out = {}
    ev_f = OUT / "events" / f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}" / f"{stamp[8:]}.parquet"
    if not ev_f.exists():
        blob = _get(f"{BASE}{stamp}.export.CSV.zip")
        if blob is None:
            return {}
        ev = africa_events(_read_zip(blob, EVENT_COLS))
        _save(ev, "events", stamp)
        out["events"] = len(ev)
    if gkg:
        gk_f = OUT / "gkg" / f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}" / f"{stamp[8:]}.parquet"
        if not gk_f.exists():
            blob = _get(f"{BASE}{stamp}.gkg.csv.zip")
            if blob is not None:
                gk = africa_gkg(_read_zip(blob, GKG_COLS))
                _save(gk, "gkg", stamp)
                out["gkg"] = len(gk)
    return out


def stamps(a: pd.Timestamp, b: pd.Timestamp) -> list:
    t = pd.date_range(a.floor("15min"), b.floor("15min"), freq="15min")
    return [x.strftime("%Y%m%d%H%M%S") for x in t]


def live() -> int:
    """Catch up from the last stored batch (or the last 6 hours) to GDELT's latest."""
    txt = _get(BASE + "lastupdate.txt")
    if txt is None:
        print("lastupdate unavailable")
        return 1
    latest = re.search(r"(\d{14})\.export", txt.decode()).group(1)
    state = OUT / "live_state.json"
    last = json.loads(state.read_text())["last"] if state.exists() else None
    start = pd.Timestamp(last) + pd.Timedelta(minutes=15) if last else pd.Timestamp(latest) - pd.Timedelta(hours=6)
    todo = stamps(start, pd.Timestamp(latest))[-96 * 3:]               # at most 3 days of catch-up
    n = {"events": 0, "gkg": 0, "batches": 0}
    for s in todo:
        c = batch(s, gkg=True)
        if c:
            n["batches"] += 1
            n["events"] += c.get("events", 0)
            n["gkg"] += c.get("gkg", 0)
    OUT.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"last": str(pd.Timestamp(latest)), "updated": str(pd.Timestamp.utcnow())}))
    print(f"{pd.Timestamp.utcnow():%Y-%m-%d %H:%M} live: {n['batches']} batches, {n['events']} African events, "
          f"{n['gkg']} African resource/disruption articles (latest {latest})", flush=True)
    return 0


def backfill_day(day: pd.Timestamp) -> int:
    """All 96 batches of one day, filtered to Africa, saved as ONE file: events_daily/YYYY-MM-DD.parquet."""
    f = OUT / "events_daily" / f"{day:%Y-%m-%d}.parquet"
    if f.exists():
        return -1
    parts = []
    for s in stamps(day, day + pd.Timedelta(hours=23, minutes=45)):
        blob = _get(f"{BASE}{s}.export.CSV.zip")
        if blob is not None:
            try:
                parts.append(africa_events(_read_zip(blob, EVENT_COLS)).assign(batch=s))
            except Exception:
                continue
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=EVENT_COLS + ["batch"])
    f.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f.with_suffix(".tmp"), index=False)
    f.with_suffix(".tmp").replace(f)
    return len(out)


def backfill(a: str, b: str, workers: int = 8) -> int:
    days = [d for d in pd.date_range(a, b, freq="D") if not (OUT / "events_daily" / f"{d:%Y-%m-%d}.parquet").exists()]
    print(f"backfill: {len(days):,} days to fetch ({a} .. {b}), {workers} in parallel", flush=True)
    done = [0, 0]

    def one(d):
        n = backfill_day(d)
        done[0] += 1
        done[1] += max(n, 0)
        if done[0] % 50 == 0:
            print(f"  {done[0]:,}/{len(days):,} days, {done[1]:,} African events so far", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, days))
    print(f"backfill done: {done[0]:,} days, {done[1]:,} African events", flush=True)
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "live":
        sys.exit(live())
    if sys.argv[1] == "backfill":
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
        sys.exit(backfill(sys.argv[2], sys.argv[3], w))
