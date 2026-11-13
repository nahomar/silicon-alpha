"""Daily weather at Africa's producing sites, from NASA POWER (free, no key, US government data).

Why NASA POWER and not Open-Meteo: Open-Meteo's free tier is for non-commercial use only. NASA
POWER is public with no commercial restriction (cite: "NASA Langley Research Center POWER
Project"). It lags about 2-4 days, which is fine for the slow physical signals it serves (rain on
the cocoa belt, drought behind the Kariba / GERD dams, rain-season stops at the bauxite and
copper mines, heat at the uranium mines).

Stored through the vintage store (chokepoint.record.supply.append_observations), so the first
value NASA printed for a day is kept even after NASA revises it. The 2015 -> today backfill is
NASA's CURRENT values (captured on the backfill date), not first prints: research that needs
point-in-time weather must lag it by >= 4 days, and only data captured by the daily poll is
truly first-print.

    python3 -m chokepoint.data.weather backfill        # 2015-01-01 -> today, all sites
    python3 -m chokepoint.data.weather poll            # last 45 days (also run by the daily supply poll)
    python3 -m chokepoint.data.weather sites
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = ("PRECTOTCORR", "T2M_MAX", "T2M_MIN", "WS10M_MAX")   # rain mm/day, max/min 2 m temp C, max 10 m wind m/s
SOURCE = "nasa_power"
PAUSE = 1.0
START = "2015-01-01"

# site: (lat, lon, what it matters for). Approximate public coordinates of producing areas.
SITES = {
    # cocoa (Cote d'Ivoire + Ghana ~60% of world supply; Cameroon, Nigeria)
    "civ_soubre": (5.78, -6.60, "cocoa"), "civ_daloa": (6.88, -6.45, "cocoa"),
    "civ_san_pedro": (4.75, -6.64, "cocoa port"), "civ_abengourou": (6.73, -3.49, "cocoa"),
    "gha_kumasi": (6.69, -1.62, "cocoa"), "gha_sefwi_wiawso": (6.21, -2.49, "cocoa"),
    "gha_goaso": (6.80, -2.52, "cocoa"), "cmr_mbalmayo": (3.52, 11.50, "cocoa"),
    "nga_ondo": (7.10, 4.84, "cocoa"),
    # coffee, tea
    "eth_jimma": (7.67, 36.83, "coffee"), "eth_yirgalem": (6.75, 38.41, "coffee"),
    "uga_masaka": (-0.33, 31.73, "coffee"), "uga_mbale": (1.08, 34.18, "coffee"),
    "ken_nyeri": (-0.42, 36.95, "coffee"), "ken_kericho": (-0.37, 35.28, "tea"),
    # copper / cobalt (Copperbelt: rain season floods pits and cuts roads)
    "zmb_kitwe": (-12.80, 28.21, "copper"), "cod_kolwezi": (-10.71, 25.47, "copper, cobalt"),
    "cod_lubumbashi": (-11.66, 27.48, "copper, cobalt"),
    # hydro catchments (Zambia/Zimbabwe power -> mine output; Ethiopia, Ghana, Mozambique)
    "zambezi_lukulu": (-14.38, 23.24, "Kariba inflow"), "zambezi_victoria_falls": (-17.93, 25.84, "Kariba inflow"),
    "kariba_dam": (-16.52, 28.80, "hydro"), "cahora_bassa": (-15.59, 32.70, "hydro"),
    "eth_bahir_dar": (11.60, 37.38, "Blue Nile / GERD"), "eth_gerd": (11.21, 35.09, "hydro"),
    "gha_akosombo": (6.30, 0.06, "hydro, aluminium"),
    # grains (South Africa maize belt), tobacco, cotton, cashew, vanilla
    "zaf_bothaville": (-27.39, 26.62, "maize"), "zaf_bethal": (-26.45, 29.47, "maize"),
    "mwi_lilongwe": (-13.96, 33.79, "tobacco"), "mli_koutiala": (12.39, -5.46, "cotton"),
    "bfa_bobo_dioulasso": (11.18, -4.29, "cotton"), "ben_kandi": (11.13, 2.94, "cotton"),
    "civ_korhogo": (9.46, -5.63, "cashew, cotton"), "mdg_sambava": (-14.27, 50.17, "vanilla"),
    # mines
    "zaf_rustenburg": (-25.67, 27.24, "platinum"), "zaf_steelpoort": (-24.73, 30.20, "platinum, chrome"),
    "zaf_hotazel": (-27.20, 22.96, "manganese"), "gab_moanda": (-1.57, 13.20, "manganese"),
    "gin_boke": (10.94, -14.29, "bauxite"), "gin_kamsar": (10.65, -14.61, "bauxite port"),
    "mrt_zouerat": (22.73, -12.47, "iron ore"), "ner_arlit": (18.74, 7.39, "uranium"),
    "nam_rossing": (-22.48, 15.04, "uranium"), "gha_obuasi": (6.20, -1.67, "gold"),
    "mli_sadiola": (13.89, -11.70, "gold"), "bwa_jwaneng": (-24.60, 24.73, "diamonds"),
    "zwe_bikita": (-20.08, 31.62, "lithium"), "moz_balama": (-13.35, 38.57, "graphite"),
    # oil and gas
    "nga_port_harcourt": (4.82, 7.03, "oil"), "ago_soyo": (-6.13, 12.37, "oil, LNG"),
    "lby_ras_lanuf": (30.50, 18.57, "oil port"), "dza_hassi_messaoud": (31.68, 6.07, "oil"),
    # shipping chokepoints (wind)
    "egy_suez": (29.97, 32.55, "Suez Canal"), "bab_el_mandeb": (12.60, 43.33, "Red Sea strait"),
}


def fetch(site: str, start: str, end: str, session: requests.Session | None = None) -> pd.DataFrame:
    """Long frame (series, period, value) for one site; NASA's fill value (-999) dropped."""
    lat, lon, _ = SITES[site]
    s = session or requests.Session()
    q = {"parameters": ",".join(PARAMS), "community": "AG", "latitude": lat, "longitude": lon,
         "start": pd.Timestamp(start).strftime("%Y%m%d"), "end": pd.Timestamp(end).strftime("%Y%m%d"),
         "format": "JSON"}
    for attempt in range(4):
        r = s.get(URL, params=q, timeout=180)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        break
    else:
        r.raise_for_status()
    d = r.json()
    fill = d.get("header", {}).get("fill_value", -999.0)
    rows = []
    for p, by_day in d["properties"]["parameter"].items():
        for day, v in by_day.items():
            if v is not None and v != fill:
                rows.append((f"{site}:{p}", pd.Timestamp(day), float(v)))
    return pd.DataFrame(rows, columns=["series", "period", "value"])


def collect(start: str, end: str, sites=None) -> tuple[pd.DataFrame, list[str]]:
    s = requests.Session()
    s.headers["User-Agent"] = "silicon-alpha research (python-requests)"
    out, failed = [], []
    for site in (sites or SITES):
        try:
            out.append(fetch(site, start, end, s))
        except Exception as e:  # one bad site must not lose the rest
            log.warning("weather %s failed: %s", site, e)
            failed.append(site)
        time.sleep(PAUSE)
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["series", "period", "value"])), failed


def poll(days: int = 45):
    """Last `days` days for every site, into the vintage store (first prints + revisions)."""
    from chokepoint.record.supply import PollReport, append_observations
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    obs, failed = collect(end - pd.Timedelta(days=days), end)
    if obs.empty:
        return PollReport(source=SOURCE, error=f"no data ({len(failed)} sites failed)")
    rep = append_observations(SOURCE, obs)
    rep.revised_periods = rep.revised_periods[:6]
    if failed:
        rep.error = "" if len(failed) < len(SITES) else "all sites failed"
        log.warning("weather: %d sites failed: %s", len(failed), ", ".join(failed))
    return rep


def backfill(start: str = START):
    from chokepoint.record.supply import append_observations
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    obs, failed = collect(start, end)
    rep = append_observations(SOURCE, obs)
    print(f"backfill {start} -> {end:%Y-%m-%d}: {len(obs):,} values, {rep.first_prints:,} new, "
          f"{rep.revisions:,} revised; failed sites: {failed or 'none'}")
    return rep


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["backfill", "poll", "sites"])
    ap.add_argument("--start", default=START)
    a = ap.parse_args()
    if a.mode == "sites":
        for k, (lat, lon, why) in SITES.items():
            print(f"{k:24s} {lat:7.2f} {lon:7.2f}  {why}")
    elif a.mode == "poll":
        print(poll().summary())
    else:
        backfill(a.start)


if __name__ == "__main__":
    main()
