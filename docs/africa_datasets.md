# CHOKEPOINT stage 2: dataset catalog (all 54 countries, all resources)

Plan and gate: docs/africa_realtime_plan.md. Rule: collect broadly, but **claim nothing until a
pre-registered lag probe passes** for a commodity.

## Running
| Dataset | Coverage | Mode | Where |
|---|---|---|---|
| GDELT 2.0 events | 54 countries + W. Sahara, every 15 min, 100+ languages | backfill 2015-02-19 →, one file per day | data/africa/gdelt/events_daily/ |
| GDELT 2.0 events + resource/disruption GKG | same | live catch-up (`chokepoint.data.gdelt live`); 15-min job in infra/live/com.siliconalpha.africa.gdelt.plist (installed 2026-11-12; moves to the server timer silicon-gdelt) | data/africa/gdelt/{events,gkg}/ |
| NASA POWER daily weather (rain, max/min temperature, max wind) | 53 producing sites: cocoa belt, coffee/tea, Copperbelt, Zambezi/Kariba, GERD, Akosombo, maize belt, mines (PGM, manganese, bauxite, iron, uranium, gold, lithium, graphite), oil fields, Suez/Bab el-Mandeb; 2015 → | backfill + daily poll inside the supply poll (`chokepoint.data.weather`), vintages kept. Replaces Open-Meteo (non-commercial terms); NASA data has no commercial restriction. Lags 2-4 days | data/supply/nasa_power.parquet (untracked) |

## Next: free, no account needed
| Dataset | What | Why |
|---|---|---|
| IMF PortWatch | daily port calls and trade estimates, all African ports; chokepoints (Suez, Bab el-Mandeb, Cape) | logistics disruptions |
| USGS Mineral Commodity Summaries (world production by country) | every mineral × every producing country | the dependency map (chokepoint/countries.py) |
| Energy Institute Statistical Review | oil and gas production by country | oil exposure (Libya, Nigeria, Angola, Algeria, Egypt) |
| FAOSTAT | crop production by country (cocoa, coffee, cashew, tea, cotton, sugar) | agricultural chokepoints |
| Prices | US equities/ETFs (data/prices/xnas_daily.parquet: CCJ, URA, USO, XLE, AA, FCX, …); FRED daily Brent/WTI | the probes' price side |

## Need the owner (free registration; Claude cannot create accounts)
| Dataset | What | Key goes in |
|---|---|---|
| ACLED | dated, geocoded conflict and protest events, all of Africa (the best curated event list) | ~/.config/silicon-alpha/acled.env |
| EIA API | oil production by country (monthly), outages | ~/.config/silicon-alpha/eia.env |
| NASA FIRMS | satellite fire detections (gas flaring as an oil-activity proxy: Nigeria, Libya, Algeria) | ~/.config/silicon-alpha/firms.env |
| UN Comtrade | trade by commodity × country × month | ~/.config/silicon-alpha/comtrade.env |
| Copernicus CDS (ERA5) | research-grade weather reanalysis | ~/.cdsapirc |

## Paid: quote first, owner approval required
- Intraday futures for the probes: Brent / WTI (ICE / CME), cocoa (ICE) from Databento.
- Commercial AIS ship tracking (tanker and bulk loadings): vendor quote; only after a probe passes.

## Not used
- Sources whose terms or robots forbid automated access (e.g. NASA POWER/CHIRPS automated paths,
  BCC Condensé reproduction) unless the owner downloads them manually or obtains permission.
- Anything non-public (company insiders, officials): insider information, never used.
