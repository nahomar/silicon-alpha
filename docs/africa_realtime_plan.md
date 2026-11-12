# CHOKEPOINT stage 2: real-time African supply events → global prices (plan, 2026-11-11)

Owner: "across all the countries in Africa ... collect real-time data that the US relies on
Africa for ... those could affect the market."

## What we already know (two probes, both no lag)
- Eskom loadshedding → SA PGM miners (docs/chokepoint_eskom_probe_result.md): no lag.
- Zambia monthly copper exports → copper, miners (docs/chokepoint_zambia_probe.md): no
  post-release drift.
- **Lesson:** slow official statistics (monthly, lagged) are priced, mostly through company
  disclosures. Any edge must come from **real-time events**, detected at their first local
  trace, in commodities where **Africa's share is large**, with a **liquid instrument**.

## Candidates (ranked by share × tradability × event frequency)
1. **Cocoa:** Côte d'Ivoire + Ghana ≈ 60% of supply; ICE cocoa futures; weather, arrivals,
   disease and policy news.
2. **Oil:** Libya, Nigeria, Angola outages → Brent/WTI; frequent, well-dated events.
3. **Uranium:** Niger, Namibia → Cameco / URA / URNM.
4. **Cobalt:** DRC ≈ 70%; export ban and quota news; weak US instruments, so research only.
5. PGMs and copper: tested, no lag; deprioritised.
- All shares are to be verified against USGS MCS / ICCO / EIA before any use.

## Gate: a lag probe before any 54-country ingestion
- **Events:** about 10 years of African supply events per candidate, each stamped with its
  **first public trace**, from GDELT 2.0 (global news, 15-minute updates, 100+ languages,
  free) plus a hand-checked seed list. Each gets a date and a direction (supply down / up).
- **Prices:** the instrument's return from the first trace to +1 h / +1 d / +5 d, against
  the move before the trace. Intraday where data allow; Databento futures and US equities
  can be quoted.
- **Question:** is there drift after the first trace (the market slow), or does the move
  come at or before it (priced, or already known elsewhere)?
- PASS = a significant post-trace drift of the expected sign at a horizon a small account can
  trade. Only then build a live collector for that commodity, then test it forward,
  pre-registered, like the trading models.
- FAIL = logged, and stop that commodity.
- Each probe gets its own pre-registration (sign, horizon, test) before its data are examined.

## Priority
- After the execution study (docs/execution_plan.md), which addresses money lost today.
- Cocoa and oil first.
