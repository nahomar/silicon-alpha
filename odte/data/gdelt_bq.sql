-- GDELT GKG 2.0: articles per 15-minute batch, total and per topic (theme tags).
-- Theme sets fixed 2026-11-08 from a one-day theme listing, before any result.
SELECT
  DATE AS batch,                                   -- YYYYMMDDHHMMSS, UTC, 15-min batch stamp
  COUNT(*) AS n_all,
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'(^|;)ECON_STOCKMARKET,')) AS n_market,
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'(^|;)(EPU_CATS_TRADE_POLICY|WB_775_TRADE_POLICY_AND_INTEGRATION),')) AS n_tariff,
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'(^|;)(ECON_CENTRALBANK|ECON_INTEREST_RATES|EPU_POLICY_INTEREST_RATES?),')) AS n_fed,
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'(^|;)(ARMEDCONFLICT|SANCTIONS),')) AS n_geo
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('2025-10-24') AND _PARTITIONTIME < TIMESTAMP('2026-11-09')
GROUP BY batch
ORDER BY batch
