-- GDELT GKG 2.0: headlines (page titles) of finance / Fed / inflation / trade-policy
-- articles from major outlets, with the 15-minute batch stamp GDELT processed them in.
-- A batch stamped s is treated as visible from s + 15 min (publication lag).
SELECT
  DATE AS batch,
  SourceCommonName AS source,
  REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title,
  CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS tone
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME >= TIMESTAMP('{start}') AND _PARTITIONTIME < TIMESTAMP('{end}')
  AND REGEXP_CONTAINS(V2Themes, r'(^|;)(ECON_STOCKMARKET|ECON_CENTRALBANK|ECON_INTEREST_RATES|EPU_CATS_TRADE_POLICY|ECON_INFLATION),')
  AND SourceCommonName IN ('reuters.com', 'bloomberg.com', 'cnbc.com', 'wsj.com', 'marketwatch.com',
    'ft.com', 'apnews.com', 'finance.yahoo.com', 'yahoo.com', 'barrons.com', 'investing.com',
    'foxbusiness.com', 'nytimes.com', 'washingtonpost.com', 'businessinsider.com', 'benzinga.com',
    'thestreet.com', 'forexlive.com', 'fxstreet.com', 'axios.com', 'politico.com', 'cnn.com',
    'bbc.co.uk', 'bbc.com', 'theguardian.com', 'fortune.com', 'forbes.com', 'seekingalpha.com',
    'zerohedge.com', 'kitco.com')
