# CHOKEPOINT probe 2: Zambia monthly copper exports → copper prices (fixed 2026-11-11, before running)

- **Source:** a private research collection shared by the owner (read only, no code run; not
  redistributed here). Used: ZamStats *The Monthly* refined copper export volume per month (public
  statistics), with each issue's date (vintage_date). Kept locally, untracked.
- That collection's own verdict was that market transmission is untested and the information is
  probably priced via company releases. This probe tests exactly that, cheaply.
- **Events:** each issue that reports a monthly volume (variable refined_copper_exports_volume,
  the latest month in the issue).
  - surprise = log(value) − log(median of the previous 6 monthly values as published in
    earlier issues), i.e. point-in-time.
- **Returns** (data/prices/xnas_daily.parquet, through 2026-08-31): from the close one trading
  day AFTER the issue date (a guard against the issue date preceding web availability) to
  1, 5 and 20 trading days later.
  - CPER (copper futures ETF), raw; COPX, FCX and SCCO minus SPY.
- **Hypothesis (sign fixed now):** lower Zambian supply → higher copper → corr(surprise,
  future return) < 0.
- **Primary:** CPER at 5 days; Spearman correlation with a 90% bootstrap CI over events.
  PASS = CI upper bound < 0.
- Reported: all horizons and instruments, the number of events, the largest surprises.
- PASS → a pre-registered forward test on new issues (and DRC/company data next). FAIL →
  logged, like the Eskom probe.

## Result (2026-11-11): FAIL
- 67 monthly releases with a point-in-time surprise; 49 with prices (2020-04 .. 2026-07).
- **Primary, CPER at 5 days: Spearman −0.11 [−0.34, +0.11].** The sign is as hypothesised, but
  the effect is indistinguishable from 0.
- All 12 instrument × horizon cells have CIs spanning 0: CPER +0.01 / −0.11 / −0.06; COPX
  −0.09 / −0.06 / −0.14; FCX ≈ 0; SCCO −0.06 / −0.01 / −0.14.
- Power: with 49 events only |rho| above about 0.3 is detectable.
- Consistent with the Eskom probe: no detectable post-release drift from African supply data.
