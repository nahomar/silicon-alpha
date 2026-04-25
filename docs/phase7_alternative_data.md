# Phase-7 — Alternative Data (design spec, no code)

**Status**: design-only. No code in the repo corresponds to this doc.
Phase 7 layers in three "physical economy" alt-data feeds that are
independent of orderbook tape: real-time pricing, consumer transaction
flow, and corporate hiring signals. In 2026, ~94% of asset managers
already use AI for alt-data; the edge here comes from structured
ingestion + integration into the same forecaster, not from any single
exclusive feed.

Reference the Silicon Alpha goal in `memory/project_silicon_alpha_goal.md`.

## Why this phase exists

Layer 1 (TradeFM) plus Phase 6 (fundamental + symbolic) capture
"what's in the tape" and "what's in the filings." They miss everything
that happens **before** earnings prints:

- A retailer raising prices ahead of a tariff announcement (margin
  signal, weeks before earnings)
- A subscription service losing 3% of receipts week-over-week
  (revenue miss, weeks before guidance)
- A company posting 200 senior-engineer roles in a region adjacent to
  a competitor (M&A or capacity-build, months before announcement)

Phase 7 captures these signals at daily-to-weekly cadence and emits
per-ticker scalar features that condition the forecaster.

## Three feeds

### Feed A — Real-time pricing power (Bright Data web scrapers)

- **Source**: Bright Data managed scrapers across **250+ global
  e-commerce domains** (Amazon, Walmart, Target, Costco, Carrefour,
  Tesco, Mercado Libre, Rakuten, Coupang, etc.).
- **Granularity**: Daily price-by-SKU snapshots for high-volume SKUs
  per merchant. ~1-10M SKUs total.
- **Edge thesis**: in 2026's tariff regime (~16% effective average
  tariff burden), the ability to **pass tariff costs through to
  consumers without demand destruction** is a primary alpha driver.
  Identify firms that lift prices on key SKUs and observe whether
  unit volumes hold (via Feed B receipts data) — passthrough
  capability is the alpha.
- **Output**: per-(ticker, week) features:
  ```
  weighted_price_change_pct,
  sku_price_dispersion,
  tariff_passthrough_score,  # demands Feed B to compute volume side
  competitive_price_gap_vs_peers,
  promo_intensity_score,
  ```

### Feed B — Consumer intent + transaction flow (YipitData / Measurable AI)

- **Source**: YipitData OR Measurable AI panels — **2M+ de-identified
  consumers**, email-receipt ingestion + bank/card aggregation.
- **Granularity**: Daily-to-weekly SKU-level transactions. Coverage
  varies by category (high for D2C, e-commerce, subscriptions; lower
  for B2B and brick-and-mortar).
- **Edge thesis**: revenue-line forecasts at a higher cadence than
  earnings calls. Combined with Feed A pricing data, you compute
  unit-volume × price = revenue runs at weekly granularity.
- **Output**: per-(ticker, week) features:
  ```
  unit_volume_yoy_pct,
  unit_volume_qoq_run_rate,
  revenue_run_rate_qoq,
  basket_size_trend,
  customer_retention_proxy,
  ```

### Feed C — Talent flow (PredictLeads)

- **Source**: PredictLeads job-posting aggregation across LinkedIn,
  Indeed, company career pages, etc.
- **Granularity**: Daily new-posting feed, tagged by role family,
  seniority, geography, company.
- **Edge thesis**: a **3-6 month lead** on official growth /
  headcount announcements. A surge in specialized roles (e.g.,
  "RF engineer" at a defense contractor, "cathode engineer" at a
  battery company) signals capacity build or new-product investment
  before management discloses it.
- **Output**: per-(ticker, week) features:
  ```
  net_postings_qoq_pct,
  senior_role_ratio,
  specialized_role_surge_flag,  # +5σ in any role family
  geo_expansion_signal,
  ```

## Integration into the forecaster

Same `modality_vocab` mechanism used for Phase 2.5 (ES) and Phase 6:
modality 6 = pricing, modality 7 = transactions, modality 8 = talent.
All three feed conditioning tokens at the per-ticker level, refreshed
weekly.

Sequence layout per training example:
```
[modality_id=6 (pricing) tokens for ticker X this week]
[modality_id=7 (txn) tokens for ticker X this week]
[modality_id=8 (talent) tokens for ticker X this week]
[modality_id=5 (symbolic factors) tokens for ticker X this tick]
[modality_id=4 (fundamental) tokens for ticker X this quarter]
[modality_id=0 (OPRA) microstructure tokens for next K seconds]
```

The model learns to attend across modalities — does the pricing-power
signal predict next-day options-flow? Does talent flow predict
implied-vol expansion? — without manual feature engineering.

## Cost reality

Approximate annual subscription costs (2026 institutional pricing):

| Feed | Vendor | Cost (USD/yr) |
|---|---|---|
| A: Bright Data, 250+ domains daily | Bright Data | $5k-$50k |
| B: YipitData panel (full coverage) | YipitData | $100k-$300k |
| B (alt): Measurable AI panel | Measurable AI | $30k-$100k |
| C: PredictLeads | PredictLeads | $5k-$30k |
| **Total alt-data baseline** | | **$140k-$480k** |

These numbers gate the phase. They're also the reason ~94% of asset
managers using AI for alt-data are running >$50M AUM — anyone smaller
can't amortize the data subscription.

## Dependencies — gate before starting

1. Phase-2 + Phase-6 working (otherwise the alt-data has no model to
   feed).
2. AUM or revenue trajectory that justifies $140k+/yr alt-data spend.
   Realistic threshold: ~$5-10M AUM **OR** demonstrable Phase-2-only
   PnL of $50k+/month for 6 months.
3. Storage + compute for the alt-data ETL: ~10-50 TB/yr, ~$2-10k/yr
   on cloud object storage.

## Build order when the time comes

1. **Feed C (talent flow) first**. Cheapest, highest signal-to-noise
   for tech/biotech/defense names, low integration cost.
2. **Feed A (pricing)** second. Cheap data at the bottom of the cost
   range; useful even without Feed B since pricing-power signal stands
   alone for some thesis types.
3. **Feed B (transactions)** last. Most expensive, highest-value, but
   only worth it after Feed A is integrated (pricing × volume = revenue
   needs both).

## What this does NOT include

- **Satellite imagery** (parking lots, oil tanks): high value, very
  expensive ($50k-500k+/yr for institutional-quality coverage).
  Defer to a hypothetical Phase 7.5 if AUM justifies.
- **Geolocation panels** (Foursquare, SafeGraph): captures
  brick-and-mortar foot traffic. Same cost tier as YipitData.
- **Credit-card panel data** (Earnest, Bloomberg Second Measure): often
  redundant with YipitData/Measurable AI for the consumer signals we
  care about.

## Related files

- [`docs/architecture.md`](architecture.md) — diagram showing all three
  Phase-7 feeds entering the model
- [`docs/phase6_alpha_factor_discovery.md`](phase6_alpha_factor_discovery.md)
  — Phase-6 fundamental + symbolic spec; Phase-7 sits above it
- [`docs/cross_asset_fusion.md`](cross_asset_fusion.md) — modality
  embedding mechanism reused for all alt-data channels
