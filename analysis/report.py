"""Produce a Markdown analysis report pulling together:
  - regulatory / event context (static, hand-curated from real sources)
  - market correlation & anomaly snapshot
  - social sentiment + ticker-mention spikes
  - cross-source patterns
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


EVENT_CONTEXT_APR_17_2026 = """
### Regulatory & policy events on / around April 17, 2026 (from public sources)

- **FINRA Rule 4210 amendment approved (SEC order, April 17, 2026)** — replaces
  day-trading margin provisions with **intraday margin standards**. 12-month
  implementation window once effective date is announced. Affects broker-dealers
  and active retail traders.
- **Nasdaq 23-hour trading approved (April 15, 2026)** — Nasdaq may extend U.S.
  equities trading to 23 hours/day, 5 days/week; order routing and risk
  controls have to be re-architected across the street before launch.
- **NYSE proprietary market-data fee change (April 2, 2026)** — immediately
  effective, raising costs for market-data consumers.
- **Federal Reserve enforcement action (April 16, 2026)** and release of
  February 9 / March 18 discount-rate minutes (April 14, 2026).
- **Section 122 universal 10% tariff in force** (post-Feb-2026 Supreme Court
  ruling); administration discussing a 15% headline rate.
- **Iran / Strait of Hormuz**: two-week ceasefire reached April 7; Strait
  re-opened on April 16/17 — oil gave up much of the March spike.

### What's expected Monday morning, April 20, 2026

- **Earnings pre-open / AH** (from week-ahead coverage): heavy financials +
  regional banks; streaming / media (Netflix already reported April 17 after
  close with a 10% AH drop on soft Q2 guidance); semis sentiment shaped by
  ASML's Wednesday beat-but-China-drag and TSMC's read-through.
- **Macro calendar**: Conference Board Leading Index (Monday), plus a heavy
  Tuesday–Thursday block of flash PMIs, existing-home sales, and jobless
  claims. **April 28–29 FOMC** meeting is the dominant forward risk.
- **Geopolitics**: Hormuz ceasefire expiry tension — any re-escalation drives
  an energy / defense beta trade.
- **Rule effective-date watch**: FINRA intraday-margin regime not yet live on
  Monday, but prime-brokerage desks will be updating client notices.
"""


def _format_corr_pairs(pairs, k: int = 15) -> str:
    rows = ["| A | B | ρ |", "|---|---|---|"]
    for a, b, v in pairs[:k]:
        rows.append(f"| {a} | {b} | {v:+.2f} |")
    return "\n".join(rows)


def _format_df(df: pd.DataFrame, k: int = 15) -> str:
    if df is None or df.empty:
        return "_no data_"
    return df.head(k).to_markdown(index=False, floatfmt=".3f")


def build_report(
    prices,
    ret,
    corr_pairs,
    anomalies,
    cluster_map,
    tagged_items,
    sentiment_df,
    lead_lag_df,
    spikes,
) -> Path:
    ts = time.strftime("%Y-%m-%d %H:%M")

    cluster_lines = []
    if cluster_map:
        by_c: dict[int, list[str]] = {}
        for t, c in cluster_map.items():
            by_c.setdefault(c, []).append(t)
        for c, members in sorted(by_c.items()):
            cluster_lines.append(f"- **Cluster {c}**: " + ", ".join(sorted(members)))

    last_row = prices.iloc[-1].dropna() if prices is not None and not prices.empty else pd.Series(dtype=float)
    prev_row = prices.iloc[-2].dropna() if prices is not None and len(prices) > 1 else pd.Series(dtype=float)
    summary_rows: list[str] = []
    for t in last_row.index:
        if t in prev_row.index:
            chg = last_row[t] / prev_row[t] - 1
            summary_rows.append(f"| {t} | {last_row[t]:.2f} | {chg:+.2%} |")
    price_table = "| Ticker | Last | 1d % |\n|---|---|---|\n" + "\n".join(summary_rows) if summary_rows else "_no prices_"

    sample_items = "\n".join(
        f"- [{(it.get('source') or '?')[:12]}] sent={it.get('sentiment', 0):+.2f} "
        f"tickers={','.join((it.get('tickers') or [])[:4])}: "
        f"{(it.get('title') or it.get('text') or '')[:140]}"
        for it in tagged_items[:20]
    ) or "_no items scraped_"

    body = f"""# Market Pattern Report — {ts}

{EVENT_CONTEXT_APR_17_2026}

---

## 1. Price snapshot
{price_table}

## 2. Anomaly screen (z-score of today's return vs trailing 63-day window)
{_format_df(anomalies, 15)}

## 3. Highly correlated pairs (|ρ| ≥ 0.8, 6-month daily returns)
{_format_corr_pairs(corr_pairs, 20)}

## 4. Return-based clusters (KMeans on daily returns)
{chr(10).join(cluster_lines) or '_no clusters_'}

## 5. Social mention spikes (ticker mentions ≥ 2σ above 14-day avg)
{_format_df(spikes.reset_index() if hasattr(spikes, 'reset_index') else spikes, 20)}

## 6. Sentiment → return lead/lag (by ticker)
_Positive `best_lag_days` ⇒ social sentiment leads price; negative ⇒ price leads chatter._

{_format_df(lead_lag_df, 20)}

## 7. Sample scraped items
{sample_items}
"""
    out = REPORTS_DIR / f"report_{time.strftime('%Y%m%dT%H%M%S')}.md"
    out.write_text(body)
    return out
