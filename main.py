"""Entry point: scrape → score → analyze → report.

Usage:
    python -m main                 # full pipeline
    python -m main --quick         # fewer tickers / posts, faster
    python -m main --no-scrape     # reuse last stored run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python main.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from scrapers import scrape_reddit, scrape_x, scrape_news  # noqa: E402
from data import load_prices, returns, default_universe, save_items, load_history  # noqa: E402
from ml import (  # noqa: E402
    score_and_tag,
    pairwise_corr,
    correlated_pairs,
    top_anomalies,
    cluster_by_corr,
    items_to_daily_sentiment,
    align_sentiment_and_returns,
    lead_lag_per_ticker,
    mention_spikes,
)
from analysis import build_report  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("main")


def run(quick: bool = False, no_scrape: bool = False) -> Path:
    if no_scrape:
        log.info("Reusing history from storage")
        hist = load_history()
        items = hist.to_dict("records") if not hist.empty else []
    else:
        reddit_limit = 15 if quick else 50
        news_limit = 10 if quick else 25
        x_limit = 15 if quick else 40

        log.info("Scraping Reddit…")
        r_items = scrape_reddit(limit=reddit_limit)
        save_items("reddit", r_items)
        log.info("  got %d reddit items", len(r_items))

        log.info("Scraping news…")
        n_items = scrape_news(limit=news_limit)
        save_items("news", n_items)
        log.info("  got %d news items", len(n_items))

        log.info("Scraping X…")
        x_items = scrape_x(limit=x_limit)
        save_items("x", x_items)
        log.info("  got %d x items", len(x_items))

        items = r_items + n_items + x_items

    log.info("Scoring sentiment + tagging tickers on %d items", len(items))
    tagged = score_and_tag(items)

    log.info("Loading prices…")
    universe = default_universe()
    prices = load_prices(universe, period="6mo")
    ret = returns(prices)

    log.info("Running correlation + anomaly screens…")
    corr_pairs = correlated_pairs(ret, threshold=0.8)
    anomalies = top_anomalies(ret, window=63, n=15)
    try:
        clusters = cluster_by_corr(ret, k=4)
    except Exception as e:
        log.warning("Cluster step failed: %s", e)
        clusters = {}

    log.info("Aligning sentiment with returns…")
    sentiment_df = items_to_daily_sentiment(tagged)
    merged = align_sentiment_and_returns(sentiment_df, prices)
    lead_lag_df = lead_lag_per_ticker(merged, max_lag=5) if not merged.empty else None
    spikes = mention_spikes(sentiment_df, window=14, z=2.0)

    # Attempt hourly fill-model retrain if we have live mm data accumulated.
    try:
        import mm_fill_trainer
        log.info("Running fill-probability trainer (hourly)…")
        mm_fill_trainer.main(hours=24, horizon_ms=2000, min_rows=500)
    except Exception as e:
        log.warning("fill trainer skipped: %s", e)

    log.info("Building report…")
    path = build_report(
        prices=prices,
        ret=ret,
        corr_pairs=corr_pairs,
        anomalies=anomalies,
        cluster_map=clusters,
        tagged_items=tagged,
        sentiment_df=sentiment_df,
        lead_lag_df=lead_lag_df,
        spikes=spikes,
    )
    log.info("Report → %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-scrape", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick, no_scrape=args.no_scrape)


if __name__ == "__main__":
    main()
