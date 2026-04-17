"""News scraper.

Uses:
  1. NewsAPI.org if NEWSAPI_KEY is set (paid but has generous dev tier).
  2. Always: Google News RSS + major financial RSS feeds (free, reliable).
"""
from __future__ import annotations

import os
import logging
import urllib.parse
from dataclasses import dataclass, asdict
from typing import Iterable, List

import feedparser
import requests

log = logging.getLogger(__name__)

DEFAULT_QUERIES = [
    "stock market",
    "Federal Reserve",
    "SEC rule",
    "FINRA margin",
    "tariff",
    "earnings",
    "Strait of Hormuz oil",
]

# Free, no-auth financial RSS feeds.
STATIC_FEEDS = [
    "https://www.sec.gov/rss/news/press.xml",
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # Top News
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",   # Economy
    "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
]


@dataclass
class Article:
    source: str
    title: str
    summary: str
    url: str
    published: str
    query: str = ""

    def to_dict(self):
        return asdict(self)


def _via_newsapi(queries: Iterable[str], limit: int) -> List[Article]:
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        return []
    out: List[Article] = []
    for q in queries:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": q,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": min(max(limit, 10), 100),
                    "apiKey": key,
                },
                timeout=20,
            )
            if r.status_code != 200:
                log.warning("NewsAPI %s for %r", r.status_code, q)
                continue
            for a in r.json().get("articles", []):
                out.append(
                    Article(
                        source="newsapi:" + (a.get("source", {}).get("name") or "?"),
                        title=a.get("title") or "",
                        summary=a.get("description") or "",
                        url=a.get("url") or "",
                        published=a.get("publishedAt") or "",
                        query=q,
                    )
                )
        except Exception as e:
            log.warning("NewsAPI error: %s", e)
    return out


def _google_news_rss(queries: Iterable[str], limit: int) -> List[Article]:
    out: List[Article] = []
    for q in queries:
        url = (
            "https://news.google.com/rss/search?q="
            + urllib.parse.quote(q)
            + "&hl=en-US&gl=US&ceid=US:en"
        )
        d = feedparser.parse(url)
        for e in d.entries[:limit]:
            out.append(
                Article(
                    source="google_news",
                    title=e.get("title", ""),
                    summary=e.get("summary", ""),
                    url=e.get("link", ""),
                    published=e.get("published", ""),
                    query=q,
                )
            )
    return out


def _static_feeds(limit: int) -> List[Article]:
    out: List[Article] = []
    for url in STATIC_FEEDS:
        try:
            d = feedparser.parse(url)
            for e in d.entries[:limit]:
                out.append(
                    Article(
                        source=f"rss:{urllib.parse.urlparse(url).netloc}",
                        title=e.get("title", ""),
                        summary=e.get("summary", ""),
                        url=e.get("link", ""),
                        published=e.get("published", ""),
                    )
                )
        except Exception as e:
            log.warning("RSS %s failed: %s", url, e)
    return out


def scrape_news(queries: Iterable[str] | None = None, limit: int = 25) -> List[dict]:
    qs = list(queries) if queries else DEFAULT_QUERIES
    articles = _via_newsapi(qs, limit)
    articles += _google_news_rss(qs, limit)
    articles += _static_feeds(limit)
    # dedupe by URL
    seen, unique = set(), []
    for a in articles:
        if a.url in seen or not a.url:
            continue
        seen.add(a.url)
        unique.append(a)
    return [a.to_dict() for a in unique]
