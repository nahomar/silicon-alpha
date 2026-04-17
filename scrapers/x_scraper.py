"""X / Twitter scraper.

X locks down most free scraping. We support three paths in order:
  1. X API v2 recent-search (needs X_BEARER_TOKEN).
  2. snscrape (no auth; may break as X changes html).
  3. Nitter mirror as a last-resort RSS fallback.
"""
from __future__ import annotations

import os
import logging
import urllib.parse
from dataclasses import dataclass, asdict
from typing import Iterable, List

import requests
import feedparser

log = logging.getLogger(__name__)

DEFAULT_QUERIES = [
    "$SPY OR $QQQ OR $SPX",
    "FOMC OR \"Jerome Powell\"",
    "tariff OR tariffs",
    "earnings beat OR earnings miss",
    "SEC OR FINRA rule",
]

NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
]


@dataclass
class Tweet:
    source: str
    id: str
    query: str
    text: str
    author: str
    created_at: str
    url: str
    likes: int = 0
    retweets: int = 0

    def to_dict(self):
        return asdict(self)


def _via_api(queries: Iterable[str], limit: int) -> List[Tweet]:
    token = os.getenv("X_BEARER_TOKEN")
    if not token:
        return []
    out: List[Tweet] = []
    headers = {"Authorization": f"Bearer {token}"}
    for q in queries:
        params = {
            "query": q + " lang:en -is:retweet",
            "max_results": min(max(limit, 10), 100),
            "tweet.fields": "created_at,public_metrics,author_id",
        }
        try:
            r = requests.get(
                "https://api.twitter.com/2/tweets/search/recent",
                headers=headers,
                params=params,
                timeout=25,
            )
            if r.status_code != 200:
                log.warning("X API %s for %r", r.status_code, q)
                continue
            for t in r.json().get("data", []):
                pm = t.get("public_metrics", {})
                out.append(
                    Tweet(
                        source="x",
                        id=t["id"],
                        query=q,
                        text=t.get("text", ""),
                        author=t.get("author_id", ""),
                        created_at=t.get("created_at", ""),
                        url=f"https://x.com/i/web/status/{t['id']}",
                        likes=int(pm.get("like_count", 0)),
                        retweets=int(pm.get("retweet_count", 0)),
                    )
                )
        except Exception as e:
            log.warning("X API error: %s", e)
    return out


def _via_snscrape(queries: Iterable[str], limit: int) -> List[Tweet]:
    try:
        import snscrape.modules.twitter as sntwitter  # type: ignore
    except Exception:
        return []
    out: List[Tweet] = []
    for q in queries:
        try:
            for i, t in enumerate(sntwitter.TwitterSearchScraper(q + " lang:en").get_items()):
                if i >= limit:
                    break
                out.append(
                    Tweet(
                        source="x",
                        id=str(t.id),
                        query=q,
                        text=t.rawContent or "",
                        author=t.user.username if t.user else "",
                        created_at=t.date.isoformat() if t.date else "",
                        url=t.url,
                        likes=int(t.likeCount or 0),
                        retweets=int(t.retweetCount or 0),
                    )
                )
        except Exception as e:
            log.warning("snscrape error for %r: %s", q, e)
    return out


def _via_nitter(queries: Iterable[str], limit: int) -> List[Tweet]:
    out: List[Tweet] = []
    for q in queries:
        encoded = urllib.parse.quote(q)
        for base in NITTER_INSTANCES:
            url = f"{base}/search/rss?f=tweets&q={encoded}"
            try:
                d = feedparser.parse(url)
                if not d.entries:
                    continue
                for e in d.entries[:limit]:
                    out.append(
                        Tweet(
                            source="x",
                            id=e.get("id", e.link),
                            query=q,
                            text=e.get("title", ""),
                            author=e.get("author", ""),
                            created_at=e.get("published", ""),
                            url=e.get("link", ""),
                        )
                    )
                break  # first working instance
            except Exception as e:
                log.debug("Nitter %s failed: %s", base, e)
    return out


def scrape_x(queries: Iterable[str] | None = None, limit: int = 50) -> List[dict]:
    qs = list(queries) if queries else DEFAULT_QUERIES
    tweets = _via_api(qs, limit)
    if not tweets:
        tweets = _via_snscrape(qs, limit)
    if not tweets:
        tweets = _via_nitter(qs, limit)
    return [t.to_dict() for t in tweets]
