"""Reddit scraper.

Two backends:
  1. PRAW (preferred) when REDDIT_CLIENT_ID/SECRET are set.
  2. Public JSON fallback via old.reddit.com (no auth, rate-limited).
"""
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, asdict
from typing import Iterable, List

import requests

log = logging.getLogger(__name__)

DEFAULT_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "StockMarket",
    "options",
    "SecurityAnalysis",
    "economy",
]


@dataclass
class Post:
    source: str
    id: str
    subreddit: str
    title: str
    body: str
    score: int
    comments: int
    created_utc: float
    url: str
    author: str

    def to_dict(self):
        return asdict(self)


def _praw_client():
    try:
        import praw  # type: ignore
    except ImportError:
        return None
    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT", "silicon-alpha/0.1")
    if not cid or not csec:
        return None
    return praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)


def _scrape_praw(subreddits: Iterable[str], limit: int) -> List[Post]:
    reddit = _praw_client()
    if reddit is None:
        return []
    out: List[Post] = []
    for sub in subreddits:
        try:
            for s in reddit.subreddit(sub).hot(limit=limit):
                out.append(
                    Post(
                        source="reddit",
                        id=s.id,
                        subreddit=sub,
                        title=s.title or "",
                        body=s.selftext or "",
                        score=int(s.score or 0),
                        comments=int(s.num_comments or 0),
                        created_utc=float(s.created_utc or 0),
                        url=f"https://reddit.com{s.permalink}",
                        author=str(s.author) if s.author else "[deleted]",
                    )
                )
        except Exception as e:
            log.warning("PRAW failure in r/%s: %s", sub, e)
    return out


def _scrape_json(subreddits: Iterable[str], limit: int) -> List[Post]:
    headers = {"User-Agent": os.getenv("REDDIT_USER_AGENT", "silicon-alpha/0.1")}
    out: List[Post] = []
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/hot.json?limit={limit}"
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code != 200:
                log.warning("r/%s HTTP %s", sub, r.status_code)
                continue
            for child in r.json().get("data", {}).get("children", []):
                d = child.get("data", {})
                out.append(
                    Post(
                        source="reddit",
                        id=d.get("id", ""),
                        subreddit=sub,
                        title=d.get("title", ""),
                        body=d.get("selftext", ""),
                        score=int(d.get("score", 0)),
                        comments=int(d.get("num_comments", 0)),
                        created_utc=float(d.get("created_utc", 0)),
                        url="https://reddit.com" + d.get("permalink", ""),
                        author=d.get("author", ""),
                    )
                )
            time.sleep(1.0)  # be polite
        except Exception as e:
            log.warning("JSON failure r/%s: %s", sub, e)
    return out


def scrape_reddit(subreddits: Iterable[str] | None = None, limit: int = 50) -> List[dict]:
    subs = list(subreddits) if subreddits else DEFAULT_SUBREDDITS
    posts = _scrape_praw(subs, limit)
    if not posts:
        log.info("PRAW unavailable; falling back to public JSON")
        posts = _scrape_json(subs, limit)
    return [p.to_dict() for p in posts]
