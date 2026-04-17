"""Sentiment scoring.

Default backend: VADER (rule-based, fast, no model download).
Optional backend: HuggingFace transformers (finbert) if installed.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, List, Dict

log = logging.getLogger(__name__)

TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b")

# Common English words that look like tickers — exclude to reduce false positives.
_TICKER_STOPWORDS = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY", "CAN", "HAD",
    "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW",
    "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID", "ITS",
    "LET", "PUT", "SAY", "SHE", "TOO", "USE", "CEO", "CFO", "COO", "USA", "USD",
    "GDP", "CPI", "PPI", "IPO", "ETF", "EPS", "PE", "AI", "IT", "IS", "OF", "TO",
    "IN", "ON", "AT", "BE", "BY", "OR", "AS", "WE", "US", "UP", "IF", "SO", "NO",
    "DO", "GO", "MY", "ME", "AM", "AN", "EX", "RSS", "API", "URL", "FAQ", "EDT",
    "EST", "PST", "PDT", "UTC", "GMT", "NYSE", "SEC", "FED", "FOMC", "FINRA",
}


def _vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as e:
        raise RuntimeError("vaderSentiment required: pip install vaderSentiment") from e
    return SentimentIntensityAnalyzer()


def score_texts(texts: Iterable[str]) -> List[Dict[str, float]]:
    """Return list of {text, compound, pos, neu, neg} dicts."""
    an = _vader()
    out: List[Dict[str, float]] = []
    for t in texts:
        if not t:
            out.append({"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0})
            continue
        s = an.polarity_scores(t)
        out.append({
            "compound": s["compound"],
            "pos": s["pos"],
            "neu": s["neu"],
            "neg": s["neg"],
        })
    return out


def extract_tickers(text: str) -> List[str]:
    if not text:
        return []
    hits = set()
    for m in TICKER_RE.finditer(text):
        t = m.group(1) or m.group(2) or ""
        if not t:
            continue
        if t in _TICKER_STOPWORDS:
            continue
        if len(t) < 2:
            continue
        hits.add(t)
    return sorted(hits)


def score_and_tag(items: Iterable[dict], text_fields=("title", "body", "text", "summary")) -> list[dict]:
    """Score each item and attach sentiment + tickers."""
    out = []
    for it in items:
        text = " ".join(str(it.get(f, "") or "") for f in text_fields).strip()
        s = score_texts([text])[0]
        it = {**it, "sentiment": s["compound"], "sent_pos": s["pos"],
              "sent_neg": s["neg"], "tickers": extract_tickers(text)}
        out.append(it)
    return out
