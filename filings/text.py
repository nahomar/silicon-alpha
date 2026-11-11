"""10-K text fingerprints: word-set MinHash (Jaccard similarity year over year) + cover-page ticker.

Only compact features are kept (about 1 KB per filing); raw documents are never stored.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

N_PERM = 256
_P = (1 << 61) - 1
_rng = np.random.default_rng(12345)
_A = _rng.integers(1, _P - 1, N_PERM, dtype=np.uint64)
_B = _rng.integers(0, _P - 1, N_PERM, dtype=np.uint64)
WORD = re.compile(r"[a-z]{3,}")
TAG = re.compile(r"<[^>]+>")
TICKER = re.compile(r"<ix:nonNumeric[^>]*name=\"dei:TradingSymbol\"[^>]*>(.*?)</ix:nonNumeric>", re.I | re.S)
EXCH = re.compile(r"<ix:nonNumeric[^>]*name=\"dei:SecurityExchangeName\"[^>]*>(.*?)</ix:nonNumeric>", re.I | re.S)


def html_to_words(html: str) -> list:
    text = re.sub(r"(?is)<(script|style|ix:header).*?</\1>", " ", html)
    text = TAG.sub(" ", text)
    text = re.sub(r"&[a-z#0-9]+;", " ", text).lower()
    return WORD.findall(text)


def _h64(w: str) -> int:
    return int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")


def minhash(words) -> np.ndarray:
    """256 MinHash values of the word SET (uint32). Equal fraction of minima ~ Jaccard similarity."""
    s = np.fromiter((_h64(w) % _P for w in set(words)), dtype=np.uint64)
    if not len(s):
        return np.full(N_PERM, np.iinfo(np.uint32).max, np.uint32)
    out = np.empty(N_PERM, np.uint64)
    for i in range(0, N_PERM, 32):                      # chunks keep memory small for long documents
        a, b = _A[i:i + 32, None], _B[i:i + 32, None]
        out[i:i + 32] = ((a * s[None, :] + b) % _P).min(axis=1)  # uint64 wraps; fine for hashing
    return (out & 0xFFFFFFFF).astype(np.uint32)


def jaccard(m1: np.ndarray, m2: np.ndarray) -> float:
    return float(np.mean(m1 == m2))


def cover(html: str) -> dict:
    t = TICKER.search(html)
    e = EXCH.search(html)
    clean = lambda m: TAG.sub("", m.group(1)).strip().upper() if m else None
    return {"ticker": clean(t), "exchange": clean(e)}


def features(html: str) -> dict:
    words = html_to_words(html)
    return {"n_words": len(words), "n_unique": len(set(words)), "minhash": minhash(words), **cover(html)}
