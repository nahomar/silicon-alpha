"""Transformer text scores for Stocktwits messages and Reddit posts.

Two pretrained transformers, both already in the local Hugging Face cache and
loaded offline (HF_HUB_OFFLINE=1), so nothing is downloaded:

  cardiffnlp/twitter-roberta-base-sentiment-latest
      RoBERTa fine-tuned on ~124M tweets; short informal posts are exactly
      its domain. Gives p_neg / p_neu / p_pos per text.
  sentence-transformers/all-MiniLM-L6-v2
      sentence embeddings, mean-pooled. Each text is scored by cosine
      similarity to fixed FEAR and GREED prototype phrases.

The prototypes below were written before any result was seen and are part
of the pre-registration. Changing them after looking at the study would turn
the NLP test into a search over phrasings.

Scores are cached per message id under data/social/nlp/, so re-running only
scores new text. Aggregation into point-in-time window features lives in
social.features (nlp_* columns).

Usage:
  python3 -m social.nlp score            # score everything not yet scored
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "social" / "nlp"
STW = ROOT / "data" / "social" / "stocktwits"
RDT = ROOT / "data" / "social" / "reddit"
SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Pre-registered 2026-11-08, before any NLP result was seen.
FEAR = ["the market is crashing", "buying puts before the drop",
        "this is going to dump hard", "VIX is spiking, panic selling",
        "recession is coming, sell everything"]
GREED = ["calls are printing", "going to the moon", "buy the dip, all in",
         "easy money on 0DTE calls", "the market only goes up"]
ZERO_DTE = re.compile(r"\b(0\s?dte|zero\s?dte|same[- ]day expir|spx\s+(calls?|puts?)|spy\s+(calls?|puts?))\b", re.I)


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Scorer:
    def __init__(self, batch: int = 64):
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.dev = _device()
        self.batch = batch
        self.st_tok = AutoTokenizer.from_pretrained(SENT_MODEL)
        self.st = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL).to(self.dev).eval()
        labels = {v.lower(): k for k, v in self.st.config.id2label.items()}
        self.ix = [labels["negative"], labels["neutral"], labels["positive"]]
        self.em_tok = AutoTokenizer.from_pretrained(EMB_MODEL)
        self.em = AutoModel.from_pretrained(EMB_MODEL).to(self.dev).eval()
        self.fear = self._embed(FEAR)
        self.greed = self._embed(GREED)

    def _embed(self, texts):
        t = self.torch
        enc = self.em_tok(texts, padding=True, truncation=True, max_length=128,
                          return_tensors="pt").to(self.dev)
        with t.no_grad():
            h = self.em(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return t.nn.functional.normalize(v, dim=-1)

    def score(self, texts: list) -> pd.DataFrame:
        t = self.torch
        rows = []
        for i in range(0, len(texts), self.batch):
            chunk = [(x or "")[:1000] for x in texts[i:i + self.batch]]
            enc = self.st_tok(chunk, padding=True, truncation=True, max_length=128,
                              return_tensors="pt").to(self.dev)
            with t.no_grad():
                p = t.softmax(self.st(**enc).logits, dim=-1)[:, self.ix].cpu().numpy()
            e = self._embed(chunk)
            fear = (e @ self.fear.T).max(dim=1).values.cpu().numpy()
            greed = (e @ self.greed.T).max(dim=1).values.cpu().numpy()
            for j, x in enumerate(chunk):
                rows.append((p[j, 0], p[j, 1], p[j, 2], fear[j], greed[j],
                             bool(ZERO_DTE.search(x))))
        return pd.DataFrame(rows, columns=["p_neg", "p_neu", "p_pos", "fear", "greed", "zero_dte"])


def _texts_stocktwits(streams=("SPX", "ES_F", "VIX")) -> pd.DataFrame:
    parts = []
    for s in streams:
        for f in sorted((STW / s).glob("20??-??-??.parquet")):
            parts.append(pd.read_parquet(f, columns=["id", "created_utc", "body"]))
    if not parts:
        return pd.DataFrame(columns=["id", "created_utc", "text"])
    d = pd.concat(parts).drop_duplicates("id").rename(columns={"body": "text"})
    d["id"] = d["id"].astype(str)
    return d


def _texts_reddit_posts(subs=("wallstreetbets", "options", "thetagang")) -> pd.DataFrame:
    parts = []
    for s in subs:
        for f in sorted((RDT / s / "posts").glob("20??-??-??.parquet")):
            d = pd.read_parquet(f, columns=["id", "created_utc", "title", "text", "removed"])
            d["sub"] = s
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["id", "created_utc", "text", "sub"])
    d = pd.concat(parts).drop_duplicates("id")
    body = d["text"].where(~d["removed"].fillna(False), "")
    d["text"] = (d["title"].fillna("") + ". " + body.fillna("")).str.slice(0, 1000)
    return d[["id", "created_utc", "text", "sub"]]


def _texts_truth() -> pd.DataFrame:
    p = ROOT / "data" / "alt" / "truth.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["id", "created_utc", "text"])
    d = pd.read_parquet(p)
    d = d[d["text"].fillna("").str.len() > 0]            # image-only posts have no text
    return d[d["created_utc"] >= 1759276800]             # 2025-10-01 onward


def score_all() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    scorer = None
    done = {}
    for name, loader in (("stocktwits", _texts_stocktwits), ("reddit_posts", _texts_reddit_posts),
                         ("truth", _texts_truth)):
        src = loader()
        p = OUT / f"{name}.parquet"
        old = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["id"])
        todo = src[~src["id"].isin(set(old["id"]))]
        if len(todo):
            scorer = scorer or Scorer()
            sc = scorer.score(todo["text"].tolist())
            new = pd.concat([todo.drop(columns=["text"]).reset_index(drop=True), sc], axis=1)
            old = pd.concat([old, new], ignore_index=True) if len(old) else new
            old.to_parquet(p, index=False)
        done[name] = (len(todo), len(old))
        print(f"{name}: scored {len(todo):,} new, {len(old):,} total", flush=True)
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("score")
    ap.parse_args()
    score_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
