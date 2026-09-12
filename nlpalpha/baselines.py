"""Non-deep baselines the transformer has to beat to justify itself.

Three rungs of a ladder, cheapest first:

  trivial    always-up and momentum. A directional model on US equities that
             cannot beat "buy every day" is not a model -- the unconditional
             up-rate is above 50%, so accuracy alone always flatters.
  lexicon    VADER, a fixed sentiment dictionary. No learning, no fitting, no
             market data. If a hand-built word list matches a trained
             transformer, the transformer has learned nothing about finance.
  classical  TF-IDF over the day's text into logistic regression. This is the
             one that matters: bag-of-words with a linear head is a genuinely
             strong text baseline, and deep models on small noisy corpora
             routinely fail to beat it. Reporting it is what separates a
             finding from a press release.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def day_documents(panel: pd.DataFrame, tweets: pd.DataFrame) -> list[str]:
    """One whitespace-joined document per ticker-day."""
    token_col = tweets["tokens"].to_numpy()
    docs = []
    for idx in panel["tweet_idx"].to_numpy():
        parts = []
        for i in idx:
            parts.extend(str(t) for t in token_col[i])
        docs.append(" ".join(parts))
    return docs


def vader_day_scores(panel: pd.DataFrame, tweets: pd.DataFrame) -> pd.DataFrame:
    """Mean/min/max VADER compound over each ticker-day's tweets.

    The lexicon ships inside the package, so this needs no network and no
    fitting -- and therefore cannot leak. Its weakness on this corpus is
    known and worth stating: VADER was tuned on social media in general, not
    on cashtag finance, where "beat", "short" and "call" carry meanings it
    does not have.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    cache: dict[int, float] = {}
    token_col = tweets["tokens"].to_numpy()

    rows = []
    for idx in panel["tweet_idx"].to_numpy():
        scores = []
        for i in idx:
            if i not in cache:
                text = " ".join(str(t) for t in token_col[i])
                cache[i] = analyzer.polarity_scores(text)["compound"]
            scores.append(cache[i])
        if scores:
            arr = np.asarray(scores)
            rows.append((arr.mean(), arr.min(), arr.max(),
                         float((arr > 0.05).mean() - (arr < -0.05).mean())))
        else:
            rows.append((0.0, 0.0, 0.0, 0.0))

    return pd.DataFrame(rows, columns=["vader_mean", "vader_min",
                                       "vader_max", "vader_polarity"],
                        index=panel.index)


def tfidf_logreg(docs: list[str], y: np.ndarray, train_mask: np.ndarray,
                 val_mask: np.ndarray, test_mask: np.ndarray,
                 seed: int = 0) -> dict:
    """TF-IDF + logistic regression, C chosen on validation.

    The vectorizer is fitted on training documents only. Fitting it on the
    whole corpus is a real leak, not a technicality: the vocabulary and the
    IDF weights would both encode which words appear in the future.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from .train import roc_auc

    docs = np.asarray(docs, dtype=object)
    vec = TfidfVectorizer(min_df=5, max_features=50000, ngram_range=(1, 2),
                          sublinear_tf=True)
    xtr = vec.fit_transform(docs[train_mask])
    xva = vec.transform(docs[val_mask])
    xte = vec.transform(docs[test_mask])

    best = {"auc": -np.inf, "C": None, "model": None}
    for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        clf.fit(xtr, y[train_mask])
        auc = roc_auc(y[val_mask], clf.predict_proba(xva)[:, 1])
        if auc > best["auc"]:
            best = {"auc": auc, "C": C, "model": clf}

    clf = best["model"]
    return {
        "val_auc": float(best["auc"]),
        "C": best["C"],
        "n_features": int(xtr.shape[1]),
        "prob_val": clf.predict_proba(xva)[:, 1],
        "prob_test": clf.predict_proba(xte)[:, 1],
    }


def momentum_signal(feat: pd.DataFrame) -> np.ndarray:
    """Yesterday's return as a directional score.

    Short-horizon single-name equity returns reverse more often than they
    continue, so this is included with the sign that reversal implies; it is a
    baseline, not a recommendation.
    """
    return -feat["ret_1"].to_numpy()
