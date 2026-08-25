"""Metrics and significance tests that respect the panel's dependence.

Daily stock returns are strongly correlated in the cross-section: on a day the
market falls, most names fall together. Treating 2,723 ticker-days as 2,723
independent observations therefore overstates significance badly -- an
"edge" that is really one market-direction bet repeated across 88 names looks
like 88 independent wins.

Everything here is built to avoid that:

  - the permutation test shuffles predictions *within each date*, which
    destroys cross-sectional skill while leaving every market-wide move
    intact. It answers the question a long/short book actually cares about:
    given what the market did, did we pick the right names?
  - the bootstrap resamples whole *days*, keeping each day's cross-section
    together, so correlated names are never split apart.
"""
from __future__ import annotations

import numpy as np

from .train import roc_auc


def accuracy(y_true, pred_label) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(pred_label)))


def matthews_corrcoef(y_true, pred_label) -> float:
    """MCC -- the metric the StockNet literature reports alongside accuracy.

    Preferred to accuracy on this task because a model that always predicts
    "up" scores 51% accuracy and an MCC of exactly 0, which is the honest
    description of it.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pred_label).astype(int)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def classification_report(y_true, prob, threshold: float = 0.5) -> dict:
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "base_rate_up": float(np.mean(y_true)),
        "accuracy": accuracy(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "auc": roc_auc(y_true, prob),
        "pred_up_rate": float(pred.mean()),
    }


def permutation_test_auc(y_true, score, dates, n_perm: int = 1000,
                         seed: int = 0) -> dict:
    """p-value for AUC under within-day shuffling of the scores.

    The null is "the score carries no cross-sectional information"; market-wide
    moves survive the shuffle, so a model that only learned "stocks went up in
    Q4" gains nothing here.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    dates = np.asarray(dates)
    observed = roc_auc(y, s)

    order = np.argsort(dates, kind="mergesort")
    groups, start = [], 0
    sorted_dates = dates[order]
    for i in range(1, len(order) + 1):
        if i == len(order) or sorted_dates[i] != sorted_dates[start]:
            groups.append(order[start:i])
            start = i

    null = np.empty(n_perm)
    shuffled = s.copy()
    for k in range(n_perm):
        for g in groups:
            if len(g) > 1:
                shuffled[g] = s[rng.permutation(g)]
        null[k] = roc_auc(y, shuffled)
    # +1 in numerator and denominator: an unbiased finite-permutation p-value
    # that can never report exactly zero.
    p = float((1 + np.sum(null >= observed)) / (n_perm + 1))
    return {"auc": float(observed), "p_value": p,
            "null_mean": float(np.mean(null)), "null_std": float(np.std(null)),
            "n_perm": int(n_perm)}


def day_block_bootstrap(daily_returns: np.ndarray, n_boot: int = 2000,
                        seed: int = 0, periods_per_year: int = 252) -> dict:
    """Resample whole days with replacement to bound the Sharpe ratio."""
    rng = np.random.default_rng(seed)
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return {"sharpe": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_days": int(len(r))}

    def sharpe(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0

    stats = np.array([sharpe(r[rng.integers(0, len(r), len(r))])
                      for _ in range(n_boot)])
    return {
        "sharpe": sharpe(r),
        "ci_low": float(np.percentile(stats, 2.5)),
        "ci_high": float(np.percentile(stats, 97.5)),
        "p_sharpe_le_0": float(np.mean(stats <= 0.0)),
        "n_days": int(len(r)),
    }


def compare_auc(y_true, score_a, score_b, dates, n_boot: int = 2000,
                seed: int = 0) -> dict:
    """Bootstrap CI for AUC(b) - AUC(a), resampling days.

    This is how H1 is judged. Two models each scoring, say, 0.52 AUC tell you
    nothing about whether the difference between them is real; the paired
    day-level bootstrap does.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    dates = np.asarray(dates)
    uniq = np.unique(dates)
    index = {d: np.flatnonzero(dates == d) for d in uniq}

    observed = roc_auc(y, b) - roc_auc(y, a)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([index[d] for d in pick])
        if len(np.unique(y[rows])) < 2:
            diffs[k] = np.nan
            continue
        diffs[k] = roc_auc(y[rows], b[rows]) - roc_auc(y[rows], a[rows])
    diffs = diffs[np.isfinite(diffs)]
    return {
        "delta_auc": float(observed),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_delta_le_0": float(np.mean(diffs <= 0.0)),
    }
