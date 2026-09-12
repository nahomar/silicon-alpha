"""Significance tests for a panel where rows are not independent.

An option chain snapshot is the extreme case of cross-sectional dependence:
every contract at a timestamp is a function of the *same* underlying price. A
signal that merely predicts the next move in SPX will look like it predicts
hundreds of individual option returns, and any test that treats those rows as
independent will report overwhelming significance for what is one bet.

Two devices, both used throughout `option_edge`:

  within-timestamp permutation
      shuffles the signal among the contracts quoted at the same instant.
      Whatever the underlying did is preserved exactly; only the ability to
      rank *contracts against each other* is destroyed. This is the null that
      matters for a market-neutral options book.

  day-block bootstrap
      resamples whole trading days, keeping each day's rows together. Options
      P&L is fat-tailed and autocorrelated within a session, so resampling
      individual rows would understate the variance badly.

The equity study in `nlpalpha/evaluate.py` implements the same two ideas for a
daily stock panel; when that branch lands, it should import from here rather
than keep a second copy.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def spearman_ic(signal: np.ndarray, forward_ret: np.ndarray) -> float:
    """Rank correlation between a signal and the return it predicts.

    Rank-based because option returns are violently non-normal -- a 0DTE
    contract can go +400% or -100% in one bar, and a Pearson correlation on
    those levels measures the outliers, not the signal.
    """
    s = np.asarray(signal, dtype=float)
    r = np.asarray(forward_ret, dtype=float)
    ok = np.isfinite(s) & np.isfinite(r)
    if ok.sum() < 3:
        return float("nan")
    rs = _rankdata(s[ok])
    rr = _rankdata(r[ok])
    rs -= rs.mean()
    rr -= rr.mean()
    denom = np.sqrt((rs ** 2).sum() * (rr ** 2).sum())
    return float((rs * rr).sum() / denom) if denom > _EPS else 0.0


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def group_indices(keys: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by key, without assuming the keys are sorted."""
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    groups, start = [], 0
    for i in range(1, len(order) + 1):
        if i == len(order) or sorted_keys[i] != sorted_keys[start]:
            groups.append(order[start:i])
            start = i
    return groups


def permutation_ic(signal: np.ndarray, forward_ret: np.ndarray,
                   timestamps: np.ndarray, n_perm: int = 1000,
                   seed: int = 0) -> dict:
    """p-value for the IC under within-timestamp shuffling of the signal.

    Reports `null_mean` alongside the p-value, and the two should be read
    together. A null centered well away from zero is itself the finding: it
    means the raw IC is dominated by a timestamp-level effect (the underlying
    moved) rather than by choosing between contracts.
    """
    rng = np.random.default_rng(seed)
    s = np.asarray(signal, dtype=float)
    r = np.asarray(forward_ret, dtype=float)
    observed = spearman_ic(s, r)

    groups = [g for g in group_indices(np.asarray(timestamps)) if len(g) > 1]
    if not groups:
        return {"ic": observed, "p_value": float("nan"), "null_mean": float("nan"),
                "null_std": float("nan"), "n_perm": 0,
                "note": "no timestamp has more than one contract; nothing to permute"}

    null = np.empty(n_perm)
    shuffled = s.copy()
    for k in range(n_perm):
        for g in groups:
            shuffled[g] = s[rng.permutation(g)]
        null[k] = spearman_ic(shuffled, r)
    # +1 top and bottom: a finite-permutation p-value that never reads as 0.
    p = float((1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1))
    return {"ic": float(observed), "p_value": p,
            "null_mean": float(np.mean(null)), "null_std": float(np.std(null)),
            "n_perm": int(n_perm)}


def day_block_bootstrap(daily_pnl: np.ndarray, n_boot: int = 2000,
                        seed: int = 0, periods_per_year: int = 252) -> dict:
    """Sharpe with a confidence interval, resampling whole days."""
    rng = np.random.default_rng(seed)
    r = np.asarray(daily_pnl, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return {"sharpe": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_sharpe_le_0": float("nan"),
                "n_days": int(len(r))}

    def sharpe(x):
        sd = x.std(ddof=1)
        return float(x.mean() / sd * np.sqrt(periods_per_year)) if sd > _EPS else 0.0

    stats = np.array([sharpe(r[rng.integers(0, len(r), len(r))])
                      for _ in range(n_boot)])
    return {"sharpe": sharpe(r),
            "ci_low": float(np.percentile(stats, 2.5)),
            "ci_high": float(np.percentile(stats, 97.5)),
            "p_sharpe_le_0": float(np.mean(stats <= 0.0)),
            "n_days": int(len(r))}
