"""Point-in-time tests for the free alternative-data features."""
import numpy as np
import pandas as pd

from odte.data import altdata as ad


def test_counts_include_before_and_exclude_at_t():
    ts = np.array([100, 200, 300, 400])
    assert ad._count(ts, 100, 300) == 2          # [100, 300): 100 and 200, not 300


def test_kalshi_ignores_prices_posted_after_t(tmp_path, monkeypatch):
    monkeypatch.setattr(ad, "ALT", tmp_path)
    (tmp_path / "kalshi_inx").mkdir()
    day = "2026-03-18"
    t = ad._et_utc(day, "12:00")
    rows = []
    for i, lo in enumerate(range(6000, 6250, 25)):
        rows.append((f"B{lo}", "between", lo, lo + 24.99, t - 1800, 0.08, 0.12))
        rows.append((f"B{lo}", "between", lo, lo + 24.99, t + 1800, 0.90 if i == 0 else 0.0, 0.95))
    pd.DataFrame(rows, columns=["ticker", "strike_type", "floor", "cap", "ts", "bid", "ask"]) \
      .to_parquet(tmp_path / "kalshi_inx" / f"{day}.parquet")
    a = ad._kalshi_inx(day, t, 6125.0, 30.0)
    # The after-t row (all weight in the lowest bucket) must not move the answer.
    assert np.isfinite(a[0]) and abs(a[1] - 0.5) < 0.11


def test_fed_entropy_uses_prior_close_only():
    meet = "2026-03-18T18:00:00Z"
    day = pd.Timestamp("2026-03-18")
    cut = ad._et_utc(day, "09:30")
    base = [("KXFED-26MAR", f"T{x}", x, meet, cut - 86400, p - 0.01, p + 0.01)
            for x, p in ((4.0, 0.9), (4.25, 0.5), (4.5, 0.1))]
    later = [("KXFED-26MAR", f"T{x}", x, meet, cut + 3600, 0.99, 1.0) for x in (4.0, 4.25, 4.5)]
    cols = ["event", "ticker", "threshold", "meeting_close", "ts", "bid", "ask"]
    h1 = ad._fed_entropy(pd.DataFrame(base, columns=cols), day)
    h2 = ad._fed_entropy(pd.DataFrame(base + later, columns=cols), day)
    assert np.isfinite(h1) and h1 == h2
