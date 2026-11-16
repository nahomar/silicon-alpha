"""Tests for the Reddit recorder's paging, storage and leakage rules.

The fake archive mimics what was measured on the real one: `after` is
exclusive, pages come back oldest-first, and page size varies.
"""
import pandas as pd
import pytest

from social import reddit


def _fake_archive(items, page_sizes):
    sizes = iter(page_sizes)

    def _get(path, params):
        lo, hi = params["after"], params["before"]
        hits = sorted((x for x in items if lo < x["created_utc"] < hi),
                      key=lambda x: (x["created_utc"], x["id"]))
        return hits[: next(sizes, 100)]
    return _get


def _items():
    # Many same-second ties, which is where a naive cursor drops rows.
    out, i = [], 0
    for sec in range(1000, 1040):
        for _ in range(1 + (sec % 7)):
            out.append({"id": f"c{i}", "created_utc": sec, "body": "x",
                        "author": "someone", "score": 999,
                        "retrieved_on": sec + 20})
            i += 1
    return out


def test_paging_is_complete_with_ties_and_variable_pages(monkeypatch):
    # Pages vary but always exceed the busiest second (7 items), which is
    # the regime of the real archive (>= 100 per page, a few per second).
    items = _items()
    monkeypatch.setattr(reddit, "_get", _fake_archive(items, [9, 12, 8, 15, 10] * 50))
    monkeypatch.setattr(reddit, "PAUSE_S", 0)
    rows = list(reddit.fetch_range("wallstreetbets", "comments", 1000, 1040))
    assert sorted(r["id"] for r in rows) == sorted(x["id"] for x in items)
    assert len({r["id"] for r in rows}) == len(rows)


def test_range_bounds_are_inclusive_exclusive(monkeypatch):
    items = _items()
    monkeypatch.setattr(reddit, "_get", _fake_archive(items, [100] * 50))
    monkeypatch.setattr(reddit, "PAUSE_S", 0)
    rows = list(reddit.fetch_range("wallstreetbets", "comments", 1010, 1020))
    ts = {r["created_utc"] for r in rows}
    assert min(ts) == 1010 and max(ts) == 1019


def test_rows_drop_future_votes_and_hash_authors():
    r = reddit._row({"id": "a", "created_utc": 5, "retrieved_on": 25,
                     "author": "real_name", "body": "hi", "score": 4000,
                     "subreddit": "options"}, "comments")
    assert "score" not in r and set(r) == set(reddit.COLUMNS)
    assert r["author_hash"] != "real_name" and len(r["author_hash"]) == 16


def test_write_dedupes_across_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(reddit, "OUT", tmp_path)
    row = reddit._row({"id": "a", "created_utc": 1790863201, "body": "hi",
                       "author": "x", "subreddit": "options"}, "comments")
    assert reddit._write("options", "comments", [row]) == 1
    assert reddit._write("options", "comments", [row]) == 0
    f = tmp_path / "options" / "comments" / "2026-10-01.parquet"
    assert len(pd.read_parquet(f)) == 1


def test_page_inside_one_second_is_flagged(monkeypatch, capsys):
    items = [{"id": f"c{i}", "created_utc": 1000, "body": "x", "author": "a"}
             for i in range(10)]
    monkeypatch.setattr(reddit, "_get", _fake_archive(items, [4] * 10))
    monkeypatch.setattr(reddit, "PAUSE_S", 0)
    list(reddit.fetch_range("wallstreetbets", "comments", 1000, 1001))
    assert "WARN" in capsys.readouterr().out


def _fake_sync_env(monkeypatch, tmp_path, items):
    monkeypatch.setattr(reddit, "OUT", tmp_path)
    monkeypatch.setattr(reddit, "PAUSE_S", 0)
    monkeypatch.setattr(reddit, "_get", _fake_archive(items, [50] * 100000))


def _sync_items():
    # One comment per 10 minutes over 3 days starting 2026-10-01 00:00Z.
    t0 = 1790812800
    return [{"id": f"x{i}", "created_utc": t0 + 600 * i, "body": "b",
             "author": "a", "subreddit": "options"} for i in range(432)]


def _all_ids(tmp_path):
    files = sorted((tmp_path / "options" / "comments").glob("*.parquet"))
    return sorted(pd.concat(pd.read_parquet(f) for f in files)["id"]) if files else []


def test_backward_fill_reaches_start_after_plain_catch_up(monkeypatch, tmp_path):
    items = _sync_items()
    _fake_sync_env(monkeypatch, tmp_path, items)
    monkeypatch.setattr(reddit.time, "time", lambda: 1790812800 + 600 * 432)
    reddit.sync(subs=("options",))                       # no --start: last hour only
    assert len(_all_ids(tmp_path)) == 6
    reddit.sync(start="2026-10-01", subs=("options",))   # must fill everything before
    assert _all_ids(tmp_path) == sorted(x["id"] for x in items)


def test_interrupted_backward_fill_leaves_no_middle_hole(monkeypatch, tmp_path):
    items = _sync_items()
    _fake_sync_env(monkeypatch, tmp_path, items)
    monkeypatch.setattr(reddit.time, "time", lambda: 1790812800 + 600 * 432)
    reddit.sync(subs=("options",))
    real = reddit.fetch_range
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 4:                  # die partway through the backfill
            raise RuntimeError("network")
        return real(*a, **k)
    monkeypatch.setattr(reddit, "fetch_range", flaky)
    with pytest.raises(RuntimeError):
        reddit.sync(start="2026-10-01", subs=("options",))
    saved = _all_ids(tmp_path)
    idx = sorted(int(i[1:]) for i in saved)
    assert idx == list(range(idx[0], 432))   # one unbroken block ending at now
    monkeypatch.setattr(reddit, "fetch_range", real)
    reddit.sync(start="2026-10-01", subs=("options",))
    assert _all_ids(tmp_path) == sorted(x["id"] for x in items)
