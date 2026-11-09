"""Score GDELT finance headlines and turn them into point-in-time features.

Input: CSV from odte/data/gdelt_titles.sql (batch, source, title, tone).
Each unique headline is kept once, at the first batch it appeared in, and
scored by the same offline models as the social text (social/nlp.py):
positive-minus-negative sentiment and similarity to the fixed FEAR / GREED
phrases. A batch stamped s is visible from s + 15 minutes.

Features at decision time t (US/Eastern, naive):
  nw_n15, nw_n60      headlines in the last 15 / 60 minutes
  nw_sent15, nw_sent60   mean sentiment (p_pos - p_neg)
  nw_fg60             mean fear - greed similarity, last 60 min
  nw_tone60           mean GDELT tone, last 60 min
  nw_sent_night, nw_n_night   previous 16:00 -> today 09:30
  nw_sent_surprise    nw_sent60 minus the mean sentiment of the previous 5 days

Usage:
  python3 -m odte.data.news_titles score data/alt/gdelt_titles_2025-26.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCORED = ROOT / "data" / "alt" / "gdelt_titles_scored.parquet"
LAG = pd.Timedelta(minutes=15)
NEWS_FEATURES = ["nw_n15", "nw_n60", "nw_sent15", "nw_sent60", "nw_fg60", "nw_tone60",
                 "nw_sent_night", "nw_n_night", "nw_sent_surprise"]


def score(csv_path) -> pd.DataFrame:
    from social.nlp import Scorer
    g = pd.read_csv(csv_path).dropna(subset=["title"])
    g["visible_utc"] = pd.to_datetime(g["batch"].astype(str), format="%Y%m%d%H%M%S") + LAG
    g = g.sort_values("visible_utc").drop_duplicates("title", keep="first").reset_index(drop=True)
    print(f"{len(g):,} unique headlines; scoring...", flush=True)
    parts_dir = SCORED.parent / "gdelt_titles_scored_parts"
    parts_dir.mkdir(exist_ok=True)
    sc, step = None, 20_000
    for i in range(0, len(g), step):                 # saved in parts: resumable
        part = parts_dir / f"{i:07d}.parquet"
        if part.exists():
            continue
        sc = sc or Scorer(batch=128)
        sc.score(g["title"].iloc[i:i + step].tolist()).to_parquet(part, index=False)
        print(f"  {min(i + step, len(g)):,}/{len(g):,}", flush=True)
    s = pd.concat([pd.read_parquet(f) for f in sorted(parts_dir.glob("*.parquet"))], ignore_index=True)
    out = pd.concat([g[["visible_utc", "source", "title", "tone"]], s], axis=1)
    out["sent"] = out["p_pos"] - out["p_neg"]
    out.to_parquet(SCORED, index=False)
    print(f"scored -> {SCORED}")
    return out


def features_for(rows: pd.DataFrame, news: pd.DataFrame | None = None) -> pd.DataFrame:
    """rows: DataFrame with 'day' and 't' (US/Eastern naive). One feature row each."""
    n = news if news is not None else pd.read_parquet(SCORED)
    vis = (n["visible_utc"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
           .dt.tz_localize(None).to_numpy())
    o = np.argsort(vis)
    vis = vis[o]
    sent = n["sent"].to_numpy(float)[o]
    fg = (n["fear"] - n["greed"]).to_numpy(float)[o]
    tone = n["tone"].to_numpy(float)[o]
    cs = {k: np.r_[0, np.cumsum(v)] for k, v in (("sent", sent), ("fg", fg), ("tone", tone))}

    def window(a, b):
        i, j = np.searchsorted(vis, a, side="right"), np.searchsorted(vis, b, side="right")
        cnt = j - i
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = {k: np.where(cnt > 0, (c[j] - c[i]) / cnt, np.nan) for k, c in cs.items()}
        return cnt, mean
    t = rows["t"].to_numpy("datetime64[ns]")
    day = pd.to_datetime(rows["day"]).dt.normalize().to_numpy("datetime64[ns]")
    c15, m15 = window(t - np.timedelta64(15, "m"), t)
    c60, m60 = window(t - np.timedelta64(60, "m"), t)
    prev_close = day - np.timedelta64(1, "D") + np.timedelta64(16 * 60, "m")
    prev_close = np.where(pd.to_datetime(day).dayofweek == 0, prev_close - np.timedelta64(2, "D"), prev_close)
    cn, mn = window(prev_close, day + np.timedelta64(9 * 60 + 30, "m"))
    c5, m5 = window(day - np.timedelta64(5, "D"), day)
    return pd.DataFrame({"nw_n15": c15, "nw_n60": c60, "nw_sent15": m15["sent"],
                         "nw_sent60": m60["sent"], "nw_fg60": m60["fg"], "nw_tone60": m60["tone"],
                         "nw_sent_night": mn["sent"], "nw_n_night": cn,
                         "nw_sent_surprise": m60["sent"] - m5["sent"]}, index=rows.index)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "score":
        score(sys.argv[2])
    else:
        print(__doc__)
