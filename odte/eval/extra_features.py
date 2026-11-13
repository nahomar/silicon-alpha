"""Attach calendar, cross-asset and social features to expiry_edge decision rows.

Kept out of expiry_edge.py so the base study stays self-contained and its
self-test needs no network data. Each group is optional: if its source data
is missing the group is skipped and named in `skipped`, never silently
zero-filled.

Feature sets are cumulative: base -> +cal -> +cal+ctx -> +cal+ctx+soc, so
each step shows what that layer adds on top of the previous ones.

Groups (feature names are defined next to their sources):
  calendar  odte.data.calendar.CAL_FEATURES    scheduled events, events ahead of t
  context   odte.data.context.CTX_FEATURES     vol surface + cross-asset, prior close
            + iv_vs_vix1d                      intraday straddle vol / VIX1D at prior close
  social    social.features.SOCIAL_FEATURES    Reddit + Stocktwits activity z-scores
  nlp       social.features.NLP_FEATURES       transformer sentiment / fear-vs-greed of the text
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .expiry_edge import FEATURES


def attach(ds: pd.DataFrame, groups=("calendar", "context", "social", "nlp", "alt", "gex", "es")) -> tuple:
    ds = ds.sort_values(["day", "t"]).reset_index(drop=True)
    sets = {"base": list(FEATURES)}
    skipped = {}
    added = []

    if "calendar" in groups:
        try:
            from odte.data import calendar as cal
            ev = cal.load()
            f = pd.DataFrame([cal.features_for(ev, d, h) for d, h in zip(ds["day"], ds["hhmm"])],
                             index=ds.index)
            ds = pd.concat([ds, f[cal.CAL_FEATURES]], axis=1)
            added += cal.CAL_FEATURES
            sets["+cal"] = list(FEATURES) + list(cal.CAL_FEATURES)
        except FileNotFoundError as e:
            skipped["calendar"] = str(e)

    if "context" in groups:
        try:
            from odte.data import context as ctx
            wide = ctx.load()
            per_day = {d: ctx.features_for(wide, d) for d in ds["day"].unique()}
            f = pd.DataFrame([per_day[d] for d in ds["day"]], index=ds.index)
            # Today's intraday straddle, annualized, vs yesterday's VIX1D close:
            # is the market pricing today richer or cheaper than it said it would?
            ann = [(im / 0.8) / math.sqrt(max(m, 1.0) / (390 * 252)) * 100.0
                   for im, m in zip(ds["implied_move"], ds["minutes_to_close"])]
            f["iv_vs_vix1d"] = np.array(ann) / f["vix1d_prev"].replace(0, np.nan)
            cols = list(ctx.CTX_FEATURES) + ["iv_vs_vix1d"]
            ds = pd.concat([ds, f[cols]], axis=1)
            added += cols
            base = sets.get("+cal", list(FEATURES))
            sets["+cal+ctx" if "+cal" in sets else "+ctx"] = base + cols
        except FileNotFoundError as e:
            skipped["context"] = str(e)

    if "social" in groups:
        from social.features import SOCIAL_FEATURES, social_features
        days = sorted(ds["day"].unique())
        prev = {d: (days[i - 1] if i else pd.NaT) for i, d in enumerate(days)}
        keys = pd.DataFrame({"day": ds["day"], "hhmm": ds["hhmm"],
                             "prev_day": [prev[d] for d in ds["day"]]}, index=ds.index)
        f = social_features(keys)
        coverage = f.notna().mean()
        if coverage.max() == 0:
            skipped["social"] = "no Reddit aggregates or Stocktwits data on disk"
        else:
            ds = pd.concat([ds, f], axis=1)
            added += SOCIAL_FEATURES
            prev_name = max(sets, key=lambda k: len(sets[k]))
            base = sets[prev_name]
            sets[(prev_name if prev_name != "base" else "") + "+soc"] = base + list(SOCIAL_FEATURES)
            skipped["social_coverage"] = {k: round(float(v), 3) for k, v in coverage.items()}
    if "nlp" in groups:
        from social.features import NLP_FEATURES, nlp_features
        days = sorted(ds["day"].unique())
        prev = {d: (days[i - 1] if i else pd.NaT) for i, d in enumerate(days)}
        keys = pd.DataFrame({"day": ds["day"], "hhmm": ds["hhmm"],
                             "prev_day": [prev[d] for d in ds["day"]]}, index=ds.index)
        f = nlp_features(keys)
        coverage = f.notna().mean()
        if coverage.max() == 0:
            skipped["nlp"] = "no transformer scores on disk (run: python3 -m social.nlp score)"
        else:
            ds = pd.concat([ds, f], axis=1)
            prev_name = max(sets, key=lambda k: len(sets[k]))
            sets[(prev_name if prev_name != "base" else "") + "+nlp"] = sets[prev_name] + list(NLP_FEATURES)
            skipped["nlp_coverage"] = {k: round(float(v), 3) for k, v in coverage.items()}
    if "alt" in groups:
        from odte.data.altdata import ALT_LAYERS, alt_features
        f = alt_features(ds)
        ds = pd.concat([ds, f], axis=1)
        cov = {}
        for layer, cols in ALT_LAYERS.items():
            c = f[cols].notna().mean()
            cov[layer] = round(float(c.mean()), 3)
            if c.max() == 0:
                skipped[layer] = "no data on disk yet"
                continue
            prev_name = max(sets, key=lambda k: len(sets[k]))
            sets[(prev_name if prev_name != "base" else "") + layer] = sets[prev_name] + cols
        skipped["alt_coverage"] = cov
    if "gex" in groups:
        from odte.data.gamma import GEX_FEATURES, gamma_features
        f = gamma_features(ds)
        cov = float(f.notna().any(axis=1).mean())
        if cov == 0:
            skipped["gex"] = "no open-interest files (run: python3 -m odte.data.fetch_oi)"
        else:
            ds = pd.concat([ds, f], axis=1)
            base = sets.get("+cal+ctx", sets.get("+cal", list(FEATURES)))
            sets["+cal+ctx+gex" if "+cal+ctx" in sets else "+gex"] = base + list(GEX_FEATURES)
            skipped["gex_coverage"] = round(cov, 3)
    if "es" in groups:
        from odte.data.futures import ES_FEATURES, es_features
        per_day = es_features(sorted(ds["day"].unique()))
        f = per_day.reindex(pd.to_datetime(ds["day"]).dt.normalize()).set_axis(ds.index)
        cov = float(f.notna().any(axis=1).mean())
        if cov == 0:
            skipped["es"] = "no ES bars (run: python3 -m odte.data.futures fetch)"
        else:
            ds = pd.concat([ds, f[ES_FEATURES]], axis=1)
            base = sets.get("+cal+ctx", sets.get("+cal", list(FEATURES)))
            sets["+cal+ctx+es" if "+cal+ctx" in sets else "+es"] = base + list(ES_FEATURES)
            skipped["es_coverage"] = round(cov, 3)
    return ds, sets, skipped
