"""Edge #15 Q2 'calm seller', shadow only (docs/expiry_edge_plan.md).

At decision t the fly's mid credit is noted. At t + 5 min: if the fly's touch credit
reached that mid + $0.05 in any of the 5 minute snapshots (a burst: informed flow, the
adverse-selection signal of edge #14), stand aside; otherwise shadow-sell the same fly
at the t + 5 quotes. Backtest: +$23/day vs always-sell [-21, +67] (near-miss).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BURST = 0.05


def fly_quote(snap: pd.DataFrame, K: float, W: float) -> dict | None:
    q = snap.drop_duplicates(["option_type", "strike"], keep="last").set_index(["option_type", "strike"])
    try:
        cb, ca = float(q.at[("C", K), "bid"]), float(q.at[("C", K), "ask"])
        pb, pa = float(q.at[("P", K), "bid"]), float(q.at[("P", K), "ask"])
        wcb, wca = float(q.at[("C", K + W), "bid"]), float(q.at[("C", K + W), "ask"])
        wpb, wpa = float(q.at[("P", K - W), "bid"]), float(q.at[("P", K - W), "ask"])
    except KeyError:
        return None
    v = np.array([cb, ca, pb, pa, wcb, wca, wpb, wpa])
    if not np.isfinite(v).all():
        return None
    mid = (cb + ca + pb + pa) / 2 - (wcb + wca + wpb + wpa) / 2
    return {"K": K, "W": W, "fly_credit_mid": mid, "fly_credit": cb + pb - wca - wpa,
            "fly_debit": ca + pa - wcb - wpb}


def decide(pf: dict, chain: list) -> tuple:
    """(side, row at t+5 or None, reason) for one pending decision."""
    snaps = [s for s in chain if len(s) and pf["t0"] < pd.Timestamp(s["quote_datetime"].iloc[0]) <= pf["t5"]]
    if not snaps:
        return 0, None, "no quotes in the 5 minutes"
    burst = any((q := fly_quote(s, pf["K"], pf["W"])) is not None and q["fly_credit"] >= pf["mid0"] + BURST
                for s in snaps)
    last = fly_quote(snaps[-1], pf["K"], pf["W"])
    if last is None:
        return 0, None, "fly not quoted at t+5"
    if pf["recent"]:
        return 0, last, "within 60 min after the release"
    return (0, last, "burst: informed flow") if burst else (1, last, "calm")
