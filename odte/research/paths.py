"""Minute paths of each session and path-shape features (development days only).

Per (index, day): a (390, 3) float32 array, minutes 09:31 .. 16:00:
  0 log forward (put-call parity), 1 ATM straddle / forward, 2 ATM straddle relative half-spread.

Path-shape features at each decision time t (only the path up to t):
  pth_ac1        autocorrelation of 1-minute returns since 09:31 (trend > 0 > reversion)
  pth_vr5        variance ratio of 5-minute vs 1-minute returns (trend > 1 > reversion)
  pth_volsurp    (straddle_t / straddle_open) / sqrt(minutes_left_t / minutes_left_open):
                 > 1 = implied vol rose during the day, < 1 = it fell
  pth_volmom15   straddle change over the last 15 minutes (relative)
  pth_since_hi, pth_since_lo   minutes since the day's high / low
  pth_dd_hi      drop from the day's high, in units of the current implied move

  python3 -m odte.research.paths build SPX     # or NDX / RUT
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import exam_guard as eg
from odte.research.library import DIRS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "research"
N_MIN = 390
PATH_FEATURES = ["pth_ac1", "pth_vr5", "pth_volsurp", "pth_volmom15", "pth_since_hi", "pth_since_lo", "pth_dd_hi"]


def day_path(chain: pd.DataFrame, day: pd.Timestamp) -> np.ndarray:
    from odte.eval.expiry_edge import implied_forward, minute_grid
    g = minute_grid(chain)
    F = implied_forward(g)
    idx = pd.date_range(day + pd.Timedelta(hours=9, minutes=31), periods=N_MIN, freq="min")
    out = np.full((N_MIN, 3), np.nan, np.float32)
    f = F.reindex(idx)
    out[:, 0] = np.log(f.to_numpy(float))
    b, a = g["bid"].reindex(idx), g["ask"].reindex(idx)
    ok = (b > 0) & (a >= b)
    mid = ((b + a) / 2).where(ok)
    half = ((a - b) / 2).where(ok)
    strikes = np.array(sorted({k for (r, k) in mid.columns if r == "C"} & {k for (r, k) in mid.columns if r == "P"}), float)
    if not len(strikes):
        return out
    C = mid.reindex(columns=pd.MultiIndex.from_product([["C"], strikes])).to_numpy(float)
    P = mid.reindex(columns=pd.MultiIndex.from_product([["P"], strikes])).to_numpy(float)
    HC = half.reindex(columns=pd.MultiIndex.from_product([["C"], strikes])).to_numpy(float)
    HP = half.reindex(columns=pd.MultiIndex.from_product([["P"], strikes])).to_numpy(float)
    fv = f.to_numpy(float)
    good = np.isfinite(fv)
    k = np.zeros(N_MIN, int)
    k[good] = np.abs(strikes[None, :] - fv[good, None]).argmin(axis=1)
    rows = np.arange(N_MIN)
    st = C[rows, k] + P[rows, k]
    hs = HC[rows, k] + HP[rows, k]
    out[:, 1] = np.where(good, st / fv, np.nan)
    out[:, 2] = np.where(good & (st > 0), hs / st, np.nan)
    return out


def path_features(path: np.ndarray, minute: int, implied_move: float) -> dict:
    """Features from minutes [0, minute] of one day's path (minute = index of the decision minute)."""
    lf = path[: minute + 1, 0].astype(float)
    st = path[: minute + 1, 1].astype(float)
    ok = np.isfinite(lf)
    out = {k: np.nan for k in PATH_FEATURES}
    if ok.sum() < 20:
        return out
    x = pd.Series(lf).ffill().to_numpy()
    r1 = np.diff(x)
    r1 = r1[np.isfinite(r1)]
    if len(r1) > 10 and r1.std() > 0:
        out["pth_ac1"] = float(np.corrcoef(r1[:-1], r1[1:])[0, 1])
        r5 = x[5::5] - x[:-5:5] if len(x) > 10 else np.array([])
        r5 = r5[np.isfinite(r5)]
        if len(r5) > 3:
            out["pth_vr5"] = float(r5.var() / (5 * r1.var()))
    left_open, left_now = N_MIN, N_MIN - minute
    s = pd.Series(st).ffill().to_numpy()
    if np.isfinite(s[0]) and np.isfinite(s[-1]) and s[0] > 0:
        out["pth_volsurp"] = float((s[-1] / s[0]) / np.sqrt(max(left_now, 1) / left_open))
    if len(s) > 15 and np.isfinite(s[-16]) and s[-16] > 0:
        out["pth_volmom15"] = float(s[-1] / s[-16] - 1)
    hi, lo = int(np.nanargmax(x)), int(np.nanargmin(x))
    out["pth_since_hi"], out["pth_since_lo"] = float(minute - hi), float(minute - lo)
    if np.isfinite(implied_move) and implied_move > 0:
        out["pth_dd_hi"] = float((np.nanmax(x) - x[-1]) / implied_move)
    return out


def build(sym: str) -> None:
    folder = DIRS[sym]
    files = [p for p in sorted(folder.glob("20??-??-??.parquet")) if pd.Timestamp(p.stem) < eg.EXAM_FROM]
    eg.check([p.stem for p in files])
    days, arrs = [], []
    for p in files:
        day = pd.Timestamp(p.stem)
        arrs.append(day_path(pd.read_parquet(p), day))
        days.append(day)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"paths_{sym}.npz", paths=np.stack(arrs), days=np.array([d.strftime("%Y-%m-%d") for d in days]))
    print(f"{sym}: {len(days)} days -> paths_{sym}.npz")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "build":
        build(sys.argv[2])
