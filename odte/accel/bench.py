"""Bench Numba-jitted hot paths vs numpy.

Produces a table like:
    fused_bin        numpy=  12.31 ms   numba=   0.21 ms   speedup=  58.6×
    microprice       numpy=   2.43 ms   numba=   0.01 ms   speedup= 243.0×
    ofi              numpy=  48.71 ms   numba=   0.31 ms   speedup= 157.1×

Run with:
    python -m odte.accel.bench --n 1000000
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


def _time(fn: Callable, *args, repeat: int = 5) -> float:
    best = float("inf")
    # warmup
    fn(*args)
    for _ in range(repeat):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


def bench(n: int = 1_000_000) -> dict:
    from odte.accel.numba_kernels import (
        fused_bin_numba, microprice_numba, ofi_numba, markout_numba, warmup,
    )
    rng = np.random.default_rng(0)
    edges = np.concatenate([[-np.inf], np.linspace(-3, 3, 63), [np.inf]])
    vals = rng.normal(size=n)
    bid_px = 100 + rng.normal(size=n) * 0.01
    ask_px = bid_px + np.abs(rng.normal(size=n) * 0.01) + 0.01
    bid_sz = rng.gamma(2, 5, size=n)
    ask_sz = rng.gamma(2, 5, size=n)
    fill_px = rng.uniform(99, 101, n)
    side = rng.choice([-1, 1], n).astype(np.int64)
    fwd_mid = fill_px + rng.normal(size=n) * 0.1

    # numpy references
    def np_fused_bin(v, e):
        return np.searchsorted(e[1:-1], v, side="right").astype(np.int16)

    def np_microprice(bp, ap, bs, as_):
        t = bs + as_
        t = np.where(t == 0, 1.0, t)
        return (ap * bs + bp * as_) / t

    def np_ofi(bp, bs, ap, as_):
        e_bid = np.zeros_like(bp); e_ask = np.zeros_like(bp)
        for i in range(1, len(bp)):
            if bp[i] > bp[i - 1]: e_bid[i] = bs[i]
            elif bp[i] == bp[i - 1]: e_bid[i] = bs[i] - bs[i - 1]
            else: e_bid[i] = -bs[i - 1]
            if ap[i] < ap[i - 1]: e_ask[i] = as_[i]
            elif ap[i] == ap[i - 1]: e_ask[i] = as_[i] - as_[i - 1]
            else: e_ask[i] = -as_[i - 1]
        return e_bid - e_ask

    def np_markout(fp, sd, fm):
        return np.sign(sd) * (fm - fp)

    warmup()
    out = {}
    for name, np_fn, nb_fn, args in [
        ("fused_bin",   np_fused_bin, fused_bin_numba, (vals, edges)),
        ("microprice",  np_microprice, microprice_numba, (bid_px, ask_px, bid_sz, ask_sz)),
        ("ofi",         np_ofi, ofi_numba, (bid_px, bid_sz, ask_px, ask_sz)),
        ("markout",     np_markout, markout_numba, (fill_px, side, fwd_mid)),
    ]:
        np_t = _time(np_fn, *args)
        nb_t = _time(nb_fn, *args)
        out[name] = {"numpy_s": np_t, "numba_s": nb_t,
                     "speedup": np_t / max(nb_t, 1e-12)}
    return out


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500_000)
    a = ap.parse_args()
    r = bench(a.n)
    print(f"{'kernel':<14} {'numpy':>10} {'numba':>10} {'speedup':>10}")
    print("-" * 48)
    for k, v in r.items():
        print(f"{k:<14} {v['numpy_s']*1000:>9.2f}ms {v['numba_s']*1000:>9.2f}ms "
              f"{v['speedup']:>9.1f}×")


if __name__ == "__main__":
    _cli()
