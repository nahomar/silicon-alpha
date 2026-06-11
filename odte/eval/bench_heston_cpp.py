"""Benchmark the C++ Heston engine against the numpy reference.

Reports, over a set of 0DTE-ish specs:
  1. Price agreement -- |C++ - numpy| vs the combined MC standard error
     (they use independent RNG streams, so agreement is statistical, not
     bit-exact; "within ~3 combined sigma" is the pass condition).
  2. Wall-clock speedup at equal path budget.

The C++ engine additionally uses antithetic variates (the numpy reference does
not), so at equal paths it also returns a lower-variance estimate -- a second,
separate advantage on top of the raw speedup reported here.

Usage:
    PYTHONPATH=. python -m odte.eval.bench_heston_cpp
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from odte.synth_options import HestonParams
from odte.train.train_dml import heston_mc_call_price

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "cpp" / "heston_mc"


def _ensure_built() -> bool:
    if BIN.exists():
        return True
    try:
        subprocess.run(["make", "-C", str(ROOT / "cpp")], check=True,
                       capture_output=True)
        return BIN.exists()
    except Exception as e:  # pragma: no cover
        print(f"[bench] could not build C++ engine: {e}")
        return False


def cpp_price(S, K, T, sigma, n_paths, n_steps, h: HestonParams):
    out = subprocess.run(
        [str(BIN), "--price", f"{S}", f"{K}", f"{T}", f"{sigma}",
         f"{n_paths}", f"{n_steps}", f"{h.kappa}", f"{h.theta}",
         f"{h.xi}", f"{h.rho}", f"{0.0}"],
        capture_output=True, text=True, check=True)
    price, stderr = out.stdout.split()
    return float(price), float(stderr)


def main(n_paths: int = 200_000, n_steps: int = 48):
    if not _ensure_built():
        print("[bench] C++ engine unavailable; skipping.")
        return

    h = HestonParams()  # kappa=4, theta=0.04, xi=0.6, rho=-0.7
    specs = [
        (5500, 5500, 1.0 / 365, 0.20), (5500, 5600, 1.0 / 365, 0.20),
        (5500, 5450, 3.0 / 365, 0.35), (5500, 5500, 7.0 / 365, 0.15),
        (5500, 5400, 14.0 / 365, 0.25), (5500, 5650, 30.0 / 365, 0.30),
    ]
    S = np.array([s[0] for s in specs], float)
    K = np.array([s[1] for s in specs], float)
    T = np.array([s[2] for s in specs], float)
    SIG = np.array([s[3] for s in specs], float)
    R = np.zeros(len(specs))

    # --- numpy reference (batched) ---
    t0 = time.time()
    np_prices = heston_mc_call_price(S, K, T, SIG, R, h,
                                     n_paths=n_paths, n_steps=n_steps, seed=0)
    np_secs = time.time() - t0

    # --- C++ engine (one invocation per spec) ---
    cpp_prices, cpp_se = [], []
    t0 = time.time()
    for (s, k, t, sig) in specs:
        p, se = cpp_price(s, k, t, sig, n_paths, n_steps, h)
        cpp_prices.append(p); cpp_se.append(se)
    cpp_secs = time.time() - t0
    cpp_prices = np.array(cpp_prices); cpp_se = np.array(cpp_se)

    # numpy MC stderr (rough, single batch): estimate from a 2nd independent run
    np_b = heston_mc_call_price(S, K, T, SIG, R, h, n_paths=n_paths,
                                n_steps=n_steps, seed=104729)
    np_se = np.abs(np_prices - np_b) / np.sqrt(2.0)

    print(f"\n=== Heston MC: C++ vs numpy  ({n_paths:,} paths x {n_steps} steps, "
          f"{len(specs)} specs) ===")
    print(f"{'K/S':>6} {'tau(d)':>7} {'sig':>5} {'numpy':>10} {'C++':>10} "
          f"{'|diff|':>8} {'comb_se':>8} {'z':>6}")
    combined_se = np.sqrt(np_se ** 2 + cpp_se ** 2) + 1e-9
    for i, (s, k, t, sig) in enumerate(specs):
        z = abs(np_prices[i] - cpp_prices[i]) / combined_se[i]
        print(f"{k/s:>6.3f} {t*365:>7.1f} {sig:>5.2f} {np_prices[i]:>10.4f} "
              f"{cpp_prices[i]:>10.4f} {abs(np_prices[i]-cpp_prices[i]):>8.4f} "
              f"{combined_se[i]:>8.4f} {z:>6.2f}")

    max_z = float(np.max(np.abs(np_prices - cpp_prices) / combined_se))
    speedup = np_secs / cpp_secs if cpp_secs > 0 else float("nan")
    print(f"\nprice agreement : max |z| = {max_z:.2f}  "
          f"({'PASS' if max_z < 4 else 'CHECK'} — within MC noise if < ~4)")
    print(f"numpy time      : {np_secs*1000:.0f} ms")
    print(f"C++ time        : {cpp_secs*1000:.0f} ms  (incl. {len(specs)} process spawns)")
    print(f"speedup         : {speedup:.1f}x")
    print("note: C++ also uses antithetic variates -> lower variance at equal "
          "paths, a second advantage beyond this raw-time speedup.")


if __name__ == "__main__":
    main()
