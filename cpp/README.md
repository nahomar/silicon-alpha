# cpp/ — native numerics

Compilable, benchmarked C++ for the hot paths the Python stack is slow at. (The
`.cu` files under `odte/kernels/` are GPU scaffolds; this is the CPU code that
actually builds and runs today.)

## `heston_mc` — Heston Monte-Carlo European-call pricer

The Heston MC reference is the bottleneck of the Phase-0 validation harness
([`odte/eval/validate_dml.py`](../odte/eval/validate_dml.py) spends most of its
wall-clock there). This is a drop-in-faster engine for it.

- **Exact-scheme parity** with the numpy reference
  ([`odte/synth_options.py`](../odte/synth_options.py)): same Euler
  full-truncation variance scheme and log-Euler price update, so the two agree
  within Monte-Carlo error (independent RNG streams → statistical, not
  bit-exact).
- **Antithetic variates** for variance reduction the numpy path lacks.
- **OpenMP** parallelism over paths (auto-detected Homebrew `libomp`; serial
  fallback otherwise).
- **xoshiro256++** PRNG + Box–Muller normals.
- **Self-validating**: the `xi → 0` limit collapses Heston to Black–Scholes at
  constant vol, and the self-test checks the MC price against that closed form.

### Build & run

```bash
cd cpp
make                 # auto-detects libomp; serial fallback otherwise
./heston_mc --selftest    # BS-limit correctness + variance-reduction check
./heston_mc --benchmark   # throughput
./heston_mc --price 5500 5500 0.0822 0.20 2000000 64   # -> "price stderr"
```

### Measured results (Apple M-series, 12 threads)

Self-test — MC vs analytic Black–Scholes in the `xi → 0` limit:

```
S=5500 K=5500 T=0.0822 sig=0.20  MC=125.70  BS=125.79  |z|=1.27  OK
S=5500 K=5600 T=0.0274 sig=0.20  MC= 33.85  BS= 33.89  |z|=1.11  OK
S=5500 K=5400 T=0.0137 sig=0.35  MC=147.79  BS=147.85  |z|=1.20  OK
S=5500 K=5500 T=0.1644 sig=0.10  MC= 88.89  BS= 88.96  |z|=1.27  OK
antithetic variance reduction @ 2,000,000 paths: 2.35x
```

Throughput: **~8.2 M paths/s** at 64 steps (4M paths, 12 threads).

Speed vs the numpy reference
([`odte/eval/bench_heston_cpp.py`](../odte/eval/bench_heston_cpp.py), 200k paths
× 48 steps, 6 specs):

| metric | value |
|---|---|
| price agreement (C++ vs numpy) | max \|z\| = 2.2 (within MC noise) |
| wall-clock | numpy 2895 ms → C++ 180 ms |
| **speedup** | **16×** (incl. process-spawn overhead) |
| extra | + 2.35× variance reduction from antithetic variates |

Reproduce: `PYTHONPATH=. python -m odte.eval.bench_heston_cpp`.
Gate test: `pytest tests/odte/test_heston_cpp.py`.
