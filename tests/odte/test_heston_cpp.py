"""Build + correctness gate for the C++ Heston MC engine.

Compiles cpp/heston_mc.cpp and runs its self-test, which checks the Monte-Carlo
price against the closed-form Black-Scholes limit (xi -> 0, v0 = sigma^2) to
within 5 standard errors, and verifies that antithetic variates actually reduce
variance at an equal path budget.

Skipped automatically when no C++ toolchain (make / clang++) is available, so it
never blocks a pure-Python environment.

Run:
    PYTHONPATH=. pytest tests/odte/test_heston_cpp.py -xvs
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CPP_DIR = ROOT / "cpp"


@pytest.mark.skipif(shutil.which("make") is None or
                    (shutil.which("clang++") is None and shutil.which("g++") is None),
                    reason="no C++ toolchain (make / clang++ / g++)")
def test_cpp_heston_selftest_passes():
    # Build (clean to be deterministic about flags).
    subprocess.run(["make", "-C", str(CPP_DIR), "clean"], check=True,
                   capture_output=True)
    build = subprocess.run(["make", "-C", str(CPP_DIR)], capture_output=True, text=True)
    assert build.returncode == 0, f"build failed:\n{build.stderr}"

    # Run the self-test (BS-limit correctness + antithetic variance reduction).
    res = subprocess.run([str(CPP_DIR / "heston_mc"), "--selftest"],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, f"selftest failed:\n{res.stdout}\n{res.stderr}"
    assert "PASS" in res.stdout, res.stdout
    # Antithetic must have reduced variance (engine prints the measured factor).
    assert "variance reduction" in res.stdout
