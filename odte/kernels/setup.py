"""Build the odte C++/CUDA extension — H100/GH200 hosts only.

Usage:
    cd odte/kernels
    python setup.py build_ext --inplace
    # or:
    pip install -e .

The resulting `odte_kernels_cu` .so is what fused_bin.py,
persistent_decode.py, and rdma_ingest.py try to import. On Mac it is never
built; the Python wrappers detect the missing module and fall back.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    raise SystemExit("pip install torch before building odte_kernels_cu")

ROOT = Path(__file__).resolve().parent

# On Hopper, target SM 90a for TMA / wgmma.async instructions.
NVCC_FLAGS = [
    "-O3", "--use_fast_math", "-std=c++17",
    "-gencode=arch=compute_90a,code=sm_90a",       # H100 / GH200
    "-gencode=arch=compute_80,code=sm_80",         # A100 fallback
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]

CXX_FLAGS = ["-O3", "-std=c++17"]

# Opt-in RDMA: set ODTE_BUILD_RDMA=1 in the environment. Requires
# rdma-core headers + nv_peer_memory kernel module.
srcs = [str(ROOT / "fused_bin.cu"), str(ROOT / "persistent_decode.cu"),
        str(ROOT / "bindings.cpp")]
libs: list[str] = []
if os.environ.get("ODTE_BUILD_RDMA") == "1":
    srcs.append(str(ROOT / "rdma_ingest.cu"))
    libs += ["ibverbs", "rdmacm"]

ext_modules = [
    CUDAExtension(
        name="odte_kernels_cu",
        sources=srcs,
        libraries=libs,
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
    )
]

setup(
    name="odte_kernels_cu",
    version="0.1.0",
    description="Hopper-native kernels for the 0DTE stack",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
