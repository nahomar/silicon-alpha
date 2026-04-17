#!/usr/bin/env bash
# deploy/nccl_env.sh — source this on every node of the 24x H100 cluster
# before launching torchrun. The high-bandwidth ens1 interface is forced
# so NCCL can't silently fall back to the 1Gb management NIC and blow up
# with connection timeouts during real-time inference.

# Interface pinning
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens1}"

# Debug — noisy but worth it until the first full-session trades clean
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH}"

# InfiniBand / RDMA tuning: use SL 3, disable P2P-level PCI checks on
# Hopper (WARNINGs about P2P are benign for FP8 traffic)
export NCCL_IB_SL=3
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL          # NVLink-only P2P to avoid PCI hops

# Prevent MPI init from hanging on rendezvous with no remote peer
export CUDA_MODULE_LOADING=LAZY
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# GPUDirect RDMA: enable kernel module discovery (nv_peer_memory)
export NCCL_NET_GDR_LEVEL=5
export NCCL_NET_GDR_READ=1

# Performance
export NCCL_BUFFSIZE=8388608        # 8 MiB buffer
export NCCL_LAUNCH_MODE=PARALLEL

# Sanity log
if [[ -t 1 ]]; then
    echo "[nccl_env] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    echo "[nccl_env] NCCL_IB_HCA=$NCCL_IB_HCA  NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL"
    echo "[nccl_env] NCCL_DEBUG=$NCCL_DEBUG  subsys=$NCCL_DEBUG_SUBSYS"
fi
