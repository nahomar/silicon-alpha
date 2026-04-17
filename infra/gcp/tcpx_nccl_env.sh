#!/usr/bin/env bash
# GCP A3 H100 NCCL environment.
#
# CRUCIAL: GCP does NOT use InfiniBand. The deploy/nccl_env.sh file is
# tuned for Mellanox ConnectX-7 IB fabric and should NOT be sourced on
# GCP A3 instances. Source THIS file instead.
#
# A3 High  : 8× H100 80GB, 200 Gbps TCPX (8 NICs × 25 Gbps)
# A3 Mega  : 8× H100 80GB, 1600 Gbps GPUDirect-TCPX (8 NICs × 200 Gbps)
# A3 Ultra : 8× H200 141GB, RoCE v2 over 3.2 Tbps
#
# For A3 Mega (what you want for 24x H100 at ~$90-120/hr/node) use the
# google-provided container image `us-central1-docker.pkg.dev/deeplearning-
# platform-release/pytorch-cu124/pytorch-cu124-transformerengine-flashattn`
# which ships the GIB NCCL plugin preconfigured.

# --- TCPX / GIB plugin discovery --------------------------------------
# The plugin path is stable across GCP's container releases.
export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-libnccl-tuner.so}"
export NCCL_TUNER_CONFIG_PATH="${NCCL_TUNER_CONFIG_PATH:-/usr/local/gib/configs/tuner_config.txtpb}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-/usr/local/gib/lib64/libnccl-net.so}"

# --- TCPX-specific tuning --------------------------------------------
export NCCL_CROSS_NIC=0
export NCCL_ALGO=Ring,Tree
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=4
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_BUFFSIZE=8388608

# --- GPUDirect over TCPX (not IB) ------------------------------------
# The gve0 / gve1 pairs are GCP's virtual NICs — NOT the InfiniBand devs.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0,eth1,eth2,eth3,eth4,eth5,eth6,eth7}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
# GCP doesn't use NCCL_IB_HCA; leave it unset
unset NCCL_IB_HCA

# --- GPUDirect-TCPX topology ------------------------------------------
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_NET_GDR_LEVEL=PIX

# --- Debug while we're still bringing it up ---------------------------
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"         # switch to INFO if init hangs
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"

# --- PyTorch / CUDA ---------------------------------------------------
export CUDA_MODULE_LOADING=LAZY
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1024

if [[ -t 1 ]]; then
    echo "[tcpx_nccl_env] plugin=$NCCL_NET_PLUGIN"
    echo "[tcpx_nccl_env] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    echo "[tcpx_nccl_env] NCCL_DEBUG=$NCCL_DEBUG  subsys=$NCCL_DEBUG_SUBSYS"
fi
