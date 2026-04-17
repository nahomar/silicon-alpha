// fused_bin.cu — persistent binning kernel for the 0DTE tokenizer.
//
// One resident block per feature. Each block loads its sorted edges array
// (<= 128 entries) into shared memory ONCE at startup, then loops: poll a
// ring-buffer head pointer in HBM3, bin the new samples with a per-thread
// binary search, write int16 token ids to the output ring, advance tail.
//
// Designed for:
//   - Hopper H100/GH200 (Hopper async copy + TMA not used here; the payload
//     is small enough that synchronous shared-mem loads win)
//   - Pair with odte/kernels/rdma_ingest.cu: consumer of its output ring.
//
// Build:
//   see odte/kernels/setup.py
//
// Performance notes:
//   - Vocab 64..128 → log2 ≈ 7 comparisons per bin → ~1ns per thread on H100.
//   - Binary search over a 129-float SMEM array fits in 1 L1 transaction.
//   - The persistent design eliminates the 10–20µs kernel-launch overhead
//     that a stateless launch would incur per tick.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

extern "C" {

// Persistent kernel. Launch once per trading day; spin until `run_flag` → 0.
//
//   values       : (N,)   float32 in HBM3 ring-buffer
//   edges        : (V+1,) float32  — ±inf sentinels at indices 0, V
//   tokens_out   : (N,)   int16    ring-buffer for downstream inference
//   head / tail  : producer/consumer indices (atomic)
//   run_flag     : set to 0 by host to stop the kernel at EOD
//   ring_size    : capacity of values / tokens_out rings (power of two)
__global__ void persistent_fused_bin(
    const float* __restrict__ values,
    const float* __restrict__ edges,
    int16_t* __restrict__ tokens_out,
    volatile unsigned long long* head,
    volatile unsigned long long* tail,
    volatile int* run_flag,
    int vocab,
    int ring_size)
{
    extern __shared__ float s_edges[];

    // One-time load: cooperative fetch of edges into SMEM (≤ 129 floats)
    for (int i = threadIdx.x; i < vocab + 1; i += blockDim.x) {
        s_edges[i] = edges[i];
    }
    __syncthreads();

    const unsigned int mask = ring_size - 1;      // ring_size must be pow2
    while (*run_flag) {
        const unsigned long long h = *head;
        const unsigned long long t = *tail;
        if (h == t) {
            // Empty — busy-wait. One __nanosleep avoids hammering L2.
            __nanosleep(100);
            continue;
        }
        // Each thread consumes one slot in the window (h - t).
        for (unsigned long long idx = t + threadIdx.x + blockIdx.x * blockDim.x;
             idx < h;
             idx += blockDim.x * gridDim.x) {
            const unsigned ring_idx = idx & mask;
            const float v = values[ring_idx];

            // Binary search over sorted interior edges (s_edges[1..vocab-1]).
            int lo = 1, hi = vocab;
            while (lo < hi) {
                int m = (lo + hi) >> 1;
                if (v < s_edges[m]) hi = m; else lo = m + 1;
            }
            tokens_out[ring_idx] = static_cast<int16_t>(lo - 1);
        }
        __syncthreads();

        // Block 0 thread 0 advances tail.
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *tail = h;
        }
    }
}

// Host-callable launch helper (not exported; call via pybind wrapper).
void launch_persistent_fused_bin(
    const float* values, const float* edges, int16_t* tokens_out,
    unsigned long long* head, unsigned long long* tail, int* run_flag,
    int vocab, int ring_size, cudaStream_t stream)
{
    const int block = 128;
    const int grid = 1;                          // persistent: 1 block/feature
    const size_t smem = (vocab + 1) * sizeof(float);
    persistent_fused_bin<<<grid, block, smem, stream>>>(
        values, edges, tokens_out, head, tail, run_flag, vocab, ring_size);
}

}  // extern "C"
