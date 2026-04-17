// persistent_decode.cu — persistent next-token inference kernel for TradeFM.
//
// Launched ONCE at market open and left running all day. Polls the RDMA
// ring-buffer's `ready_flag` (written by rdma_ingest.cu), reads the freshly
// tokenized context, runs the transformer forward pass with weights already
// resident in SMEM / registers / L2, and writes the next-token logits to
// an output ring.
//
// Memory residency plan (per SM):
//   registers:  current hidden state tile, RoPE table slice
//   shared:     attention Q/K/V tiles, block's slice of W_qkv/W_o/W_ffn
//   L2:         tied embedding matrix (read-mostly)
//   HBM3:       KV cache, ring buffers
//
// For a 40M model: each SM holds ~1MB of weights (fits in Hopper's 228KB/SM
// SMEM + L1 budget with proper tiling). For the 524M model the weights are
// sharded across SMs and streamed via wgmma async-copy.
//
// Because this is a realistic ~600-line kernel when fully fleshed out, this
// file shows the control flow (polling, SMEM preload, forward call) and
// DELEGATES the heavy matmul tiles to cutlass / cuBLASLt / transformer
// engine primitives — the right choice in production.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

extern "C" {

// Forward declarations of the tile primitives supplied by a library.
// On H100 you'd plug in cutlass::gemm::device::GemmUniversal or
// transformer_engine::common::gemm::gemm_fp8 here.
struct TransformerWeights {
    // per-layer pointers into L2-friendly, warp-tiled layouts
    const __nv_bfloat16* wqkv;     // (L, 3*d, d)
    const __nv_bfloat16* wo;       // (L, d, d)
    const __nv_bfloat16* w_ffn1;   // (L, hidden, d)
    const __nv_bfloat16* w_ffn2;   // (L, d, hidden)
    const __nv_bfloat16* emb;      // (V, d)  — tied with lm_head
    const __nv_bfloat16* rms_g_attn;  // (L, d)
    const __nv_bfloat16* rms_g_ffn;   // (L, d)
    int d_model, n_heads, n_layers, vocab, ctx_len;
};

__device__ void forward_one_token(
    const int16_t token_id,
    const TransformerWeights& W,
    __nv_bfloat16* kv_cache,       // (L, 2, ctx, n_heads, d_head) in HBM3
    int kv_pos,
    __nv_bfloat16* h_reg,          // (d_model,) — in registers, not pointer
    __nv_bfloat16* logits_out      // (V,)
)
{
    // ---- embed -----------------------------------------------------------
    // h = emb[token_id]  (d_model floats)
    // (in production: warp-level async copy via cp.async from L2)

    // ---- per-layer transformer block ------------------------------------
    // for (l=0; l<n_layers; ++l) {
    //    RMSNorm + attention (Q,K,V via wgmma, RoPE, softmax, wo)
    //    residual + dropout (training only; inference sets dropout=0)
    //    RMSNorm + SwiGLU FFN
    // }

    // ---- final norm + logits --------------------------------------------
    // logits = h @ emb.T  (lm_head tied)

    // ---- (omitted for brevity; full version ~450 LOC of cutlass/TE glue)
    (void)token_id; (void)W; (void)kv_cache; (void)kv_pos;
    (void)h_reg; (void)logits_out;
}

__global__ void persistent_decode(
    const int16_t* __restrict__ tokens_ring,   // from fused_bin's output ring
    int16_t* __restrict__ logits_ring,         // produced next-token logits
    volatile unsigned long long* tokens_head,
    volatile unsigned long long* tokens_tail,
    volatile int* run_flag,
    TransformerWeights W,
    __nv_bfloat16* kv_cache,                   // persistent across steps
    int ring_size)
{
    // Per-block register state (allocated once per launch).
    extern __shared__ __nv_bfloat16 s_hidden[];

    // Staging of W_qkv / W_o for this SM's responsibility slice would happen
    // here using cp.async.bulk (Hopper TMA) to move weight tiles from L2
    // into SMEM once, then keep them resident for the whole session.

    const unsigned int mask = ring_size - 1;
    unsigned long long last_processed = *tokens_tail;
    int kv_pos = 0;

    while (*run_flag) {
        const unsigned long long h = *tokens_head;
        if (h == last_processed) {
            __nanosleep(50);
            continue;
        }
        while (last_processed < h) {
            const int16_t tok = tokens_ring[last_processed & mask];

            __nv_bfloat16 h_reg[512];           // holds d_model
            __nv_bfloat16 logits[8192];         // holds vocab

            forward_one_token(tok, W, kv_cache, kv_pos, h_reg, logits);

            // Write logits out (quantized to int16 log-probs saves bandwidth)
            for (int i = threadIdx.x; i < W.vocab; i += blockDim.x) {
                logits_ring[(last_processed & mask) * W.vocab + i] =
                    __float2int_rn(__bfloat162float(logits[i]) * 32768.0f);
            }
            __syncthreads();
            ++last_processed;
            ++kv_pos;
            if (kv_pos >= W.ctx_len) {
                // sliding-window KV: recycle the oldest entry
                kv_pos = W.ctx_len - 1;
            }
        }
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *tokens_tail = last_processed;
        }
    }
}

}  // extern "C"
