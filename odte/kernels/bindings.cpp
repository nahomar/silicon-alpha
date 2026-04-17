// bindings.cpp — pybind11 surface for odte_kernels_cu
//
// Exports three entry points consumed by the Python wrappers:
//   fused_bin(values, edges) -> tokens            (fused_bin.py)
//   persistent_decode_step(tokens, staged_weights) -> logits
//   rdma_open / rdma_drain / rdma_close           (rdma_ingest.py)
//   stage_weights_to_smem(state_dict) -> handle   (persistent_decode.py)

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// Forward decls for device-side launch helpers (defined in the .cu files).
extern "C" void launch_persistent_fused_bin(
    const float*, const float*, int16_t*,
    unsigned long long*, unsigned long long*, int*,
    int, int, cudaStream_t);

// A realistic persistent_decode API needs a dedicated context object; this
// stub shows the pybind surface only.
struct StagedWeights {
    at::Tensor flattened;          // opaque view the kernel understands
    int d_model, n_heads, n_layers, vocab;
};

torch::Tensor fused_bin(torch::Tensor values, torch::Tensor edges) {
    TORCH_CHECK(values.is_cuda(), "values must be on CUDA");
    TORCH_CHECK(edges.is_cuda(),  "edges must be on CUDA");
    TORCH_CHECK(values.dtype() == at::kFloat,
                "values must be float32");

    auto tokens = torch::empty({values.size(0)},
        values.options().dtype(at::kShort));
    // In production: wire to a one-shot (non-persistent) launch for eager
    // mode use-cases. The persistent daemon is spawned separately.
    // Placeholder: throw to make missing implementation explicit.
    TORCH_CHECK(false, "eager fused_bin: not implemented in scaffold");
    return tokens;
}

StagedWeights stage_weights_to_smem(pybind11::dict state_dict) {
    // Reorder PyTorch state_dict tensors into the SM-tiled layout
    // persistent_decode expects, copy to device, return handle.
    // Placeholder.
    TORCH_CHECK(false, "stage_weights_to_smem: not implemented in scaffold");
    return {};
}

torch::Tensor persistent_decode_step(torch::Tensor tokens, StagedWeights) {
    TORCH_CHECK(false, "persistent_decode_step: not implemented in scaffold");
    return torch::empty({tokens.size(0), 0}, tokens.options());
}

#ifdef ODTE_BUILD_RDMA
// Defined in rdma_ingest.cu
extern "C" struct RDMAHandle* rdma_open(const char*, int, size_t, const char*);
extern "C" int rdma_drain(struct RDMAHandle*, int, uint64_t*, uint32_t*);
extern "C" void rdma_close(struct RDMAHandle*);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bin", &fused_bin, "Fused quantile/log binning (persistent)");

    pybind11::class_<StagedWeights>(m, "StagedWeights");
    m.def("stage_weights_to_smem", &stage_weights_to_smem,
          "Reorder TradeFM weights for the persistent_decode kernel");
    m.def("persistent_decode_step", &persistent_decode_step,
          "One next-token step using persistent kernel");

#ifdef ODTE_BUILD_RDMA
    m.def("rdma_open", [](const std::string& mcg, int port,
                           size_t ring_bytes, const std::string& iface) {
        return (uint64_t) rdma_open(mcg.c_str(), port, ring_bytes, iface.c_str());
    });
    m.def("rdma_drain", [](uint64_t handle, int max_events) {
        std::vector<uint64_t> offs(max_events);
        std::vector<uint32_t> lens(max_events);
        int n = rdma_drain((RDMAHandle*)handle, max_events, offs.data(), lens.data());
        offs.resize(n); lens.resize(n);
        return pybind11::make_tuple(offs, lens);
    });
    m.def("rdma_close", [](uint64_t handle) { rdma_close((RDMAHandle*)handle); });
#endif
}
