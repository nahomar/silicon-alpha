// rdma_ingest.cu — GPUDirect RDMA receiver for OPRA/Cboe multicast feeds.
//
// Binds a ConnectX-7 (or CX-6/CX-5) NIC to a CUDA stream, registers an
// HBM3 ring buffer via nvidia_p2p_dmabuf / nv_peer_memory, and configures
// the NIC to write incoming UDP packets directly into GPU memory.
//
// No CPU involvement after the initial setup:
//   - Packets arrive from the switch into the NIC
//   - NIC writes payload into HBM3 ring (GPUDirect RDMA)
//   - fused_bin.cu's persistent kernel consumes the ring on the same GPU
//
// Full implementation is ~400 LOC and uses IBVerbs + CUDA-aware MPI
// primitives. The scaffold below shows the init / drain / close lifecycle
// and leaves the NIC-specific setup to Mellanox's `rdma-core`.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

// NOTE: a real build pulls these from /usr/include/infiniband/* and
// Mellanox's nv_peer_memory kernel interface.
// #include <infiniband/verbs.h>
// #include <rdma/rdma_cma.h>

struct RDMAHandle {
    void* hbm_ring;             // device ptr in HBM3
    size_t ring_bytes;
    unsigned long long head;    // producer (written by NIC via RDMA)
    unsigned long long tail;    // consumer (read by fused_bin kernel)
    int run;                    // set to 0 to shut down

    // NIC state (opaque here)
    void* qp;
    void* cq;
    void* mr;
};

extern "C" {

/// Allocate HBM ring, register with NIC, subscribe to multicast group.
///   returns: opaque handle pointer for later calls
RDMAHandle* rdma_open(const char* multicast_group, int port,
                      size_t ring_bytes, const char* iface_name)
{
    auto* h = new RDMAHandle{};
    h->ring_bytes = ring_bytes;
    cudaMalloc(&h->hbm_ring, ring_bytes);
    cudaMemset(h->hbm_ring, 0, ring_bytes);
    h->head = 0; h->tail = 0; h->run = 1;

    // TODO(phase-3 build): open ibv context by iface_name, create PD/CQ/QP,
    // ibv_reg_mr(... h->hbm_ring ...) using nv_peer_memory, subscribe to
    // multicast via rdma_join_multicast, post N receive WRs.
    // ibv_post_recv (ring_bytes / mtu) buffers.

    (void)multicast_group; (void)port; (void)iface_name;
    return h;
}

/// Drain up to `max_events` completed receives. Each event is the offset
/// into the HBM ring where the payload landed; the Python wrapper decodes
/// the PITCH/OPRA message from the bytes there.
int rdma_drain(RDMAHandle* h, int max_events, uint64_t* offsets_out,
               uint32_t* lengths_out)
{
    if (!h || !h->run) return 0;
    // TODO: ibv_poll_cq → update h->head atomically; offsets_out[i] = ring index
    (void)h; (void)max_events; (void)offsets_out; (void)lengths_out;
    return 0;
}

/// Close subscription, deregister MR, free ring.
void rdma_close(RDMAHandle* h)
{
    if (!h) return;
    h->run = 0;
    // TODO: ibv_destroy_qp, ibv_dereg_mr, ibv_destroy_cq, ibv_close_device
    if (h->hbm_ring) cudaFree(h->hbm_ring);
    delete h;
}

}  // extern "C"
