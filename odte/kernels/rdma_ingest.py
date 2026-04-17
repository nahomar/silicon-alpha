"""GPUDirect RDMA ingest — Python wrapper.

On H100 + Mellanox ConnectX-7 (with nv_peer_memory / dmabuf kernel module):
delegates to the C++ extension which registers an HBM3 ring buffer with the
NIC, configures a UDP multicast subscription on the OPRA/Cboe feed, and
pumps incoming packets directly into GPU memory via RDMA writes. Zero CPU
copies; ≤ 1µs NIC→HBM3 latency.

On Mac / generic host: the class falls back to a userspace asyncio UDP
socket so integration tests work end-to-end. Producers (real NIC OR a
replay client) push event dicts into the same ring buffer.

The HAS_RDMA env var must be set to '1' for the class to attempt the real
path. That gate exists so a misconfigured Linux box doesn't accidentally
claim to have RDMA and then crash when it tries to talk to the NIC.
"""
from __future__ import annotations

import asyncio
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterable, Optional

log = logging.getLogger(__name__)

HAS_CUDA_RDMA = False
_cuda_ext = None
if os.environ.get("ODTE_HAS_RDMA") == "1":
    try:
        import odte_kernels_cu  # type: ignore
        _cuda_ext = odte_kernels_cu
        HAS_CUDA_RDMA = hasattr(_cuda_ext, "rdma_open")
    except Exception as e:  # pragma: no cover
        log.warning("odte_kernels_cu unavailable (%s)", e)


@dataclass
class ConnectX7Receiver:
    """GPUDirect RDMA receiver. Supports real NIC + UDP fallback."""

    multicast_group: str = "224.0.62.2"   # example OPRA PITCH line
    port: int = 30001
    hbm_ring_bytes: int = 1 << 24          # 16 MiB
    iface: Optional[str] = None            # e.g. "mlx5_0"
    _handle: object = field(default=None, init=False)
    _queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=100_000),
                                  init=False)

    def __post_init__(self):
        if HAS_CUDA_RDMA and self.iface:
            self._handle = _cuda_ext.rdma_open(
                self.multicast_group, self.port,
                self.hbm_ring_bytes, self.iface)
            log.info("rdma_ingest: opened %s @ %s:%d (hbm_ring=%d B)",
                     self.iface, self.multicast_group, self.port,
                     self.hbm_ring_bytes)
        else:
            log.info("rdma_ingest: CPU/UDP fallback on %s:%d",
                     self.multicast_group, self.port)

    # ------------------------------------------------------------------
    # async iteration
    # ------------------------------------------------------------------
    async def __aiter__(self) -> AsyncIterator[dict]:
        if HAS_CUDA_RDMA and self._handle is not None:
            async for ev in self._iter_rdma():
                yield ev
        else:
            async for ev in self._iter_udp():
                yield ev

    async def _iter_rdma(self) -> AsyncIterator[dict]:  # pragma: no cover
        """Spin on the HBM ring buffer. The kernel decodes the packet header
        into the event dict before we see it (or we decode in Python if the
        extension returns a raw view)."""
        while True:
            evs = await asyncio.to_thread(_cuda_ext.rdma_drain, self._handle, 1024)
            for e in evs:
                yield e

    async def _iter_udp(self) -> AsyncIterator[dict]:
        """Plain UDP multicast listener so Mac dev is unbroken."""
        loop = asyncio.get_event_loop()
        sock = await self._open_udp_socket()
        while True:
            data = await loop.sock_recv(sock, 65535)
            if not data:
                continue
            yield self._decode_udp_packet(data)

    async def _open_udp_socket(self):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                             socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.port))
        # multicast group join
        mreq = struct.pack("4sl", socket.inet_aton(self.multicast_group),
                           socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.setblocking(False)
        return sock

    @staticmethod
    def _decode_udp_packet(data: bytes) -> dict:
        """Minimal placeholder decoder. Real PITCH/OPRA codecs live in
        a separate module; here we just echo a raw event for integration.
        """
        return {"kind": "raw", "ts": int(time.time() * 1000),
                "bytes": data[:64]}

    def close(self) -> None:
        if HAS_CUDA_RDMA and self._handle is not None:
            _cuda_ext.rdma_close(self._handle)
            self._handle = None
