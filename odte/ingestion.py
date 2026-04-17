"""OPRAIngest — pluggable async feed for 0DTE options.

Provides the same async-iterator-of-dicts contract as feeds/coinbase_feed.py
with additional option-chain fields. Two implementations:

  PythonOPRAIngest : asyncio queue, CPU. Ingests from any upstream that puts
                     dicts into the queue (e.g. feeds/databento_opra.py).
  RDMAStubIngest   : Phase-3 placeholder. Raises NotImplementedError unless
                     HAS_RDMA is True AND odte.kernels.rdma_ingest is built.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from . import HAS_RDMA

log = logging.getLogger(__name__)


# Unified event schema for options:
#   kind: "tob" | "trade" | "chain"
#   ts:   int ms
#   underlying: "SPX"
#   strike: float
#   expiry: str (YYYY-MM-DD or "0DTE")
#   cp_flag: "C" | "P"
#   (plus TOB / trade / chain-specific fields)


class OPRAIngest(ABC):
    """Abstract base. Implementations must be async-iterable."""

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[dict]: ...


@dataclass
class PythonOPRAIngest(OPRAIngest):
    """Asyncio-queue-backed ingest. Upstream puts dicts in `queue`."""

    queue: asyncio.Queue = None  # type: ignore

    def __post_init__(self):
        if self.queue is None:
            self.queue = asyncio.Queue(maxsize=100_000)

    async def put(self, event: dict) -> None:
        await self.queue.put(event)

    async def __aiter__(self) -> AsyncIterator[dict]:  # type: ignore[override]
        while True:
            item = await self.queue.get()
            if item is None:
                return
            yield item


class RDMAStubIngest(OPRAIngest):
    """Phase 3: GPUDirect RDMA from Mellanox ConnectX-7 → H100 HBM3 ring buffer.

    On any non-H100 host this raises immediately so the dev is never confused
    about what's actually running.
    """

    def __init__(self, ring_size: int = 1 << 20):
        self.ring_size = ring_size
        if not HAS_RDMA:
            raise NotImplementedError(
                "RDMAStubIngest requires ODTE_HAS_RDMA=1 + odte.kernels.rdma_ingest. "
                "On the Mac / any non-Hopper host, use PythonOPRAIngest instead."
            )

    async def __aiter__(self) -> AsyncIterator[dict]:  # pragma: no cover
        # Phase-3 implementation will import the CUDA extension and pump events
        # out as dicts with the same schema.
        from odte.kernels import rdma_ingest as _rdma  # type: ignore
        async for ev in _rdma.iter_events(self.ring_size):
            yield ev
