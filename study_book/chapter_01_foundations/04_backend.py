"""04 — backend / context.

Mirrors ``backends/common/backend`` and ``backends/metal/context.rs``.

On Apple Silicon the context owns the Metal device, the residency set, the
command queue, the allocator and the kernel table.  Here the context owns a
NumPy buffer pool and a table of reference kernels.  ``supports_mxu`` is kept as
a flag so you can see where the real engine branches on M5+.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from study_book.chapter_01_foundations.dtypes import DataType
from study_book.chapter_01_foundations.buffer import Buffer, CommandBuffer, BufferPool
from study_book.chapter_01_foundations.kernels import (
    attention,
    causal_mask,
    depthwise_causal_conv,
    embedding_lookup,
    gather_rows,
    l2_normalize,
    linear,
    rms_norm,
    rope_apply,
    sample_from_logits,
    softmax,
    split,
)


class CPUKernels:
    """Kernel table; in Metal this is a set of compiled ``MTLComputePipelineState``."""

    linear = staticmethod(linear)
    rms_norm = staticmethod(rms_norm)
    attention = staticmethod(attention)
    causal_mask = staticmethod(causal_mask)
    depthwise_causal_conv = staticmethod(depthwise_causal_conv)
    embedding_lookup = staticmethod(embedding_lookup)
    gather_rows = staticmethod(gather_rows)
    l2_normalize = staticmethod(l2_normalize)
    rope_apply = staticmethod(rope_apply)
    sample = staticmethod(sample_from_logits)
    softmax = staticmethod(softmax)
    split = staticmethod(split)


@dataclass
class DeviceCapabilities:
    device_name: str = "python-cpu"
    gpu_core_count: int = 8
    apple_gpu_family: str = "Apple9"
    supports_mxu: bool = False


class CPUContext:
    def __init__(self, capabilities: Optional[DeviceCapabilities] = None) -> None:
        self.capabilities = capabilities or DeviceCapabilities()
        self.device_name = self.capabilities.device_name
        self.gpu_core_count = self.capabilities.gpu_core_count
        self.apple_gpu_family = self.capabilities.apple_gpu_family
        self.supports_mxu = self.capabilities.supports_mxu
        self.pool = BufferPool()
        self.kernels = CPUKernels()

    def create_command_buffer(
        self,
        name: Optional[str] = None,
        allocation_pool: Optional[BufferPool] = None,
    ) -> CommandBuffer:
        return CommandBuffer(self, name=name, pool=allocation_pool or self.pool)

    def copyout(self, buf: Buffer) -> np.ndarray:
        return buf.data.copy()

    def buffer(self, data: np.ndarray, name: str = "", dtype: DataType = DataType.F32) -> Buffer:
        return Buffer(np.asarray(data, dtype=dtype.numpy_dtype), name=name, dtype=dtype)

    def sync(self) -> None:
        """No-op: eager execution is already complete."""


class Backend:
    """Small facade matching the Rust generic ``Backend`` trait."""

    def __init__(self, context: Optional[CPUContext] = None) -> None:
        self.context = context or CPUContext()

    @property
    def kernels(self) -> CPUKernels:
        return self.context.kernels

    def command_buffer(self, name: str | None = None) -> CommandBuffer:
        return self.context.create_command_buffer(name)
