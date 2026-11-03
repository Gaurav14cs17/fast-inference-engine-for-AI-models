"""02 — buffers, allocation pool, command buffer.

Mirrors:

* ``backends/common/buffer`` — typed GPU buffers / scratch buffers
* ``backends/common/allocator`` — block/bump/pool allocators
* ``backends/common/command_buffer`` — encoded command buffer

The real engine records many kernels into one ``MTLCommandBuffer`` and submits
it once, which is a major source of speed.  For study this class executes
eagerly, but keeps the same API shape: ``allocate_scratch``, ``encode_copy``,
``debug_group``, ``end_encoding().submit()``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from study_book.chapter_01_foundations.dtypes import DataType


@dataclass
class Buffer:
    data: np.ndarray
    name: str = ""
    dtype: DataType = DataType.F32
    # In the real engine a Buffer is a handle to a Metal allocation; this flag
    # tells the allocator whether the storage may be recycled.
    scratch: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def size(self) -> int:
        return int(self.data.size)

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    def copy(self, name: str | None = None) -> "Buffer":
        return Buffer(self.data.copy(), name or self.name, self.dtype, self.scratch)

    def __repr__(self) -> str:
        return f"Buffer(shape={self.shape}, dtype={self.dtype.value}, name={self.name!r})"


ScratchBuffer = Buffer
GlobalGpuBuffer = Buffer
ConstantBuffer = Buffer


class BufferPool:
    """Recycle scratch allocations by (shape, dtype).

    This is the simplified counterpart of the block allocator.  Reusing buffers
    avoids per-token malloc/free and keeps peak memory flat.
    """

    def __init__(self) -> None:
        self._free: dict[tuple[tuple[int, ...], str], list[Buffer]] = defaultdict(list)
        self._live: set[int] = set()
        self.peak_bytes = 0
        self.live_bytes = 0

    def allocate(
        self,
        shape: Iterable[int],
        dtype: DataType = DataType.F32,
        name: str = "scratch",
        zero: bool = False,
    ) -> Buffer:
        shape = tuple(int(x) for x in shape)
        key = (shape, dtype.value)
        if self._free[key]:
            buf = self._free[key].pop()
            if zero:
                buf.data.fill(0)
            buf.name = name
            self._live.add(id(buf))
            return buf
        data = np.zeros(shape, dtype=dtype.numpy_dtype) if zero else np.empty(shape, dtype=dtype.numpy_dtype)
        buf = Buffer(data, name, dtype, scratch=True)
        self._live.add(id(buf))
        self.live_bytes += buf.nbytes
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)
        return buf

    def free(self, buf: Buffer) -> None:
        if id(buf) not in self._live:
            return
        self._live.discard(id(buf))
        self.live_bytes -= buf.nbytes
        self._free[(buf.shape, buf.dtype.value)].append(buf)


@dataclass
class EncodedOp:
    name: str
    detail: str = ""


class CommandBuffer:
    """Eager stand-in for an encoded Metal command buffer."""

    def __init__(self, context: Any, name: str | None = None, pool: BufferPool | None = None) -> None:
        self.context = context
        self.name = name or "command buffer"
        self.pool = pool or context.pool
        self.ops: list[EncodedOp] = []
        self.groups: list[str] = []
        self._submitted = False

    # -- allocation -------------------------------------------------------
    def allocate_scratch(self, shape: Iterable[int], dtype: DataType = DataType.F32) -> ScratchBuffer:
        return self.pool.allocate(shape, dtype)

    # -- encoding ---------------------------------------------------------
    def encode_copy(self, src: Buffer, dst: Buffer) -> None:
        np.copyto(dst.data, src.data)
        self.ops.append(EncodedOp("copy", f"{src.shape} -> {dst.shape}"))

    def encode_fill(self, dst: Buffer, value: float) -> None:
        dst.data.fill(value)
        self.ops.append(EncodedOp("fill", f"{value}"))

    def encode(self, fn, *args, **kwargs):
        """Handy for study: call a Python kernel and record it.

        Real code calls ``kernel.encode(arguments, command_buffer)``; the kernel
        appends a GPU dispatch instead of executing immediately.
        """
        self.ops.append(EncodedOp(getattr(fn, "__name__", str(fn))))
        return fn(*args, **kwargs)

    def debug_group(self, name: str) -> "CommandBuffer":
        self.groups.append(name)
        return self

    def debug_point(self, message: str) -> None:
        self.ops.append(EncodedOp("debug", message))

    # -- submit -----------------------------------------------------------
    def end_encoding(self) -> "CommandBuffer":
        self._submitted = True
        return self

    def submit(self) -> "CommandBuffer":
        self._submitted = True
        return self

    def __enter__(self) -> "CommandBuffer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.end_encoding().submit()

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for op in self.ops:
            counts[op.name] += 1
        return dict(counts)
