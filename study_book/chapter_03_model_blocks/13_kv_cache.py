"""13 — KV cache and attention state.

Mirrors ``encodable_block/mixer/attention/state.rs``.

The cache is preallocated (capacity = max context) and appended in place.  After
speculative verification only the accepted rows survive: ``accept(indices)``
compacts the last appended chunk, dropping rejected tree branches.  This is the
Python equivalent of the ``kv_cache_update`` / accept kernels.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from study_book.chapter_01_foundations.dtypes import DataType


class KVCache:
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        capacity: int,
        dtype: DataType = DataType.F32,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.k = np.zeros((capacity, num_heads, head_dim), dtype=dtype.numpy_dtype)
        self.v = np.zeros((capacity, num_heads, head_dim), dtype=dtype.numpy_dtype)
        self.length = 0
        self._last_start = 0

    def reset(self) -> None:
        self.length = 0
        self._last_start = 0

    def append(self, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = k.shape[0]
        if self.length + n > self.capacity:
            raise ValueError(f"KV cache overflow: {self.length}+{n} > {self.capacity}")
        self._last_start = self.length
        self.k[self.length : self.length + n] = k
        self.v[self.length : self.length + n] = v
        self.length += n
        return self.view()

    def view(self) -> tuple[np.ndarray, np.ndarray]:
        return self.k[: self.length], self.v[: self.length]

    def accept(self, indices: Sequence[int]) -> None:
        """Keep only ``indices`` from the last appended chunk."""
        indices = np.asarray(list(indices), dtype=np.int64)
        start = self._last_start
        keep = start + indices
        n = len(keep)
        # Move accepted rows to the end of the prefix.  Use copies because the
        # source may overlap the destination.
        self.k[start : start + n] = self.k[keep].copy()
        self.v[start : start + n] = self.v[keep].copy()
        self.length = start + n
        self._last_start = self.length

    def __len__(self) -> int:
        return self.length


class AttentionState:
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        capacity: int,
        dtype: DataType = DataType.F32,
        shared: Optional["AttentionState"] = None,
        read_only: bool = False,
    ) -> None:
        self.shared = shared
        self.read_only = read_only
        self.cache = None if shared is not None else KVCache(num_heads, head_dim, capacity, dtype)

    @property
    def context_length(self) -> int:
        return self.shared.context_length if self.shared is not None else self.cache.length

    def prepare(self, context_length: int, suffix_length: int) -> None:
        # Preallocation makes prepare a no-op in this port.  The real engine
        # uses it to size ring buffers / reset per-layer state.
        pass

    def append(self, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = self.shared if self.shared is not None else self
        assert target.cache is not None
        if self.read_only:
            return target.cache.view()
        return target.cache.append(k, v)

    def view(self) -> tuple[np.ndarray, np.ndarray]:
        target = self.shared if self.shared is not None else self
        assert target.cache is not None
        return target.cache.view()

    def accept(self, indices: Sequence[int]) -> None:
        target = self.shared if self.shared is not None else self
        assert target.cache is not None
        if self.read_only:
            return
        target.cache.accept(indices)

    def reset(self) -> None:
        target = self.shared if self.shared is not None else self
        assert target.cache is not None
        target.cache.reset()
