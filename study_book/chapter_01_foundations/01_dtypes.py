"""01 — data types.

Mirrors ``crates/uzu-engine/src/data_type.rs``.

The Rust engine has BF16/F16/F32 and integer types because Metal kernels are
generated per type.  The Python port keeps everything in float32/int8 and only
models the *rounding* behaviour of BF16, which is enough to study the pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


class DataType(Enum):
    BF16 = "bf16"
    F16 = "f16"
    F32 = "f32"
    INT8 = "int8"
    U32 = "u32"

    @property
    def numpy_dtype(self) -> np.dtype:
        return {
            DataType.BF16: np.dtype(np.float32),
            DataType.F16: np.dtype(np.float16),
            DataType.F32: np.dtype(np.float32),
            DataType.INT8: np.dtype(np.int8),
            DataType.U32: np.dtype(np.uint32),
        }[self]

    @property
    def is_float(self) -> bool:
        return self in (DataType.BF16, DataType.F16, DataType.F32)


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even BF16, stored in a float32 container.

    BF16 is just the upper 16 bits of an IEEE-754 float32.  The real GPU stores
    two bytes; here we keep a float32 so NumPy arithmetic stays simple.
    """
    a = np.asarray(x, dtype=np.float32)
    bits = a.view(np.uint32).copy()
    lsb = (bits >> 16) & 1
    rounding = 0x7FFF + lsb
    bits = (bits + rounding) & 0xFFFF0000
    return bits.view(np.float32)


def cast(x: Any, dtype: DataType) -> np.ndarray:
    arr = np.asarray(x)
    if dtype == DataType.BF16:
        return to_bf16(arr)
    if dtype == DataType.INT8:
        return np.clip(np.rint(arr), -127, 127).astype(np.int8)
    if dtype == DataType.U32:
        return arr.astype(np.uint32)
    return arr.astype(dtype.numpy_dtype)


def symmetric_int8_quantize(x: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8 quantization plus scale (used by A8 kernels)."""
    x = np.asarray(x, dtype=np.float32)
    scale = np.max(np.abs(x), axis=axis, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    q = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
    return q, scale


def symmetric_int8_dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale.astype(np.float32)
