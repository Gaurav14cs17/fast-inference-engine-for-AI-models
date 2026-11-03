"""06 — quantization and incoherence processing.

Mirrors the parts of:

* ``config/weight_matrix/{int_spec,mlx_spec,hybrid_spec,d4s4_spec}.rs``
* ``backends/common/gpu_types/trellis.rs``
* ``backends/metal/kernel/matmul/gemm/common/quantized/trellis.h``

The real engine supports 4/8-bit group quantization, int8 activations, RHT
(random Hadamard transform) for incoherence, and trellis-coded weights decoded
inside the GEMM prologue.  This file keeps the *ideas* in a few readable NumPy
functions.  It is not bit-compatible with the Metal kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from study_book.chapter_01_foundations.dtypes import DataType
from study_book.chapter_01_foundations.kernels import fast_hadamard


# ---------------------------------------------------------------------------
# group quantization (W4 / W8, group size 32/64)
# ---------------------------------------------------------------------------
@dataclass
class GroupQuant:
    q: np.ndarray          # int8 codes
    scales: np.ndarray     # one scale per group
    bits: int
    group_size: int
    shape: tuple[int, ...]

    def dequantize(self) -> np.ndarray:
        return dequantize_group(self.q, self.scales, self.bits, self.shape, self.group_size)

    @property
    def bits_per_weight(self) -> float:
        # codes + scale overhead (fp16 scale per group)
        return self.bits + 16.0 / self.group_size


def quantize_group(x: np.ndarray, group_size: int = 32, bits: int = 4) -> GroupQuant:
    x = np.asarray(x, dtype=np.float32)
    shape = x.shape
    last = shape[-1]
    pad = (-last) % group_size
    if pad:
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
    flat = x.reshape(-1, group_size)
    qmax = float(2 ** (bits - 1) - 1)
    scale = np.max(np.abs(flat), axis=1, keepdims=True) / qmax
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    q = np.clip(np.rint(flat / scale), -qmax, qmax).astype(np.int8)
    return GroupQuant(q, scale, bits, group_size, shape)


def dequantize_group(
    q: np.ndarray,
    scales: np.ndarray,
    bits: int,
    shape: tuple[int, ...],
    group_size: int,
) -> np.ndarray:
    flat = q.astype(np.float32) * scales.astype(np.float32)
    flat = flat.reshape(-1, group_size)
    out = flat.reshape(-1)[: int(np.prod(shape))]
    return out.reshape(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# RHT / incoherence processing
# ---------------------------------------------------------------------------
class RHT:
    """Randomized Hadamard transform used before low-bit quantization.

    QuIP/QTIP-style methods rotate weights and activations so that outliers are
    spread out; low-bit group quantization then has a much smaller error.  The
    Rust engine fuses the input transform into the linear layer
    (``encodable_block/linear/rht_wrapper.rs``).
    """

    def __init__(self, dim: int, seed: int = 0) -> None:
        if dim & (dim - 1):
            raise ValueError("RHT dimension must be a power of two")
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dim)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return fast_hadamard(x, self.signs)

    def apply_inverse(self, x: np.ndarray) -> np.ndarray:
        return fast_hadamard(x, self.signs)


# ---------------------------------------------------------------------------
# educational trellis codec
# ---------------------------------------------------------------------------
class TrellisCodec:
    """A tiny DPCM/trellis-style codec, *not* the real QTIP trellis.

    The real kernel stores 16-bit states and 2-bit transitions and walks the
    trellis while dequantizing in registers.  This version stores an initial
    level and 2-bit deltas, then reconstructs by cumulative sum.  It shows why
    trellis quantization can represent a smooth weight row with very few bits.
    """

    LEVELS = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)

    @classmethod
    def encode(cls, x: np.ndarray, step: Optional[float] = None) -> dict:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if step is None:
            step = float(np.mean(np.abs(x))) / 1.5 + 1e-6
        levels = cls.LEVELS * step
        state = np.float32(0.0)
        codes = np.empty(x.size, dtype=np.uint8)
        for i, value in enumerate(x):
            target = value - state
            code = int(np.argmin(np.abs(levels - target)))
            codes[i] = code
            state = state + levels[code]
        return {
            "codes": codes,
            "step": np.float32(step),
            "shape": x.shape,
            "state_bits": 16,
            "transition_bits": 2,
        }

    @classmethod
    def decode(cls, encoded: dict) -> np.ndarray:
        levels = cls.LEVELS * float(encoded["step"])
        codes = encoded["codes"]
        return np.cumsum(levels[codes]).astype(np.float32).reshape(encoded["shape"])


# ---------------------------------------------------------------------------
# weight formats exposed to the linear layer
# ---------------------------------------------------------------------------
@dataclass
class QuantizedWeight:
    format: str                     # "int4_g32" | "int8" | "trellis" | "d4s4"
    data: object
    shape: tuple[int, ...]

    def dequantize(self) -> np.ndarray:
        if self.format.startswith("int"):
            return self.data.dequantize()
        if self.format == "trellis":
            return TrellisCodec.decode(self.data).reshape(self.shape)
        if self.format == "d4s4":
            # D4S4 is a lookup-only embedding table; study version is plain fp32.
            return np.asarray(self.data, dtype=np.float32).reshape(self.shape)
        raise ValueError(f"unknown quant format {self.format}")


def quantize_weights(
    weights: np.ndarray,
    format: str = "int4_g32",
    group_size: int = 32,
) -> QuantizedWeight:
    weights = np.asarray(weights, dtype=np.float32)
    if format == "int4_g32":
        return QuantizedWeight(format, quantize_group(weights, group_size, bits=4), weights.shape)
    if format == "int8_g32":
        return QuantizedWeight(format, quantize_group(weights, group_size, bits=8), weights.shape)
    if format == "int8":
        return QuantizedWeight("int8", quantize_group(weights, weights.shape[-1], bits=8), weights.shape)
    if format == "trellis":
        return QuantizedWeight(format, TrellisCodec.encode(weights), weights.shape)
    if format == "d4s4":
        return QuantizedWeight(format, weights.astype(np.float32), weights.shape)
    raise ValueError(f"unknown quant format {format}")


def bits_for_format(format: str) -> DataType:
    return DataType.INT8 if format.startswith("int8") else DataType.BF16
