"""12 — rotary position embedding (RoPE).

Mirrors ``encodable_block/mixer/attention/rope.rs``.

The Rust engine precomputes cos/sin for the token positions of the current batch
(``PrecalculatedRoPE``) so the attention kernels do not recompute them per layer.
With tree speculation the positions are the tree depths, not a contiguous range,
which is why ``precalculate`` takes an explicit position array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from study_book.chapter_01_foundations.kernels import rope_apply
from study_book.chapter_02_configuration.config import RoPEConfig


@dataclass
class PrecalculatedRoPE:
    cos: np.ndarray  # [T, dim/2]
    sin: np.ndarray  # [T, dim/2]

    @classmethod
    def precalculate(cls, config: RoPEConfig, positions: np.ndarray) -> "PrecalculatedRoPE":
        positions = np.asarray(positions, dtype=np.float32)
        dim = int(config.dim)
        if dim % 2:
            raise ValueError("RoPE dim must be even")
        inv_freq = 1.0 / (float(config.base) ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        angles = positions[:, None] * inv_freq[None, :]
        return cls(np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32))

    def apply(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.cos.shape[0]:
            raise ValueError(f"RoPE has {self.cos.shape[0]} positions but x has {x.shape[0]} rows")
        return rope_apply(x, self.cos, self.sin)
