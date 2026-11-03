"""22 — sampling.

Mirrors ``encodable_block/sampling/*``.

The real engine runs top-k/top-p/min-p, the RNG and the final token choice on
the GPU, one threadgroup per batch row.  The important study point is that
sampling is part of the same command buffer as the forward pass — no logits are
copied to the CPU during normal generation.

``SamplingConfig`` lives in file 05; this file implements the algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from study_book.chapter_01_foundations.kernels import sample_from_logits
from study_book.chapter_02_configuration.config import SamplingConfig
from study_book.chapter_04_generation.trie import apply_grammar_mask
from study_book.chapter_04_generation.batch_topology import BatchTopology


class PRng:
    """Small deterministic splitmix64-style RNG (the real engine uses a GPU PRNG)."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int = 0) -> None:
        self.state = seed & self.MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return (z ^ (z >> 31)) & self.MASK

    def derive(self, salt: int) -> "PRng":
        return PRng(self.state ^ (salt * 0x9E3779B97F4A7C15 & self.MASK))


class Sampling:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def encode(
        self,
        logits: np.ndarray,
        topology: Optional[BatchTopology],
        indices: Sequence[int],
        config: SamplingConfig,
        allowed: Optional[np.ndarray] = None,
        prng: Optional[PRng] = None,
    ) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 1:
            logits = logits[None, :]
        prng = prng or PRng(config.seed)
        out = []
        for row in indices:
            row_logits = apply_grammar_mask(logits[row], allowed)
            rng = np.random.default_rng(prng.derive(int(row)).next_u64())
            out.append(
                sample_from_logits(
                    row_logits,
                    rng,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    min_p=config.min_p,
                    greedy=config.greedy,
                )
            )
        return np.asarray(out, dtype=np.int64)

    def sample_one(
        self,
        logits: np.ndarray,
        config: SamplingConfig,
        allowed: Optional[np.ndarray] = None,
        prng: Optional[PRng] = None,
    ) -> int:
        return int(self.encode(logits[None, :], None, [0], config, allowed, prng)[0])
