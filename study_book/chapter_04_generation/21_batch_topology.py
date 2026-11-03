"""21 — batch topology for tree/speculative batches.

Mirrors ``encodable_block/batch_topology.rs``.

A normal decode batch is a chain: token i attends to prefix + tokens 0..i.  A
speculative batch is a tree: node i attends to prefix + its ancestors, not to
its siblings.  ``BatchTopology`` turns the flat trie into

* ``positions`` = context_length + depth,
* ``ancestor_mask`` [num_nodes, context_length + num_nodes].

That mask is what the Python attention kernel uses; in Metal the equivalent is
``ancestor_attention.metal``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from study_book.chapter_04_generation.trie import TrieNode


@dataclass
class BatchTopology:
    parents: list[int]
    depths: list[int]
    token_ids: list[int]

    @classmethod
    def from_nodes(cls, nodes: Sequence[TrieNode]) -> "BatchTopology":
        return cls(
            parents=[n.parent for n in nodes],
            depths=[n.depth for n in nodes],
            token_ids=[n.token_id for n in nodes],
        )

    @classmethod
    def chain(cls, token_ids: Sequence[int]) -> "BatchTopology":
        parents = [-1] + [i - 1 for i in range(1, len(token_ids))]
        depths = list(range(len(token_ids)))
        return cls(parents, depths, [int(t) for t in token_ids])

    def size(self) -> int:
        return len(self.parents)

    def heights(self) -> np.ndarray:
        return np.asarray(self.depths, dtype=np.int32)

    def positions(self, context_length: int) -> np.ndarray:
        return (context_length + np.asarray(self.depths, dtype=np.float32)).astype(np.float32)

    def ancestor_mask(self, context_length: int) -> np.ndarray:
        n = self.size()
        mask = np.zeros((n, context_length + n), dtype=bool)
        # Every node may read the already-generated prefix.
        mask[:, :context_length] = True
        for i in range(n):
            j = i
            while j != -1:
                mask[i, context_length + j] = True
                j = self.parents[j]
        return mask

    def ancestor_indices(self, index: int) -> list[int]:
        out: list[int] = []
        while index != -1:
            out.append(index)
            index = self.parents[index]
        return list(reversed(out))

    def validate(self) -> None:
        n = self.size()
        if n == 0:
            raise ValueError("empty topology")
        for i, parent in enumerate(self.parents):
            if i == 0:
                if parent != -1:
                    raise ValueError("root parent must be -1")
            elif not (0 <= parent < i):
                raise ValueError(f"node {i}: parent {parent} is not earlier")
            elif self.depths[parent] + 1 != self.depths[i]:
                raise ValueError(f"node {i}: depth is inconsistent with parent")
