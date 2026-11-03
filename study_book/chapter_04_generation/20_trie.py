"""20 — token trie.

Mirrors ``src/trie.rs`` and ``backends/common/gpu_types/trie.rs``.

A trie is used for two things in the engine:

1. structured output / grammar masks — allowed next tokens are a trie of token
   ids;
2. speculative decoding — the draft tree is uploaded as a flat trie so one
   target forward can verify every branch.

Here the trie is a flat list of nodes with index-based children.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass
class TrieNode:
    token_id: int
    depth: int = 0
    logprob: float = 0.0
    parent: int = -1
    children: list[int] = field(default_factory=list)


def new_root(token_id: int = 0) -> list[TrieNode]:
    return [TrieNode(token_id=token_id, depth=0, parent=-1)]


def add_node(nodes: list[TrieNode], parent: int, token_id: int, logprob: float = 0.0) -> int:
    index = len(nodes)
    nodes.append(
        TrieNode(
            token_id=token_id,
            depth=nodes[parent].depth + 1,
            logprob=logprob,
            parent=parent,
        )
    )
    nodes[parent].children.append(index)
    return index


def add_sequence(nodes: list[TrieNode], sequence: Sequence[int], logprob: float = 0.0) -> int:
    current = 0
    for token in sequence:
        found = next((c for c in nodes[current].children if nodes[c].token_id == token), None)
        if found is None:
            found = add_node(nodes, current, int(token), logprob)
        current = found
    return current


def path_to(nodes: list[TrieNode], index: int) -> list[int]:
    path: list[int] = []
    while index >= 0:
        path.append(index)
        index = nodes[index].parent
    return list(reversed(path))


def flatten(nodes: list[TrieNode]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (parents, depths, token_ids) arrays for the GPU trie."""
    parents = np.asarray([n.parent for n in nodes], dtype=np.int32)
    depths = np.asarray([n.depth for n in nodes], dtype=np.int32)
    tokens = np.asarray([n.token_id for n in nodes], dtype=np.int32)
    return parents, depths, tokens


def validate(nodes: list[TrieNode]) -> None:
    if not nodes:
        raise ValueError("empty trie")
    for i, node in enumerate(nodes):
        if i == 0:
            if node.parent != -1:
                raise ValueError("root must have parent -1")
            continue
        if not (0 <= node.parent < i):
            raise ValueError(f"node {i} has invalid parent {node.parent}")
        if nodes[node.parent].depth + 1 != node.depth:
            raise ValueError(f"node {i} has inconsistent depth")


def grammar_mask(allowed_token_ids: Iterable[int], vocab_size: int) -> np.ndarray:
    mask = np.zeros(vocab_size, dtype=bool)
    for token in allowed_token_ids:
        if 0 <= token < vocab_size:
            mask[int(token)] = True
    return mask


def apply_grammar_mask(logits: np.ndarray, allowed: Optional[np.ndarray]) -> np.ndarray:
    if allowed is None:
        return logits
    out = np.asarray(logits, dtype=np.float32).copy()
    out[..., ~allowed] = -np.inf
    return out
