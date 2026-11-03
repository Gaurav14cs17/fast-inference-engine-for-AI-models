"""24 — Weaver tree construction.

Mirrors ``encodable_block/weaver.rs`` and
``backends/metal/kernel/weaver_selection.metal``.

Given DFlash's block logits, Weaver builds a *tree* of candidate continuations
instead of a single argmax chain.  It expands the frontier for a few rounds,
scores each edge with its log-probability, and prunes unlikely branches.  The
target model then verifies the whole tree in one forward pass.

The study version uses the draft logits directly.  The real model has learned
Weaver weights and a GPU frontier/pruning kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from study_book.chapter_01_foundations.kernels import log_softmax
from study_book.chapter_02_configuration.config import WeaverConfig
from study_book.chapter_04_generation.trie import TrieNode


@dataclass
class ProposalNode:
    token_id: int
    depth: int
    logprob: float = 0.0
    parent: int = -1
    children: list[int] = field(default_factory=list)


@dataclass
class WeaverTreeShape:
    tree_budget: int
    max_depth: int
    dflash_depth: int


class Weaver:
    def __init__(self, config: WeaverConfig) -> None:
        self.config = config

    @property
    def max_depth(self) -> int:
        return self.config.max_depth

    def encode_tree(
        self,
        draft_logits: np.ndarray,
        root_token: int,
        shape: WeaverTreeShape,
        target_output_norm: Optional[np.ndarray] = None,
        draft_hidden: Optional[np.ndarray] = None,
        target_embedding=None,
        depth_seeds: Optional[Sequence[int]] = None,
        prune_noise_scale: Optional[float] = None,
    ) -> list[ProposalNode]:
        """Return a flat tree; node 0 is the already-generated root token."""
        budget = max(2, int(shape.tree_budget))
        max_depth = max(1, min(int(shape.max_depth), int(draft_logits.shape[0])))
        nodes = [ProposalNode(token_id=int(root_token), depth=0, logprob=0.0, parent=-1)]
        frontier = [0]
        sigma = prune_noise_scale if prune_noise_scale is not None else self.config.prune_sigma

        for depth in range(1, max_depth + 1):
            remaining = budget - len(nodes)
            if remaining <= 0 or not frontier:
                break
            row = np.asarray(draft_logits[depth - 1], dtype=np.float32)
            logp = log_softmax(row)
            top = np.argsort(-logp)[: max(1, self.config.expand_width)]
            candidates: list[tuple[float, int, int, float]] = []
            for parent in frontier:
                for token in top:
                    score = nodes[parent].logprob + float(logp[token])
                    candidates.append((score, parent, int(token), float(logp[token])))
            if not candidates:
                break
            candidates.sort(key=lambda item: -item[0])

            if sigma is not None and len(candidates) > 2:
                scores = np.asarray([c[0] for c in candidates], dtype=np.float32)
                threshold = float(scores.mean() - sigma * scores.std())
                kept = [c for c in candidates if c[0] >= threshold]
                candidates = kept or candidates[:1]

            take = min(self.config.expand_per_round, remaining, len(candidates))
            new_frontier: list[int] = []
            for score, parent, token, _ in candidates[:take]:
                index = len(nodes)
                nodes.append(
                    ProposalNode(
                        token_id=token,
                        depth=depth,
                        logprob=score,
                        parent=parent,
                    )
                )
                nodes[parent].children.append(index)
                new_frontier.append(index)
            frontier = new_frontier

        return nodes

    @staticmethod
    def read_nodes(nodes: list[ProposalNode]) -> list[ProposalNode]:
        """Real engine copies the packed tree back from the GPU; already here."""
        return nodes

    @staticmethod
    def to_trie(nodes: list[ProposalNode]) -> list[TrieNode]:
        return [
            TrieNode(
                token_id=n.token_id,
                depth=n.depth,
                logprob=n.logprob,
                parent=n.parent,
                children=list(n.children),
            )
            for n in nodes
        ]
