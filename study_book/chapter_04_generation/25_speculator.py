"""25 — DFlash + Weaver speculator.

Mirrors ``speculators/dflash_tfm.rs``.

``DFlashTfmSpeculator`` glues the draft model to the tree builder:

    target features ──► DFlash ──► block logits ──► Weaver ──► proposal tree

The stream (file 28) then runs the target model on the whole tree once and
accepts the longest matching path.  The real engine also handles grammar masks,
sampling seeds, and target hidden-feature capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from study_book.chapter_02_configuration.config import SpeculatorConfig
from study_book.chapter_02_configuration.safetensors import ParameterTree
from study_book.chapter_03_model_blocks.embedding import Embedding
from study_book.chapter_04_generation.trie import TrieNode
from study_book.chapter_04_generation.sampling import PRng
from study_book.chapter_04_generation.dflash import DFlash, DFlashState
from study_book.chapter_04_generation.weaver import ProposalNode, Weaver, WeaverTreeShape


@dataclass
class ProposalTree:
    nodes: list[ProposalNode]

    @property
    def root(self) -> ProposalNode:
        return self.nodes[0]

    def tokens(self) -> list[int]:
        return [n.token_id for n in self.nodes]


class DFlashTfmSpeculator:
    def __init__(
        self,
        config: SpeculatorConfig,
        params: ParameterTree,
        target_embedding: Embedding,
    ) -> None:
        self.config = config
        self.draft = DFlash(config.draft_config, params.subtree("draft_model"))
        self.weaver = Weaver(config.weaver_config) if config.weaver_config is not None else None
        self.target_embedding = target_embedding

    @property
    def hidden_feature_layer_indices(self) -> list[int]:
        return list(self.config.draft_config.target_layer_ids)

    def empty_state(self, context_capacity: int = 0) -> DFlashState:
        return self.draft.create_empty_state(context_capacity)

    def make_shape(self, max_depth: Optional[int] = None) -> Optional[WeaverTreeShape]:
        depth = self.config.max_tree_depth
        if max_depth is not None:
            if max_depth < 2:
                return None
            depth = min(depth, max_depth)
        return WeaverTreeShape(
            tree_budget=self.config.tree_budget,
            max_depth=depth,
            dflash_depth=min(self.config.draft_config.block_size, depth),
        )

    def propose_tree(
        self,
        state: DFlashState,
        target_output_token: int,
        target_features: Sequence[np.ndarray],
        shape: WeaverTreeShape,
        prng: Optional[PRng] = None,
        grammar_allowed: Optional[np.ndarray] = None,
    ) -> ProposalTree:
        draft_out = self.draft.encode_draft(
            self.target_embedding,
            target_output_token,
            shape.dflash_depth,
            state,
            target_features,
        )
        logits = draft_out.logits
        if grammar_allowed is not None:
            logits = np.where(grammar_allowed[None, :], logits, -np.inf)

        if self.weaver is not None and self.config.use_weaver:
            nodes = self.weaver.encode_tree(
                draft_logits=logits,
                root_token=target_output_token,
                shape=shape,
            )
        else:
            nodes = [ProposalNode(token_id=int(target_output_token), depth=0, logprob=0.0, parent=-1)]
            for depth in range(1, min(shape.max_depth, logits.shape[0]) + 1):
                token = int(np.argmax(logits[depth - 1]))
                parent = len(nodes) - 1
                index = len(nodes)
                nodes.append(ProposalNode(token_id=token, depth=depth, logprob=0.0, parent=parent))
                nodes[parent].children.append(index)
        return ProposalTree(nodes)

    def encode_accept(
        self,
        state: DFlashState,
        accepted_indices: Sequence[int],
        target_features: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        self.draft.encode_accept(state, accepted_indices)
