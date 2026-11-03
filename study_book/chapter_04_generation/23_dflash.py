"""23 — DFlash draft model.

Mirrors ``encodable_block/dflash.rs``.

DFlash is a small *block* draft model.  It consumes features from selected
target-model layers (``target_layer_ids``) plus a learned mask token, runs a
tiny causal transformer, and emits logits for several future positions in one
forward pass.  Those logits seed the Weaver tree.

The real implementation also keeps a draft KV state and projects target K/V into
it (``state_kv_projection``).  Here each proposal is a fresh local block, which
keeps the data flow easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from study_book.chapter_01_foundations.kernels import causal_mask
from study_book.chapter_02_configuration.config import (
    AttentionConfig,
    DenseMLPConfig,
    DFlashConfig,
    NormalizationConfig,
    RoPEConfig,
    TransformerConfig,
    TransformerLayerConfig,
)
from study_book.chapter_02_configuration.safetensors import ParameterTree
from study_book.chapter_03_model_blocks.linear import Linear, UntiedReadout, new_linear
from study_book.chapter_03_model_blocks.embedding import Embedding
from study_book.chapter_03_model_blocks.normalization import Normalization
from study_book.chapter_03_model_blocks.transformer import Transformer, TransformerState


@dataclass
class DFlashOutput:
    draft_hidden: np.ndarray   # [depth, model_dim]
    logits: np.ndarray         # [depth, vocab]


class DFlashState:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.context_length = 0

    def reset(self) -> None:
        self.context_length = 0


class DFlash:
    def __init__(self, config: DFlashConfig, params: ParameterTree) -> None:
        self.config = config
        self.model_dim = config.model_dim

        self.target_feature_projection: Linear = new_linear(
            config.target_feature_dim,
            config.model_dim,
            params.subtree("target_feature_projection"),
            bias=False,
        )
        self.projected_feature_norm = Normalization.new(
            config.model_dim,
            NormalizationConfig(),
            params.subtree("projected_feature_norm"),
        )
        # Present in the real model; kept as a study placeholder.
        self.state_kv_projection: Optional[Linear] = None

        rope = config.rope or RoPEConfig(dim=config.head_dim, max_sequence_length=config.block_size)
        layer_config = TransformerLayerConfig(
            mixer=AttentionConfig(
                num_heads=config.num_heads,
                num_groups=config.num_groups,
                head_dim=config.head_dim,
                rope=rope,
            ),
            mlp=DenseMLPConfig(),
            hidden_dim=config.hidden_dim,
        )
        transformer_config = TransformerConfig(
            model_dim=config.model_dim,
            hidden_dim=config.hidden_dim,
            layer_configs=[layer_config for _ in range(config.num_layers)],
        )
        self.transformer = Transformer(
            config.model_dim,
            transformer_config,
            params.subtree("transformer"),
        )
        self.output_norm = Normalization.new(
            config.model_dim,
            NormalizationConfig(),
            params.subtree("output_norm"),
        )
        self.readout = UntiedReadout.new(
            config.model_dim,
            config.vocab_size,
            params.subtree("readout"),
        )

    @property
    def block_size(self) -> int:
        return self.config.block_size

    def create_empty_state(self, context_capacity: int = 0) -> DFlashState:
        return DFlashState(context_capacity)

    def encode_draft(
        self,
        target_embedding: Embedding,
        target_output_token: int,
        depth: int,
        state: Optional[DFlashState],
        target_features: Sequence[np.ndarray],
    ) -> DFlashOutput:
        if depth <= 0 or depth > self.block_size:
            raise ValueError(f"depth {depth} outside 1..{self.block_size}")

        # Features are [T, D] per selected target layer; use the last position
        # (the token that was just produced).
        feature = np.concatenate([np.asarray(f, dtype=np.float32)[-1:] for f in target_features], axis=-1)
        projected = self.target_feature_projection.encode(feature)
        projected, _ = self.projected_feature_norm.apply(projected, None)
        mask_embedding = target_embedding.encode_lookup(np.asarray([self.config.mask_token_id]))[0]
        x = np.repeat((projected[0] + mask_embedding)[None, :], depth, axis=0).astype(np.float32)

        positions = np.arange(depth, dtype=np.float32)
        topology_state = self.transformer.create_empty_state(depth, None)
        mask = causal_mask(depth, depth, offset=0)
        out = self.transformer.encode(
            x,
            positions,
            topology_state,
            output_range=(0, depth),
            mask=mask,
        )
        assert out.output is not None
        hidden = out.output
        logits = self.readout.encode(hidden)
        if state is not None:
            state.context_length = depth
        return DFlashOutput(hidden, logits)

    def encode_accept(self, state: DFlashState, accepted_indices: Sequence[int]) -> None:
        # A real draft state would compact its KV cache here.  The local-block
        # study version has nothing to compact.
        state.context_length += len(accepted_indices)
