"""27 — decoder: embeddings -> transformer -> logits.

Mirrors ``encodable_block/decoder.rs``.

This is the top of the model graph:

    token ids
      -> embedding lookup (+ optional embedding norm, per-layer embedding)
      -> transformer stack
      -> output norm (inside Transformer.encode)
      -> readout / lm head
      -> logits

Everything above this file is generation logic; everything below is a block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from study_book.chapter_02_configuration.config import DecoderConfig
from study_book.chapter_02_configuration.safetensors import ParameterTree
from study_book.chapter_03_model_blocks.linear import Linear, UntiedReadout
from study_book.chapter_03_model_blocks.embedding import Embedding, PerLayerEmbedding
from study_book.chapter_03_model_blocks.normalization import Normalization
from study_book.chapter_03_model_blocks.transformer import Transformer, TransformerState


@dataclass
class DecoderEncodeOutput:
    logits: Optional[np.ndarray]
    hidden_features: Optional[list[np.ndarray]]
    final_hidden: Optional[np.ndarray]


class Decoder:
    def __init__(self, config: DecoderConfig, params: ParameterTree) -> None:
        self.config = config
        self.embedding = Embedding.new(config.vocab_size, config.transformer_config.model_dim, params.subtree("embedding"))
        self.embedding_norm = (
            Normalization.new(
                config.transformer_config.model_dim,
                config.embedding_norm_config,
                params.subtree("embedding_norm"),
            )
            if config.embedding_norm_config is not None
            else None
        )
        self.per_layer_embedding = (
            PerLayerEmbedding.new(
                config.transformer_config.model_dim,
                config.ple_model_config.num_layers,
                config.ple_model_config.table_size,
                params.subtree("per_layer_embedding"),
            )
            if config.ple_model_config is not None
            else None
        )
        self.transformer = Transformer(
            config.transformer_config.model_dim,
            config.transformer_config,
            params.subtree("transformer"),
        )
        if config.readout_config.tied:
            self.readout: Optional[Linear] = None  # use embedding table
        else:
            self.readout = UntiedReadout.new(
                config.transformer_config.model_dim,
                config.vocab_size,
                params.subtree("readout"),
            )

    # -- properties -------------------------------------------------------
    @property
    def speculation_supported(self) -> bool:
        return self.transformer.speculation_supported

    def max_context_length(self) -> Optional[int]:
        return self.transformer.max_context_length()

    def prefill_cache_skips_trailing_layers(self) -> bool:
        return self.transformer.prefill_cache_skips_trailing_layers()

    def create_empty_state(self, max_context_length: Optional[int], context) -> TransformerState:
        return self.transformer.create_empty_state(max_context_length, context)

    # -- encode -----------------------------------------------------------
    def encode(
        self,
        token_ids: Sequence[int] | np.ndarray,
        state: TransformerState,
        positions: Optional[np.ndarray] = None,
        output_range: Optional[tuple[int, int]] = None,
        hidden_feature_layer_indices: Optional[Sequence[int]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> DecoderEncodeOutput:
        ids = np.asarray(token_ids, dtype=np.int64)
        if positions is None:
            start = state.context_length
            positions = np.arange(start, start + len(ids), dtype=np.float32)

        embedded = self.embedding.encode_lookup(ids)
        if self.embedding_norm is not None:
            embedded, _ = self.embedding_norm.apply(embedded, None)

        if self.per_layer_embedding is not None:
            # The study Transformer currently ignores per-layer inputs; this
            # branch documents where they enter the real engine.
            _ = self.per_layer_embedding.encode(ids, embedded)

        transformer_out = self.transformer.encode(
            embedded,
            positions,
            state,
            output_range=output_range,
            hidden_feature_layer_indices=hidden_feature_layer_indices,
            mask=mask,
        )

        logits = None
        if transformer_out.output is not None:
            if self.readout is not None:
                logits = self.readout.encode(transformer_out.output)
            else:
                logits = transformer_out.output @ self.embedding.table.T

        return DecoderEncodeOutput(
            logits=logits,
            hidden_features=transformer_out.hidden_features,
            final_hidden=transformer_out.output,
        )
