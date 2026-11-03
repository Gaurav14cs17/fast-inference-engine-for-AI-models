"""19 — transformer stack and state.

Mirrors ``encodable_block/transformer.rs``.

``TransformerState`` owns one mixer state per layer (or a shared read-only view
of an earlier KV cache).  ``encode`` runs all layers, precomputes RoPE once for
the batch positions, and can capture residual features from chosen layers — the
latter is exactly what the DFlash speculator consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from study_book.chapter_02_configuration.config import TransformerConfig
from study_book.chapter_02_configuration.safetensors import ParameterTree
from study_book.chapter_03_model_blocks.normalization import Normalization, ShortcutMode
from study_book.chapter_03_model_blocks.rope import PrecalculatedRoPE
from study_book.chapter_03_model_blocks.kv_cache import AttentionState
from study_book.chapter_03_model_blocks.mixer import MixerState
from study_book.chapter_03_model_blocks.transformer_layer import TransformerLayer


@dataclass
class TransformerEncodeOutput:
    output: Optional[np.ndarray]
    hidden_features: Optional[list[np.ndarray]]


class TransformerState:
    def __init__(self, layer_states: list[MixerState], context_length: int = 0) -> None:
        self.layer_states = layer_states
        self.context_length = context_length

    def prepare(self, context_length: int, suffix_length: int) -> None:
        for state in self.layer_states:
            state.prepare(context_length, suffix_length)

    def encode_accept(self, accepted_indices: Sequence[int]) -> None:
        for state in self.layer_states:
            state.accept(accepted_indices)
        self.context_length += len(accepted_indices)

    def reset(self) -> None:
        for state in self.layer_states:
            state.reset()
        self.context_length = 0


class Transformer:
    def __init__(self, model_dim: int, config: TransformerConfig, params: ParameterTree) -> None:
        self.model_dim = model_dim
        self.config = config
        self.layers = [
            TransformerLayer(
                model_dim,
                config.hidden_dim,
                index,
                layer_config,
                params.subtree(f"layers.{index}"),
            )
            for index, layer_config in enumerate(config.layer_configs)
        ]
        self.output_norm = Normalization.new(
            model_dim,
            config.output_norm_config,
            params.subtree("output_norm"),
        )

    # -- properties -------------------------------------------------------
    @property
    def speculation_supported(self) -> bool:
        return all(layer.speculation_supported for layer in self.layers)

    def max_context_length(self) -> Optional[int]:
        lengths = [layer.mixer.max_context_length() for layer in self.layers]
        lengths = [x for x in lengths if x is not None]
        return min(lengths) if lengths else None

    def prefill_cache_layer_count(self) -> int:
        indices = [
            i for i, layer in enumerate(self.layers) if layer.kv_source_layer_index is None
        ]
        return max(indices) + 1 if indices else len(self.layers)

    def prefill_cache_skips_trailing_layers(self) -> bool:
        return self.prefill_cache_layer_count() < len(self.layers)

    # -- state ------------------------------------------------------------
    def create_empty_state(self, max_context_length: Optional[int], context) -> TransformerState:
        states: list[MixerState] = []
        for layer in self.layers:
            if layer.kv_source_layer_index is None:
                states.append(layer.create_empty_state(max_context_length, context))
            else:
                source = states[layer.kv_source_layer_index]
                if not isinstance(source, AttentionState):
                    raise TypeError("shared KV cache requires an attention source layer")
                states.append(
                    AttentionState(
                        source.cache.num_heads,
                        source.cache.head_dim,
                        source.cache.capacity,
                        shared=source,
                        read_only=True,
                    )
                )
        return TransformerState(states)

    # -- encode -----------------------------------------------------------
    def encode(
        self,
        input_embeddings: np.ndarray,
        positions: np.ndarray,
        state: TransformerState,
        output_range: Optional[tuple[int, int]] = None,
        hidden_feature_layer_indices: Optional[Sequence[int]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> TransformerEncodeOutput:
        positions = np.asarray(positions, dtype=np.float32)
        rope_cache: dict[tuple[int, float, int], PrecalculatedRoPE] = {}

        def rope_for(layer: TransformerLayer) -> Optional[PrecalculatedRoPE]:
            cfg = getattr(layer.config.mixer, "rope", None)
            if cfg is None:
                return None
            key = (cfg.dim, cfg.base, cfg.max_sequence_length)
            if key not in rope_cache:
                rope_cache[key] = PrecalculatedRoPE.precalculate(cfg, positions)
            return rope_cache[key]

        hidden = input_embeddings
        shortcut: Optional[np.ndarray] = None
        wanted = list(hidden_feature_layer_indices or [])
        features: dict[int, np.ndarray] = {}

        layer_count = len(self.layers)
        for index in range(layer_count):
            layer = self.layers[index]
            hidden, shortcut = layer.encode(
                hidden,
                shortcut,
                rope_for(layer),
                state.layer_states[index],
                mask,
            )
            if index in wanted:
                features[index] = (shortcut + hidden).astype(np.float32)

        if output_range is None:
            return TransformerEncodeOutput(None, [features[i] for i in wanted] if wanted else None)

        assert shortcut is not None
        out, _ = self.output_norm.apply(hidden, shortcut, ShortcutMode.ADD)
        start, end = output_range
        return TransformerEncodeOutput(
            out[start:end],
            [features[i] for i in wanted] if wanted else None,
        )
