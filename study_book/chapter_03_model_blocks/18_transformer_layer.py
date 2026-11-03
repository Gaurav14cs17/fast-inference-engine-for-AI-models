"""18 — one transformer layer.

Mirrors ``encodable_block/transformer_layer.rs``.

Order inside the layer (pre-norm):

    shortcut = shortcut + input          (Copy on layer 0, Add afterwards)
    h = norm(shortcut)
    h = mixer(h)                         (attention / delta net / short conv)
    shortcut = shortcut + h
    h = norm(shortcut)
    h = mlp(h)
    return h                             (residual is still in ``shortcut``)

The next layer (or the output norm) adds the returned ``h`` to the residual.
This is why the function returns both values.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from study_book.chapter_02_configuration.config import DenseMLPConfig, TransformerLayerConfig
from study_book.chapter_02_configuration.safetensors import ParameterTree
from study_book.chapter_03_model_blocks.normalization import Normalization, ShortcutMode
from study_book.chapter_03_model_blocks.mlp import DenseMlp
from study_book.chapter_03_model_blocks.rope import PrecalculatedRoPE
from study_book.chapter_03_model_blocks.mixer import Mixer, MixerState, build_mixer, mixer_speculation_supported


class TransformerLayer:
    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        layer_index: int,
        config: TransformerLayerConfig,
        params: ParameterTree,
    ) -> None:
        self.layer_index = layer_index
        self.model_dim = model_dim
        self.config = config
        self.kv_source_layer_index = config.kv_source_layer_index
        self.hidden_dim = config.hidden_dim or hidden_dim

        if config.pre_mixer_norm_config is not None:
            self.pre_mixer_norm: Optional[Normalization] = Normalization.new(
                model_dim,
                config.pre_mixer_norm_config,
                params.subtree("pre_mixer_norm"),
            )
        else:
            if layer_index != 0:
                raise ValueError("only layer 0 may omit pre_mixer_norm")
            self.pre_mixer_norm = None

        self.mixer: Mixer = build_mixer(model_dim, config.mixer, params.subtree("mixer"))
        self.post_mixer_norm = (
            Normalization.new(model_dim, config.post_mixer_norm_config, params.subtree("post_mixer_norm"))
            if config.post_mixer_norm_config is not None
            else None
        )
        self.pre_mlp_norm = Normalization.new(
            model_dim,
            config.pre_mlp_norm_config,
            params.subtree("pre_mlp_norm"),
        )
        self.post_mlp_norm = (
            Normalization.new(model_dim, config.post_mlp_norm_config, params.subtree("post_mlp_norm"))
            if config.post_mlp_norm_config is not None
            else None
        )
        mlp_config = config.mlp or DenseMLPConfig()
        self.mlp = DenseMlp.new(model_dim, self.hidden_dim, mlp_config, params.subtree("mlp"))

    @property
    def speculation_supported(self) -> bool:
        return mixer_speculation_supported(self.mixer)

    def create_empty_state(self, max_context_length: int | None, context) -> MixerState:
        return self.mixer.create_empty_state(max_context_length, context)

    def encode(
        self,
        x: np.ndarray,
        shortcut: Optional[np.ndarray],
        rope: Optional[PrecalculatedRoPE],
        state: MixerState,
        mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.pre_mixer_norm is not None:
            mode = ShortcutMode.COPY if self.layer_index == 0 else ShortcutMode.ADD
            h, shortcut = self.pre_mixer_norm.apply(x, shortcut, mode)
        else:
            shortcut = x.copy()
            h = x

        h = self.mixer.encode(h, rope, state, mask)

        if self.post_mixer_norm is not None:
            h, _ = self.post_mixer_norm.apply(h, None, ShortcutMode.NONE)

        h, shortcut = self.pre_mlp_norm.apply(h, shortcut, ShortcutMode.ADD)
        h = self.mlp.encode(h)

        if self.post_mlp_norm is not None:
            h, _ = self.post_mlp_norm.apply(h, None, ShortcutMode.NONE)

        return h, shortcut
