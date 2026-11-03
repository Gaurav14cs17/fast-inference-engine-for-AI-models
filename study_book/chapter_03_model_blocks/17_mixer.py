"""17 — mixer dispatch.

Mirrors ``encodable_block/mixer/mod.rs``.

A transformer layer does not know whether its token mixer is full attention, a
gated delta net, or a short convolution.  The dispatch lives here, exactly like
``AnyTokenMixerConfig`` in the Rust engine.
"""

from __future__ import annotations

from typing import Any, Union

from study_book.chapter_02_configuration.config import AnyMixerConfig, AttentionConfig, DeltaNetConfig, ShortConvConfig
from study_book.chapter_02_configuration.safetensors import ParameterTree
from study_book.chapter_03_model_blocks.attention import Attention
from study_book.chapter_03_model_blocks.short_conv import ShortConv
from study_book.chapter_03_model_blocks.delta_net import DeltaNet

Mixer = Union[Attention, DeltaNet, ShortConv]
MixerState = Any


def build_mixer(model_dim: int, config: AnyMixerConfig, params: ParameterTree) -> Mixer:
    if isinstance(config, AttentionConfig):
        return Attention(model_dim, config, params)
    if isinstance(config, DeltaNetConfig):
        return DeltaNet(model_dim, config, params)
    if isinstance(config, ShortConvConfig):
        return ShortConv(model_dim, config, params)
    raise TypeError(f"unsupported mixer config: {type(config).__name__}")


def mixer_speculation_supported(mixer: Mixer) -> bool:
    return bool(getattr(mixer, "speculation_supported", False))


def create_mixer_state(mixer: Mixer, max_context_length: int | None, context) -> MixerState:
    return mixer.create_empty_state(max_context_length, context)
