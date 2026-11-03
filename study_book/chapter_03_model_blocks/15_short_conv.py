"""15 — short convolution mixer.

Mirrors ``encodable_block/mixer/short_conv.rs`` and
``backends/metal/kernel/convolution/separable_causal_conv.metal``.

A depthwise causal convolution over the token axis.  It keeps a small history
(``kernel_size - 1`` tokens) as its state, so prefill and decode use the same
kernel.  The real engine fuses the pre/post convolution with the mixer
projection; here the pieces are separate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from study_book.chapter_01_foundations.kernels import depthwise_causal_conv, silu, split
from study_book.chapter_02_configuration.config import ShortConvConfig
from study_book.chapter_02_configuration.safetensors import ParameterTree
from study_book.chapter_03_model_blocks.linear import Linear, new_linear


class ShortConvState:
    def __init__(self, kernel_size: int, channels: int) -> None:
        self.history = np.zeros((max(kernel_size - 1, 0), channels), dtype=np.float32)

    def reset(self) -> None:
        self.history.fill(0.0)

    def accept(self, indices) -> None:
        # Speculation is disabled for this study mixer; see class docstring.
        pass


class ShortConv:
    def __init__(
        self,
        model_dim: int,
        config: ShortConvConfig,
        params: ParameterTree,
    ) -> None:
        self.model_dim = model_dim
        self.config = config
        self.in_projection: Linear = new_linear(
            model_dim,
            2 * model_dim,
            params.subtree("in_projection"),
            bias=False,
        )
        self.out_projection: Linear = new_linear(
            model_dim,
            model_dim,
            params.subtree("out_projection"),
            bias=False,
        )
        self.conv_weight = params.read("conv.weights", (config.groups, config.kernel_size))
        self.conv_bias = params.read("conv.bias", (model_dim,), init="zeros")
        self.speculation_supported = False

    def create_empty_state(self, max_context_length: Optional[int], context) -> ShortConvState:
        return ShortConvState(self.config.kernel_size, self.model_dim)

    def max_context_length(self) -> Optional[int]:
        return None

    def encode(self, x: np.ndarray, rope, state: ShortConvState, mask=None) -> np.ndarray:
        t = x.shape[0]
        fused = self.in_projection.encode(x)
        gate, value = split(fused, (self.model_dim, self.model_dim), axis=-1)
        full = np.concatenate((state.history, value), axis=0)
        convolved = depthwise_causal_conv(full, self.conv_weight, self.conv_bias, groups=self.config.groups)
        convolved = convolved[-t:]
        state.history = full[-(self.config.kernel_size - 1) :].copy() if self.config.kernel_size > 1 else full[:0]
        out = silu(gate) * convolved
        return self.out_projection.encode(out)
