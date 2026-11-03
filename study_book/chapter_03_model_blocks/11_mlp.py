"""11 — dense MLP (SwiGLU / GeGLU).

Mirrors ``encodable_block/mlp/dense.rs`` and ``mlp/gate_act_mul.rs``.

The Rust block fuses the up-projection of both gate and value into one matmul
(``2 * hidden_dim``), then runs a ``gated_act_mul`` kernel.  This is exactly the
classic SwiGLU:

    y = down( act(gate) * up )

with clipping applied before the activation.
"""

from __future__ import annotations

import numpy as np

from study_book.chapter_01_foundations.kernels import gelu, silu, split
from study_book.chapter_02_configuration.config import DenseMLPConfig
from study_book.chapter_02_configuration.safetensors import ParameterTree
from study_book.chapter_03_model_blocks.linear import Linear, new_linear


class Mlp:
    def encode(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GateActMul:
    def __init__(self, config: DenseMLPConfig) -> None:
        self.config = config
        self.activation = {"silu": silu, "gelu": gelu, "swish": silu}[config.activation]

    def __call__(self, fused: np.ndarray) -> np.ndarray:
        hidden = fused.shape[-1] // 2
        gate, up = split(fused, (hidden, hidden), axis=-1)
        if self.config.gate_clipping is not None:
            gate = np.clip(gate, -self.config.gate_clipping, self.config.gate_clipping)
        if self.config.up_clipping is not None:
            up = np.clip(up, -self.config.up_clipping, self.config.up_clipping)
        return (self.activation(gate) * up).astype(np.float32)


class DenseMlp(Mlp):
    def __init__(self, up: Linear, down: Linear, gate: GateActMul) -> None:
        self.up = up       # model_dim -> 2 * hidden_dim
        self.down = down   # hidden_dim -> model_dim
        self.gate = gate

    @classmethod
    def new(
        cls,
        model_dim: int,
        hidden_dim: int,
        config: DenseMLPConfig,
        params: ParameterTree,
    ) -> "DenseMlp":
        up = new_linear(
            model_dim,
            2 * hidden_dim,
            params.subtree("up_projection"),
            bias=config.has_up_biases,
        )
        down = new_linear(
            hidden_dim,
            model_dim,
            params.subtree("down_projection"),
            bias=config.has_down_biases,
        )
        return cls(up, down, GateActMul(config))

    def encode(self, x: np.ndarray) -> np.ndarray:
        fused = self.up.encode(x)
        hidden = self.gate(fused)
        return self.down.encode(hidden)
