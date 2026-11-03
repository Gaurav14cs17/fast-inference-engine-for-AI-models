"""16 — gated delta-rule linear attention (DeltaNet).

Mirrors ``encodable_block/mixer/delta_net.rs`` and
``backends/metal/kernel/gdn/*``.

Instead of a growing KV cache, this mixer keeps a fixed-size recurrent state
``S[head, d_k, d_v]``.  Per token:

    v_new = v - S^T k
    S     = alpha * S + beta * k outer v_new
    y     = S^T q

The real engine runs a *chunked* version of this recurrence (and a special
tree-verify variant) so prefill is parallel.  The loop below is the clearest
possible version for study.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from study_book.chapter_01_foundations.kernels import depthwise_causal_conv, l2_normalize, sigmoid
from study_book.chapter_02_configuration.config import DeltaNetConfig
from study_book.chapter_02_configuration.safetensors import ParameterTree
from study_book.chapter_03_model_blocks.linear import Linear, new_linear


class DeltaNetState:
    def __init__(self, num_heads: int, head_dim: int, kernel_size: int) -> None:
        self.state = np.zeros((num_heads, head_dim, head_dim), dtype=np.float32)
        self.conv_history = np.zeros((max(kernel_size - 1, 0), 3 * num_heads * head_dim), dtype=np.float32)

    def reset(self) -> None:
        self.state.fill(0.0)
        self.conv_history.fill(0.0)

    def accept(self, indices) -> None:
        # Speculation is disabled for the simplified DeltaNet state.
        pass


class DeltaNet:
    def __init__(
        self,
        model_dim: int,
        config: DeltaNetConfig,
        params: ParameterTree,
    ) -> None:
        self.model_dim = model_dim
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        dim = self.num_heads * self.head_dim

        self.q_projection: Linear = new_linear(model_dim, dim, params.subtree("q_projection"), bias=False)
        self.k_projection: Linear = new_linear(model_dim, dim, params.subtree("k_projection"), bias=False)
        self.v_projection: Linear = new_linear(model_dim, dim, params.subtree("v_projection"), bias=False)
        self.beta_projection: Linear = new_linear(
            model_dim, self.num_heads, params.subtree("beta_projection"), bias=False
        )
        self.alpha_projection: Linear = new_linear(
            model_dim, self.num_heads, params.subtree("alpha_projection"), bias=False
        )
        self.out_projection: Linear = new_linear(model_dim, model_dim, params.subtree("out_projection"), bias=False)
        self.conv_weight = params.read("conv.weights", (1, config.conv_kernel_size))
        self.conv_bias = params.read("conv.bias", (3 * dim,), init="zeros")
        self.speculation_supported = False

    def create_empty_state(self, max_context_length: Optional[int], context) -> DeltaNetState:
        return DeltaNetState(self.num_heads, self.head_dim, self.config.conv_kernel_size)

    def max_context_length(self) -> Optional[int]:
        return None

    def encode(self, x: np.ndarray, rope, state: DeltaNetState, mask=None) -> np.ndarray:
        t = x.shape[0]
        q = self.q_projection.encode(x).reshape(t, self.num_heads, self.head_dim)
        k = self.k_projection.encode(x).reshape(t, self.num_heads, self.head_dim)
        v = self.v_projection.encode(x).reshape(t, self.num_heads, self.head_dim)

        if self.config.use_conv:
            qkv = np.concatenate((q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1)), axis=-1)
            full = np.concatenate((state.conv_history, qkv), axis=0)
            convolved = depthwise_causal_conv(full, self.conv_weight, self.conv_bias, groups=1)[-t:]
            q, k, v = np.split(convolved, 3, axis=-1)
            q = q.reshape(t, self.num_heads, self.head_dim)
            k = k.reshape(t, self.num_heads, self.head_dim)
            v = v.reshape(t, self.num_heads, self.head_dim)
            keep = self.config.conv_kernel_size - 1
            state.conv_history = full[-keep:].copy() if keep > 0 else full[:0]

        q = l2_normalize(q)
        k = l2_normalize(k)
        beta = sigmoid(self.beta_projection.encode(x))  # [T, H]
        alpha = sigmoid(self.alpha_projection.encode(x))  # [T, H]

        out = np.zeros((t, self.num_heads, self.head_dim), dtype=np.float32)
        s = state.state
        for i in range(t):
            k_i = k[i]          # [H, Dk]
            v_i = v[i]          # [H, Dv]
            q_i = q[i]          # [H, Dk]
            # S^T k -> [H, Dv]
            predicted = np.einsum("hkd,hk->hd", s, k_i, optimize=True)
            v_new = v_i - predicted
            s = (
                alpha[i, :, None, None] * s
                + beta[i, :, None, None] * k_i[:, :, None] * v_new[:, None, :]
            )
            out[i] = np.einsum("hkd,hk->hd", s, q_i, optimize=True)
        state.state = s
        return self.out_projection.encode(out.reshape(t, self.model_dim))
