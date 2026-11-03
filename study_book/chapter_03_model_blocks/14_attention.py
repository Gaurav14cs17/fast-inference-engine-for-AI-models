"""14 — grouped-query attention mixer.

Mirrors ``encodable_block/mixer/attention/mod.rs`` and
``backends/metal/kernel/attention/*``:

* fused QKV(+gate) projection,
* optional QK RMSNorm,
* RoPE,
* KV-cache append,
* causal / sliding-window attention (single-pass for short context),
* optional per-head sinks and sigmoid output gate,
* output projection.

The real kernel chooses between single-pass, two-pass, GEMM-grouped, int8 and
ancestor-attention variants based on context length and speculation.  Here one
NumPy function covers all of them.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from study_book.chapter_01_foundations.kernels import attention as attention_kernel
from study_book.chapter_01_foundations.kernels import causal_mask, rms_norm, sigmoid, split
from study_book.chapter_02_configuration.config import AttentionConfig
from study_book.chapter_02_configuration.safetensors import ParameterTree
from study_book.chapter_03_model_blocks.linear import Linear, new_linear
from study_book.chapter_03_model_blocks.rope import PrecalculatedRoPE
from study_book.chapter_03_model_blocks.kv_cache import AttentionState


class Attention:
    def __init__(
        self,
        model_dim: int,
        config: AttentionConfig,
        params: ParameterTree,
    ) -> None:
        self.model_dim = model_dim
        self.config = config
        self.num_heads = config.num_heads
        self.num_groups = config.num_groups
        self.head_dim = config.head_dim
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_groups * self.head_dim

        qkv_out = self.q_dim + 2 * self.kv_dim
        if config.has_gate:
            qkv_out += self.q_dim
        self.qkvg_projection: Linear = new_linear(
            model_dim,
            qkv_out,
            params.subtree("qkvg_projection"),
            bias=config.has_qkvg_biases,
        )
        self.out_projection: Linear = new_linear(
            model_dim,
            model_dim,
            params.subtree("out_projection"),
            bias=False,
        )
        self.q_norm_scales = (
            params.read("q_norm.scales", (self.head_dim,), init="ones") if config.qk_norm else None
        )
        self.k_norm_scales = (
            params.read("k_norm.scales", (self.head_dim,), init="ones") if config.qk_norm else None
        )
        self.sinks = params.read("sinks", (self.num_heads,), init="zeros") if config.use_sinks else None
        self.speculation_supported = True

    # -- state ------------------------------------------------------------
    def create_empty_state(self, max_context_length: Optional[int], context) -> AttentionState:
        capacity = max_context_length or 4096
        return AttentionState(self.num_groups, self.head_dim, capacity)

    def max_context_length(self) -> Optional[int]:
        return self.config.rope.max_sequence_length if self.config.rope else None

    # -- encode -----------------------------------------------------------
    def encode(
        self,
        x: np.ndarray,
        rope: Optional[PrecalculatedRoPE],
        state: AttentionState,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        t = x.shape[0]
        fused = self.qkvg_projection.encode(x)
        sizes = [self.q_dim, self.kv_dim, self.kv_dim]
        if self.config.has_gate:
            sizes.append(self.q_dim)
        parts = split(fused, sizes, axis=-1)
        q, k, v = parts[0], parts[1], parts[2]
        gate = parts[3] if self.config.has_gate else None

        q = q.reshape(t, self.num_heads, self.head_dim)
        k = k.reshape(t, self.num_groups, self.head_dim)
        v = v.reshape(t, self.num_groups, self.head_dim)

        if rope is not None:
            q = rope.apply(q)
            k = rope.apply(k)
        if self.q_norm_scales is not None:
            q = rms_norm(q, self.q_norm_scales)
        if self.k_norm_scales is not None:
            k = rms_norm(k, self.k_norm_scales)

        context_before = state.context_length
        k_all, v_all = state.append(k, v)

        if mask is None:
            mask = causal_mask(
                t,
                k_all.shape[0],
                offset=context_before,
                window=self.config.sliding_window_size,
            )

        out = attention_kernel(q, k_all, v_all, mask=mask, sinks=self.sinks)
        if gate is not None:
            gate = gate.reshape(t, self.num_heads, self.head_dim)
            out = (out * sigmoid(gate)).astype(np.float32)
        out = out.reshape(t, self.model_dim)
        return self.out_projection.encode(out)
