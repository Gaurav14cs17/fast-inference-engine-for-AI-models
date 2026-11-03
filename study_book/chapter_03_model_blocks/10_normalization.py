"""10 — normalization and residual shortcut handling.

Mirrors ``encodable_block/normalization.rs``.

The Rust normalization kernel does more than RMSNorm: it can

* write the residual stream into a separate ``shortcut`` buffer (``Copy``/``Add``),
* apply an optional Hadamard/RHT transform,
* apply a scalar to the residual sum or the output.

That fused residual bookkeeping is why the transformer layer looks unusual at
first: the layer returns only the MLP output, while the residual stream lives in
``shortcut`` and is advanced by the next normalization call.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from study_book.chapter_01_foundations.kernels import rms_norm
from study_book.chapter_02_configuration.config import NormalizationConfig
from study_book.chapter_02_configuration.quantization import RHT
from study_book.chapter_02_configuration.safetensors import ParameterTree


class ShortcutMode(Enum):
    NONE = "none"
    COPY = "copy"   # shortcut <- input
    ADD = "add"     # shortcut <- shortcut + input


class PostLayerScalar(Enum):
    NONE = "none"
    SCALE_RESIDUAL_SUM = "scale_residual_sum"
    SCALE_OUTPUT = "scale_output"


class Normalization:
    def __init__(
        self,
        element_count: int,
        config: NormalizationConfig,
        scales: Optional[np.ndarray] = None,
        biases: Optional[np.ndarray] = None,
        rht: Optional[RHT] = None,
        post_layer_scalar: PostLayerScalar = PostLayerScalar.NONE,
        scalar_value: float = 1.0,
        residual_sum_scalar: float = 1.0,
    ) -> None:
        self.element_count = element_count
        self.config = config
        self.scales = scales
        self.biases = biases
        self.rht = rht
        self.post_layer_scalar = post_layer_scalar
        self.scalar_value = float(scalar_value)
        self.residual_sum_scalar = float(residual_sum_scalar)

    @classmethod
    def new(
        cls,
        element_count: int,
        config: NormalizationConfig,
        params: ParameterTree,
        rht: Optional[RHT] = None,
        post_layer_scalar: PostLayerScalar = PostLayerScalar.NONE,
        scalar_value: float = 1.0,
        residual_sum_scalar: float = 1.0,
    ) -> "Normalization":
        scales = params.read("scales", (element_count,), init="ones") if config.has_scale else None
        biases = params.read("biases", (element_count,), init="zeros") if config.has_biases else None
        return cls(
            element_count,
            config,
            scales,
            biases,
            rht,
            post_layer_scalar,
            scalar_value,
            residual_sum_scalar,
        )

    def apply(
        self,
        x: np.ndarray,
        shortcut: Optional[np.ndarray] = None,
        mode: ShortcutMode = ShortcutMode.NONE,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(normalized_input, updated_shortcut)``.

        ``shortcut`` is mutated through a new array so callers can keep the
        data-flow explicit.
        """
        x = np.asarray(x, dtype=np.float32)
        if mode == ShortcutMode.COPY:
            shortcut = x.copy()
        elif mode == ShortcutMode.ADD:
            if shortcut is None:
                raise ValueError("ADD shortcut requires an existing residual")
            shortcut = (shortcut + x).astype(np.float32)
        elif shortcut is None:
            shortcut = x

        if self.post_layer_scalar == PostLayerScalar.SCALE_RESIDUAL_SUM:
            shortcut = (shortcut * self.scalar_value).astype(np.float32)

        source = shortcut if mode != ShortcutMode.NONE else x
        out = rms_norm(
            source,
            self.scales,
            self.biases,
            eps=self.config.eps,
            subtract_mean=self.config.subtract_mean,
        )
        if self.rht is not None:
            out = self.rht.apply(out)
        if self.post_layer_scalar == PostLayerScalar.SCALE_OUTPUT:
            out = (out * self.scalar_value).astype(np.float32)
        return out, shortcut
