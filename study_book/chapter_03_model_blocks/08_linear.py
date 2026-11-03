"""08 — linear layers.

Mirrors ``encodable_block/linear``:

* ``LinearMatmul``       — dense matmul, optionally quantized
* ``RHTLinearWrapper``   — input randomized-Hadamard transform + linear
* ``UntiedReadout``      — final vocab projection
* ``Gather``             — embedding-style gather used by MoE/readout

The real engine picks a kernel (GEMV/GEMM/MXU) and an activation format from
the matmul shape.  Here ``select_activation_format`` only reports a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from study_book.chapter_01_foundations.dtypes import DataType, symmetric_int8_dequantize, symmetric_int8_quantize
from study_book.chapter_01_foundations.kernels import linear as linear_kernel
from study_book.chapter_02_configuration.quantization import QuantizedWeight, RHT, quantize_weights
from study_book.chapter_02_configuration.safetensors import ParameterTree


class Linear:
    """Interface matching the Rust ``trait Linear<B>``."""

    def encode(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def encode_input(self, x: np.ndarray) -> np.ndarray:
        return self.encode(x)

    def select_activation_format(self, batch_dim: int, context=None) -> str:
        return "bf16"


@dataclass
class Gather:
    indices: np.ndarray
    output_dim: int


class LinearMatmul(Linear):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        weights: np.ndarray,
        bias: Optional[np.ndarray] = None,
        quant_format: Optional[str] = None,
    ) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.raw_weights = np.asarray(weights, dtype=np.float32)
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
        self.quant: Optional[QuantizedWeight] = None
        self._dequantized: Optional[np.ndarray] = None
        if quant_format:
            self.quant = quantize_weights(self.raw_weights, quant_format)
        if self.raw_weights.shape != (out_dim, in_dim):
            raise ValueError(f"weight shape {self.raw_weights.shape} != {(out_dim, in_dim)}")

    @classmethod
    def new(
        cls,
        in_dim: int,
        out_dim: int,
        params: ParameterTree,
        bias: bool = False,
        quant_format: Optional[str] = None,
    ) -> "LinearMatmul":
        weights = params.read("weights", (out_dim, in_dim))
        b = params.read("bias", (out_dim,), init="zeros") if bias else None
        return cls(in_dim, out_dim, weights, b, quant_format)

    def weights(self) -> np.ndarray:
        if self.quant is None:
            return self.raw_weights
        if self._dequantized is None:
            self._dequantized = self.quant.dequantize()
        return self._dequantized

    def encode(self, x: np.ndarray) -> np.ndarray:
        return linear_kernel(x, self.weights(), self.bias)

    def select_activation_format(self, batch_dim: int, context=None) -> str:
        # Decode (small batch) uses a GEMV-style activation; prefill uses GEMM.
        if context is not None and getattr(context, "supports_mxu", False):
            return "mxu"
        return "packed4" if self.quant and self.quant.format.startswith("int4") else "bf16"


class RHTLinearWrapper(Linear):
    """Fuses an input RHT into the linear layer (``IncoherenceProcessingMode``)."""

    def __init__(self, inner: Linear, rht: RHT) -> None:
        self.inner = inner
        self.rht = rht

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.inner.encode(self.rht.apply(x))

    def select_activation_format(self, batch_dim: int, context=None) -> str:
        return self.inner.select_activation_format(batch_dim, context)


class UntiedReadout(Linear):
    """Final projection to vocabulary logits."""

    def __init__(self, linear: Linear) -> None:
        self.linear = linear

    @classmethod
    def new(
        cls,
        model_dim: int,
        vocab_size: int,
        params: ParameterTree,
        quant_format: Optional[str] = None,
    ) -> "UntiedReadout":
        return cls(LinearMatmul.new(model_dim, vocab_size, params, bias=False, quant_format=quant_format))

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.linear.encode(x)


def new_linear(
    in_dim: int,
    out_dim: int,
    params: ParameterTree,
    bias: bool = False,
    quant_format: Optional[str] = None,
    rht: Optional[RHT] = None,
) -> Linear:
    """Factory corresponding to ``<dyn Linear<B>>::new_mixed_precision``."""
    inner = LinearMatmul.new(in_dim, out_dim, params, bias=bias, quant_format=quant_format)
    return RHTLinearWrapper(inner, rht) if rht is not None else inner


class Int8ActivationLinear(Linear):
    """Example of the A8 path: quantize the activation, then dequantize.

    The real kernel multiplies int8 x int8 and applies scales in the epilogue;
    here we show the mathematical equivalent.
    """

    def __init__(self, inner: LinearMatmul) -> None:
        self.inner = inner

    def encode(self, x: np.ndarray) -> np.ndarray:
        q, scale = symmetric_int8_quantize(x)
        x_hat = symmetric_int8_dequantize(q, scale)
        return self.inner.encode(x_hat)
