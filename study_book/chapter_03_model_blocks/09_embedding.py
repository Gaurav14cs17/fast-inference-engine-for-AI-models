"""09 — token embedding and per-layer embedding.

Mirrors ``encodable_block/embedding.rs`` and ``per_layer_embedding.rs``.

The main table is a normal ``[vocab, model_dim]`` lookup.  ``PerLayerEmbedding``
(used by some Qwen-style models) adds a small per-layer token vector to the
shared embedding; the real engine fuses that into the first normalization.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from study_book.chapter_01_foundations.kernels import embedding_lookup
from study_book.chapter_02_configuration.safetensors import ParameterTree
from study_book.chapter_03_model_blocks.linear import Linear, LinearMatmul


class Embedding:
    def __init__(self, table: np.ndarray, quant_format: Optional[str] = None) -> None:
        self.table = np.asarray(table, dtype=np.float32)
        self.quant_format = quant_format

    @classmethod
    def new(cls, vocab_size: int, model_dim: int, params: ParameterTree) -> "Embedding":
        return cls(params.read("weights", (vocab_size, model_dim)))

    def encode_lookup(self, token_ids: np.ndarray) -> np.ndarray:
        return embedding_lookup(self.table, token_ids)

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        return self.encode_lookup(token_ids)


class PerLayerEmbedding:
    """Simplified PLE: ``embedding[token] + table[layer, token]``."""

    def __init__(
        self,
        table: np.ndarray,
        projection: Optional[Linear] = None,
        num_layers: int = 0,
    ) -> None:
        self.table = np.asarray(table, dtype=np.float32)  # [num_layers, table_size, model_dim]
        self.projection = projection
        self.num_layers = num_layers or self.table.shape[0]

    @classmethod
    def new(
        cls,
        model_dim: int,
        num_layers: int,
        table_size: int,
        params: ParameterTree,
    ) -> "PerLayerEmbedding":
        table = params.read("table", (num_layers, table_size, model_dim))
        projection = LinearMatmul.new(model_dim, model_dim, params.subtree("projection"), bias=False)
        return cls(table, projection, num_layers)

    def encode(self, token_ids: np.ndarray, embedded: np.ndarray) -> np.ndarray:
        # [T, L, D]
        per_layer = self.table[:, np.asarray(token_ids, dtype=np.int64), :]  # [L, T, D]
        per_layer = np.transpose(per_layer, (1, 0, 2))
        out = embedded[:, None, :] + per_layer
        if self.projection is not None:
            out = self.projection.encode(out)
        return out.astype(np.float32)
