"""05 — model configuration.

Mirrors ``crates/uzu-engine/src/config``.  Every ``*Config`` here corresponds to
a serde struct in the Rust engine (``config/attention.rs``, ``config/mlp/...``,
``config/speculator/...``).

The tiny factories at the bottom are the equivalent of a model's ``config.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------
@dataclass
class RoPEConfig:
    dim: int
    base: float = 10_000.0
    max_sequence_length: int = 4096


@dataclass
class NormalizationConfig:
    has_scale: bool = True
    has_biases: bool = False
    subtract_mean: bool = False
    eps: float = 1e-5


@dataclass
class AttentionConfig:
    num_heads: int = 4
    num_groups: int = 4  # KV heads
    head_dim: int = 16
    is_causal: bool = True
    sliding_window_size: Optional[int] = None
    has_gate: bool = False
    has_qkvg_biases: bool = False
    is_kv_sharing: bool = False
    qk_norm: bool = False
    use_sinks: bool = False
    rope: Optional[RoPEConfig] = None


@dataclass
class DeltaNetConfig:
    num_heads: int = 4
    head_dim: int = 16
    conv_kernel_size: int = 4
    use_conv: bool = True


@dataclass
class ShortConvConfig:
    kernel_size: int = 4
    groups: int = 1


AnyMixerConfig = Union[AttentionConfig, DeltaNetConfig, ShortConvConfig]


@dataclass
class DenseMLPConfig:
    activation: str = "silu"
    has_up_biases: bool = False
    has_down_biases: bool = False
    gate_clipping: Optional[float] = None
    up_clipping: Optional[float] = None


@dataclass
class TransformerLayerConfig:
    mixer: AnyMixerConfig
    mlp: DenseMLPConfig = field(default_factory=DenseMLPConfig)
    hidden_dim: Optional[int] = None
    pre_mixer_norm_config: Optional[NormalizationConfig] = field(default_factory=NormalizationConfig)
    post_mixer_norm_config: Optional[NormalizationConfig] = None
    pre_mlp_norm_config: NormalizationConfig = field(default_factory=NormalizationConfig)
    post_mlp_norm_config: Optional[NormalizationConfig] = None
    # When set, this layer reuses the KV cache of the referenced earlier layer.
    kv_source_layer_index: Optional[int] = None


@dataclass
class TransformerConfig:
    model_dim: int
    hidden_dim: int
    layer_configs: list[TransformerLayerConfig]
    output_norm_config: NormalizationConfig = field(default_factory=NormalizationConfig)


@dataclass
class EmbeddingConfig:
    tied: bool = False


@dataclass
class PerLayerEmbeddingConfig:
    num_layers: int
    table_size: int = 256


@dataclass
class ReadoutConfig:
    tied: bool = False


@dataclass
class DecoderConfig:
    vocab_size: int
    transformer_config: TransformerConfig
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    embedding_norm_config: Optional[NormalizationConfig] = None
    readout_config: ReadoutConfig = field(default_factory=ReadoutConfig)
    ple_model_config: Optional[PerLayerEmbeddingConfig] = None


# ---------------------------------------------------------------------------
# generation / speculation
# ---------------------------------------------------------------------------
@dataclass
class SamplingConfig:
    greedy: bool = True
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    seed: int = 0


@dataclass
class GenerationConfig:
    max_tokens: int = 16
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    eos_token_id: int = 256


@dataclass
class DFlashConfig:
    vocab_size: int
    model_dim: int
    num_layers: int
    num_heads: int
    num_groups: int
    head_dim: int
    hidden_dim: int
    block_size: int = 4
    mask_token_id: int = 257
    target_layer_ids: list[int] = field(default_factory=lambda: [0])
    target_feature_dim: int = 64
    rope: Optional[RoPEConfig] = None


@dataclass
class WeaverConfig:
    max_depth: int = 4
    tree_budget: int = 8
    rounds: int = 3
    expand_per_round: int = 4
    expand_width: int = 2
    prune_sigma: Optional[float] = 1.5


@dataclass
class SpeculatorConfig:
    draft_config: DFlashConfig
    weaver_config: Optional[WeaverConfig] = None
    tree_budget: int = 8
    max_tree_depth: int = 4
    use_weaver: bool = True


@dataclass
class ModelConfig:
    decoder_config: DecoderConfig
    generation_config: GenerationConfig
    speculator_config: Optional[SpeculatorConfig] = None
    tokenizer_vocab_size: int = 258


# ---------------------------------------------------------------------------
# tiny factories (the equivalent of a model pack's config.json)
# ---------------------------------------------------------------------------
def _rope(dim: int = 16, max_seq: int = 256) -> RoPEConfig:
    return RoPEConfig(dim=dim, max_sequence_length=max_seq)


def _attention_layer(model_dim: int = 64, head_dim: int = 16) -> TransformerLayerConfig:
    heads = model_dim // head_dim
    return TransformerLayerConfig(
        mixer=AttentionConfig(
            num_heads=heads,
            num_groups=heads,
            head_dim=head_dim,
            rope=_rope(head_dim),
        ),
    )


def tiny_config() -> ModelConfig:
    """2 attention layers; supports DFlash + tree verification."""
    model_dim = 64
    head_dim = 16
    vocab = 258
    transformer = TransformerConfig(
        model_dim=model_dim,
        hidden_dim=128,
        layer_configs=[_attention_layer(model_dim, head_dim) for _ in range(2)],
    )
    decoder = DecoderConfig(vocab_size=vocab, transformer_config=transformer)
    draft = DFlashConfig(
        vocab_size=vocab,
        model_dim=model_dim,
        num_layers=1,
        num_heads=4,
        num_groups=4,
        head_dim=head_dim,
        hidden_dim=128,
        block_size=4,
        mask_token_id=257,
        target_layer_ids=[0, 1],
        target_feature_dim=model_dim * 2,
        rope=_rope(head_dim),
    )
    spec = SpeculatorConfig(
        draft_config=draft,
        weaver_config=WeaverConfig(),
        tree_budget=8,
        max_tree_depth=4,
    )
    return ModelConfig(
        decoder_config=decoder,
        generation_config=GenerationConfig(max_tokens=16, sampling=SamplingConfig(greedy=True), eos_token_id=256),
        speculator_config=spec,
        tokenizer_vocab_size=vocab,
    )


def hybrid_config() -> ModelConfig:
    """Attention + DeltaNet, like the hybrid Qwen3.5-style stacks.

    Speculation is disabled because the simplified DeltaNet state cannot be
    rewound for a rejected branch (the real engine has ``gdn/tree_verify``).
    """
    model_dim = 64
    head_dim = 16
    vocab = 258
    layers = [
        _attention_layer(model_dim, head_dim),
        TransformerLayerConfig(
            mixer=DeltaNetConfig(num_heads=4, head_dim=head_dim, conv_kernel_size=4),
        ),
    ]
    transformer = TransformerConfig(model_dim=model_dim, hidden_dim=128, layer_configs=layers)
    decoder = DecoderConfig(vocab_size=vocab, transformer_config=transformer)
    return ModelConfig(
        decoder_config=decoder,
        generation_config=GenerationConfig(max_tokens=16, sampling=SamplingConfig(greedy=True), eos_token_id=256),
        speculator_config=None,
        tokenizer_vocab_size=vocab,
    )
