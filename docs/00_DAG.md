# uzu → Python study port (DAG order)

This is a compact, runnable Python translation of the architecture described in
`crates/uzu-engine`. It is **not** a bit-exact port and it does not include Metal,
MXU, Slang, or the real trellis/QTIP codec. Everything is NumPy so you can read
one file at a time.

The number is the topological (DAG) order. Files live in topic folders but keep
their number:

```
chapter_01_foundations/   01-04    foundation
chapter_02_configuration/ 05-07    config, quantization, safetensors
chapter_03_model_blocks/  08-19    the model graph
chapter_04_generation/    20-25    trie, sampling, DFlash, Weaver
chapter_05_runtime/       26-29    tokenizer, decoder, stream, engine
```

```text
01  chapter_01_foundations/01_dtypes.py           dtypes
02  chapter_01_foundations/02_buffer.py           buffers + allocator + command buffer
03  chapter_01_foundations/03_kernels.py          NumPy reference kernels
04  chapter_01_foundations/04_backend.py          CPU context / kernel table
05  chapter_02_configuration/05_config.py         all model/config dataclasses
06  chapter_02_configuration/06_quantization.py   int4/int8 groups, RHT, toy trellis
07  chapter_02_configuration/07_safetensors.py    safetensors + ParameterTree
08  chapter_03_model_blocks/08_linear.py         LinearMatmul / RHT wrapper / readout
09  chapter_03_model_blocks/09_embedding.py      token + per-layer embedding
10  chapter_03_model_blocks/10_normalization.py  RMSNorm + residual shortcuts
11  chapter_03_model_blocks/11_mlp.py            SwiGLU / GeGLU
12  chapter_03_model_blocks/12_rope.py           precomputed RoPE
13  chapter_03_model_blocks/13_kv_cache.py       KV cache + accept/compact
14  chapter_03_model_blocks/14_attention.py      GQA attention
15  chapter_03_model_blocks/15_short_conv.py     depthwise causal conv
16  chapter_03_model_blocks/16_delta_net.py      gated delta-rule linear attention
17  chapter_03_model_blocks/17_mixer.py          mixer dispatch
18  chapter_03_model_blocks/18_transformer_layer.py
19  chapter_03_model_blocks/19_transformer.py    stack + state + hidden features
20  chapter_04_generation/20_trie.py       token trie + grammar mask
21  chapter_04_generation/21_batch_topology.py  tree positions + ancestor mask
22  chapter_04_generation/22_sampling.py   sampling + PRng
23  chapter_04_generation/23_dflash.py     block draft model
24  chapter_04_generation/24_weaver.py     tree expansion/pruning
25  chapter_04_generation/25_speculator.py DFlash + Weaver glue
26  chapter_05_runtime/26_tokenizer.py     byte tokenizer + chat template
27  chapter_05_runtime/27_decoder.py       embeddings -> transformer -> logits
28  chapter_05_runtime/28_stream.py        prefill / decode / tree verify
29  chapter_05_runtime/29_engine.py        Engine / LanguageModel / ChatSession
```

Import rule: **a module may only import modules with a smaller number**.
`tools/check_dag.py` enforces it. Because Python identifiers cannot start with a
digit, each topic folder has an `__init__.py` that loads its numbered files and
registers clean aliases:

```
chapter_01_foundations/01_dtypes.py       -> study_book.chapter_01_foundations.dtypes
chapter_01_foundations/03_kernels.py      -> study_book.chapter_01_foundations.kernels
chapter_02_configuration/05_config.py     -> study_book.chapter_02_configuration.config
chapter_03_model_blocks/14_attention.py  -> study_book.chapter_03_model_blocks.attention
chapter_04_generation/24_weaver.py -> study_book.chapter_04_generation.weaver
chapter_05_runtime/29_engine.py    -> study_book.chapter_05_runtime.engine
```

## Mapping to the Rust source

All paths below are relative to `crates/uzu-engine/src` in the upstream
repository: **<https://github.com/trymirai/uzu>**. (What is *not* mapped here
is listed in `../docs/01_COVERAGE.md`.)

| # | Python file | Mirrors |
|---|---|---|
| 01 | `chapter_01_foundations/01_dtypes.py` | `src/data_type.rs` |
| 02 | `chapter_01_foundations/02_buffer.py` | `backends/common/buffer`, `backends/common/allocator` |
| 03 | `chapter_01_foundations/03_kernels.py` | `backends/cpu/kernel/*`, `kernel/generated/*` |
| 04 | `chapter_01_foundations/04_backend.py` | `backends/common/backend`, `backends/metal/context.rs` |
| 05 | `chapter_02_configuration/05_config.py` | `src/config/*` |
| 06 | `chapter_02_configuration/06_quantization.py` | `config/weight_matrix/*`, `gpu_types/trellis.rs` |
| 07 | `chapter_02_configuration/07_safetensors.py` | `src/safetensors.rs` |
| 08 | `chapter_03_model_blocks/08_linear.py` | `encodable_block/linear/*` |
| 09 | `chapter_03_model_blocks/09_embedding.py` | `encodable_block/embedding.rs` |
| 10 | `chapter_03_model_blocks/10_normalization.py` | `encodable_block/normalization.rs` |
| 11 | `chapter_03_model_blocks/11_mlp.py` | `encodable_block/mlp/dense.rs` |
| 12 | `chapter_03_model_blocks/12_rope.py` | `encodable_block/mixer/attention/rope.rs` |
| 13 | `chapter_03_model_blocks/13_kv_cache.py` | `encodable_block/mixer/attention/state.rs` |
| 14 | `chapter_03_model_blocks/14_attention.py` | `encodable_block/mixer/attention/mod.rs` |
| 15 | `chapter_03_model_blocks/15_short_conv.py` | `encodable_block/mixer/short_conv.rs` |
| 16 | `chapter_03_model_blocks/16_delta_net.py` | `encodable_block/mixer/delta_net.rs` |
| 17 | `chapter_03_model_blocks/17_mixer.py` | `encodable_block/mixer/mod.rs` |
| 18 | `chapter_03_model_blocks/18_transformer_layer.py` | `encodable_block/transformer_layer.rs` |
| 19 | `chapter_03_model_blocks/19_transformer.py` | `encodable_block/transformer.rs` |
| 20 | `chapter_04_generation/20_trie.py` | `src/trie.rs`, `gpu_types/trie.rs` |
| 21 | `chapter_04_generation/21_batch_topology.py` | `encodable_block/batch_topology.rs` |
| 22 | `chapter_04_generation/22_sampling.py` | `encodable_block/sampling/*` |
| 23 | `chapter_04_generation/23_dflash.py` | `encodable_block/dflash.rs` |
| 24 | `chapter_04_generation/24_weaver.py` | `encodable_block/weaver.rs` |
| 25 | `chapter_04_generation/25_speculator.py` | `speculators/dflash_tfm.rs` |
| 26 | `chapter_05_runtime/26_tokenizer.py` | tokenizer / chat template (legacy `hanashi`) |
| 27 | `chapter_05_runtime/27_decoder.py` | `encodable_block/decoder.rs` |
| 28 | `chapter_05_runtime/28_stream.py` | `engine/language_model/stream/stream.rs` |
| 29 | `chapter_05_runtime/29_engine.py` | `engine/mod.rs`, `engine/language_model/mod.rs` |

## Dataflow (what to read in what order)

```text
Engine.load_random(config)
  └─ ParameterTree (07)
       └─ Decoder (27)
            ├─ Embedding (09)
            ├─ Transformer (19)
            │    └─ TransformerLayer (18)
            │         ├─ Normalization (10)
            │         ├─ Mixer (17)
            │         │    ├─ Attention (14) ── RoPE (12) + KVCache (13)
            │         │    ├─ DeltaNet (16)
            │         │    └─ ShortConv (15)
            │         └─ DenseMlp (11)
            └─ Readout (08)

GenerationStream (28)
  ├─ prefill: Decoder.encode -> Sampling (22)
  ├─ speculate: Speculator (25)
  │    ├─ DFlash (23) -> draft logits
  │    └─ Weaver (24) -> proposal tree
  ├─ verify: BatchTopology (21) ancestor mask -> Decoder.encode
  └─ accept: KVCache.accept -> next token
```

## What is intentionally simplified

- **Metal/MXU**: `chapter_01_foundations/04_backend.py` is a CPU context. The real engine selects
  `GemmEngine::Mxu` on M5+ and uses `supports_mxu` / `MTLResidencySet`.
- **Quantization**: real 4/8-bit group layouts, RHT and trellis-coded weights are
  represented by a small NumPy codec in `chapter_02_configuration/06_quantization.py`.
- **DeltaNet / GDN**: a per-token recurrent loop, not the chunked/tree-verify
  Metal kernels.
- **Speculation**: the tree is built and verified in Python. One target forward
  sees the whole tree with an ancestor mask, the same idea as
  `ancestor_attention.metal`.
- **Weights**: the demo uses random weights; real weights come from a `lalamo`
  converted safetensors file (see [trymirai/lalamo](https://github.com/trymirai/lalamo)
  and the upstream engine at <https://github.com/trymirai/uzu>).
