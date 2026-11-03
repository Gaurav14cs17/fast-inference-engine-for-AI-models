# Deep check — what this port covers vs upstream uzu

Upstream: **<https://github.com/trymirai/uzu>** (Rust, MIT license,
[docs](https://docs.trymirai.com)). This document records a file-by-file
comparison between this NumPy study book and the upstream `main` branch
(checked 2026-11-03: 299 `.rs` files under `crates/uzu-engine/src`, plus
`crates/uzu-engine-macros`, a `crates/legacy/*` crate set and `apps/*`).

State of the study book at check time:

- `python tools/check_dag.py` — **OK** (29 numbered files, 98 import edges,
  DAG only goes forward).
- `python tools/run_demo.py` — **runs end-to-end** (attention model, hybrid
  attention + DeltaNet model, quantization primitives, tree verification).

`docs/00_DAG.md` lists the 29 modules that *are* ported and the Rust files
they mirror. This document is the complement: what is **not** here.

## Summary

| Area | Status |
|---|---|
| dtypes, buffers, backend abstraction, kernels | covered (CPU/NumPy only) |
| config, group quantization, RHT, trellis | covered conceptually (toy codecs) |
| linear, embedding, per-layer embedding, norm, SwiGLU MLP | covered |
| RoPE | one variant only (unscaled); scaling variants missing |
| KV cache, GQA attention (QK-norm, sinks, gate, sliding window, KV sharing) | covered (dense cache) |
| short conv, DeltaNet (per-token recurrence), mixer dispatch | covered |
| transformer layer/stack, decoder, readout | covered |
| trie + grammar mask, batch topology, sampling, DFlash, Weaver, speculator | covered |
| tokenizer | byte-level only; real token codecs missing |
| stream (prefill / decode / tree verify), engine, chat session | covered |
| MoE, classifier/decision models, grammar engine, tool calls | **missing** |
| Metal, WebGPU, sparse KV storage, int8 attention | **missing** (partly intentional) |
| server, CLI, downloads, bindings, telemetry, apps | **missing** (out of scope) |

## 1. Model architecture and config — missing

| Missing feature | Upstream files (`crates/uzu-engine/src/…`) | Study book today |
|---|---|---|
| **Mixture of Experts (MoE)** — expert routing, top-k gate, expert combine | `config/mlp/mixture_of_experts.rs`, `config/mlp/routing_function/softmax_routing.rs`, `backends/cpu/kernel/moe/{router_top_k,gemv,combine}.rs` | `11_mlp.py` implements dense SwiGLU/GeGLU only; `DenseMLPConfig` has no expert fields |
| **RoPE scaling variants** — Llama linear scaling, LongRoPE, YaRN, unscaled | `config/rope/{llama_rope,longrope,yarn_rope,linear_scaling_rope,unscaled_rope}.rs` | `12_rope.py` has a single precomputed RoPE (base + max length, i.e. the unscaled variant) |
| **Classifier and decision model types** | `config/model/{classifier_model,decision_model}.rs`, `config/classifier.rs`, `encodable_block/{classifier,decision,prediction_head}.rs`, `engine/{classifier_model,decision_model}.rs` | only the generation/language-model path exists (`27_decoder.py`, `29_engine.py`) |
| **Convolution token mixer** as a standalone mixer | `config/token_mixer/convolutions.rs`, `encodable_block/convolution.rs` | a depthwise `15_short_conv.py` exists, but the separable causal conv block/config is not modeled |
| **Weaver as a model layer** | `encodable_block/weaver_layer.rs` | `24_weaver.py` implements the tree builder only |
| **Standalone embedding table block** | `encodable_block/embedding_table.rs` | embedding is folded into `09_embedding.py` |
| **Standalone gated-act-mul block** | `encodable_block/mlp/gate_act_mul.rs` | fused inside the SwiGLU MLP |
| **KV storage modes** — cache mode selection, shared/sparse storage | `encodable_block/mixer/attention/{kv_storage,mode}.rs`, `backends/{cpu,metal,webgpu}/buffer/sparse.rs` | `13_kv_cache.py` is a single dense ring buffer; `is_kv_sharing`/`kv_source_layer_index` exist but no storage abstraction |
| **Weight-matrix specs** — d4s4, MLX, hybrid, full precision, real trellis/QTIP | `config/weight_matrix/{d4s4_spec,mlx_spec,hybrid_spec,full_precision_spec,int_spec}.rs`, `gpu_types/trellis.rs`, `backends/metal/kernel/matmul/gemm/trellis.rs` | `06_quantization.py` has toy int4/int8 group, RHT and DPCM trellis codecs (intentional simplification) |
| **Token codecs** — real tokenizer integration and raw-text codec | `config/token_codec/{chat_codec,raw_text_codec}.rs`, legacy `hanashi` crate | `26_tokenizer.py` is a byte-level tokenizer with a fixed chat template |

Partially covered (present but reduced): activation variants
(`config/activation/{gelu,silu,identity}.rs` vs one `activation: str` field),
clipping (`config/clipping.rs` vs per-MLP clip fields), attention int8 /
two-pass / GEMM-grouped dispatch (documented in `14_attention.py` but executed
as one NumPy function).

## 2. Sampling and logits — missing

| Missing feature | Upstream files | Study book today |
|---|---|---|
| **Repetition penalty** | `backends/cpu/kernel/sampling/repetition_penalty.rs` | `22_sampling.py` has temperature / top-k / top-p / min-p / greedy, no penalty term |
| **Gumbel (key/value) sampling** | `encodable_block/sampling/gumbel.rs` | not implemented; `sample_from_logits` covers greedy + filtered sampling |
| **Unified GPU sampling + context ring update** | `backends/cpu/kernel/sampling/{unified_sampling,context_ring_update}.rs` | sampling runs eagerly per row in NumPy |
| **Logit transform block** (final-logits processing, e.g. softcapping) | `backends/cpu/kernel/logit_transform/*`, `encodable_block` wiring | no equivalent |

## 3. Grammar, structured output, tool calls — missing

| Missing feature | Upstream files | Study book today |
|---|---|---|
| **Grammar engine** — JSON-schema-constrained generation (upstream README: "chat with structured output") | `engine/language_model/grammar/{mod,grammar,config,engagement,error}.rs` | `20_trie.py` builds a token trie and a static `grammar_mask`; there is no grammar compiler or engagement state machine |
| **Tool calls** — stream parsing into structured tool invocations | legacy `token-stream-parser`, `json-transform` crates | not modeled |

## 4. Runtime and engine — missing

| Missing feature | Upstream files | Study book today |
|---|---|---|
| **Language-model session state** management | `engine/language_model/state.rs` | `ChatSession` keeps one `GenerationStream`; no state object |
| **Model download / registry** | legacy `download-manager`, `mock-registry`, `utils/file.rs` | `29_engine.py` loads random weights or a local safetensors file only |
| **OpenAI-compatible HTTP server** (`POST /v1/chat/completions`, `GET /v1/models`) | `cargo run --release -p cli -- server` | none |
| **CLI / GUI apps** | `cargo run --release -p cli`, `apps/{mirai-gui,playground,uzu-pi-extension}` | none |
| **Cloud backend** | legacy `backend-remote` | none |
| **Benchmarks + Apple telemetry** | `cli bench`, legacy `keisoku` (IOReport/SMC power, energy, DRAM counters via `kanka`'s dlsym bindings) | none |
| **Cross-platform OS layer** (tokio on native, OPFS/wasm in browser) | legacy `kiban` | none |
| **Language bindings** — Swift (uniffi), Python (pyo3), TypeScript (napi-rs) | `crates/legacy/uzu/bindings/*`, `nagare`, `shoji`, `bindings-types` | none (pure-Python package) |

## 5. Backends and kernels — missing

- **Metal backend**: 70 hand-written `.metal` shaders under
  `backends/metal/kernel/**`, MXU (`GemmEngine::Mxu` on M5+), heaps,
  `MTLResidencySet`, capture (`capture.rs`), timestamp recorder/spans,
  `metal_extensions/*`. The study book's `04_backend.py` is a CPU context —
  this is a documented intentional simplification.
- **WebGPU backend**: `backends/webgpu/*` plus `utils/{wasm_file,wasm_future}.rs`
  (browser/wasm target, native OPFS storage via `kiban`). Entirely absent here.
- **Traceable-correctness tooling**: `backends/common/capture_matcher.rs`,
  `timestamp_span.rs`, `device_capabilities.rs`.
- **GPU PRNG**: sampling RNG on device (`gpu_types/ring.rs`,
  `kernel/sampling/*`); the study book uses a CPU splitmix64-style `PRng`.
- **Derive macros**: `crates/uzu-engine-macros` (config/block boilerplate).

## 6. Gaps inside the study port itself

1. **LICENSE** — the GitHub repository has an MIT license; it is now also in
   the local working copy.
2. **No unit tests.** Only `tools/check_dag.py` (import-order check) and the
   random-weight demo run. Upstream has a `tests` suite and CI
   (`.github/workflows/tests.yml`).
3. **No CI** (no `.github/` folder at all).
4. **Stale path names**: README/docstrings say `python_study/`, but the actual
   directory is `fast-inference-engine-for-AI-models/` (fixed alongside this
   document).
5. **No upstream reference** anywhere in README, docs, or `pyproject.toml`
   (fixed alongside this document — see the References section of `README.md`).

## Not missing — intentionally simplified

These are documented in `docs/00_DAG.md` § "What is intentionally simplified"
and are *by design*, not gaps: eager NumPy execution instead of one Metal
command buffer; no MXU/Slang/residency sets; conceptual quantization codecs;
per-token DeltaNet recurrence instead of chunked/tree-verify kernels; random
weights instead of `lalamo`-converted ones.

## References

- This study port: <https://github.com/Gaurav14cs17/fast-inference-engine-for-AI-models>
- Upstream uzu: <https://github.com/trymirai/uzu>
- uzu docs: <https://docs.trymirai.com>
- lalamo (model export): <https://github.com/trymirai/lalamo>
