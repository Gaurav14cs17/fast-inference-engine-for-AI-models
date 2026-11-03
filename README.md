# fast-inference-engine-for-AI-models

A compact NumPy reimplementation of the **fast-inference-engine-for-AI-models** inference engine architecture,
organized as a numbered study book. It is meant for studying the code, not for
production inference.

The real engine is Rust + hand-written Metal/Slang kernels. This port keeps the
same structure (buffers, backend, linear, attention, KV cache, DFlash, Weaver,
stream, engine) and replaces the GPU kernels with readable NumPy functions.

## Study-book layout

```text
python_study/
├── pyproject.toml
├── README.md
├── study_book/
│   ├── __init__.py                      # loads chapters in DAG order
│   ├── _loader.py                       # 01_dtypes.py -> chapter_01_foundations.dtypes
│   ├── chapter_01_foundations/          # 01-04
│   │   ├── __init__.py
│   │   ├── 01_dtypes.py
│   │   ├── 02_buffer.py
│   │   ├── 03_kernels.py
│   │   └── 04_backend.py
│   ├── chapter_02_configuration/        # 05-07
│   │   ├── __init__.py
│   │   ├── 05_config.py
│   │   ├── 06_quantization.py
│   │   └── 07_safetensors.py
│   ├── chapter_03_model_blocks/         # 08-19
│   │   ├── __init__.py
│   │   ├── 08_linear.py
│   │   ├── 09_embedding.py
│   │   ├── 10_normalization.py
│   │   ├── 11_mlp.py
│   │   ├── 12_rope.py
│   │   ├── 13_kv_cache.py
│   │   ├── 14_attention.py
│   │   ├── 15_short_conv.py
│   │   ├── 16_delta_net.py
│   │   ├── 17_mixer.py
│   │   ├── 18_transformer_layer.py
│   │   └── 19_transformer.py
│   ├── chapter_04_generation/           # 20-25
│   │   ├── __init__.py
│   │   ├── 20_trie.py
│   │   ├── 21_batch_topology.py
│   │   ├── 22_sampling.py
│   │   ├── 23_dflash.py
│   │   ├── 24_weaver.py
│   │   └── 25_speculator.py
│   └── chapter_05_runtime/              # 26-29
│       ├── __init__.py
│       ├── 26_tokenizer.py
│       ├── 27_decoder.py
│       ├── 28_stream.py
│       └── 29_engine.py
├── tools/
│   ├── check_dag.py                     # imports may only go forward
│   └── run_demo.py                      # tiny random model + speculation demo
└── docs/
    └── 00_DAG.md                        # full order + Rust source mapping
```

Each numbered file keeps its number prefix, e.g.
`study_book/chapter_03_model_blocks/14_attention.py`. Since Python module names
cannot start with a digit, `study_book/__init__.py` registers them under clean
aliases:

```python
from study_book.chapter_03_model_blocks.attention import Attention
from study_book.chapter_05_runtime.engine import Engine
```

## Run

From `python_study/`:

```bash
python tools/check_dag.py
python tools/run_demo.py
```

Optional editable install:

```bash
pip install -e .
python -c "import study_book; print(study_book.__all__)"
```

The demo uses **random weights**, so the text is gibberish. What it demonstrates
is the dataflow: embeddings → transformer layers → KV cache → sampling, plus the
DFlash/Weaver proposal tree and one-pass tree verification.

## Reading order

Read the chapters and numbered files in order:

```text
Chapter 01  foundations    01-04   dtypes, buffers, kernels, backend
Chapter 02  configuration  05-07   config, quantization, weight loading
Chapter 03  model blocks   08-19   linear .. transformer
Chapter 04  generation     20-25   trie .. speculator
Chapter 05  runtime        26-29   tokenizer, decoder, stream, engine
```

See `docs/00_DAG.md` for the mapping to `crates/uzu-engine/src`.

## Math notes

Every folder and chapter has its own `README.md` with the formulas and
derivations for that part of the engine:

| Folder | Math notes |
|---|---|
| `study_book/` | global dataflow, parameter counts, complexity |
| `chapter_01_foundations/` | IEEE-754/BF16, quantization error, softmax, RoPE, attention, Hadamard |
| `chapter_02_configuration/` | group quantization, RHT, trellis/DPCM, bits per weight |
| `chapter_03_model_blocks/` | RMSNorm, SwiGLU, RoPE, GQA, KV-cache memory, DeltaNet recurrence |
| `chapter_04_generation/` | tree masks, sampling filters, DFlash, Weaver scoring, expected speedup |
| `chapter_05_runtime/` | decoder equations, prefill/decode cost, speculative acceptance |
| `docs/` | notation, shape conventions, DAG definition |
| `tools/` | DAG check algorithm, demo math |

Conventions used everywhere:

- uppercase matrices `X, W, Q, K, V`; lowercase vectors `x, q, k, v`;
- `T` = sequence length, `B` = batch, `d` = model dimension, `H` = heads,
  `d_h` = head dimension, `V` = vocabulary;
- `⊙` = elementwise product, `σ` = sigmoid, `‖·‖` = L2 norm;
- tensors are `[T, d]` unless a shape is stated.

## Important simplifications

- NumPy executes eagerly; the real engine encodes many kernels into one Metal
  command buffer.
- No MXU (`supports_mxu` is only a flag), no Metal residency sets, no Slang.
- Quantization is conceptual: group int4/int8, RHT and a toy trellis codec.
- DeltaNet uses a per-token recurrence instead of chunked/tree-verify kernels.
- The live random-weight draft rarely matches the target, so live speculation
  accepts 0 draft tokens. The synthetic verification demo in `run_demo.py`
  shows the accept path producing 3 draft tokens + 1 bonus token.
