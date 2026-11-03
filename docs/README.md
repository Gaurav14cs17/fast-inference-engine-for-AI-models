# Notation and conventions

This folder contains the cross-cutting notes that apply to every chapter. Each
chapter folder also has its own `README.md` with the full derivations for its
files.

## Tensor shapes

| Symbol | Meaning | Typical shape |
|---|---|---|
| `B` | batch size / number of tree nodes in a verify batch | scalar |
| `T` | sequence length (tokens) | scalar |
| `d` | model dimension / hidden size | scalar |
| `H` | query heads | scalar |
| `H_kv` | key/value heads (GQA groups) | scalar |
| `d_h` | head dimension | scalar (`d = H·d_h`) |
| `V` | vocabulary size | scalar |
| `F` | MLP hidden dimension | scalar |
| `L` | number of transformer layers | scalar |
| `X` | activations | `[T, d]` |
| `W` | weight matrix | `[out, in]` |
| `Q, K, V` | attention tensors | `[T, H, d_h]`, `[T, H_kv, d_h]` |
| `S` | recurrent state (DeltaNet) | `[H, d_h, d_h]` |
| `C` | KV cache | `[capacity, H_kv, d_h]` |

## Core operators

$$
\operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}},
\qquad
\sigma(z) = \frac{1}{1+e^{-z}},
\qquad
\operatorname{silu}(z) = z\,\sigma(z).
$$

Elementwise product is `⊙`. L2 norm is `‖x‖₂ = sqrt(Σ x_i²)`.

## DAG definition

The study book is a directed acyclic graph (DAG) where:

- each vertex is a numbered file;
- an edge `u → v` exists when file `v` imports an alias from file `u`;
- the number is a topological order, so every edge points from a smaller number
  to a larger one.

A topological order exists iff the graph has no directed cycle. The loader
executes files in numeric order, and `tools/check_dag.py` tests the condition

$$
\forall (u \to v):\quad \operatorname{num}(u) < \operatorname{num}(v).
$$

## Complexity summary

### Prefill (prompt of length `T`)

| Operation | Time | Memory |
|---|---|---|
| Linear `[T,d] × [d,F]` | `O(T d F)` | `O(dF)` weights |
| Full attention scores | `O(T² d)` | `O(T²)` scores |
| Causal attention | `O(T² d / 2)` | `O(T²)` scores |
| KV cache write | `O(T H_kv d_h)` | `O(T H_kv d_h)` |

### Decode (one token, context `T`)

| Operation | Time | Memory traffic |
|---|---|---|
| GEMV weights | `O(d F)` | `O(dF)` bytes read |
| Attention | `O(T H d_h)` | `O(T H_kv d_h)` cache |
| KV cache append | `O(H_kv d_h)` | one row per layer |

### Speculative decode

If a draft proposes `D` tokens and the target accepts `a` of them per target
forward, the token yield per target pass is `1 + a` (the `1` is the bonus
token). The ideal speedup is

$$
\text{speedup} \approx \frac{1 + \mathbb{E}[a]}{1 + c},
$$

where `c` is the draft cost expressed as a fraction of one target forward.

## Numerical stability

- Softmax subtracts `max(z)` before `exp` (the result is mathematically
  identical and prevents overflow).
- RMSNorm uses `eps` inside the square root.
- Sampling filters use `-inf` for masked logits, so their probability is zero.
- Quantization scales are clamped to at least `1` when all values are zero.

## Reading paths

1. **Fast path**: `chapter_01_foundations/README.md` → `chapter_03_model_blocks/README.md` → `chapter_05_runtime/README.md`.
2. **Full path**: chapters 01 → 05 in order.
3. **Speculation path**: `chapter_04_generation/README.md` then `chapter_05_runtime/README.md` (stream verification).
