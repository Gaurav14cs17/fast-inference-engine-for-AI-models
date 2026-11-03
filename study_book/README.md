# study_book — the whole engine in DAG order

`study_book` is the importable package. Each chapter is a folder; each file in a
chapter is numbered. The numbers are a topological order, so a file only imports
files with a smaller number.

## Chapters

| Chapter | Files | Content |
|---|---|---|
| `chapter_01_foundations` | 01-04 | data types, buffers, reference kernels, backend |
| `chapter_02_configuration` | 05-07 | config, quantization, safetensors |
| `chapter_03_model_blocks` | 08-19 | linear, embedding, norm, MLP, RoPE, cache, attention, DeltaNet, transformer |
| `chapter_04_generation` | 20-25 | trie, batch topology, sampling, DFlash, Weaver, speculator |
| `chapter_05_runtime` | 26-29 | tokenizer, decoder, stream, engine |

Each chapter has its own `README.md` with the detailed maths. This page gives
the global equations that connect the chapters.

## The forward pass

Let `x = (x_1, …, x_T)` be token ids and `E ∈ R^{V×d}` the embedding table.

$$
H^{(0)}_t = E[x_t] \in \mathbb{R}^{d}.
$$

For layer `ℓ = 0 … L-1`, with pre-norm residuals:

$$
\begin{aligned}
A^{(\ell)} &= \operatorname{Mixer}^{(\ell)}\!\left(\operatorname{Norm}(H^{(\ell)})\right),\\
H^{(\ell+\frac12)} &= H^{(\ell)} + A^{(\ell)},\\
M^{(\ell)} &= \operatorname{MLP}^{(\ell)}\!\left(\operatorname{Norm}(H^{(\ell+\frac12)})\right),\\
H^{(\ell+1)} &= H^{(\ell+\frac12)} + M^{(\ell)}.
\end{aligned}
$$

The final logits are

$$
z_t = W_{\text{readout}}\,\operatorname{Norm}(H^{(L)}_t), \qquad
p(x_{t+1}=i \mid x_{\le t}) = \operatorname{softmax}(z_t)_i.
$$

In the Python code the residual stream lives in `shortcut`; a layer returns only
`M^{(\ell)}`, and the next normalization adds it to `shortcut`. This mirrors
`10_normalization.py` and `18_transformer_layer.py`.

## Three mixers

The `Mixer` slot in a layer can be:

1. **Attention** (`14_attention.py`) — content-based lookup with a KV cache.
2. **DeltaNet** (`16_delta_net.py`) — fixed-size recurrent state.
3. **ShortConv** (`15_short_conv.py`) — depthwise causal convolution.

A hybrid model interleaves them. Attention has `O(T²)` prefill cost and a
growing cache; DeltaNet has `O(T d_h²)` prefill cost and `O(H d_h²)` state.
That is why hybrid stacks are attractive for long contexts.

## Parameter count

For one attention layer with model dimension `d`, query heads `H`, KV heads
`H_kv`, head dimension `d_h = d/H`:

$$
\underbrace{d^2}_{\text{Q}} + \underbrace{2 d H_{kv} d_h}_{\text{K,V}} + \underbrace{d^2}_{\text{O}}.
$$

For full multi-head attention `H_kv = H`, this is `4d²`; for grouped-query
attention with `H_kv < H`, the K/V term shrinks.

For a dense SwiGLU MLP with hidden dimension `F`:

$$
\underbrace{2dF}_{\text{gate + up}} + \underbrace{Fd}_{\text{down}} = 3dF.
$$

The total is approximately

$$
N_{\text{params}} \approx Vd + L\left(4d^2 + 3dF + 2d\right)
$$

for MHA, plus embeddings and norms. Quantizing weights to `b` bits makes the
file size roughly `N_params · b / 8` bytes.

## Compute and memory

### Prefill (`T` tokens at once)

$$
\text{FLOPs}_{\text{linear}} \approx 2T\,N_{\text{params}}.
$$

Attention adds `O(T² d)` (causal: about half). Prefill is compute-bound on a GPU.

### Decode (one token, context `T`)

Each weight is read once per token:

$$
\text{bytes per token} \approx \frac{N_{\text{params}} \cdot b}{8}, \qquad
t_{\text{token}} \gtrsim \frac{\text{bytes}}{BW}.
$$

That is why decode is bandwidth-bound and why quantization helps so much:
halving the weight bits roughly halves the lower bound on per-token time.

### KV cache

Per token per layer, for K and V:

$$
\text{bytes} = 2 \cdot H_{kv} \cdot d_h \cdot \text{sizeof(dtype)}.
$$

For a `T`-token context and `L` layers, multiply by `T·L`. GQA reduces `H_kv`
and therefore the cache.

## Speculative decoding

The target model is expensive; a small draft model proposes `D` tokens, and one
target forward verifies them all:

$$
\text{tokens per target forward} = 1 + a,
$$

where `a` is the number of accepted draft tokens (the `+1` is the bonus token).
If `c` is the draft cost as a fraction of a target forward, the ideal speedup is

$$
\text{speedup} \approx \frac{1 + \mathbb{E}[a]}{1 + c}.
$$

`chapter_04_generation/README.md` derives DFlash and Weaver; the stream and
accept rule are in `chapter_05_runtime/README.md`.

## Loading and aliases

`__init__.py` imports the five chapter packages in order. Each chapter's
`__init__.py` calls `load_numbered`, which registers

```text
chapter_03_model_blocks/14_attention.py -> study_book.chapter_03_model_blocks.attention
```

so the numbered files remain visible while imports use readable names.

## Where to start

- New to the engine: `chapter_01_foundations/README.md`.
- Want the model maths: `chapter_03_model_blocks/README.md`.
- Want the speed tricks: `chapter_04_generation/README.md` and
  `chapter_05_runtime/README.md`.
- Want the exact order: `../docs/00_DAG.md`.
