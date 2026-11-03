# Chapter 03 — model blocks

Files: `08_linear.py` … `19_transformer.py`.

This is the mathematical core: linear algebra, normalization, MLP, RoPE, KV
cache, attention, short convolution, DeltaNet, and the transformer stack.

---

## 08 — linear layers

For `X ∈ R^{T×d_in}`, `W ∈ R^{d_out×d_in}`, `b ∈ R^{d_out}`:

$$
Y = XW^\top + \mathbf{1}b^\top.
$$

FLOPs: `2 T d_in d_out` (multiply and add). The weight memory is
`d_in d_out · sizeof(dtype)`.

The `RHTLinearWrapper` computes

$$
Y = \left(\frac{1}{\sqrt d} H D X\right) W^\top + b,
$$

which is algebraically the same linear map but with an incoherent input. The
`RHT` is orthogonal, so a full-precision model would be unchanged; it only
changes the quantization error.

`UntiedReadout` is the same formula with `d_out = V`.

---

## 09 — embeddings

Token embedding is a table lookup:

$$
H_t = E[x_t], \qquad E \in \mathbb{R}^{V×d}.
$$

A per-layer embedding adds a second table:

$$
H_t^{(\ell)} = E[x_t] + P_{\ell}[x_t],
$$

where `P ∈ R^{L×V×d}`. The study version optionally projects the sum with a
linear map.

---

## 10 — normalization and residual stream

### RMSNorm

$$
\operatorname{RMSNorm}(x) =
\frac{x}{\sqrt{\frac1d \sum_i x_i^2 + \varepsilon}} \odot g + b.
$$

Unlike LayerNorm, there is no mean subtraction by default. The Rust kernel can
also do `subtract_mean=True`, which turns it into a LayerNorm-style normalization.

### Residual shortcuts

The layer uses a **pre-norm residual stream**. With `x` the layer input and
`h` the mixer/MLP output, the normalization performs one of:

$$
\text{Copy: } s \leftarrow x, \qquad
\text{Add: } s \leftarrow s + x, \qquad
\text{None: } s \leftarrow s.
$$

Then it returns `Norm(s)`. This is why the transformer layer only returns the
MLP output: the residual sum stays in `s` and is advanced by the next
normalization call. The final output norm does one last `Add` before the readout.

If `post_layer_scalar = ScaleResidualSum(v)`, the stored shortcut is multiplied
by `v`; if `ScaleOutput(v)`, the normalized output is multiplied by `v`. These
are the `PostLayerScalar` options.

---

## 11 — MLP (SwiGLU / GeGLU)

For input `x`, up-projection `W_up ∈ R^{2F×d}`, down-projection
`W_down ∈ R^{d×F}`:

$$
\begin{pmatrix} g \\ u \end{pmatrix} = W_{up} x,
\qquad
h = \operatorname{act}(g) \odot u,
\qquad
y = W_{down} h.
$$

The activation is usually SiLU:

$$
\operatorname{silu}(z) = z\,\sigma(z),
$$

or GELU (tanh approximation):

$$
\operatorname{GELU}(z) \approx \frac12 z\left(1 + \tanh\!\left(\sqrt{\frac2\pi}(z + 0.044715 z^3)\right)\right).
$$

Optional clipping applies `clip(g, -c, c)` before the activation and
`clip(u, -c, c)` before the product.

The parameter count is

$$
\underbrace{2dF}_{\text{up}} + \underbrace{Fd}_{\text{down}} = 3dF.
$$

FLOPs per token: `2·d·2F + 2·F·d = 6dF`.

---

## 12 — RoPE

For each head vector `x ∈ R^{d_h}` and position `p`, pair the coordinates
`(x_{2i}, x_{2i+1})`:

$$
\theta_i = p \cdot \text{base}^{-2i/d_h},
$$

$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix}
=
\begin{pmatrix} \cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}.
$$

Properties:

1. **Norm preservation** per pair.
2. **Relative position**: `R(p)^T R(q) = R(q-p)`, so `q_p · k_q` depends on
   `q-p`.
3. **No learned parameters**; only `base` and `d_h` matter.

The Python code precomputes `cos/sin` for the current batch positions. For a
speculative tree the positions are `context_length + depth`, not contiguous.

---

## 13 — KV cache

For each layer, store keys and values:

$$
K_{\text{cache}} \in \mathbb{R}^{T_{\max}×H_{kv}×d_h},
\qquad
V_{\text{cache}} \in \mathbb{R}^{T_{\max}×H_{kv}×d_h}.
$$

Memory per layer:

$$
M_{\text{layer}} = T_{\max} \cdot H_{kv} \cdot d_h \cdot \text{sizeof(dtype)} \cdot 2.
$$

For `L` layers and context `T`:

$$
M_{\text{KV}} = 2 L T H_{kv} d_h s.
$$

GQA with `H_kv = H/G` reduces this by a factor `G` compared with full MHA.
A 4-bit weight model usually keeps the KV cache in BF16/FP16, so long contexts
can be cache-bound even when weights are small.

`accept(indices)` keeps only the rows on the accepted path. If the last batch
appended `N` rows and `A` are accepted, the cache length changes by

$$
T \leftarrow T_{\text{prefix}} + |\text{accepted}|.
$$

---

## 14 — attention

### Projections

$$
Q = XW_Q^\top, \quad K = XW_K^\top, \quad V = XW_V^\top,
$$

with shapes `Q:[T,H,d_h]`, `K,V:[T,H_kv,d_h]`. Optional QK RMSNorm is applied per
head before RoPE:

$$
Q \leftarrow \operatorname{RMSNorm}_{d_h}(Q), \qquad
K \leftarrow \operatorname{RMSNorm}_{d_h}(K).
$$

### Scores

$$
S = \frac{QK^\top}{\sqrt{d_h}} \in \mathbb{R}^{H×T×T}.
$$

For GQA, repeat each KV head `H/H_kv` times along the head axis.

### Mask

Causal:

$$
M_{ij} = \begin{cases}0 & j \le i\\ -\infty & j > i\end{cases}
$$

Sliding window `w`: allowed iff `i-w < j ≤ i`. For a speculative tree, the mask is
the ancestor mask from chapter 04.

### Softmax and output

$$
P = \operatorname{softmax}(S + M),
\qquad
O = PV.
$$

### Sinks

A per-head sink logit `s_h` is appended as an extra column with value `0`:

$$
S' = [S,\; s_h], \qquad
P = \operatorname{softmax}(S'), \qquad
O = P_{:,:T} V.
$$

The sink absorbs probability mass but contributes zero to the output.

### Output gate

If `has_gate`, a gate vector `G ∈ R^{T×H×d_h}` is produced by the same fused
projection:

$$
O \leftarrow O \odot \sigma(G).
$$

### Output projection

$$
Y = O W_O^\top, \qquad O \text{ flattened to } [T,d].
$$

### Complexity

- Prefill scores: `O(T² H d_h) = O(T² d)`.
- Causal prefill: about half.
- Decode with context `T`: `O(T H d_h)`.
- Attention memory: `O(T²)` scores for prefill; the cache is `O(T H_kv d_h)`.

---

## 15 — short convolution

For channel `c` and kernel size `K`:

$$
y_{t,c} = \sum_{k=0}^{K-1} w_{c,k}\,x_{t-(K-1-k),c} + b_c.
$$

The state is the last `K-1` rows:

$$
\text{state} = x_{t-K+1:t}.
$$

The mixer is

$$
\text{out} = W_O\left(\operatorname{silu}(G) \odot \operatorname{Conv}(V)\right),
$$

where `(G,V)` come from a fused input projection. Memory state:
`(K-1)·d` per layer.

---

## 16 — DeltaNet (gated delta rule)

DeltaNet is linear attention with a **delta-rule** update. The state
`S ∈ R^{H×d_h×d_h}` is an associative memory: it maps keys to values. The
update is derived from minimizing the reconstruction error
`‖v_t - S^T k_t‖²` with a step size `β_t`:

$$
S_t = \alpha_t S_{t-1} + \beta_t\, k_t\left(v_t - S_{t-1}^\top k_t\right)^\top.
$$

Here:

- `α_t = σ(a_t)` is the decay (forget gate),
- `β_t = σ(b_t)` is the write strength,
- `k_t` is L2-normalized,
- `v_t - S_{t-1}^T k_t` is the prediction residual.

The output reads the memory with the query:

$$
o_t = S_t^\top q_t.
$$

Shapes: `k_t,q_t ∈ R^{d_h}`, `v_t ∈ R^{d_h}`, `S_t ∈ R^{d_h×d_h}` per head.

Complexity per token per head: `O(d_h²)` for both the update and the read. For a
sequence of `T` tokens:

$$
O(T H d_h^2)
$$

versus `O(T² H d_h)` for full attention. For long `T`, DeltaNet is cheaper and
uses `O(H d_h²)` memory instead of a growing cache.

The study implementation loops over tokens; the real engine uses a chunked
algorithm that processes blocks in parallel and a special tree-verify variant.

---

## 17 — mixer dispatch

The layer's mixer is one of the three blocks above. The interface is

$$
\operatorname{Mixer}: (X, \text{positions}, \text{state}, M) \mapsto Y.
$$

`mixer_speculation_supported` is `True` for attention and `False` for the
simplified DeltaNet/ShortConv states, because their recurrent state cannot be
rewound for a rejected branch in this port.

---

## 18 — transformer layer

With `x` the input and `s` the residual stream:

$$
\begin{aligned}
h &= \operatorname{Norm}(s + x) && \text{(Copy on layer 0, Add later)}\\
a &= \operatorname{Mixer}(h)\\
s &\leftarrow s + a\\
h &= \operatorname{Norm}(s)\\
m &= \operatorname{MLP}(h)\\
s &\leftarrow s + m\\
\text{return } m.
\end{aligned}
$$

The layer returns only `m`; the caller keeps `s`. The next layer's first
normalization performs the `s + m` step. At the very end, the output norm does

$$
y = \operatorname{Norm}(s + m_{\text{last}}).
$$

Optional `post_mixer_norm` / `post_mlp_norm` insert an extra normalization
before the residual add.

### Parameter count per layer

For full attention + SwiGLU:

$$
\underbrace{4d^2}_{\text{QKVO}} + \underbrace{3dF}_{\text{MLP}} + \underbrace{2d}_{\text{norms}}.
$$

GQA reduces the K/V term to `2 d H_kv d_h`.

---

## 19 — transformer stack

The stack applies `L` layers, precomputes RoPE once per batch, and can capture
features from selected layers:

$$
\text{feature}^{(\ell)} = s^{(\ell)} + m^{(\ell)}.
$$

The DFlash speculator consumes the features at `target_layer_ids`. This is why
the stream passes `hidden_feature_layer_indices`.

### State

`TransformerState` holds one mixer state per layer and a `context_length`. A
layer with `kv_source_layer_index` shares an earlier layer's KV cache
(read-only). `encode_accept(indices)` advances every state and increments the
context length by the number of accepted rows.

### Prefill vs decode

- Prefill: `T` tokens, attention scores `O(T²)`, big matmuls.
- Decode: one token (or a small tree batch), attention `O(T)`, GEMV weights.

### Output

Only the requested rows are normalized and projected to logits:

$$
z_{t} = W_{\text{readout}}\operatorname{Norm}(s_t + m_t).
$$

---

## Chapter summary

| File | Formula |
|---|---|
| `08_linear.py` | `Y = XW^T + b` |
| `09_embedding.py` | table lookup, per-layer add |
| `10_normalization.py` | RMSNorm, residual Copy/Add |
| `11_mlp.py` | `down(act(gate) ⊙ up)` |
| `12_rope.py` | rotation by `p·base^{-2i/d}` |
| `13_kv_cache.py` | `2 L T H_kv d_h s` bytes |
| `14_attention.py` | `softmax(QK^T/√d_h + M)V` |
| `15_short_conv.py` | causal depthwise conv |
| `16_delta_net.py` | `S_t = αS + βk(v - S^Tk)^T` |
| `17_mixer.py` | dispatch |
| `18_transformer_layer.py` | pre-norm residual equations |
| `19_transformer.py` | stack, state, features |
