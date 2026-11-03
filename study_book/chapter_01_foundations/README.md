# Chapter 01 — foundations

Files: `01_dtypes.py`, `02_buffer.py`, `03_kernels.py`, `04_backend.py`.

This chapter is the mathematical and memory foundation. Everything later is a
composition of these operations.

---

## 01 — data types

### IEEE-754 and BF16

A float32 has 1 sign bit, 8 exponent bits and 23 mantissa bits:

$$
x = (-1)^s \cdot 2^{e-127} \cdot (1 + m/2^{23}).
$$

BF16 keeps the 8 exponent bits and only 7 mantissa bits. It is the upper 16 bits
of the float32 bit pattern. With `u` the float32 bits,

$$
u_{\text{bf16}} = (u + \texttt{0x7FFF} + ((u \gg 16)\ \&\ 1))\ \&\ \texttt{0xFFFF0000}.
$$

The `+ ((u >> 16) & 1)` term implements round-to-nearest-even. The relative
precision is about `2^{-8} ≈ 0.4%`, but the exponent range is the same as float32
(no overflow at normal model scales).

### Symmetric int8 quantization

For a vector/row `x`, choose

$$
s = \frac{\max_i |x_i|}{127}, \qquad
q_i = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x_i}{s}\right), -127, 127\right),
\qquad
\hat{x}_i = s\,q_i.
$$

The quantization error is bounded by half a step:

$$
|x_i - \hat{x}_i| \le \frac{s}{2}
= \frac{\max_j |x_j|}{254}.
$$

This is the per-row int8 path used for activations (`A8`).

### Why BF16 and int8 together

BF16 is the accumulation/storage type; int8 is a bandwidth optimization. A
rough model of decode time is

$$
t \approx \frac{\text{bytes read}}{BW},
$$

so reading 8-bit weights instead of 16-bit halves the memory time (ignoring
dequantization math).

---

## 02 — buffers, allocator, command buffer

### Buffer size

For a tensor of shape `(n_1, …, n_k)` and element size `s` bytes:

$$
\text{bytes} = s \prod_{i=1}^k n_i.
$$

The `Buffer` class stores a NumPy array plus a `DataType`; `nbytes` is exactly
the formula above.

### Pool reuse

A pool caches free buffers by `(shape, dtype)`. If a request matches a cached
shape, allocation is `O(1)` and no new memory is touched. Peak memory is

$$
M_{\text{peak}} = \max_t \sum_{\text{live at } t} \text{bytes}.
$$

Reusing buffers keeps `M_peak` flat across tokens instead of growing with the
number of allocations. This is the same idea as the Rust block allocator.

### Command buffer

The real engine records many kernel dispatches, then submits once:

$$
\text{CPU overhead} \approx n_{\text{submit}} \cdot c_{\text{submit}},
$$

so batching `N` kernels into one command buffer reduces overhead from
`N·c_submit` to `c_submit`. The Python `CommandBuffer` executes eagerly but keeps
the same recording API (`debug_group`, `encode_copy`, `submit`).

---

## 03 — reference kernels

### Linear layer

For `X ∈ R^{T×d_in}`, `W ∈ R^{d_out×d_in}`, `b ∈ R^{d_out}`:

$$
Y = XW^\top + \mathbf{1}b^\top, \qquad Y \in \mathbb{R}^{T×d_{out}}.
$$

FLOPs (multiply + add counted as 2):

$$
\text{FLOPs} = 2\,T\,d_{in}\,d_{out}.
$$

### RMSNorm

With scale vector `g`, bias `b`, and epsilon `ε`:

$$
\mathrm{RMSNorm}(x) =
\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \varepsilon}} \odot g + b.
$$

If `subtract_mean=True`, replace `x` by `x - mean(x)` first (LayerNorm-style).
The sum of squares is computed in float32 even when the input is BF16, because
`Σ x_i²` easily loses precision in low precision.

### Softmax and log-softmax

$$
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

The numerically stable implementation is

$$
\mathrm{softmax}(z)_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \qquad m = \max_j z_j.
$$

The subtraction cancels in the ratio, so it changes nothing mathematically but
prevents `exp` overflow. Log-softmax is

$$
\log \mathrm{softmax}(z)_i = z_i - m - \log\sum_j e^{z_j - m}.
$$

### RoPE (rotary position embedding)

For a pair of coordinates `(x_{2i}, x_{2i+1})` at position `p`, define

$$
\theta_i = p \cdot \text{base}^{-2i/d}, \qquad
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix}
=
\begin{pmatrix} \cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}.
$$

This is a rotation by `θ_i`, so it preserves the pair norm:

$$
x_{2i}'^2 + x_{2i+1}'^2 = x_{2i}^2 + x_{2i+1}^2.
$$

The key property for attention: for positions `p` and `q`,

$$
R(p)^\top R(q) = R(q-p),
$$

so the dot product depends only on the **relative** offset `q-p`.

### Scaled dot-product attention

$$
\mathrm{Attn}(Q,K,V) =
\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V,
$$

where `M` is `0` for allowed keys and `-∞` for masked keys. The `1/√d_h` factor
keeps the logits variance roughly constant as `d_h` grows: if `q·k` has variance
`d_h`, dividing by `√d_h` gives unit variance.

For grouped-query attention (GQA), each KV head is shared by `H/H_kv` query
heads. This reduces both compute and cache size without changing the formula.

### Causal mask

For query position `i` and key position `j`:

$$
M_{ij} = \begin{cases}
0 & j \le i,\\
-\infty & j > i.
\end{cases}
$$

With a sliding window `w`:

$$
M_{ij} = 0 \iff i-w < j \le i.
$$

### Depthwise causal convolution

For channel `c` and kernel `w_c[0..K-1]`:

$$
y_{t,c} = \sum_{k=0}^{K-1} w_c[k]\,x_{t-(K-1-k),c} + b_c.
$$

Only the past is used, so the state is the last `K-1` inputs. This is the
short-conv mixer and part of DeltaNet's input processing.

### Hadamard transform / RHT

The Hadamard matrix is defined recursively:

$$
H_1 = [1], \qquad
H_{2n} = \begin{pmatrix} H_n & H_n \\ H_n & -H_n \end{pmatrix}.
$$

The normalized transform is `H/√n`, which is orthogonal:

$$
\left(\frac{H}{\sqrt n}\right)^\top \left(\frac{H}{\sqrt n}\right) = I,
\qquad
\left\|\frac{Hx}{\sqrt n}\right\|_2 = \|x\|_2.
$$

The randomized Hadamard transform (RHT) multiplies by a random sign diagonal `D`
first:

$$
\mathrm{RHT}(x) = \frac{1}{\sqrt n} H D x.
$$

`D` makes the transform incoherent: energy is spread across coordinates, so
outliers no longer dominate a low-bit group quantizer.

### Sampling filters

Given logits `z` and temperature `T`:

$$
p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}.
$$

- **top-k**: keep the `k` largest `z_i`, set the rest to `-∞`.
- **top-p**: sort by probability, keep the smallest set whose cumulative
  probability is `≥ p`, set the rest to `-∞`.
- **min-p**: keep `i` with `p_i ≥ min_p · max_j p_j`.

Greedy sampling is `argmax_i z_i`.

---

## 04 — backend and context

The context owns the device, the allocator, the command queue and the kernel
table. The important mathematical idea is the **roofline**:

$$
\text{time} \ge \max\left(\frac{\text{FLOPs}}{\text{peak FLOP/s}},
\frac{\text{bytes}}{\text{bandwidth}}\right).
$$

Prefill is usually compute-bound (large `T`, big matmuls). Decode is usually
bandwidth-bound (one token, all weights read). That is why the engine has
separate GEMV/GEMM kernels and why quantization is central.

`supports_mxu` is a flag for the Apple M5 matrix units; when true the real engine
selects `GemmEngine::Mxu` with larger tiles. The Python port always uses the CPU
path.

---

## File-by-file summary

| File | Maths |
|---|---|
| `01_dtypes.py` | BF16 rounding, int8 scale/error |
| `02_buffer.py` | byte sizes, pool reuse, command batching |
| `03_kernels.py` | linear, RMSNorm, softmax, RoPE, attention, conv, filters, Hadamard |
| `04_backend.py` | roofline, kernel dispatch, device capabilities |
