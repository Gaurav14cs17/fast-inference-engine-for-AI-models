# Chapter 02 — configuration, quantization, weight loading

Files: `05_config.py`, `06_quantization.py`, `07_safetensors.py`.

This chapter turns a model description and a weight file into typed arrays. The
main maths is quantization: how many bits per weight, and how much error does
that introduce?

---

## 05 — configuration

The config dataclasses mirror the Rust serde structs. The only formula worth
writing here is the shape bookkeeping:

$$
d = H \cdot d_h,
\qquad
\text{QKV out} = d + 2 H_{kv} d_h + \underbrace{d}_{\text{optional gate}},
$$

and for a dense MLP:

$$
\text{up out} = 2F, \qquad \text{down out} = d.
$$

`tiny_config()` uses `d = 64`, `H = 4`, `d_h = 16`, `F = 128`, two layers. The
hybrid config replaces the second layer with DeltaNet, whose state is
`[H, d_h, d_h]` instead of a KV cache.

---

## 06 — group quantization

### The quantizer

For a group of `g` weights `x_1 … x_g`, with `b` bits and `q_max = 2^{b-1}-1`:

$$
s = \frac{\max_i |x_i|}{q_{\max}},
\qquad
q_i = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{x_i}{s}\right), -q_{\max}, q_{\max}\right),
\qquad
\hat{x}_i = s q_i.
$$

The reconstruction error is

$$
|x_i - \hat{x}_i| \le \frac{s}{2}
= \frac{\max_j |x_j|}{2 q_{\max}}.
$$

For `b=4`, `q_max=7`, so the step is about `max|x|/14`. For `b=8`,
`q_max=127`, about `max|x|/254`.

### Bits per weight

Each group stores `g` codes plus one scale (16-bit float in this port):

$$
\text{bits/weight} = b + \frac{16}{g}.
$$

Examples:

| format | `b` | `g` | bits/weight |
|---|---:|---:|---:|
| int4, group 32 | 4 | 32 | 4.50 |
| int4, group 64 | 4 | 64 | 4.25 |
| int8, group 32 | 8 | 32 | 8.50 |
| int8, per-row | 8 | `d` | `8 + 16/d` |

The group size trades accuracy against overhead. A smaller `g` gives a tighter
`max|x|` per group and therefore lower error, but more scale bits.

### Why groups help

If one weight is an outlier `10×` larger than the rest, a per-row scale is set by
the outlier, so all other weights get a coarse step. Grouping localizes the
outlier to its group; the other groups keep a fine step. This is the same reason
block quantization (Q4_K, etc.) works.

### Quantization error model

If the rounding error is approximately uniform on `[-s/2, s/2]`, its variance is

$$
\mathrm{Var}(e) = \frac{s^2}{12}.
$$

With `s ∝ max|x|/q_max`, increasing bits by 1 halves `s`, so the error variance
drops by `4×` (about 6 dB). This is the standard 6 dB/bit rule for uniform
quantization.

### Int8 activations

The same formula with `b=8` and `g = d` is used for the activation `A8` path.
The kernel multiplies int8×int8 and applies scales in the epilogue; the Python
port dequantizes first to show the equivalent maths.

---

## 06 — RHT (randomized Hadamard transform)

### Definition

With `H_n` the Sylvester Hadamard matrix and `D = diag(±1)`:

$$
y = \frac{1}{\sqrt n} H_n D x.
$$

`H_n/√n` is orthogonal, and `D` is orthogonal, so the composition is orthogonal:

$$
\|y\|_2 = \|x\|_2.
$$

The inverse is the same transform (up to the sign ordering):

$$
x = D^\top \frac{1}{\sqrt n} H_n^\top y.
$$

### Incoherence

Let `x` have one large coordinate and the rest small. After RHT, that coordinate's
energy is spread over all `n` outputs. Informally, the maximum absolute value
drops roughly like

$$
\max_i |y_i| \lesssim \frac{\|x\|_2}{\sqrt n} \cdot O(\sqrt{\log n})
$$

with high probability over the random signs. A smaller dynamic range means the
group quantizer's `max|x|` is smaller, so the step `s` is smaller and the error
drops. The Rust engine fuses this transform into the linear layer
(`RHTLinearWrapper`), so it costs one extra Hadamard kernel, not an extra matmul.

### Fast transform cost

The recursive structure gives the fast Walsh–Hadamard transform:

$$
T(n) = 2T(n/2) + O(n) = O(n \log n).
$$

A naive matrix multiply would be `O(n²)`; the butterfly version is why the
transform is cheap enough to run on activations.

---

## 06 — trellis / DPCM toy codec

The real engine uses a trellis-coded quantizer (QTIP-style) with 16-bit states
and 2-bit transitions. The study version is scalar DPCM:

### Encoder

$$
s_0 = 0, \qquad
c_t = \arg\min_{c \in \{0,1,2,3\}} \left| (x_t - s_{t-1}) - \ell_c \right|,
\qquad
s_t = s_{t-1} + \ell_{c_t},
$$

with codebook

$$
\ell = \left[-\tfrac32, -\tfrac12, \tfrac12, \tfrac32\right] \cdot \Delta.
$$

### Decoder

$$
\hat{x}_t = s_t = \sum_{k=1}^t \ell_{c_k}.
$$

Because consecutive weights in a row are correlated, the residual `x_t - s_{t-1}`
is small, so 2 bits per step can track the row. This is the intuition behind
trellis quantization: the state carries the running reconstruction, and the code
only stores a short transition.

### Bit accounting

The real trellis stores:

- one 16-bit state per block,
- `T-1` transitions of `t` bits each.

So for a block of `n` weights:

$$
\text{bits} = 16 + (n-1)t.
$$

With `t=2` and large `n`, this approaches 2 bits/weight plus a small state
overhead. The study codec uses `t=2` with a 16-bit initial state and scalar
steps, which is enough to see the cumulative-sum decode.

---

## 07 — safetensors and the parameter tree

A safetensors file is

```text
[8-byte little-endian header length][JSON header][raw tensor bytes]
```

Each tensor's JSON entry has `dtype`, `shape`, and `data_offsets = [start, end]`.
The byte offset formula is

$$
\text{offset}_{i+1} = \text{offset}_i + \prod_j \text{shape}_{i,j} \cdot \text{sizeof(dtype)}.
$$

Loading is therefore a set of zero-copy views (or one copy, depending on the
reader). `ParameterTree` is a flat `name → array` map with a prefix; `subtree("a")`
maps `name` to `a.name`. Random init uses

$$
w \sim \mathcal{N}(0, 0.02^2)
$$

for weights, `1` for norm scales, and `0` for biases.

---

## File-by-file summary

| File | Maths |
|---|---|
| `05_config.py` | shape bookkeeping, parameter count |
| `06_quantization.py` | group quant, error bound, bits/weight, RHT, trellis DPCM |
| `07_safetensors.py` | header offsets, byte sizes, random init |
