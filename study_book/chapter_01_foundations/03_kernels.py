"""03 — CPU reference kernels.

Mirrors the family of kernels under ``backends/cpu/kernel`` and the portable
Slang/Metal kernels, but written with NumPy so every operation is readable.

The point is to see *what* each GPU kernel computes.  In the real engine these
are dispatched per type/format (bf16, int8, packed 4-bit, trellis) and fused.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def as_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = as_f32(x)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    x = as_f32(x)
    return (x * sigmoid(x)).astype(np.float32)


def gelu(x: np.ndarray) -> np.ndarray:
    x = as_f32(x)
    return (0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))).astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = as_f32(x)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = as_f32(x)
    x = x - np.max(x, axis=axis, keepdims=True)
    return (x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))).astype(np.float32)


def l2_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.sqrt(np.sum(as_f32(x) ** 2, axis=-1, keepdims=True))
    return (as_f32(x) / np.maximum(n, eps)).astype(np.float32)


# ---------------------------------------------------------------------------
# matrix / normalization
# ---------------------------------------------------------------------------
def linear(x: np.ndarray, weights: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
    """``weights`` is [out, in], like PyTorch nn.Linear."""
    y = np.matmul(as_f32(x), as_f32(weights).T)
    if bias is not None:
        y = y + as_f32(bias)
    return y.astype(np.float32)


def rms_norm(
    x: np.ndarray,
    scales: Optional[np.ndarray] = None,
    biases: Optional[np.ndarray] = None,
    eps: float = 1e-5,
    subtract_mean: bool = False,
) -> np.ndarray:
    x = as_f32(x)
    if subtract_mean:
        x = x - np.mean(x, axis=-1, keepdims=True)
    var = np.mean(x * x, axis=-1, keepdims=True)
    y = x / np.sqrt(var + eps)
    if scales is not None:
        y = y * as_f32(scales)
    if biases is not None:
        y = y + as_f32(biases)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# positional encoding
# ---------------------------------------------------------------------------
def rope_apply(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate-half RoPE.

    x: [T, H, D]; cos/sin: [T, D/2].
    """
    x = as_f32(x)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    return np.concatenate((x1 * c - x2 * s, x2 * c + x1 * s), axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------
def causal_mask(
    query_len: int,
    key_len: int,
    offset: int = 0,
    window: Optional[int] = None,
) -> np.ndarray:
    """Boolean [query_len, key_len]; True means the key is visible."""
    if query_len == 0 or key_len == 0:
        return np.zeros((query_len, key_len), dtype=bool)
    qpos = offset + np.arange(query_len)[:, None]
    kpos = np.arange(key_len)[None, :]
    allowed = kpos <= qpos
    if window is not None and window > 0:
        allowed &= kpos > (qpos - window)
    return allowed


def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
    sinks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Grouped-query scaled dot-product attention.

    q: [Tq, Hq, D]
    k/v: [Tk, Hkv, D]      (Hq must be a multiple of Hkv)
    mask: [Tq, Tk]         (True = allowed)
    sinks: [Hq]            (optional per-head sink logits)
    """
    q = as_f32(q)
    k = as_f32(k)
    v = as_f32(v)
    tq, hq, d = q.shape
    tk, hkv, _ = k.shape
    if hkv != hq:
        repeat = hq // hkv
        k = np.repeat(k, repeat, axis=1)
        v = np.repeat(v, repeat, axis=1)

    scale = float(scale if scale is not None else 1.0 / np.sqrt(d))
    scores = np.einsum("qhd,khd->hqk", q, k, optimize=True) * scale  # [Hq, Tq, Tk]

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 2:
            m = m[None, :, :]
        scores = np.where(m, scores, np.float32(-1e30))

    if sinks is not None:
        sink_col = np.broadcast_to(as_f32(sinks)[:, None, None], (hq, tq, 1))
        scores = np.concatenate((scores, sink_col), axis=-1)

    probs = softmax(scores, axis=-1)
    if sinks is not None:
        probs = probs[..., :-1]  # sink value is zero, only its probability is dropped

    out = np.einsum("hqk,khd->qhd", probs, v, optimize=True)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# convolution (short conv / GDN)
# ---------------------------------------------------------------------------
def depthwise_causal_conv(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    groups: int = 1,
) -> np.ndarray:
    """Causal depthwise convolution.

    x: [T, C]
    weight: [groups, kernel]  (each group is shared by C/groups channels)
    """
    x = as_f32(x)
    t, c = x.shape
    kernel = weight.shape[-1]
    out = np.zeros_like(x)
    channels_per_group = c // groups
    for g in range(groups):
        w = as_f32(weight[g])
        lo = g * channels_per_group
        hi = lo + channels_per_group
        for tap in range(kernel):
            # y[t] += w[tap] * x[t - (kernel - 1 - tap)]
            shift = kernel - 1 - tap
            if shift == 0:
                out[:, lo:hi] += x[:, lo:hi] * w[tap]
            else:
                out[shift:, lo:hi] += x[:-shift, lo:hi] * w[tap]
    if bias is not None:
        out += as_f32(bias)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    out = np.full_like(logits, -np.inf)
    idx = np.argpartition(logits, -k, axis=-1)[..., -k:]
    np.put_along_axis(out, idx, np.take_along_axis(logits, idx, axis=-1), axis=-1)
    return out


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    if p >= 1.0:
        return logits
    order = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, order, axis=-1)
    probs = softmax(sorted_logits, axis=-1)
    cumulative = np.cumsum(probs, axis=-1)
    keep = cumulative - probs < p
    keep[..., 0] = True
    sorted_logits = np.where(keep, sorted_logits, -np.inf)
    out = np.empty_like(logits)
    np.put_along_axis(out, order, sorted_logits, axis=-1)
    return out


def min_p_filter(logits: np.ndarray, min_p: float) -> np.ndarray:
    if min_p <= 0:
        return logits
    probs = softmax(logits, axis=-1)
    threshold = np.max(probs, axis=-1, keepdims=True) * min_p
    return np.where(probs >= threshold, logits, -np.inf)


def sample_from_logits(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    greedy: bool = False,
) -> int:
    logits = as_f32(logits).reshape(-1)
    if greedy or temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / float(temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    logits = min_p_filter(logits, min_p)
    probs = softmax(logits)
    # Guard against a fully filtered distribution.
    if not np.isfinite(probs).all() or probs.sum() <= 0:
        return int(np.argmax(logits))
    return int(rng.choice(probs.shape[0], p=probs / probs.sum()))


# ---------------------------------------------------------------------------
# quantization support (Hadamard / RHT)
# ---------------------------------------------------------------------------
def fast_hadamard(x: np.ndarray, signs: Optional[np.ndarray] = None) -> np.ndarray:
    """In-place-style fast Walsh-Hadamard transform on the last axis.

    The last dimension must be a power of two.  ``signs`` implements the
    randomized part of RHT (one +/-1 per coordinate).
    """
    y = as_f32(x).copy()
    if signs is not None:
        y = y * as_f32(signs)
    original_shape = y.shape
    n = y.shape[-1]
    if n & (n - 1):
        raise ValueError("hadamard dimension must be a power of two")
    y = y.reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :].copy()
        b = y[:, :, 1, :].copy()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        y = y.reshape(-1, n)
        h *= 2
    return (y.reshape(original_shape) / np.sqrt(n)).astype(np.float32)


# ---------------------------------------------------------------------------
# indexing
# ---------------------------------------------------------------------------
def embedding_lookup(table: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return as_f32(table)[np.asarray(ids, dtype=np.int64)]


def gather_rows(x: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return as_f32(x)[np.asarray(list(indices), dtype=np.int64)].copy()


def concat(parts: Sequence[np.ndarray], axis: int = -1) -> np.ndarray:
    return np.concatenate([as_f32(p) for p in parts], axis=axis)


def split(x: np.ndarray, sizes: Sequence[int], axis: int = -1) -> list[np.ndarray]:
    cuts = np.cumsum(sizes)[:-1]
    return [np.ascontiguousarray(p) for p in np.split(x, cuts, axis=axis)]
