# Chapter 04 — generation structures

Files: `20_trie.py`, `21_batch_topology.py`, `22_sampling.py`,
`23_dflash.py`, `24_weaver.py`, `25_speculator.py`.

This chapter builds the data structures and probability machinery for
speculative decoding. The central question is: **how do we verify many candidate
tokens with one target forward pass?**

---

## 20 — trie and grammar mask

A trie over token sequences is a prefix tree. A node at depth `d` represents a
prefix of length `d`; its children are the allowed next tokens.

For a vocabulary mask `m ∈ {0,1}^V`, the allowed logits are

$$
\tilde z_i = \begin{cases} z_i & m_i = 1,\\ -\infty & m_i = 0.\end{cases}
$$

After softmax, masked tokens have probability zero:

$$
p_i = \frac{e^{\tilde z_i}}{\sum_j e^{\tilde z_j}}.
$$

The same structure stores a speculative proposal tree: each node has
`(token_id, depth, logprob, parent, children)`. Flattening gives arrays
`parent[i]`, `depth[i]`, `token_id[i]`.

---

## 21 — batch topology

A normal decode batch is a chain. A speculative batch is a tree. Let

- `L` = number of tokens already in the KV cache (the prefix),
- `N` = number of nodes in the current batch (root + proposals),
- `π(i)` = parent of node `i`, with `π(0) = -1`,
- `d_i` = depth of node `i`, `d_0 = 0`.

The absolute position of node `i` is

$$
p_i = L + d_i.
$$

Node `i` may attend to:

1. every prefix token `0 … L-1`, and
2. every ancestor of `i` in the tree, including itself.

Define the mask `M ∈ {0,1}^{N×(L+N)}`:

$$
M_{i,j} =
\begin{cases}
1 & j < L,\\
1 & j \ge L \text{ and } (j-L) \in \mathrm{Anc}(i),\\
0 & \text{otherwise}.
\end{cases}
$$

For a chain, `Anc(i) = {0,…,i}`, so the mask is exactly causal attention. For a
tree, siblings are masked from each other, which is what makes one batched
forward equivalent to verifying all branches.

### Size and cost

The mask has `N(L+N)` entries. For a small tree (`N` ≤ 8 in the demo) it is tiny
compared with the model weights. Building it costs `O(N·depth)` by walking parent
pointers, or `O(N²)` in the naive loop.

The cache accept step keeps only the accepted path:

$$
\text{cache} \leftarrow \text{cache}[:L] \;\|\; \text{cache}[L + \text{accepted}].
$$

---

## 22 — sampling

### From logits to probabilities

For temperature `T > 0`:

$$
p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}.
$$

`T → 0` approaches argmax (greedy); `T = 1` is the model distribution.

### Filters

- **top-k**: keep the `k` largest logits, set the rest to `-∞`.
- **top-p (nucleus)**: sort by `p`, keep the smallest prefix with cumulative
  probability `≥ p`.
- **min-p**: keep `i` if `p_i ≥ min_p · max_j p_j`.

Let the filtered distribution be `p̃`. Sampling draws

$$
x \sim \mathrm{Categorical}(\tilde p).
$$

### Determinism

The PRNG is splitmix64:

$$
z \leftarrow (z + \phi)\ \text{mod}\ 2^{64}, \qquad
\text{output} = f(z),
$$

with the usual xor-shift/multiply finalizer. Deriving a per-row seed makes
sampling reproducible for a given request.

---

## 23 — DFlash draft model

DFlash is a **block** draft model. Given the target hidden features and the last
target token, it predicts a block of `B` future tokens:

$$
q_\theta(x_{t+1}, \dots, x_{t+B} \mid h_t^{(\text{target})}, x_t).
$$

### Input construction

Let `h_t^{(i)}` be the target residual feature at layer `i ∈ target_layer_ids`.
Concatenate them and project:

$$
u_t = W_{\text{feat}}\,\mathrm{concat}_i(h_t^{(i)}) \in \mathbb{R}^{d}.
$$

The draft input is the projected feature plus a learned mask token `m`:

$$
x^{(0)}_j = u_t + E[m], \qquad j = 1 \dots B.
$$

### Draft transformer

A small causal transformer processes the block:

$$
H^{(0)} = \begin{pmatrix} u_t + E[m] \\ \vdots \\ u_t + E[m] \end{pmatrix},
\qquad
H^{(\ell+1)} = \mathrm{Layer}^{(\ell)}(H^{(\ell)}).
$$

The output logits are

$$
z_j = W_{\text{readout}}\mathrm{Norm}(H^{(L)}_j),
\qquad
q_j = \mathrm{softmax}(z_j).
$$

The draft is cheap because it is shallow, has a small `d`, and runs only `B`
positions. In the study port `B = 4` and the draft is one tiny layer.

### Why features, not just tokens

Target features carry information about the whole prefix. Conditioning on them
makes the draft much more accurate than a standalone small language model. This
is the DFlash idea: reuse the target's hidden state.

---

## 24 — Weaver tree

Weaver turns block logits into a **tree** of candidates instead of a single
chain. Each node `v` has a token `x_v`, a depth `d_v`, and a parent.

### Node score

For a root with score `s_0 = 0`, each edge adds a log-probability:

$$
s_v = s_{\pi(v)} + \log q(x_v \mid \text{parent}).
$$

Since `q` is a probability, `s_v ≤ 0`; a higher score means a more likely branch.
The branch probability is

$$
P(v) = \exp(s_v).
$$

### Expansion

At each round `r = 1 … R`, for every frontier node `u` keep the top-`k` tokens
from the draft logits at depth `r`:

$$
\mathcal{C}_r = \bigcup_{u \in \text{frontier}} \mathrm{TopK}\!\left(\log q(\cdot \mid u), k\right).
$$

Then keep the `E` highest-scoring candidates (`expand_per_round`) and add them
as children.

### Pruning

With `prune_sigma = σ`, compute the mean `μ` and standard deviation `sd` of the
candidate scores and drop candidates below

$$
s_{\min} = \mu - \sigma \cdot sd.
$$

This keeps the tree budget for plausible branches and discards low-probability
tails.

### Tree budget

If the budget is `N` nodes (including the root), expansion stops when

$$
1 + \sum_{r=1}^{R} |\mathcal{C}_r| = N.
$$

The demo uses `N = 8`, `R = 4`, `k = 2`, `E = 4`, producing a root + 7 draft
nodes.

---

## 25 — speculator: DFlash + Weaver

`DFlashTfmSpeculator` composes the two:

$$
\underbrace{h_t^{(\text{target})}}_{\text{features}}
\;\xrightarrow{\text{DFlash}}\;
\{q_1, \dots, q_B\}
\;\xrightarrow{\text{Weaver}}\;
\text{tree}.
$$

### Argmax mode

If Weaver is disabled, the tree is a chain:

$$
x_{t+j} = \arg\max_i q_j(i), \qquad j = 1 \dots B.
$$

This is the simplest draft.

### Weaver mode

The tree is a set of paths. The target verifies all nodes in one forward pass.

### Expected accepted tokens

Let `A_d` be the event that the first `d` draft tokens are accepted. Then

$$
\mathbb{E}[a] = \sum_{d=1}^{D} P(A_d),
$$

where `P(A_d)` is the probability that the target's sampled token matches the
draft path at every step. If the draft is perfect, `P(A_d)=1` and
`E[a] = D`; if it is independent, `P(A_d)` decays quickly.

### Speedup

If the draft costs `c` target forwards and the target accepts `a` tokens, the
ideal speedup is

$$
S = \frac{1 + \mathbb{E}[a]}{1 + c}.
$$

For example, `E[a] = 3`, `c = 0.2` gives `S = 4/1.2 ≈ 3.33×`. The real gain is
lower because of verification overhead and imperfect acceptance.

---

## File-by-file summary

| File | Maths |
|---|---|
| `20_trie.py` | prefix tree, grammar mask |
| `21_batch_topology.py` | positions `L+d_i`, ancestor mask |
| `22_sampling.py` | temperature, top-k/p/min-p, categorical draw |
| `23_dflash.py` | feature projection, block logits |
| `24_weaver.py` | path score `Σ log q`, expansion, `μ - σ·sd` pruning |
| `25_speculator.py` | draft → tree, expected accepted tokens, speedup |
