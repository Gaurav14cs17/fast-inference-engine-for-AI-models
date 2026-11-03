# Chapter 05 — runtime

Files: `26_tokenizer.py`, `27_decoder.py`, `28_stream.py`, `29_engine.py`.

This chapter connects the model graph to a generation loop. The maths is mostly
probability, cache bookkeeping, and cost accounting.

---

## 26 — tokenizer

The study tokenizer maps UTF-8 bytes directly to ids:

$$
\text{id}(b) = b \in \{0,\dots,255\},
$$

with `256 = EOS` and `257 = mask/pad`. The vocabulary is `V = 258`.

Encoding is

$$
\mathrm{encode}(s) = [\,b_1, b_2, \dots, b_n\,], \qquad s = \text{UTF-8 bytes}.
$$

Decoding reverses the map and ignores special ids:

$$
\mathrm{decode}([i_1,\dots,i_n]) = \text{UTF-8}([i_1,\dots,i_n]).
$$

A chat template is a deterministic string concatenation:

```text
<|system|>\n...\n<|user|>\n...\n<|assistant|>\n
```

The real engine uses a HuggingFace tokenizer plus `hanashi` chat encoding; the
maths is the same: text → ids → model → ids → text.

---

## 27 — decoder

The decoder is

$$
H_0 = \mathrm{Embed}(x),
$$

$$
H_{\ell+1} = \mathrm{Layer}_\ell(H_\ell), \qquad \ell=0\dots L-1,
$$

$$
z_t = W_{\text{readout}}\mathrm{Norm}(H_{L,t}).
$$

If the readout is tied, `W_readout = E^T`, so

$$
z_t = E^\top \mathrm{Norm}(H_{L,t}),
$$

which reuses the embedding matrix and saves `V·d` parameters.

### Output range

The decoder can compute logits for only the last `k` positions:

$$
z_{T-k:T} = W_{\text{readout}}\mathrm{Norm}(H_{L,T-k:T}).
$$

This is what the prefill does: it only needs the last token's logits to start
generation. The study port computes all rows when `output_range=(0,T)` and slices
when needed.

---

## 28 — generation stream

### Prefill

Given prompt `x_1 … x_L`:

$$
H = \mathrm{Decoder}(x_{1:L}), \qquad
x_{L+1} \sim \mathrm{softmax}(z_L).
$$

The KV cache is filled with the prompt rows. The cache length becomes `L`.

### Normal decode

At each step, the current token `x_t` is appended:

$$
z_t = \mathrm{Decoder}(x_t \mid \text{cache}),
\qquad
x_{t+1} \sim \mathrm{softmax}(z_t).
$$

The cache grows by one row. Cost per token is dominated by reading all weights
(see chapter 03), not by the token count.

### Speculative decode

The draft proposes a tree with root `x_t` and nodes `v_1 … v_N`. The target runs
**one** forward over the batch

$$
[x_t, x_{v_1}, \dots, x_{v_N}]
$$

with positions `L + d_i` and the ancestor mask from chapter 04. This produces
target logits `z_i` for every node.

#### Accept rule

Starting at the root, sample a target token

$$
\hat x_i \sim \mathrm{softmax}(z_i).
$$

If a child of node `i` has token `\hat x_i`, accept that child and continue.
Otherwise stop and use `\hat x_i` as the bonus token. The accepted rows are

$$
\mathcal{A} = \{0\} \cup \{\text{accepted children}\}.
$$

The output tokens are the accepted draft tokens plus the bonus:

$$
\text{output} = (x_{v_1}, \dots, x_{v_a}, \hat x).
$$

#### Cache update

Only the accepted rows survive. After verification:

$$
L \leftarrow L + |\mathcal{A}|.
$$

Rejected branches are dropped by compacting the last cache chunk. This is why
the KV cache can verify a tree without permanently storing it.

### Metrics

- `tokens_count_output` = number of emitted tokens.
- `num_prefill_forward_passes` = 1 (in this port).
- `num_decode_forward_passes` = number of target forwards after prefill.
- `tokens_per_forward_pass`:

$$
\text{tokens/forward} = \frac{\text{tokens}}{\text{prefill passes} + \text{decode passes}}.
$$

- `accepted_draft_tokens` = total accepted draft tokens.
- Acceptance rate:

$$
\text{accept rate} = \frac{\text{accepted draft tokens}}{\text{proposed draft tokens}}.
$$

- Ideal speedup:

$$
S \approx \frac{1 + \mathbb{E}[a]}{1 + c}.
$$

### Why one forward can verify a tree

Attention is a set of independent dot products plus a mask. If each node only
attends to its ancestors, the computation for every branch is independent of the
other branches. Stacking them in the batch dimension gives the same result as
running each branch separately, but with one weight read and one kernel launch.

---

## 29 — engine

The engine owns:

- a `CPUContext` (device, allocator, kernel table),
- loaded `LanguageModel` instances,
- chat sessions.

A `LanguageModel` is

$$
\text{Model} = (\text{Decoder}, \text{Speculator}, \text{Sampling}, \text{Tokenizer}).
$$

A `ChatSession` renders messages to ids, creates a `GenerationStream`, and
decodes the output back to text.

### Memory accounting

For a model with `N` parameters at `b` bits and a context of `T` tokens:

$$
M_{\text{weights}} \approx \frac{N b}{8},
\qquad
M_{\text{KV}} \approx 2 L T H_{kv} d_h s,
$$

plus activations `O(B T d)` and the scratch pool. At batch 1, weights dominate
for short contexts; the KV cache dominates for very long contexts.

### Cost accounting

$$
\text{prefill} \sim O(T^2 d + T N),
\qquad
\text{decode} \sim O(N) \text{ bytes read}.
$$

Speculation changes decode from one token per target forward to `1 + a` tokens
per target forward, at the cost of one draft forward.

---

## File-by-file summary

| File | Maths |
|---|---|
| `26_tokenizer.py` | byte ↔ id map, chat template |
| `27_decoder.py` | embedding → stack → readout |
| `28_stream.py` | prefill, decode, tree accept, metrics |
| `29_engine.py` | composition, memory/cost accounting |

---

## Worked example from `run_demo.py`

The synthetic verification example uses a chain:

```text
root(10) → 11 → 12 → 13 → bonus(14)
```

The target logits are constructed so that

$$
\arg\max z_0 = 11,\quad
\arg\max z_1 = 12,\quad
\arg\max z_2 = 13,\quad
\arg\max z_3 = 14.
$$

The verifier accepts rows `[0,1,2,3]`, draft tokens `[11,12,13]`, and bonus `14`.
So one target forward produced `3 + 1 = 4` tokens. With real DFlash weights this
is the source of the speedup; with random weights the draft is independent and
the expected number of accepted tokens is small.
