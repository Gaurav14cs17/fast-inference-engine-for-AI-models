# Tools

Two small scripts. Neither is part of the model; they only help you study it.

## `check_dag.py`

Verifies that the numbered files form a DAG in numeric order.

### What it reads

For each file `NN_name.py` it collects imports of the form

```python
from study_book.chapter_XX_... import alias
from study_book.chapter_XX_....alias import Symbol
import study_book.chapter_XX_....alias
```

It maps `alias` back to the number of the file that defines it (for example
`dtypes → 01`, `attention → 14`), then checks

$$
\forall (u \to v):\quad \mathrm{num}(u) < \mathrm{num}(v).
$$

### Algorithm

1. Build `alias → number` by scanning all numbered files.
2. For each file, parse its imports.
3. For every edge `(dep, current)`, fail if `dep >= current`.
4. Print the edge list and a summary.

Complexity: `O(V + E)` for `V` files and `E` import edges (plus regex parsing).
This is exactly the topological-sort feasibility check; a cycle would produce at
least one backward or self edge.

### Example

```text
u14 -> u18   18_transformer_layer.py
u18 -> u19   19_transformer.py
...
OK: 29 numbered files, 98 import edges, DAG only goes forward.
```

## `run_demo.py`

Runs a tiny random model end-to-end. It demonstrates four things.

### 1. Forward pass and sampling

$$
H_0 = \mathrm{Embed}(x), \qquad
H_{\ell+1} = \mathrm{Layer}_\ell(H_\ell), \qquad
z = W_{\text{readout}}\mathrm{Norm}(H_L).
$$

The script samples from `z` with the greedy sampler and prints the generated
token ids. With random weights the text is gibberish; the point is the shape and
data flow.

### 2. Attention + DeltaNet

The first run uses two attention layers (`tiny_config`). The second uses one
attention layer and one DeltaNet layer (`hybrid_config`). DeltaNet keeps a
fixed-size state `S ∈ R^{H×d_h×d_h}` instead of a growing KV cache.

### 3. Quantization primitives

- int4 group quantization error,
- RHT norm preservation `‖HDx‖₂ = ‖x‖₂`,
- toy trellis (DPCM) reconstruction error.

### 4. Tree verification

The synthetic demo creates a chain

```text
root(10) → 11 → 12 → 13 → bonus(14)
```

and target logits whose argmax follows that chain. The verifier accepts

```text
accepted rows        [0, 1, 2, 3]
accepted draft tokens [11, 12, 13]
bonus token          14
```

so one target forward produced `3 + 1 = 4` tokens. In the live random-weight
model the independent draft rarely matches the target, so `accepted_draft` is
usually zero; real DFlash weights are trained to match.

## Run

```bash
# from the project root
python tools/check_dag.py
python tools/run_demo.py
```
