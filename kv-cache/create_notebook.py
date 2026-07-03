"""Generates kvcache_explainer.ipynb — run once (make kv-cache-regen), then open the .ipynb.

Maintenance tooling, not learner-facing. The .ipynb is the deliverable.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Learning (shared)", "language": "python", "name": "learning"},
    "language_info": {"name": "python", "version": "3.11"},
})

def md(src):   return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

cells = []

cells.append(md(r"""# The KV-Cache — Make Generation Fast Without Changing the Answer

**Builds on [`../nanogpt-from-scratch`](../nanogpt-from-scratch).** We take that exact trained
GPT and add the single most important inference optimization: the **key/value cache**.

By the end you will:
- See *why* naive generation is O(n²) and where the wasted work is
- Build a KV-cache that makes per-step attention O(1) in sequence length → O(n) total
- **Prove** the cache changes speed, not output (logits identical to ~1e-15)
- Measure the real speedup, and see why KV-cache *memory* is the next bottleneck
  (the link to `../turbo-quant`)
"""))

cells.append(md(r"""---
## 0 · The waste: recomputing the past, every step

To generate token *t*, a transformer needs attention over tokens 0…*t*. The naive loop
runs a **full forward over the whole prefix every step**:

```
step 1:  [t0]                 → compute K,V for 1 token
step 2:  [t0 t1]              → recompute K,V for t0 (again!) + t1
step 3:  [t0 t1 t2]          → recompute K,V for t0, t1 (again!) + t2
...
```

But the K and V vectors for past tokens **never change** — the weights are fixed and each
past token's input is fixed. Recomputing them is pure waste. Total work to emit *n* tokens
is 1+2+…+n = **O(n²)**.
"""))

cells.append(code(r"""import time
import numpy as np
import matplotlib.pyplot as plt
from kvcache import (
    _train_tiny, verify_equivalence, decode_step, KVCache,
    generate_naive, generate_cached,
)

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#2a2d3e", "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0", "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
    "axes.titlecolor": "#e2e8f0", "grid.color": "#2a2d3e", "figure.dpi": 110,
})

model, tok = _train_tiny()   # a small GPT trained on a tiny corpus (real weights)
print("model ready — vocab", tok.vocab_size, "| block_size", model.block_size)"""))

cells.append(md(r"""---
## 1 · The idea: cache K and V, and the mask disappears

Store each layer's K and V for every past token. To take a step:
1. Compute Q, K, V for **only the new token**.
2. Append its K, V to the cache.
3. Attend the new Q against **all cached K, V**.

Because the cache only ever contains the past, you never build a causal mask — "can't see
the future" is automatic. Read `decode_step` in `kvcache.py`: it is one token in, next-token
logits out, with the cache carrying the history.
"""))

cells.append(md(r"""---
## 2 · Prove it first (the trust moment)

Before trusting a speedup, prove it didn't change the output. `verify_equivalence` runs a
full forward and the step-by-step cached path and compares the logits at **every** position.
"""))
cells.append(code(r"""ids = tok.encode("to be or not to be")
err = verify_equivalence(model, ids)
print(f"max |logit difference| between cached and full forward: {err:.2e}")
print("PASS — identical to machine precision." if err < 1e-6 else "FAIL")"""))

cells.append(md(r"""---
## 3 · The payoff: same text, a fraction of the time

Same seed → both paths generate the **identical string**. Only the cost differs.
"""))
cells.append(code(r"""prompt = tok.encode("to be")
results = {}
for name, fn in (("naive", generate_naive), ("cached", generate_cached)):
    t0 = time.perf_counter()
    out = fn(model, prompt, 80, np.random.default_rng(7), temperature=0.8)
    results[name] = (time.perf_counter() - t0, tok.decode(out))

for name, (dt, txt) in results.items():
    print(f"{name:6s} {dt*1000:7.1f} ms   {txt[:44]!r}")
print("\nidentical output:", results["naive"][1] == results["cached"][1])
print(f"speedup: {results['naive'][0] / results['cached'][0]:.1f}x")"""))

cells.append(md(r"""---
## 4 · Where the O(n²) vs O(n) shows up

Time the **per-step** cost of each approach as the sequence grows. Naive climbs linearly per
step (→ quadratic total); cached stays roughly flat.
"""))
cells.append(code(r"""def per_step_times(fn, prompt, n):
    # crude but illustrative: time generating k tokens, difference = marginal cost
    xs, ys, prev = [], [], 0.0
    for k in range(10, n + 1, 10):
        t0 = time.perf_counter()
        fn(model, prompt, k, np.random.default_rng(7))
        tot = time.perf_counter() - t0
        xs.append(k); ys.append((tot - prev)); prev = tot
    return xs, ys

xs, yn = per_step_times(generate_naive, prompt, 80)
_,  yc = per_step_times(generate_cached, prompt, 80)
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(xs, yn, "-o", color="#f472b6", label="naive (recompute prefix)")
ax.plot(xs, yc, "-o", color="#38bdf8", label="cached")
ax.set(title="Cost grows with length — unless you cache",
       xlabel="tokens generated", ylabel="marginal time (s)")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()"""))

cells.append(md(r"""---
## 5 · The catch: the cache costs memory

The speedup isn't free — you now store K and V for every layer, head, and past token:

```
KV bytes ≈ 2 (K and V) × n_layer × n_head × head_dim × seq_len × batch × dtype_bytes
```

For a real 7B model with long context and big batches, the KV-cache can dwarf the model
weights. That is exactly why **KV-cache compression** matters — and why you built
[`../turbo-quant`](../turbo-quant): quantizing the cached K/V is one of its headline uses.
"""))
cells.append(code(r"""def kv_megabytes(n_layer, n_head, head_dim, seq_len, batch, dtype_bytes=2):
    return 2 * n_layer * n_head * head_dim * seq_len * batch * dtype_bytes / 1e6

print("this toy model, 96 tokens, batch 1 :",
      f"{kv_megabytes(2, 4, 16, 96, 1):.3f} MB")
print("a 7B-ish model, 8k tokens, batch 16:",
      f"{kv_megabytes(32, 32, 128, 8192, 16)/1000:.1f} GB  <- bigger than the weights")"""))

cells.append(md(r"""---
## 6 · Where next

- **Paged attention** — instead of one contiguous growing array per sequence, store the KV
  cache in fixed-size *pages* (like OS virtual memory). This is vLLM's core trick; it lets
  many sequences share GPU memory without fragmentation. Next subfolder: `inference-engine`.
- **KV quantization** — compress cached K/V with `../turbo-quant` and measure the recall hit.
- **Sliding-window / RoPE** — our learned positional embeddings cap context at `block_size`.
  Rotary embeddings (`../rope`) remove that cap and interact with the cache cleanly.

You now understand the optimization that makes LLM serving affordable — from the inside.
"""))

cells.append(md(r"""---
## 7 · Summary

| Piece | Where | Job |
|---|---|---|
| `KVCache` | `kvcache.py` | grow per-layer K/V one token at a time |
| `decode_step` | `kvcache.py` | one token in → next-token logits, using the cache |
| `verify_equivalence` | `kvcache.py` | proof: cached logits == full-forward logits |
| naive vs cached | this notebook | same output, O(n²) vs O(n) cost |

**One line:** cache the past keys and values, compute only the new token, and generation
goes from quadratic to linear — with byte-identical output.
"""))

nb.cells = cells
with open("kvcache_explainer.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote kvcache_explainer.ipynb ({len(cells)} cells)")
