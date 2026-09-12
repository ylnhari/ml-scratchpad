# KV-Cache From Scratch — Learning Path

> Builds on [`../nanogpt-from-scratch`](../nanogpt-from-scratch). Takes that trained GPT and
> adds the single most important inference optimization — the key/value cache — then **proves**
> it changes speed, not output.

**Goal:** Understand why naive generation is quadratic, how the KV-cache makes it linear, and
why KV-cache *memory* becomes the next bottleneck (the reason KV compression exists).

---

## What You Will Learn

- Why autoregressive generation recomputes the whole prefix every step (O(n²))
- How caching past K/V makes per-step attention O(1) in length → O(n) total
- Why a KV-cache needs no causal mask (it only ever holds the past)
- How to *prove* an optimization is correct (logits identical to ~1e-15)
- Why the KV-cache can grow larger than the model weights — the link to `../turbo-quant`

---

## Setup

```bash
cd ..                # ml-scratchpad/ root
make venv            # once
make kernel          # once
```
Open the notebook and select **Kernel → Change Kernel → Learning (shared)**.

---

## Learning Flow

### Step 1 — Read the engine (10 min)
Open `kvcache.py`. Focus on `KVCache` (grows K/V one token at a time), `decode_step`
(one token in → next-token logits), and `verify_equivalence` (the correctness proof).

**Checkpoint:**
1. Why do past K and V never need recomputing?
2. Why does a cached decoder need no causal mask?
3. How would you prove the cache didn't change the model's output?

### Step 2 — Walk the notebook (45 min)
```bash
cd ..
make kv-cache-notebook
```

| Section | Question |
|---|---|
| §0 · The waste | Where is the recomputed work? |
| §1 · The idea | What exactly do we cache, and why no mask? |
| §2 · Prove it | Are the cached logits identical to a full forward? |
| §3 · The payoff | Same text, how much faster? |
| §4 · O(n²) vs O(n) | Watch per-step cost with and without the cache |
| §5 · The catch | How big does the cache get? |
| §6 · Where next | Paged attention, KV quantization, RoPE |

### Step 3 — Extend
Quantize the cached K/V with `../turbo-quant` and measure the recall/quality hit; or add a
sliding window so generation can exceed `block_size`.

---

## Files
```
kvcache.py                 engine — KVCache + decode_step + equivalence check (read first)
kvcache_explainer.ipynb    teaching notebook — THE deliverable
create_notebook.py         maintenance: regenerates the .ipynb (not learner-facing)
data/                      scratch outputs (gitignored)
```

Verify any time:
```bash
../.venv/Scripts/python kvcache.py   # equivalence check + naive-vs-cached timing
```

---

## License
MIT — Hari Yelesetty
