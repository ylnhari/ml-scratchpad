# TurboQuant — Learning Path

> Paper: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (ICLR 2026)](https://arxiv.org/abs/2504.19874)

**Goal:** Understand *why* TurboQuant works well enough to know where to use it in your own projects — without reading the math paper.

---

## What You Will Learn

By the end of this path you will be able to answer:

- What does TurboQuant actually do to a vector, step by step?
- Why does random rotation matter — and what breaks without it?
- How do inner products survive lossy compression?
- When should I use MSE variant vs inner-product variant?
- Where can I drop this into my existing projects (RAG, KV cache, recommenders)?

---

## Setup — Do This Once

Run from the `ml-scratchpad/` root (one shared env for all subfolders):

```bash
cd ..                # go to ml-scratchpad/ root
make venv            # creates .venv and installs all deps
make kernel          # registers "Learning (shared)" Jupyter kernel
```

Then open the notebook and select **Kernel → Change Kernel → Learning (shared)**.

---

## Learning Flow

Work through these in order. Each step builds on the previous.

---

### Step 1 — Read the Algorithm (5 min)

Open `turbo_quant.py` and read top-to-bottom.
You do not need to understand every line.
Focus on the method signatures and their docstrings:

```
__init__    → what gets generated once (Pi, S matrices)
rotate      → what the rotation does
quantize    → MSE-optimal compression
dequantize  → reconstruction
quantize_prod    → inner-product-optimal compression
dequantize_prod  → reconstruction with QJL correction
```

**Checkpoint:** Can you name the 5 steps of TurboQuant from memory?
(Normalise → Rotate → Quantise → Unrotate → QJL residual)

---

### Step 2 — Walk Through the Notebook (45–60 min)

This is the core of the learning path.

```bash
jupyter notebook turbo_quant_explainer.ipynb
```

The notebook takes **one concrete 16-dim vector** and walks it through every step.
At each step: math → code → plot → insight.

**Run cells top to bottom. Do not skip.**

| Section | What you learn |
|---------|---------------|
| §0 · The Problem | Why vector storage/retrieval costs blow up |
| §1 · Meet the vector | What a concrete embedding looks like |
| §2 · Normalise | Why unit-sphere matters; cost is one float |
| §3 · Random rotation | The key innovation — why energy must be equalised |
| §4 · Lloyd-Max quantisation | How centroids compress float → b bits |
| §5 · Dequantise | Reconstruction quality at different bit-widths |
| §6 · Inner-product bias | Why MSE variant breaks nearest-neighbour search |
| §7 · QJL residual | How 1-bit sign sketch fixes the bias |
| §8 · Unbiasedness proof | Scatter plots showing MSE vs prod variant |
| §9 · Full pipeline | Everything together; compression ratio table |
| §10.1 · Vector search | Recall@k table — what bit-width gives what quality |
| §10.2 · KV cache | Attention heatmaps before/after compression |
| §10.3 · Recommenders + RAG | Memory math for billion-item systems |
| §11 · Decision guide | ASCII flowchart: is TurboQuant right for my project? |
| §12 · Summary | Full step table + end-to-end pipeline code |

**Stop at §11.** Before running it, answer for yourself:
*"In which of my current projects do I store or query dense vectors?"*

---

All visuals — MSE-vs-bit-width curve, inner-product scatter, recall@k, KV-cache attention
heatmaps, compression trade-off, the rotation energy-equalisation plot — live **inside the
notebook**, next to the step they explain. There is no separate figure to run.

---

### Step 3 — Apply It (open-ended)

Use `turbo_quant.py` directly in your project:

```python
from turbo_quant import TurboQuant

tq = TurboQuant(d=YOUR_DIM, bits=2)

# Compress at index time
indices, norms = tq.quantize(your_vectors)          # MSE variant
# or
idx, qjl, gamma, norms = tq.quantize_prod(your_vectors)  # Prod variant

# Query at runtime — same dot product, nothing else changes
reconstructed = tq.dequantize(indices, norms)
scores = query @ reconstructed.T
```

**When to use which variant:**

| Use case | Variant |
|----------|---------|
| Minimise reconstruction error (weight storage) | MSE |
| Vector search / ANN retrieval | Prod |
| KV cache compression | Prod |
| Recommender scoring | Prod |
| RAG document index | Prod |

---

## Files

```
turbo_quant.py              core algorithm — read this first (Step 1)
turbo_quant_explainer.ipynb teaching notebook — the main deliverable (Step 2)
create_notebook.py          maintenance: regenerates the notebook (not learner-facing)
data/                       scratch outputs (gitignored)
```

---

## Quick Reference

```bash
# open the notebook
jupyter notebook turbo_quant_explainer.ipynb
```

---

## Compression Sweet Spot (d = 256)

| Bit-width | Compression | Recall@10 | Use when |
|-----------|-------------|-----------|----------|
| 1-bit | ~28× | ~70% | Memory critical, quality secondary |
| **2-bit** | **~10×** | **~95%** | **Recommended default** |
| 3-bit | ~8× | ~99% | Quality critical |
| 4-bit | ~6× | ~99.9% | Near-lossless |

---

## License

MIT — Hari Yelesetty
