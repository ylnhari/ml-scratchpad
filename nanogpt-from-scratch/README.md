# nanoGPT From Scratch — Learning Path

> Builds directly on [`../attention-transformer`](../attention-transformer). That path taught
> the transformer *forward* pass. This one adds **training**: a full GPT plus
> **backpropagation by hand**, so you can watch a model actually learn — in pure NumPy.

**Goal:** Understand a decoder-only transformer well enough to build, train, and debug one
yourself — the "framework builder, not just framework user" depth.

---

## What You Will Learn

- How next-token prediction + cross-entropy loss turns text into a training signal
- What every layer's `backward` is, and how chaining them = backpropagation
- How to *prove* hand-written gradients are correct (numerical gradient check)
- How Adam turns gradients into learning; how to read a loss curve
- How generation (autoregressive sampling) works, and what `temperature` does
- Where this is identical to real GPT-2 — and the exact next steps toward inference internals

---

## Setup

Run from the `learning/` root (one shared env for all subfolders):

```bash
cd ..                # go to learning/ root
make venv            # creates .venv and installs all deps
make kernel          # registers "Learning (shared)" Jupyter kernel
```

Then open the notebook and select **Kernel → Change Kernel → Learning (shared)**.

---

## Learning Flow

### Step 1 — Read the engine (15 min)

Open `nanogpt.py` top-to-bottom. Focus on the shape comments and on how each layer's
`backward` mirrors its `forward`. Don't memorise the gradient algebra — just see the pattern.

**Checkpoint — answer from memory:**
1. What single quantity does the model output for each position, and what is the loss?
2. What does a layer's `backward` receive, and what does it return?
3. Why must the attention be *causal* during training?
4. How would you convince a skeptic your hand-derived gradients are correct?

### Step 2 — Walk the notebook (60–90 min)

```bash
cd ..                              # from learning/ root
make nanogpt-notebook
```

Run cells top to bottom. Do not skip.

| Section | Core question answered |
|---|---|
| §0 · The one job | What does a GPT actually predict? |
| §1 · Data + tokenizer | How does text become integer tokens? |
| §2 · Build the model | What are the shapes from tokens → logits? |
| §3 · Backprop + gradient check | How do gradients flow, and how do I trust them? |
| §4 · Train | Watch the loss fall (plotted live) |
| §5 · Generate | Sample text; what does temperature do? |
| §6 · Attention pattern | The causal mask, and what training changed |
| §7 · Experiment | Turn the knobs — the whole point of pure NumPy |
| §8 · Where next | KV-cache, RoPE, a from-scratch inference engine |
| §9 · Summary | Every piece in one table |

**Stop at §7 and actually experiment.** Change `n_layer`, `block_size`, the learning rate;
break the residuals on purpose. That is where understanding turns from words into intuition.

### Step 3 — Apply / extend (open-ended)

```python
from nanogpt import GPT, Adam, CharTokenizer, get_batch
# swap in your own text, train, and generate — same code, real data
```

Then take the next step from §8: add a KV-cache to `generate`, or swap learned positional
embeddings for RoPE.

---

## Files

```
nanogpt.py                 core engine — GPT + manual backprop + Adam (read first)
nanogpt_explainer.ipynb    teaching notebook — THE deliverable (all visuals inline)
create_notebook.py         maintenance: regenerates the .ipynb (not learner-facing)
data/                      scratch outputs (gitignored)
```

Verify the engine at any time:

```bash
../.venv/Scripts/python nanogpt.py     # prints gradient-check error, trains, samples
```

---

## License

MIT — Hari Yelesetty
