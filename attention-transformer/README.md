# Attention Is All You Need — Learning Path

> Paper: [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) — the paper that started the LLM era.

**Goal:** Understand transformer self-attention well enough to recognise where it applies in your own projects.

---

## What You Will Learn

- What "attention" actually computes — as matrix operations you can trace by hand
- Why random rotation and √d_k scaling matters
- What multi-head attention adds over single-head
- How positional encoding solves the "bag of words" problem
- Where this runs under the hood in GPT, BERT, ViT, Whisper, and your own AI integrations

---

## Setup

Run from the `ml-scratchpad/` root (one shared env for all subfolders):

```bash
cd ../               # go to ml-scratchpad/ root
make venv            # creates .venv and installs all deps
make kernel          # registers "Learning (shared)" Jupyter kernel
```

Then open the notebook and select **Kernel → Change Kernel → Learning (shared)**.

---

## Learning Flow

### Step 1 — Read the Implementation (10 min)

Open `attention.py` and read top-to-bottom.
Focus on function signatures, arguments, return shapes, and the one-line comment above each function explaining *why* it exists.

**Checkpoint — answer from memory:**
1. What are Q, K, V and where do they come from?
2. Why divide by √d_k?
3. What does multi-head attention add?
4. What does the FFN do that attention doesn't?

---

### Step 2 — Walk Through the Notebook (60–90 min)

```bash
cd ../   # from ml-scratchpad/ root
make attention-transformer-notebook
```

Takes one sentence — `"The cat sat on the mat"` — through every transformer step.
Every matrix is printed and plotted. Every design choice is explained.

**Run cells top to bottom. Do not skip.**

| Section | Core question answered |
|---------|----------------------|
| §0 · The Problem | Why did RNNs fail? What does attention solve? |
| §1 · Our sentence | What are the input shapes? |
| §2 · Word embeddings | How do tokens become vectors? |
| §3 · Positional encoding | How does the model know word order? |
| §4 · Q, K, V projections | What does each projection "ask" and "offer"? |
| §5 · Attention scores QKᵀ | How is similarity computed? |
| §6 · Why √d_k? | What breaks without scaling? (plot shows it) |
| §7 · Softmax → weights | How do scores become probabilities? |
| §8 · Weighted sum V | What does "attending to" actually produce? |
| §9 · Single-head end-to-end | All steps together |
| §10 · Multi-head attention | Why multiple heads? What do they each learn? |
| §11 · Residual + LayerNorm | Why "Add & Norm"? |
| §12 · Feed-forward layer | Attention vs FFN — what each contributes |
| §13 · Full encoder block | Everything stacked |
| §14 · BERT vs GPT | Full vs causal mask — same mechanism, different use |
| §15 · Use in your projects | Where this runs in code you already write |
| §16 · Summary table | All shapes and operations in one view |

**Stop at §15.** Before running it, ask:
*"In my projects, where do I call an API that internally runs attention?"*

All visuals (positional-encoding heatmaps, attention weight maps, the √d_k saturation
plot, multi-head comparison, causal mask) live **inside the notebook**, next to the
step they explain. There is no separate figure to run.

---

### Step 3 — Apply It (open-ended)

```python
from attention import EncoderBlock, positional_encoding

block = EncoderBlock(d_model=768, n_heads=12, d_ff=3072)
X     = your_embeddings + positional_encoding(seq_len, 768)
out, attn_weights = block.forward(X)
```

**Use cases to look for in your projects:**

| You do this | Attention runs here |
|-------------|---------------------|
| Call `client.chat()` | Causal self-attention over prompt tokens |
| Embed text for RAG | Encoder self-attention → CLS token output |
| Use Whisper | Cross-attention: decoder attends to audio encoder |
| Use CLIP / ViT | Self-attention over image patches |

---

## Files

```
attention.py                  core implementation — read first
attention_explainer.ipynb     teaching notebook — the main deliverable
create_notebook.py            maintenance: regenerates the notebook (not learner-facing)
data/                         scratch outputs (gitignored)
```

---

## Quick Reference

```bash
# From ml-scratchpad/ root:
make attention-transformer-notebook   # open the explainer
```
