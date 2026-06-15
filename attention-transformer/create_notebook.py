"""Generates attention_explainer.ipynb — run once, then open the .ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
})

def md(src):   return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md(r"""# Attention Is All You Need — Step-by-Step Visual Guide

**Paper:** [Attention Is All You Need, Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

**Goal:** Take one sentence and walk every word through the full transformer encoder step by step.
See every matrix. Plot every transformation. Understand *why* each piece exists.

By the end you will know:
- What "attention" actually computes — as a matrix operation you can trace by hand
- Why random rotation / scaling matters (sqrt d_k)
- What multi-head attention adds over single-head
- How positional encoding solves "bag of words"
- Where this shows up in GPT, BERT, ViT, Whisper, and your own projects
"""))

# ── §0 The Problem ────────────────────────────────────────────────────────────
cells.append(md(r"""---
## 0 · The Problem: Sequential Processing Breaks at Scale

Before transformers, sequence models were **RNNs** (LSTM, GRU).
They read one token at a time and pass a hidden state forward.

```
"The" → h1 → "cat" → h2 → "sat" → h3 → "on" → h4 → "the" → h5 → "mat" → h6
```

**Problems:**
1. **Not parallelisable** — step n needs step n-1. GPUs sit idle.
2. **Long-range forgetting** — by the time we process "mat", information about "The" is diluted through 5 hidden states.
3. **Fixed bottleneck** — all context must pass through one vector h.

**The transformer's answer:** throw away the sequential assumption entirely.
Let every token attend to every other token *simultaneously*.

```
"The" ─┐    ┌─ "The"
"cat" ─┼────┼─ "cat"
"sat" ─┼────┼─ "sat"   ← every token sees every other token in one step
"on"  ─┼────┼─ "on"
"the" ─┼────┼─ "the"
"mat" ─┘    └─ "mat"
```
"""))

# ── §1 Setup ──────────────────────────────────────────────────────────────────
cells.append(code(r"""import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from attention import (
    softmax, layer_norm,
    scaled_dot_product_attention,
    MultiHeadAttention,
    positional_encoding,
    FeedForward,
    EncoderBlock,
)

plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#2a2d3e", "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0", "xtick.color": "#64748b", "ytick.color": "#64748b",
    "grid.color": "#2a2d3e", "grid.linewidth": 0.6, "axes.grid": False,
    "figure.dpi": 110, "font.size": 11,
})
C0, C1, C2, C3, C4 = "#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6"

np.random.seed(42)
print("Ready.")
"""))

# ── §2 Our sentence ───────────────────────────────────────────────────────────
cells.append(md(r"""---
## 1 · Our Sentence

We will use this sentence through the entire notebook:

> **"The cat sat on the mat"**

6 tokens. Small enough to visualise every matrix.
We use `d_model = 8` (embedding dimension) so we can see every number.
"""))

cells.append(code(r"""TOKENS  = ["The", "cat", "sat", "on", "the", "mat"]
SEQ_LEN = len(TOKENS)
D_MODEL = 8
N_HEADS = 2
D_FF    = 16
D_K     = D_MODEL // N_HEADS  # 4 per head

print(f"Tokens:   {TOKENS}")
print(f"Sequence length: {SEQ_LEN}")
print(f"d_model = {D_MODEL}  |  n_heads = {N_HEADS}  |  d_k = {D_K}")
print(f"Total input matrix shape: {SEQ_LEN} x {D_MODEL}")
"""))

# ── §3 Word Embeddings ────────────────────────────────────────────────────────
cells.append(md(r"""---
## 2 · Word Embeddings — Tokens Become Vectors

Each token is first looked up in an **embedding table** (a learned matrix).
Here we use random embeddings fixed by seed — in a real model these are learned.

Every word becomes a `d_model`-dimensional vector.
Similar words end up with similar vectors after training.
"""))

cells.append(code(r"""rng = np.random.default_rng(42)
# embedding table: one vector per token (in practice: vocab_size x d_model)
E = rng.standard_normal((SEQ_LEN, D_MODEL)) * 0.5
# Normalise to unit norm (common initialisation)
E = E / np.linalg.norm(E, axis=1, keepdims=True)

print("Embedding matrix E  (shape: seq_len x d_model):")
print(E.round(3))
"""))

cells.append(code(r"""fig, ax = plt.subplots(figsize=(11, 3.5))
im = ax.imshow(E, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
ax.set_xticks(range(D_MODEL))
ax.set_xticklabels([f"dim {i}" for i in range(D_MODEL)], fontsize=9)
ax.set_yticks(range(SEQ_LEN))
ax.set_yticklabels(TOKENS, fontsize=11)
ax.set_title('Word embeddings E  — each row is one token\'s 8-dim vector', pad=10)
plt.colorbar(im, ax=ax, fraction=0.02)
for i in range(SEQ_LEN):
    for j in range(D_MODEL):
        ax.text(j, i, f"{E[i,j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(E[i,j]) > 0.5 else "#0f1117")
plt.tight_layout(); plt.show()
print("Each row = one word's representation. Columns = learned features (abstract).")
"""))

# ── §4 Positional Encoding ────────────────────────────────────────────────────
cells.append(md(r"""---
## 3 · Positional Encoding — Telling the Model Where Words Are

Without position information, the transformer treats tokens as a **bag of words**.
"The cat sat on the mat" and "the mat sat on the cat" would look identical.

**Solution:** add a unique positional signal to each embedding.

$$PE_{(pos,\, 2i)}   = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

**Why sine/cosine?**
- Deterministic (no extra parameters to learn)
- Every position gets a unique pattern
- Nearby positions have similar patterns (smooth continuity)
- The model can extrapolate to sequence lengths not seen during training
"""))

cells.append(code(r"""PE = positional_encoding(SEQ_LEN, D_MODEL)

print("Positional encoding PE  (shape: seq_len x d_model):")
print(PE.round(3))
"""))

cells.append(code(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Left: heatmap of PE
ax = axes[0]
im = ax.imshow(PE, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
ax.set_xticks(range(D_MODEL))
ax.set_xticklabels([f"d{i}" for i in range(D_MODEL)], fontsize=9)
ax.set_yticks(range(SEQ_LEN))
ax.set_yticklabels([f"{i}: {t}" for i, t in enumerate(TOKENS)], fontsize=10)
ax.set_title("Positional encoding PE", pad=8)
plt.colorbar(im, ax=ax, fraction=0.02)
for i in range(SEQ_LEN):
    for j in range(D_MODEL):
        ax.text(j, i, f"{PE[i,j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(PE[i,j]) > 0.5 else "#0f1117")

# Right: show the sine waves
ax = axes[1]
dims_to_show = [0, 2, 4, 6]   # even dims (sine)
pos_range = np.arange(20)      # show first 20 positions
for d in dims_to_show:
    pe_ext = positional_encoding(20, D_MODEL)
    ax.plot(pos_range, pe_ext[:, d], marker="o", ms=5, lw=2,
            label=f"dim {d} (freq {1/10000**(d/D_MODEL):.4f})")
ax.axvspan(0, SEQ_LEN - 0.5, alpha=0.1, color=C0, label="our 6 tokens")
ax.set_xlabel("Token position"); ax.set_ylabel("Sine value")
ax.set_title("Sine waves at different frequencies\n(each dim = different frequency)")
ax.legend(fontsize=9, loc="upper right",
          facecolor="#1a1d27", edgecolor="#2a2d3e", labelcolor="#e2e8f0")

plt.suptitle("Key insight: position 0 and position 1 have similar PE → model knows they're adjacent",
             y=1.01, fontsize=11)
plt.tight_layout(); plt.show()
"""))

cells.append(code(r"""# Add positional encoding to embeddings
X = E + PE   # (seq_len, d_model) — this is the actual input to the transformer

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
for ax, mat, title in zip(axes,
    [E,  PE,  X],
    ["E (word embedding)", "PE (positional)", "X = E + PE  (transformer input)"]):
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(D_MODEL)); ax.set_xticklabels([f"d{j}" for j in range(D_MODEL)], fontsize=8)
    ax.set_yticks(range(SEQ_LEN)); ax.set_yticklabels(TOKENS)
    ax.set_title(title, pad=6)
    plt.colorbar(im, ax=ax, fraction=0.025)
plt.suptitle("X = E + PE  — each token now carries WHAT it is AND WHERE it is", y=1.02)
plt.tight_layout(); plt.show()
"""))

# ── §5 QKV ────────────────────────────────────────────────────────────────────
cells.append(md(r"""---
## 4 · Query, Key, Value — The Three Projections

This is the conceptual heart of attention. Each token produces **three** vectors:

| Vector | Analogy | Role |
|--------|---------|------|
| **Q** (Query) | "What am I looking for?" | Token asking a question |
| **K** (Key)   | "What do I contain?" | Token advertising its content |
| **V** (Value) | "What I actually give you" | Token's actual contribution |

Each is a learned linear projection of X:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

$W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_{model}}$ are learned during training.

**Intuition:** for the word "sat", its Query might ask "who/what did the sitting?"
Its Key might say "I'm a verb". When "cat"'s Query matches "sat"'s Key, attention flows.
"""))

cells.append(code(r"""mha = MultiHeadAttention(D_MODEL, N_HEADS, seed=42)

# Raw projections (before splitting into heads)
Q_full = X @ mha.W_Q   # (6, 8)
K_full = X @ mha.W_K   # (6, 8)
V_full = X @ mha.W_V   # (6, 8)

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, mat, label in zip(axes,
    [Q_full, K_full, V_full],
    ["Q = X @ W_Q\n(What am I looking for?)",
     "K = X @ W_K\n(What do I contain?)",
     "V = X @ W_V\n(What I give if attended to)"]):
    im = ax.imshow(mat, aspect="auto", cmap="RdBu")
    ax.set_xticks(range(D_MODEL)); ax.set_xticklabels([f"d{j}" for j in range(D_MODEL)], fontsize=8)
    ax.set_yticks(range(SEQ_LEN)); ax.set_yticklabels(TOKENS)
    ax.set_title(label, pad=6)
    plt.colorbar(im, ax=ax, fraction=0.025)
plt.suptitle("Three projections of the same input X — each head will use d_k=4 columns",
             y=1.02, fontsize=11)
plt.tight_layout(); plt.show()
"""))

# ── §6 Attention scores ───────────────────────────────────────────────────────
cells.append(md(r"""---
## 5 · Attention Scores — Q · Kᵀ

Now we compute how much every token should attend to every other token.

$$\text{scores} = Q \cdot K^\top \in \mathbb{R}^{seq \times seq}$$

`scores[i, j]` = dot product of token i's Query with token j's Key.

- **High score:** token i's question matches token j's advertisement → attend more
- **Low score:** token i is not interested in token j → attend less

We use head 0 (one of our 2 heads) with its `d_k = 4` slice of Q and K.
"""))

cells.append(code(r"""# Head 0 — use first d_k columns of Q and K
Q0 = Q_full[:, :D_K]   # (6, 4)
K0 = K_full[:, :D_K]   # (6, 4)
V0 = V_full[:, :D_K]   # (6, 4)

# Raw attention scores (before scaling)
scores_raw = Q0 @ K0.T   # (6, 6)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
im = ax.imshow(scores_raw, cmap="viridis")
ax.set_xticks(range(SEQ_LEN)); ax.set_xticklabels(TOKENS, rotation=30, ha="right")
ax.set_yticks(range(SEQ_LEN)); ax.set_yticklabels(TOKENS)
ax.set_title("Raw scores  Q · Kᵀ\n(before sqrt scaling)", pad=6)
plt.colorbar(im, ax=ax, fraction=0.04)
for i in range(SEQ_LEN):
    for j in range(SEQ_LEN):
        ax.text(j, i, f"{scores_raw[i,j]:.2f}", ha="center", va="center", fontsize=8,
                color="white")

# Scaled
scores_scaled = scores_raw / np.sqrt(D_K)
ax = axes[1]
im = ax.imshow(scores_scaled, cmap="viridis")
ax.set_xticks(range(SEQ_LEN)); ax.set_xticklabels(TOKENS, rotation=30, ha="right")
ax.set_yticks(range(SEQ_LEN)); ax.set_yticklabels(TOKENS)
ax.set_title(f"Scaled scores  Q · Kᵀ / √d_k  (÷ {np.sqrt(D_K):.2f})", pad=6)
plt.colorbar(im, ax=ax, fraction=0.04)
for i in range(SEQ_LEN):
    for j in range(SEQ_LEN):
        ax.text(j, i, f"{scores_scaled[i,j]:.2f}", ha="center", va="center", fontsize=8,
                color="white")

plt.suptitle("Scores[i,j] = how much token i (row) wants to attend to token j (col)",
             y=1.02, fontsize=11)
plt.tight_layout(); plt.show()

print(f"Score range BEFORE scaling: [{scores_raw.min():.3f}, {scores_raw.max():.3f}]")
print(f"Score range AFTER  scaling: [{scores_scaled.min():.3f}, {scores_scaled.max():.3f}]")
"""))

# ── §7 Why sqrt(d_k) ─────────────────────────────────────────────────────────
cells.append(md(r"""---
## 6 · Why Divide by √d_k? — The Softmax Saturation Problem

**The problem:** dot products grow with dimension.
If Q and K have random entries with variance 1, then $Q \cdot K$ has variance $d_k$.
Large values → softmax returns near-one-hot distribution → gradients vanish.

**Demonstration below:** as dot product magnitude grows, softmax "sharpens" until
one weight → 1 and all others → 0. The model stops learning.

Dividing by $\sqrt{d_k}$ keeps variance at 1 regardless of dimension.
"""))

cells.append(code(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: softmax shape at different temperatures
temps = [0.5, 1.0, 2.0, 5.0, 10.0]
x_vals = np.array([2.0, 0.5, -0.3, 1.2, -1.5, 0.1])   # 6 scores

ax = axes[0]
for t in temps:
    sm = softmax(x_vals * t)
    ax.plot(range(len(x_vals)), sm, "o-", lw=2, ms=7, label=f"scale ×{t}")
ax.set_xticks(range(len(TOKENS))); ax.set_xticklabels(TOKENS, rotation=15)
ax.set_ylabel("Softmax weight"); ax.set_ylim(0, 1.05)
ax.set_title("Effect of score magnitude on softmax\n(temperature = scale applied to same scores)")
ax.legend(fontsize=9, facecolor="#1a1d27", edgecolor="#2a2d3e", labelcolor="#e2e8f0")

# Right: max softmax weight as dimension grows (no scaling vs with scaling)
dims = np.arange(1, 200, 5)
max_weights_unscaled = []
max_weights_scaled   = []
rng2 = np.random.default_rng(0)
for d in dims:
    q = rng2.standard_normal(d)
    k = rng2.standard_normal((20, d))
    scores_u = q @ k.T                        # unscaled
    scores_s = q @ k.T / np.sqrt(d)           # scaled
    max_weights_unscaled.append(softmax(scores_u).max())
    max_weights_scaled.append(softmax(scores_s).max())

ax = axes[1]
ax.plot(dims, max_weights_unscaled, color=C3, lw=2, label="No scaling  → saturates")
ax.plot(dims, max_weights_scaled,   color=C0, lw=2, label="÷ √d_k  → stays diffuse")
ax.axhline(1/20, color="#64748b", ls="--", lw=1, label="Uniform (1/20=0.05)")
ax.set_xlabel("d_k (dimension)"); ax.set_ylabel("Max softmax weight")
ax.set_title("As d_k grows, unscaled softmax → one-hot\n÷√d_k keeps gradients alive")
ax.legend(fontsize=9, facecolor="#1a1d27", edgecolor="#2a2d3e", labelcolor="#e2e8f0")
ax.set_ylim(0, 1.05)

plt.suptitle("WHY √d_k scaling: prevents softmax saturation → enables gradient flow during training",
             y=1.02, fontsize=11)
plt.tight_layout(); plt.show()
"""))

# ── §8 Attention weights ──────────────────────────────────────────────────────
cells.append(md(r"""---
## 7 · Softmax → Attention Weights

Apply softmax row-wise so each token's attention sums to 1.

$$A = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) \in [0,1]^{seq \times seq}$$

Row $i$ tells us: **when outputting token $i$, how much do I attend to each position?**

- $A_{ij} = 0.9$ → token $i$ mostly uses information from token $j$
- Rows sum to 1.0 (probability distribution over sequence positions)
"""))

cells.append(code(r"""weights_h0 = softmax(scores_scaled, axis=-1)   # (6, 6)

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(weights_h0, cmap="Blues", vmin=0, vmax=weights_h0.max())
ax.set_xticks(range(SEQ_LEN))
ax.set_xticklabels([f"{t}\n(key)" for t in TOKENS], fontsize=10)
ax.set_yticks(range(SEQ_LEN))
ax.set_yticklabels([f"{t} (query)" for t in TOKENS], fontsize=10)
ax.set_title('Attention weights A = softmax(QKᵀ / √d_k)\nEach row sums to 1.0', pad=10)
ax.set_xlabel("Attends TO (Key token)"); ax.set_ylabel("Attends FROM (Query token)")
plt.colorbar(im, ax=ax, fraction=0.03)
for i in range(SEQ_LEN):
    for j in range(SEQ_LEN):
        ax.text(j, i, f"{weights_h0[i,j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if weights_h0[i,j] > 0.3 else "#e2e8f0")
plt.tight_layout(); plt.show()
print("Row sums:", weights_h0.sum(axis=1).round(4), "← all 1.0 ✓")
"""))

# ── §9 Weighted sum V ─────────────────────────────────────────────────────────
cells.append(md(r"""---
## 8 · Weighted Sum of Values — The Output

Finally, use the attention weights to form a weighted combination of Values:

$$\text{output} = A \cdot V$$

For token $i$:
$$\text{output}_i = \sum_j A_{ij} \cdot V_j$$

If token "cat" strongly attends to "sat" (verb), its output vector becomes
a blend of its own Value and "sat"'s Value — it has *gathered* context.

**This is attention:** contextualising each token's representation using
information from every other token in the sequence.
"""))

cells.append(code(r"""output_h0 = weights_h0 @ V0   # (6, 4)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, mat, title in zip(axes,
    [weights_h0, V0, output_h0],
    ["A (attention weights)\n(6×6)", "V (values)\n(6×4)", "Output = A @ V\n(6×4)"]):
    im = ax.imshow(mat, aspect="auto", cmap="RdBu")
    ax.set_yticks(range(SEQ_LEN)); ax.set_yticklabels(TOKENS)
    ax.set_title(title, pad=6)
    plt.colorbar(im, ax=ax, fraction=0.03)
plt.suptitle("output[i] = weighted sum of all token values, weighted by attention to each",
             y=1.02, fontsize=11)
plt.tight_layout(); plt.show()

print("Single-head attention output shape:", output_h0.shape, " (seq_len, d_k)")
print("\nFor token 'cat' (row 1):")
print(f"  Attention weights: {weights_h0[1].round(3)}")
print(f"  → its output is: A[1,:] @ V = {output_h0[1].round(3)}")
"""))

# ── §10 Full single-head ──────────────────────────────────────────────────────
cells.append(md(r"""---
## 9 · Full Single-Head Attention — All Steps Together

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

This is Equation 1 from the paper. Four operations: multiply, scale, softmax, multiply again.
"""))

cells.append(code(r"""output_single, weights_single = scaled_dot_product_attention(
    Q0[None, :, :],   # add batch dim
    K0[None, :, :],
    V0[None, :, :],
)
output_single  = output_single[0]    # (6, 4)
weights_single = weights_single[0]   # (6, 6)

print("Single-head attention:")
print(f"  Input X shape:       {X.shape}")
print(f"  Q0, K0, V0 shape:   {Q0.shape}")
print(f"  Attention weights:   {weights_single.shape}")
print(f"  Output shape:        {output_single.shape}")
print(f"\nMax attention weight per row (who each token focuses on most):")
for i, tok in enumerate(TOKENS):
    j = weights_single[i].argmax()
    print(f"  '{tok}' focuses most on '{TOKENS[j]}' ({weights_single[i,j]:.3f})")
"""))

# ── §11 Multi-head ────────────────────────────────────────────────────────────
cells.append(md(r"""---
## 10 · Multi-Head Attention — Parallel Perspectives

**Why one head is not enough:**
One attention head can only model one type of relationship at a time.
"cat" might need to find its subject relationship AND its co-reference with "the".
One head can't do both simultaneously.

**Multi-head attention:**
Split $d_{model}$ into $h$ heads, each with $d_k = d_{model}/h$.
Run attention *in parallel* in each sub-space.
Each head can specialise in a different linguistic or semantic pattern.
Concatenate outputs and project back to $d_{model}$.

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$
$$\text{where head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)$$
"""))

cells.append(code(r"""output_mha, weights_mha = mha.forward(X)
# output_mha:  (6, 8)        — same shape as input X
# weights_mha: (2, 6, 6)     — one (6,6) attention map per head

print(f"Multi-head attention output shape: {output_mha.shape}")
print(f"Attention weights shape: {weights_mha.shape}  (n_heads, seq_len, seq_len)")
"""))

cells.append(code(r"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for h, ax in enumerate(axes):
    im = ax.imshow(weights_mha[h], cmap="Blues", vmin=0, vmax=weights_mha[h].max())
    ax.set_xticks(range(SEQ_LEN))
    ax.set_xticklabels([f"{t}" for t in TOKENS], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(SEQ_LEN))
    ax.set_yticklabels(TOKENS, fontsize=10)
    ax.set_title(f"Head {h+1} attention pattern\n(different sub-space, different pattern)", pad=8)
    plt.colorbar(im, ax=ax, fraction=0.04)
    for i in range(SEQ_LEN):
        for j in range(SEQ_LEN):
            ax.text(j, i, f"{weights_mha[h,i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if weights_mha[h,i,j] > 0.3 else "#e2e8f0")

plt.suptitle(
    "Two heads, two different attention patterns\n"
    "Real models: head 1 might track syntax, head 2 might track co-reference",
    y=1.02, fontsize=11)
plt.tight_layout(); plt.show()

print("Similarity between head patterns:")
flat1, flat2 = weights_mha[0].flatten(), weights_mha[1].flatten()
print(f"  Pearson r = {np.corrcoef(flat1, flat2)[0,1]:.4f}  (low = they learned different things)")
"""))

# ── §12 Residual + LayerNorm ──────────────────────────────────────────────────
cells.append(md(r"""---
## 11 · Add & Normalise (Residual Connection + LayerNorm)

After attention, two operations happen before the FFN:

### Residual Connection
$$X' = X + \text{MultiHeadAttention}(X)$$

**Why:** ensures gradients flow directly back to early layers (solves vanishing gradients).
The block learns a *delta* on top of X, not a full transform.
Think: "what new information did attention bring, in addition to what X already knew?"

### Layer Normalisation
$$X'' = \text{LayerNorm}(X') = \frac{X' - \mu}{\sigma + \epsilon}$$

Normalises each token's embedding to zero mean, unit variance.
Keeps activations in a stable range, speeds up training.
"""))

cells.append(code(r"""X_after_attn = layer_norm(X + output_mha)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, mat, title in zip(axes,
    [X, output_mha, X_after_attn],
    ["X (input)", "MHA output", "LayerNorm(X + MHA output)"]):
    im = ax.imshow(mat, aspect="auto", cmap="RdBu")
    ax.set_yticks(range(SEQ_LEN)); ax.set_yticklabels(TOKENS)
    ax.set_xticks(range(D_MODEL)); ax.set_xticklabels([f"d{j}" for j in range(D_MODEL)], fontsize=8)
    ax.set_title(title, pad=6)
    plt.colorbar(im, ax=ax, fraction=0.025)
plt.suptitle("Residual + LayerNorm: X keeps its identity, MHA adds contextual information",
             y=1.02)
plt.tight_layout(); plt.show()

print(f"Mean per token BEFORE LayerNorm: {(X + output_mha).mean(axis=1).round(3)}")
print(f"Mean per token AFTER  LayerNorm: {X_after_attn.mean(axis=1).round(3)}  ← near 0")
print(f"Std  per token AFTER  LayerNorm: {X_after_attn.std(axis=1).round(3)}   ← near 1")
"""))

# ── §13 FFN ───────────────────────────────────────────────────────────────────
cells.append(md(r"""---
## 12 · Feed-Forward Network — Processing Each Token Independently

After attention gathers context, the FFN *processes* each token's updated representation.

$$\text{FFN}(x) = \max(0,\; x W_1 + b_1) \cdot W_2 + b_2$$

- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$
- Typically $d_{ff} = 4 \times d_{model}$ (expand then compress)
- Applied **identically and independently** to each token position
- ReLU non-linearity adds expressive power

**Attention vs FFN — what each does:**
- **Attention:** mixes information *across* tokens (context gathering)
- **FFN:** processes each token's representation *in place* (feature transformation)

Both are needed. The FFN is where most of the model's "knowledge" is stored.
"""))

cells.append(code(r"""ff = FeedForward(D_MODEL, D_FF, seed=42)
X_after_ff = ff.forward(X_after_attn)
X_final    = layer_norm(X_after_attn + X_after_ff)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, mat, title in zip(axes,
    [X_after_attn, X_after_ff, X_final],
    ["After attention+norm\n(input to FFN)",
     "FFN output\n(token-wise transformation)",
     "LayerNorm(X + FFN)\n(encoder block output)"]):
    im = ax.imshow(mat, aspect="auto", cmap="RdBu")
    ax.set_yticks(range(SEQ_LEN)); ax.set_yticklabels(TOKENS)
    ax.set_xticks(range(D_MODEL)); ax.set_xticklabels([f"d{j}" for j in range(D_MODEL)], fontsize=8)
    ax.set_title(title, pad=6)
    plt.colorbar(im, ax=ax, fraction=0.025)
plt.suptitle("FFN: each token processed independently after attention has mixed context",
             y=1.02)
plt.tight_layout(); plt.show()
"""))

# ── §14 Full encoder block ────────────────────────────────────────────────────
cells.append(md(r"""---
## 13 · The Full Encoder Block — Putting It All Together

One encoder block = Multi-Head Attention + Add&Norm + FFN + Add&Norm.

```
X (seq_len, d_model)
│
├──→ Multi-Head Self-Attention ──→ + ──→ LayerNorm ──→ X'
│                                  ↑
│                                  X (residual)
│
├──→ Feed-Forward Network ──────→ + ──→ LayerNorm ──→ output
│                                  ↑
│                                  X' (residual)
```

BERT uses 12 of these stacked. GPT-3 uses 96. Each block refines the representations.
"""))

cells.append(code(r"""block = EncoderBlock(D_MODEL, N_HEADS, D_FF, seed=42)
X_out, block_weights = block.forward(X)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
im = ax.imshow(X_out, aspect="auto", cmap="RdBu")
ax.set_yticks(range(SEQ_LEN)); ax.set_yticklabels(TOKENS)
ax.set_xticks(range(D_MODEL)); ax.set_xticklabels([f"d{j}" for j in range(D_MODEL)], fontsize=8)
ax.set_title("Encoder block output\nEach token now holds context from all others", pad=8)
plt.colorbar(im, ax=ax, fraction=0.025)

ax = axes[1]
diff = X_out - X
im = ax.imshow(diff, aspect="auto", cmap="RdBu")
ax.set_yticks(range(SEQ_LEN)); ax.set_yticklabels(TOKENS)
ax.set_xticks(range(D_MODEL)); ax.set_xticklabels([f"d{j}" for j in range(D_MODEL)], fontsize=8)
ax.set_title("Delta: output - input\n(What the encoder block added)", pad=8)
plt.colorbar(im, ax=ax, fraction=0.025)

plt.suptitle("One encoder block transforms X  →  contextualised X_out  (same shape)",
             y=1.02, fontsize=11)
plt.tight_layout(); plt.show()

print(f"Input shape:  {X.shape}")
print(f"Output shape: {X_out.shape}  ← identical shape (can stack N blocks)")
"""))

# ── §15 Use cases ─────────────────────────────────────────────────────────────
cells.append(md(r"""---
## 14 · Where This Lives in Modern AI

Every major LLM, vision model, and speech model is built from these same blocks.

| Model | Input tokens | Key use of attention |
|-------|-------------|----------------------|
| **BERT** | Words | Bidirectional: every word attends to every other |
| **GPT / LLaMA** | Words | Causal (masked): token i only attends to 0…i |
| **ViT** | Image patches | Patches attend to patches → vision understanding |
| **Whisper** | Audio spectrogram | Cross-attention: decoder attends to audio encoder |
| **DALL-E / Stable Diffusion** | Pixels + text | Cross-attention: image attends to text tokens |
| **AlphaFold 2** | Amino acids | Self-attention over protein sequence + structure |

### The two variants of attention

**Self-attention** (what we built): Q, K, V all come from the same sequence.
Used in encoder (BERT) and decoder (GPT, LLaMA).

**Cross-attention**: Q from sequence A, K/V from sequence B.
Used in encoder-decoder models (translation, Whisper, T5).
Lets the decoder "look at" the encoder's output while generating.
"""))

cells.append(code(r"""# DEMO: Causal (masked) attention — how GPT/LLaMA work
# Tokens can only attend to PAST tokens, not future ones.
# Enforced by a mask that sets future positions to -inf before softmax.

# Upper-triangular mask: position i cannot attend to position j > i
mask = np.triu(np.ones((SEQ_LEN, SEQ_LEN), dtype=bool), k=1)
print("Causal mask (True = blocked):")
print(mask.astype(int))
print("\nThis ensures autoregressive generation: predict token t using only tokens 0..t-1")

_, weights_causal = scaled_dot_product_attention(
    Q0[None], K0[None], V0[None], mask=mask[None]
)
weights_causal = weights_causal[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, w, title in zip(axes,
    [weights_single, weights_causal],
    ["Full self-attention (BERT/encoder)\nAll positions visible",
     "Causal masked attention (GPT/decoder)\nOnly past positions visible"]):
    im = ax.imshow(w, cmap="Blues", vmin=0)
    ax.set_xticks(range(SEQ_LEN)); ax.set_xticklabels(TOKENS, rotation=30, ha="right")
    ax.set_yticks(range(SEQ_LEN)); ax.set_yticklabels(TOKENS)
    ax.set_title(title, pad=8)
    plt.colorbar(im, ax=ax, fraction=0.04)
    for i in range(SEQ_LEN):
        for j in range(SEQ_LEN):
            val = w[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if val > 0.3 else "#e2e8f0")

plt.suptitle("Same mechanism, different mask = BERT vs GPT", y=1.02, fontsize=12)
plt.tight_layout(); plt.show()
"""))

# ── §16 Use in your projects ──────────────────────────────────────────────────
cells.append(md(r"""---
## 15 · Use Cases for Your Projects

### When does attention run under the hood?

| Your action | What attention computes |
|-------------|------------------------|
| `client.chat()` → LLM response | Causal self-attention over your prompt tokens |
| RAG retrieval → embed document | Encoder self-attention builds the embedding |
| `embed(text)` → vector | The [CLS] token's output after full self-attention |
| Whisper transcription | Cross-attention: decoder attends to audio encoder |
| Image classification (ViT) | Self-attention over 16×16 pixel patches |

### When you might implement attention yourself

| Scenario | What to build |
|----------|---------------|
| Custom sequence model | `EncoderBlock` from `attention.py` |
| Recommendation (users × items) | Cross-attention: user query attends to item keys |
| Multi-modal fusion | Cross-attention between two modalities |
| Time-series anomaly | Self-attention finds which time steps correlate |
| Structured data (tabular) | Attention across feature columns |

### The one-line mental model

> **Attention lets every element of a sequence ask a question (Q),
> and every other element answer (K) and contribute information (V).**
>
> Trained attention = the model has learned *which* questions to ask
> and *what information* each position should contribute.
"""))

# ── §17 Summary ───────────────────────────────────────────────────────────────
cells.append(md(r"""---
## 16 · What the Transformer Encoder Does — Full Summary

| Step | Operation | Input shape | Output shape | Why |
|------|-----------|-------------|--------------|-----|
| 1 | Embed tokens | (seq,) int | (seq, d_model) | Tokens → continuous vectors |
| 2 | Add PE | (seq, d_model) | (seq, d_model) | Inject position information |
| 3 | Project Q, K, V | (seq, d_model) | 3 × (seq, d_model) | Learn what to ask/answer/give |
| 4 | Attention scores | (seq, d_k) × (d_k, seq) | (seq, seq) | How much each token attends to each |
| 5 | Scale ÷√d_k | (seq, seq) | (seq, seq) | Prevent softmax saturation |
| 6 | Softmax | (seq, seq) | (seq, seq) | Normalise to probability over positions |
| 7 | Weighted sum V | (seq, seq) × (seq, d_k) | (seq, d_k) | Gather context |
| 8 | Concat + W_O | h × (seq, d_k) | (seq, d_model) | Merge all heads |
| 9 | Residual + LN | (seq, d_model) | (seq, d_model) | Gradient flow + stability |
| 10 | FFN | (seq, d_model) | (seq, d_model) | Per-token feature transformation |
| 11 | Residual + LN | (seq, d_model) | (seq, d_model) | Gradient flow + stability |

**Stack N of these = BERT (N=12), GPT-3 (N=96), LLaMA-70B (N=80)**
"""))

cells.append(code(r"""# Final verification: run the full encoder block on our sentence
print("FULL ENCODER BLOCK — end to end")
print("=" * 50)
print(f"Input:  'The cat sat on the mat'")
print(f"        {SEQ_LEN} tokens x {D_MODEL} dims = {SEQ_LEN*D_MODEL} numbers")
print()

block2 = EncoderBlock(D_MODEL, N_HEADS, D_FF, seed=99)
out, w  = block2.forward(X)

print(f"Output: same shape {out.shape}")
print(f"        Each token now holds context from all 6 tokens")
print()
print("Dominant attention per token (who each word focused on most):")
for i, tok in enumerate(TOKENS):
    j = w.mean(axis=0)[i].argmax()   # average across heads
    print(f"  '{tok:<5}' → '{TOKENS[j]}'  ({w.mean(axis=0)[i,j]:.3f})")
print()
print("These patterns are random (untrained weights).")
print("After training on text: 'cat' would attend to 'sat' (subject→verb)")
print("'mat' would attend to 'the' immediately before it, etc.")
"""))

nb.cells = cells

with open("attention_explainer.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Created: attention_explainer.ipynb  ({len(cells)} cells)")
