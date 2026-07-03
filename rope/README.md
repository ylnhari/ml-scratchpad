# RoPE From Scratch — Learning Path

> Follows `../nanogpt-from-scratch` (learned positions) and `../kv-cache`. Replaces the
> position *lookup table* with **rotary embeddings**: encode position by rotating Q and K, so
> attention depends on *relative* distance and works at any length.

**Goal:** Understand the positional scheme used by nearly every modern open LLM (Llama, Mistral,
Qwen, DeepSeek) — and why it enables context-length extrapolation.

---

## What You Will Learn

- Why adding learned/absolute positions caps length and hides relative position
- The 2D-rotation idea behind RoPE, and the multi-frequency "clock" of dimension pairs
- How to *prove* the score q·k depends only on the offset (j − i)
- Why RoPE extrapolates to positions far beyond training (no table), and how context-length
  tricks (PI, NTK, YaRN) build on it

---

## Setup

```bash
cd ..
make venv      # once
make kernel    # once
```
Open the notebook → **Kernel → Change Kernel → Learning (shared)**.

---

## Learning Flow

### Step 1 — Read the engine (10 min)
Open `rope.py`: `rope_tables` (angles), `apply_rope` (rotate each 2D pair),
`verify_relative_property` (the proof).

**Checkpoint:**
1. Why *rotate* Q and K instead of *adding* a position vector?
2. Why does the dot product end up depending only on (j − i)?
3. Why can RoPE handle position 100,000 when a learned table can't?

### Step 2 — Walk the notebook (40 min)
```bash
cd ..
make rope-notebook
```

| Section | Question |
|---|---|
| §0 · The problem | What's wrong with adding positions? |
| §1 · The idea | How does rotation encode position? |
| §2 · Prove it | Is the score really only about (j − i)? |
| §3 · Score vs distance | What signal does RoPE inject? |
| §4 · Extrapolation | Position 100,000 with no table |
| §5 · Where it fits | Drop-in for the GPT; composes with the KV-cache |

### Step 3 — Extend
Wire `apply_rope` into `../nanogpt-from-scratch`'s attention (rotate Q and K, drop the learned
`pos_emb`), retrain, and confirm it still learns — now without a context cap.

---

## Files
```
rope.py                 engine — rope_tables + apply_rope + relative-property proof (read first)
rope_explainer.ipynb    teaching notebook — THE deliverable (visuals inline)
create_notebook.py      maintenance: regenerates the .ipynb (not learner-facing)
data/                   scratch outputs (gitignored)
```

Verify any time:
```bash
../.venv/Scripts/python rope.py   # relative-property proof + extrapolation demo
```

---

## License
MIT — Hari Yelesetty
