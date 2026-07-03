# Inference Engine From Scratch — Learning Path

> The capstone of the inference path: `../nanogpt-from-scratch` → `../kv-cache` → `../rope` →
> here. Builds the two ideas behind modern LLM serving (vLLM, TGI): a **paged KV-cache** and
> **speculative decoding** — in NumPy, each proven to keep the output identical.

**Goal:** Understand *why* LLM serving is fast and memory-efficient — the "builder, not just
operator" of vLLM story.

---

## What You Will Learn

- Why a contiguous KV-cache wastes memory when serving many sequences, and how **paging**
  (fixed-size blocks + a block table, like OS virtual memory) fixes it — vLLM's core trick
- How **speculative decoding** uses a cheap draft model to let the big model verify k tokens
  in one forward, and why greedy speculative output is *exactly* the target's greedy output
- How to prove each optimization changes speed/memory, not the answer

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

### Step 1 — Read the engine (15 min)
Open `engine.py`: `PagedKVCache` (drop-in for `../kv-cache`'s cache), `verify_paging`,
`speculative_greedy`, `verify_speculative`.

**Checkpoint:**
1. Why does paging help when serving *many* sequences, not just one?
2. In greedy speculative decoding, why is the output guaranteed identical to target greedy?
3. Where does the speedup come from if the output is the same?

### Step 2 — Walk the notebook (50 min)
```bash
cd ..
make inference-engine-notebook
```

| Section | Question |
|---|---|
| §0 · Serving is the bottleneck | Why isn't a fast forward pass enough? |
| §1 · Paged KV-cache | How do fixed blocks + a block table remove fragmentation? |
| §2 · Paging is transparent | Same logits as a contiguous cache? |
| §3 · Speculative decoding | How does a draft model make the target faster? |
| §4 · Prove + measure | Identical output, how many fewer target forwards? |
| §5 · Where next | Continuous batching, real paged-attention kernels |

### Step 3 — Extend
Add **continuous batching** (many sequences sharing the paged pool, each with its own block
table) or plug `../rope` into the attention so the engine handles long contexts.

---

## Files
```
engine.py                      PagedKVCache + speculative decoding (read first)
inference_engine_explainer.ipynb  teaching notebook — THE deliverable
create_notebook.py             maintenance: regenerates the .ipynb (not learner-facing)
data/                          scratch outputs (gitignored)
```

Verify any time:
```bash
../.venv/Scripts/python engine.py   # paging equivalence + speculative correctness/speedup
```

---

## License
MIT — Hari Yelesetty
