# Continuous Batching From Scratch — Learning Path

> The final step of the inference path: `../nanogpt-from-scratch` → `../kv-cache` → `../rope` →
> `../inference-engine` → here. Schedule many variable-length requests so the batch stays full —
> continuous batching, the piece that (with paged attention) is the core of vLLM.

**Goal:** Understand why static batching wastes compute and how continuous batching reaches
~100% utilization without changing any output.

---

## What You Will Learn

- Why requests of different lengths make static batching leave slots idle
- How continuous batching evicts finished sequences and admits waiting ones mid-flight
- How to prove the scheduler doesn't change outputs (batched == standalone greedy)
- How to measure batch-slot utilization and the throughput gain

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

### Step 1 — Read the engine (12 min)
Open `batching.py`: `_Seq` (one live request), `run_static` vs `run_continuous`,
`verify_equivalence`, and the `occupancy_*` helpers used for the schedule pictures.

**Checkpoint:**
1. In static batching, why do short requests waste slots?
2. What does continuous batching do the instant a sequence finishes?
3. Why is each request's output identical under both schedulers?

### Step 2 — Walk the notebook (40 min)
```bash
cd ..
make continuous-batching-notebook
```

| Section | Question |
|---|---|
| §0 · The problem | Why don't requests finish together? |
| §1 · See the waste | Static vs continuous schedule grids (idle slots) |
| §2 · Prove it | Batched output == standalone greedy? |
| §3 · Measure | Utilization and slot-step savings |
| §4 · Where next | Chunked prefill, disaggregation, preemption |

### Step 3 — Extend
Add **chunked prefill** (interleave a long new prompt's processing with ongoing decodes) or
wire in `../inference-engine`'s `PagedKVCache` so slots share one paged memory pool.

---

## Files
```
batching.py                          engine — schedulers + equivalence proof (read first)
continuous_batching_explainer.ipynb  teaching notebook — THE deliverable (visuals inline)
create_notebook.py                   maintenance: regenerates the .ipynb (not learner-facing)
data/                                scratch outputs (gitignored)
```

Verify any time:
```bash
../.venv/Scripts/python batching.py   # equivalence + static-vs-continuous utilization
```

---

## License
MIT — Hari Yelesetty
