# Inference Engine learning instructions

## Role
Learning implementation: understand how modern LLM serving is fast and memory-efficient by
building the two core ideas — paged KV-cache and speculative decoding — in NumPy, each proven
to preserve output. The "builder, not just operator" of vLLM capstone.

## Context
Pure NumPy. Reuses the verified model (`../nanogpt-from-scratch/nanogpt.py`) and cache/decoder
(`../kv-cache/kvcache.py`) via sys.path inserts. Educational clarity > performance. No
server/app scaffolding (the *concepts* are the subject, not a running server). Scratch outputs
to `data/` (gitignored). See parent `../AGENTS.md` for governing principles.

## Architecture
```
engine.py                        engine — read first
  PagedKVCache                   fixed-size blocks + reassembled view; drop-in for KVCache
  verify_paging                  proof paged logits == contiguous logits
  greedy_target                  reference greedy decoding (one forward/token)
  speculative_greedy             draft proposes k, target verifies in one forward
  verify_speculative             proof spec output == target greedy; returns target-forward count
inference_engine_explainer.ipynb teaching notebook — THE deliverable (visuals inline)
create_notebook.py               maintenance: regenerates the .ipynb (not learner-facing)
data/                            scratch outputs (gitignored)
```

## Key Facts
- Paging: allocate KV in fixed blocks (page_size), keep a per-sequence block list; the attention
  step sees a normal contiguous view. Removes fragmentation when serving many sequences (vLLM).
- Because paging just reassembles the same K/V, logits are identical — `verify_paging` < 1e-6.
- Greedy speculative decoding accepts proposed tokens while they match the target's argmax, so
  the output is EXACTLY target greedy; a mismatch still yields one free correct target token.
- `verify_speculative` must return True (output identical); speedup = fewer target forwards.

## Rules
1. Numpy only. Clarity over performance. No paid APIs, no hardcoded paths.
2. All outputs to `data/` (gitignored). No auto-commit.
3. If you change paging or speculative logic, re-run `verify_paging` / `verify_speculative` —
   never let output-equivalence drift.
4. Learner journey first. Visuals inline in the notebook.

## Running
```bash
# from ml-scratchpad/ root:
make inference-engine-notebook    # open the explainer
make inference-engine-regen       # rebuild the .ipynb
../.venv/Scripts/python inference-engine/engine.py   # paging + speculative self-test
```
