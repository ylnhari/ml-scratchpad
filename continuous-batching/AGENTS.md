# Continuous Batching learning instructions

## Role
Learning implementation: understand continuous batching — the request scheduler that (with paged
attention) is the core of vLLM — by building static vs continuous schedulers in NumPy and proving
they don't change output. Final subfolder of the inference path.

## Context
Pure NumPy. Reuses the verified model (`../nanogpt-from-scratch/nanogpt.py`) and cache/decoder
(`../kv-cache/kvcache.py`) via sys.path inserts. Educational clarity > performance. No server/app
scaffolding. Scratch outputs to `data/` (gitignored). See parent `../AGENTS.md` for principles.

## Architecture
```
batching.py                          engine — read first
  Request / _Seq                     a request and its live slot (own KV-cache + next logits)
  run_static                         fixed groups of B; run until the longest finishes
  run_continuous                     rolling B slots; refill the instant one finishes
  verify_equivalence                 proof batched output == standalone greedy (correctness anchor)
  occupancy_static/continuous        slot x tick occupancy grids (model-free, for visualization)
continuous_batching_explainer.ipynb  teaching notebook — THE deliverable (visuals inline)
create_notebook.py                   maintenance: regenerates the .ipynb (not learner-facing)
data/                                scratch outputs (gitignored)
```

## Key Facts
- Static batching wastes slots: a group runs until its longest member finishes; earlier finishers idle.
- Continuous batching evicts a finished sequence and admits a waiting one the same tick → ~100% utilization.
- Scheduling changes *when* tokens are computed, not *what* — `verify_equivalence()` must stay True.
- Utilization = useful decode steps / occupied slot-steps; report the slot-step ratio as the gain.
- `occupancy_*` depend only on gen lengths + B + policy (no model) — safe for plots.

## Rules
1. Numpy only. Clarity over performance. No paid APIs, no hardcoded paths.
2. All outputs to `data/` (gitignored). No auto-commit.
3. If you change a scheduler, re-run `verify_equivalence()` — outputs must stay identical to greedy.
4. Learner journey first. Visuals inline in the notebook.
5. This notebook trains a model in-cell; if nbconvert's Windows zmq kernel crashes on it, verify
   by executing the code cells as a script with the Agg backend (that is a kernel bug, not a code bug).

## Running
```bash
# from learning/ root:
make continuous-batching-notebook   # open the explainer
make continuous-batching-regen      # rebuild the .ipynb
../.venv/Scripts/python continuous-batching/batching.py   # equivalence + utilization self-test
```
