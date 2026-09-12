# KV Cache learning instructions

## Role
Learning implementation: understand the key/value cache — the core LLM inference speedup —
by adding it to the verified GPT from `../nanogpt-from-scratch` and proving it correct.

## Context
Pure NumPy. Reuses the trained model from `../nanogpt-from-scratch/nanogpt.py` (imported via
a sys.path insert) so the same verified weights/math carry over. Educational clarity >
performance. No server/app scaffolding. Scratch outputs to `data/` (gitignored). See parent
`../AGENTS.md` for governing principles.

## Architecture
```
kvcache.py                 engine — read first
  KVCache                  per-layer growing K/V store
  decode_step              one-token incremental forward using the cache (no causal mask)
  generate_cached/_naive   cached vs recompute-the-prefix generation
  verify_equivalence       proof cached logits == full-forward logits (the trust anchor)
kvcache_explainer.ipynb    teaching notebook — THE deliverable (visuals inline)
create_notebook.py         maintenance: regenerates the .ipynb (not learner-facing)
data/                      scratch outputs (gitignored)
```

## Key Facts
- Past tokens' K/V are fixed once computed → cache them; recompute only the new token.
- A cached decoder needs no causal mask: the cache only ever contains the past.
- Learned positional embeddings cap context at `block_size`; keep demos within it (or note it).
- `verify_equivalence()` must stay < 1e-6 (it is ~1e-15 in practice) — the correctness anchor.
- KV-cache memory ≈ 2·n_layer·n_head·head_dim·seq_len·batch·dtype_bytes → motivates `../turbo-quant`.

## Rules
1. Numpy only. Clarity over performance. No paid APIs, no hardcoded paths.
2. All outputs to `data/` (gitignored). No auto-commit.
3. If you change `decode_step`, re-run `verify_equivalence()` — never let it drift from `model.forward`.
4. Learner journey first. Visuals inline in the notebook.

## Running
```bash
# from ml-scratchpad/ root:
make kv-cache-notebook     # open the explainer
make kv-cache-regen        # rebuild the .ipynb
../.venv/Scripts/python kv-cache/kvcache.py   # equivalence + timing self-test
```
