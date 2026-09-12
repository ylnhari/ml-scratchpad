# RoPE learning instructions

## Role
Learning implementation: understand Rotary Positional Embeddings (RoPE) — the positional scheme
in nearly every modern open LLM — by building it in NumPy and proving its relative-position property.

## Context
Pure NumPy, self-contained (no model import needed — RoPE is a positional transform on Q/K).
Follows `../nanogpt-from-scratch` (learned positions) and `../kv-cache`. Educational clarity >
performance. No server/app scaffolding. Scratch outputs to `data/` (gitignored). See parent
`../AGENTS.md` for governing principles.

## Architecture
```
rope.py                 engine — read first
  rope_tables           cos/sin angle tables per position & dim-pair
  apply_rope            rotate each 2D pair of a vector by position*inv_freq
  verify_relative_property  proof <RoPE(q,i),RoPE(k,j)> depends only on (j-i)
  score_vs_distance     RoPE score as a function of relative distance
rope_explainer.ipynb    teaching notebook — THE deliverable (visuals inline)
create_notebook.py      maintenance: regenerates the .ipynb (not learner-facing)
data/                   scratch outputs (gitignored)
```

## Key Facts
- Pair dims (2m, 2m+1); rotate by angle position*base^(-2m/dim), base=10000.
- Rotation makes q_i·k_j a function of (i-j) only → relative position, defined by a formula.
- No lookup table → works at any position (length extrapolation; PI/NTK/YaRN are RoPE rescalings).
- `verify_relative_property()` must stay < 1e-9 (it is ~1e-15) — the correctness anchor.
- Composes with `../kv-cache`: rotate K by its absolute position before caching.

## Rules
1. Numpy only. Clarity over performance. No paid APIs, no hardcoded paths.
2. All outputs to `data/` (gitignored). No auto-commit.
3. If you change `apply_rope`, re-run `verify_relative_property()` — never let it drift.
4. Learner journey first. Visuals inline in the notebook.

## Running
```bash
# from ml-scratchpad/ root:
make rope-notebook     # open the explainer
make rope-regen        # rebuild the .ipynb
../.venv/Scripts/python rope/rope.py   # relative-property proof + extrapolation demo
```
