# Claude Instructions — turbo-quant

## Role
Learning implementation: understand and visualize TurboQuant (ICLR 2026) vector quantization.

## Context
Paper: https://arxiv.org/abs/2504.19874
No server. Pure Python/numpy/scipy/matplotlib. All outputs go to `data/` (gitignored).
Visuals live inline in the notebook, not a separate figure script. See parent `../CLAUDE.md` for governing principles.

## Architecture
```
turbo_quant.py              core algorithm — TurboQuant class (read first)
turbo_quant_explainer.ipynb teaching notebook — THE deliverable (all visuals inline)
create_notebook.py          maintenance: regenerates the .ipynb (not learner-facing)
data/                       scratch outputs (gitignored)
```

## Key Algorithm Facts
- Rotation matrix: QR decomp of d×d Gaussian matrix → orthogonal Pi
- After rotation, each coordinate ~ N(0, 1/d) → optimal for Lloyd-Max quantization
- Lloyd-Max centroids: precomputed via Lloyd's algorithm on N(0,1) samples, then scaled /sqrt(d)
- MSE bound: sqrt(3π)/2 * 1/4^b  |  lower bound: 1/4^b
- Prod variant: (b-1)-bit MSE + 1-bit QJL on residual → unbiased inner product estimator
- QJL: sign(S·r), inverse: sqrt(π/2)/d * S^T * sign(S·r), where S ~ N(0,1) d×d matrix

## Rules
1. No paid APIs. No hardcoded paths.
2. All outputs to data/ (gitignored).
3. No auto-commit.

## Running
```bash
# from learning/ root:
make install                  # shared env
make turbo-quant-notebook     # open the explainer
```
