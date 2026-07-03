# Claude Instructions — nanogpt-from-scratch

## Role
Learning implementation: understand a trainable decoder-only transformer (GPT) by building
it and its backpropagation by hand. Builds on `../attention-transformer` (forward-only);
this adds training. Learner focus: Hari should be able to build, train, and debug a GPT
himself, and explain every gradient.

## Context
Pure NumPy — no PyTorch/TensorFlow. Every gradient is written out so it is visible and
checkable. Educational clarity > performance. No server, no app scaffolding. All scratch
outputs go to `data/` (gitignored). See parent `../CLAUDE.md` for governing principles.

## Architecture
```
nanogpt.py              core engine — read first
  Module                base: params `p`, grads `g`, children `sub`
  Linear/LayerNorm/ReLU each with forward + mirror backward
  CausalSelfAttention   masked multi-head attention, forward + backward
  MLP / Block / GPT      pre-norm decoder blocks; GPT ties it together
  Adam                   optimizer
  CharTokenizer/get_batch data
  gradient_check         numerical vs analytic grads — the correctness proof
nanogpt_explainer.ipynb teaching notebook — THE deliverable (all visuals inline)
create_notebook.py      maintenance: regenerates the .ipynb (not learner-facing)
data/                   scratch outputs (gitignored)
```

## Key Facts
- GPT-2 layout: pre-LayerNorm blocks, learned positional embeddings, weight-untied head.
- ReLU in the MLP (exact, pure-NumPy) where real GPT-2 uses GELU — noted in the notebook.
- Loss = softmax cross-entropy on next token; `dlogits = (softmax - onehot) / N`.
- Every layer's `backward(dy)` returns `dx` and fills `self.g[...]`; chaining in reverse = backprop.
- `gradient_check()` must stay < 1e-4 relative error — it is the trust anchor for the whole file.

## Rules
1. Numpy only. Clarity over performance. No paid APIs, no hardcoded paths.
2. All outputs to `data/` (gitignored). No auto-commit.
3. If you change any `forward`, change its `backward` and re-run `gradient_check()` — never
   let them drift. Re-run `nanogpt.py` and re-execute the notebook after edits.
4. Learner journey first. Visuals live inline in the notebook, not a separate figure script.

## Running
```bash
# from learning/ root:
make nanogpt-notebook     # open the explainer
make nanogpt-regen        # rebuild the .ipynb from create_notebook.py
../.venv/Scripts/python nanogpt-from-scratch/nanogpt.py   # self-test: gradient check + train + sample
```
