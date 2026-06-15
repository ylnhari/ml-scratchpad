# Claude Instructions — attention-transformer/

## Role
Teach the transformer self-attention mechanism (Vaswani et al. 2017) through concrete step-by-step examples.
Learner focus: Hari should be able to identify where to use/integrate attention in his own projects.

## Context
Paper: https://arxiv.org/abs/1706.03762
No external ML frameworks. Pure numpy so every matrix operation is visible and traceable.
Sentence used throughout: "The cat sat on the mat" (6 tokens, d_model=8 in notebook).

## Architecture
```
attention.py              core reference implementation — read first
attention_explainer.ipynb teaching notebook — THE deliverable (all visuals inline)
create_notebook.py        maintenance: regenerates the .ipynb (not learner-facing)
data/                     scratch outputs (gitignored)
```

## Key Algorithm Facts
- Attention: softmax(QKᵀ / √d_k) V
- sqrt(d_k) prevents softmax saturation (dot products grow with dimension)
- Multi-head: split d_model into h heads, parallel attention, concat + project
- Positional encoding: PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(...)
- Encoder block: MHA → Add&Norm → FFN → Add&Norm
- Causal mask: upper-triangular True matrix blocks future positions (GPT/decoder)

## Rules
1. Numpy only — no PyTorch/TensorFlow. Educational clarity > performance.
2. No paid APIs. All outputs to data/.
3. No auto-commit.
4. Learner journey first. No app scaffolding. Visuals live inline in the notebook, not a separate figure script. See parent `../CLAUDE.md` for governing principles.
