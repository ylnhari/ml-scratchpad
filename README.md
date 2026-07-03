# Learning

Personal learning lab — deep dives into algorithms, papers, and ML concepts.
Each subfolder is one topic: notebook-first, visual, step-by-step.

---

## Setup — Do This Once

One shared Python environment for all subfolders.

```bash
cd C:\Users\ylnha\Projects\learning

make venv      # creates .venv + installs all deps (numpy, scipy, matplotlib, jupyter …)
make kernel    # registers "Learning (shared)" Jupyter kernel system-wide
```

Then open any notebook and select **Kernel → Change Kernel → Learning (shared)**.

> No per-subfolder venv or kernel. One setup covers everything.

---

## Subfolders

| Folder | Topic | Open with |
|--------|-------|-----------|
| `turbo-quant/` | TurboQuant — vector quantization (ICLR 2026) | `make turbo-quant-notebook` |
| `attention-transformer/` | Attention Is All You Need (Vaswani 2017) | `make attention-transformer-notebook` |
| `nanogpt-from-scratch/` | Train a GPT + backprop by hand, in NumPy | `make nanogpt-notebook` |
| `kv-cache/` | KV-cache: make generation O(n) not O(n²) | `make kv-cache-notebook` |
| `rope/` | Rotary positional embeddings (Llama/Mistral) | `make rope-notebook` |
| `inference-engine/` | Paged KV-cache + speculative decoding (vLLM ideas) | `make inference-engine-notebook` |

---

## Adding a New Topic

1. Create subfolder under `learning/`.
2. If new deps needed: `.venv/Scripts/pip install <pkg>` (no venv re-create needed).
3. Add a `make <name>-notebook` target to `Makefile`.
4. Add a row to the table above.
5. Notebook kernels use "Learning (shared)" — no per-subfolder setup needed.

---

## Quick Reference

```bash
make venv                         # first-time env setup
make kernel                       # first-time kernel registration
make turbo-quant-notebook         # open TurboQuant explainer
make attention-transformer-notebook  # open Attention explainer
make nanogpt-notebook             # open nanoGPT (train + backprop) explainer
make kv-cache-notebook            # open KV-cache explainer
make rope-notebook                # open RoPE explainer
make inference-engine-notebook    # open inference-engine explainer
```
