# LLM inference, visualized

An interactive page of live, step-through figures. Every tensor shows its shape, and each GPU keeps the same colour throughout.

**Read it in a browser, no setup:** https://ylnhari.github.io/ml-scratchpad/llm-inference-visualized/

| Part | What you learn |
|---|---|
| I · Foundations | One batch traced from text to the next token: tokens [b, s] → embedding [b, s, h] → RoPE → LayerNorm → Q/K/V → heads [b, n_heads, s, d_head] → scores [b, n_heads, s, s] → softmax·V → residual → MLP [b, s, ffn] → × L layers → logits [b, s, V]. Toggle toy sizes ↔ Llama 2 7B numbers. |
| II · Inference optimizations | How serving systems make inference fast: prefill vs decode (staged), KV-cache calculator, static vs in-flight batching, data / pipeline / tensor / sequence parallelism (3D tensor split along b, h or s + layer walk-through with per-GPU activation memory), MHA→GQA→MQA, FlashAttention tiling, PagedAttention, quantization, 2:4 sparsity, distillation, speculative decoding. |
| III · Deeper dive | Backward pass and training memory (why activations are stored, the residual gradient highway, 16 B/param, Megatron activation formula, recomputation). What modern LLMs change: pre-norm, RMSNorm, SwiGLU, sampling (temperature / top-p), Mixture of Experts. |

## Why this is a page and not a notebook

The point is motion: watching a [b, s, h] block split along h for tensor parallelism and along s for sequence parallelism, or a layer's activations move between GPUs one collective at a time. A notebook can't play that kind of step-through. For runnable maths on the same ideas, use the notebooks next door: `attention-transformer/`, `rope/`, `nanogpt-from-scratch/` (backprop), `kv-cache/`, `continuous-batching/`, `inference-engine/`, `vllm-memory/`.

## Maintenance

`index.html` is hand-written source, not a generated export: edit it directly. It is one self-contained file (inline SVG + vanilla JS, Google Fonts only). Grid sizes are scaled down for drawing; formulas and the Llama 2 7B numbers follow the papers cited in the page.
