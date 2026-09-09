"""Generates vllm_memory_explainer.ipynb — run once (make vllm-memory-regen), then execute it.

Maintenance tooling, not learner-facing. The .ipynb is the deliverable.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Learning (shared)", "language": "python", "name": "learning"},
    "language_info": {"name": "python", "version": "3.11"},
})

def md(src):   return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

cells = []

# =============================================================================
# 0. Title + map
# =============================================================================
cells.append(md(r"""# How an LLM server uses a GPU — vLLM from power-on to the thousandth request

This notebook assumes you know *nothing* about serving. Every term is defined the first time it
appears, every claim is shown with a number, and every mechanism gets a picture.

**Where this sits:** `../kv-cache` → `../inference-engine` → `../continuous-batching` → **here**.
Those notebooks build the ideas from scratch in NumPy. This one shows how the real server, vLLM,
turns those ideas into a memory budget and a scheduler, and how each command-line flag moves
memory and speed.

**One worked example runs through everything:** Llama-3.1-8B-Instruct, one NVIDIA H100 80 GB.

| Part | Question it answers |
|---|---|
| 1 | What is a token, a parameter, a forward pass, attention, a KV cache, prefill, decode, a batch, a block? |
| 2 | What does each `vllm serve` flag mean in plain words, and what does it move? |
| 3 | Startup, second by second: what is allocated, in what order, and why one step is *measured* |
| 4 | Runtime, one request: memory and time from arrival to last token |
| 5 | Runtime, many requests: the scheduler loop, the block table, the three concurrency limits, preemption |
| 6 | Engineering: what actually limits speed in prefill vs decode, and how each flag trades memory for speed |
| 7 | Models that take images and video |
| 8 | Reading your own startup log line by line |

> **Units.** vLLM logs in **GiB** (1 GiB = 1024³ bytes). Marketing uses GB (10⁹ bytes). An
> "80 GB" H100 reports 79.6 GiB. Everything here is in GiB so it matches your logs.
"""))

cells.append(code(r"""import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
GiB, MiB, KiB = 1024**3, 1024**2, 1024

# ---------------- the worked example -------------------------------------------------
GPU_TOTAL_GiB   = 79.6        # what torch reports for an H100 80 GB
GPU_BW_TBs      = 3.35        # H100 SXM HBM3 memory bandwidth, TB/s (peak)
GPU_TFLOPS      = 989.0       # H100 SXM BF16 dense tensor-core peak, TFLOP/s
# Llama-3.1-8B-Instruct, from its HuggingFace config.json
PARAMS          = 8.03e9
LAYERS, HIDDEN, INTER, VOCAB = 32, 4096, 14336, 128_256
Q_HEADS, KV_HEADS, HEAD_DIM  = 32, 8, 128
# vLLM flags for the example (server defaults on an H100 unless stated)
UTIL            = 0.92        # --gpu-memory-utilization
MAX_MODEL_LEN   = 32_768      # --max-model-len   (we set it; the model allows 131 072)
MAX_NUM_SEQS    = 256         # --max-num-seqs    (we set it; H100 server default is 1024)
MAX_BATCH_TOK   = 8_192       # --max-num-batched-tokens (H100 server default)
BLOCK           = 16          # --block-size (default)
BYTES_W         = 2           # BF16 weights
BYTES_KV        = 2           # --kv-cache-dtype auto → BF16
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 11, "axes.labelsize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8})
print("ready")
"""))

# =============================================================================
# 1. Concepts from zero
# =============================================================================
cells.append(md(r"""---
# Part 1 — The concepts, each with a picture

## 1.1 Tokens: what the model actually reads and writes

A language model never sees letters or words. Text is first cut into **tokens**: pieces that
are usually a whole common word, or part of a rare one. English averages about 1.3 tokens per
word. The model's vocabulary for Llama-3.1 is 128 256 such pieces, each with an integer id.

The picture below shows a sentence as its tokens. Everything that follows — memory, speed,
limits — is counted in tokens, never in words or characters.
"""))

cells.append(code(r"""sentence = ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", ".",
            " Tokeni", "zation", " splits", " rare", " words", "."]
ids = [791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13, 9857, 2065, 41567, 9024, 4339, 13]
fig, ax = plt.subplots(figsize=(12, 1.6))
x = 0
for tok, i in zip(sentence, ids):
    w = 0.45 + 0.12 * len(tok)
    ax.add_patch(FancyBboxPatch((x, 0.2), w, 0.6, boxstyle="round,pad=0.02", fc="#e8f0fa", ec="#4e79a7"))
    ax.text(x + w/2, 0.5, repr(tok)[1:-1], ha="center", va="center", fontsize=9)
    ax.text(x + w/2, 0.02, str(i), ha="center", va="bottom", fontsize=7, color="gray")
    x += w + 0.08
ax.set_xlim(0, x); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title(f"{len(sentence)} tokens (ids below). 'Tokenization' became two pieces; common words are one piece each.")
plt.show()
"""))

cells.append(md(r"""## 1.2 Parameters, precision, and the size of the weights

The model is a stack of **layers**. Each layer is a handful of large matrices of numbers. Those
numbers are the **parameters** (also called **weights**): learned once during training and
never changed while serving. "8B" means 8 billion of them.

Each parameter is stored in some **precision** — how many bytes one number takes:

| Precision | Bytes per parameter | 8.03 B parameters take |
|---|---|---|
| FP32 | 4 | 29.9 GiB |
| **BF16** (the usual serving default) | 2 | **14.96 GiB** |
| FP8 / INT8 | 1 | 7.5 GiB |
| INT4 / NVFP4 / AWQ / GPTQ | 0.5 | 3.7 GiB |

Choosing a smaller precision is called **quantization**. It is the single biggest lever on
memory, and it is decided *before* the server starts. Everything else in this notebook must fit
in the room the weights leave.
"""))

cells.append(code(r"""precs = [("FP32", 4), ("BF16", 2), ("FP8 / INT8", 1), ("INT4 / NVFP4", 0.5)]
sizes = [PARAMS * b / GiB for _, b in precs]
fig, ax = plt.subplots(figsize=(8, 2.8))
bars = ax.barh([p for p, _ in precs], sizes, color=["#bab0ac", "#4e79a7", "#59a14f", "#76b7b2"])
for b, s in zip(bars, sizes): ax.text(s + 0.4, b.get_y() + b.get_height()/2, f"{s:.1f} GiB", va="center", fontsize=9)
ax.axvline(GPU_TOTAL_GiB * UTIL, color="black", ls="--", lw=1)
ax.text(GPU_TOTAL_GiB * UTIL - 0.5, 3.4, "H100 budget at util 0.92", ha="right", fontsize=8)
ax.set_xlim(0, 80); ax.invert_yaxis(); ax.set_xlabel("GiB")
ax.set_title("Weights of an 8 B model at each precision — a fixed cost paid once at startup")
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""## 1.3 A forward pass, and why attention needs a memory

Generating text is a loop. Each turn of the loop is one **forward pass**: the token ids go in at
the bottom, pass through all 32 layers, and a probability for every vocabulary word comes out at
the top. The most likely (or a sampled) word is appended, and the loop runs again.

Inside every layer, the **attention** step is the one that looks *backwards*: to produce the
next token it must compare the current position against every earlier token. To do that
comparison it needs, for every earlier token and every layer, two small vectors called the
**Key (K)** and the **Value (V)**.

Without help, each new token would recompute K and V for the whole history: token 1 000 costs
1 000 tokens of work, token 2 000 costs 2 000, and so on (quadratic). The fix is to compute K and
V once per token and **keep them on the GPU**. That storage is the **KV cache**. It is the
reason a serving GPU fills up, and the reason concurrency is limited.

How big is one token's entry? For Llama-3.1-8B:
"""))

cells.append(code(r"""kv_per_layer = 2 * KV_HEADS * HEAD_DIM * BYTES_KV              # K and V, 8 heads of 128 numbers, 2 bytes each
kv_per_token = LAYERS * kv_per_layer
print(f"per layer : 2 (K,V) × {KV_HEADS} heads × {HEAD_DIM} numbers × {BYTES_KV} B = {kv_per_layer:,} B")
print(f"per token : {LAYERS} layers × {kv_per_layer:,} B = {kv_per_token:,} B = {kv_per_token/KiB:.0f} KiB")
print(f"1 000-token prompt = {1000*kv_per_token/MiB:.0f} MiB;  32 768-token request = {MAX_MODEL_LEN*kv_per_token/GiB:.2f} GiB")

# Picture: one token's KV entry as a 32-layer stack; then growth with context.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
for l in range(LAYERS):
    a1.add_patch(Rectangle((0, l), 1, 0.9, fc="#f28e2b", ec="white")); a1.add_patch(Rectangle((1.05, l), 1, 0.9, fc="#4e79a7", ec="white"))
a1.set_xlim(-0.2, 2.6); a1.set_ylim(-0.5, LAYERS + 0.5); a1.set_xticks([0.5, 1.55]); a1.set_xticklabels(["K", "V"]); a1.set_yticks([0, 15, 31])
a1.set_ylabel("layer"); a1.set_title(f"ONE token's KV entry: {LAYERS} layers × (K + V) = {kv_per_token/KiB:.0f} KiB")
toks = np.arange(0, MAX_MODEL_LEN + 1, 512)
a2.plot(toks, toks * kv_per_token / GiB, color="#59a14f", lw=2)
a2.set_xlabel("tokens in the request (prompt + answer so far)"); a2.set_ylabel("KV cache held (GiB)")
a2.set_title("KV cache of one request grows linearly with its length")
for t in (1000, 8192, 32768): a2.annotate(f"{t:,} tok → {t*kv_per_token/GiB:.2f} GiB", xy=(t, t*kv_per_token/GiB), xytext=(t, t*kv_per_token/GiB + 0.5), fontsize=8, ha="center", arrowprops=dict(arrowstyle="->"))
a2.set_ylim(0, 5); plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""> **Why 8 heads and not 32?** Llama uses *grouped-query attention*: 32 query heads share 8 K/V
> heads. That is a 4× saving on the cache compared to the original transformer. DeepSeek goes
> further (multi-head latent attention, one compressed vector per token); Gemma uses
> sliding-window layers whose cache stops growing after a fixed window. The per-token number
> is the one thing to read off a model's `config.json` before sizing a server.

## 1.4 Prefill and decode: the two very different phases of one request

A request has two phases and they behave like two different workloads.

**Prefill.** The prompt is already known, so *all* its tokens go through the network together
in one forward pass. 1 000 prompt tokens = 1 000 tokens of matrix work in one go. This writes
1 000 KV entries. It is heavy on compute and produces the *first* output token. The time it
takes is the **time to first token (TTFT)**.

**Decode.** Every later token is produced one at a time: one forward pass, one token. Each pass
does very little arithmetic (one token's worth) but must *read* every K and V stored so far. It
is heavy on memory traffic and light on compute. The time per pass is the **inter-token latency
(ITL)** — the speed at which words appear on screen.
"""))

cells.append(code(r"""PROMPT, ANSWER = 1000, 300
fig, ax = plt.subplots(figsize=(12, 2.8))
ax.bar(0, PROMPT, color="#f28e2b", width=1); ax.bar(np.arange(1, ANSWER + 1), 1, color="#4e79a7", width=1)
ax.set_yscale("log"); ax.set_xlabel("forward pass number"); ax.set_ylabel("tokens processed in the pass (log)")
ax.set_title("One request: pass 0 is the prefill (1 000 tokens at once), passes 1..300 are decodes (1 token each)")
ax.annotate("prefill → time to first token", xy=(0, PROMPT), xytext=(40, 400), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.annotate("each decode → one inter-token latency", xy=(150, 1), xytext=(120, 8), arrowprops=dict(arrowstyle="->"), fontsize=9)
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""## 1.5 Batching: many requests share one forward pass

The GPU is enormously parallel. Running one decode pass for one request uses a fraction of it.
So the server runs one forward pass for *many* requests at once — a **batch** — and the cost of
the pass barely changes. The tokens of all the requests in the batch are stacked into one input.

vLLM uses **continuous batching**: the batch is re-assembled at *every* pass. A request that
finishes leaves immediately; a waiting one joins immediately; a request in prefill and a request
in decode can share the same pass. There is no "wait for the whole batch to finish".

One turn of that loop — assemble a batch, run one forward pass, sample one token for each
request in it — is a **step** (vLLM also says *iteration*). Two flags cap the step:
`--max-num-seqs` caps how many *requests* may be in it; `--max-num-batched-tokens` caps how many
*tokens* may be in it. The picture shows the difference.
"""))

cells.append(code(r"""fig, ax = plt.subplots(figsize=(12, 3.2))
# a step with 3 requests in decode (1 token each) and 2 prompts in prefill
parts = [("req A decode", 1, "#4e79a7"), ("req B decode", 1, "#4e79a7"), ("req C decode", 1, "#4e79a7"),
         ("req D prefill (700 tok)", 700, "#f28e2b"), ("req E prefill, first chunk (7 489 tok)", 7489, "#f28e2b")]
x = 0
for name, n, c in parts:
    w = max(n, 120)                       # decode tokens drawn wide enough to see
    ax.add_patch(Rectangle((x, 0.3), w, 0.5, fc=c, ec="white"))
    ax.text(x + w/2, 0.55, name, ha="center", va="center", fontsize=8, rotation=90 if w < 400 else 0, color="white")
    x += w
ax.axvline(x, color="black", ls="--"); ax.text(x + 100, 0.9, "--max-num-batched-tokens = 8 192 tokens reached", fontsize=9, ha="left")
ax.set_xlim(0, 11000); ax.set_ylim(0, 1.05); ax.set_yticks([]); ax.set_xlabel("tokens in this ONE step (decode boxes widened for visibility)")
ax.set_title("A single step: 5 requests (counts toward --max-num-seqs = 256), 8 192 tokens (the token cap). Req E's prompt continues next step.")
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""Request E's prompt did not fit in the tokens left, so only the first 7 489 tokens were taken.
That is **chunked prefill**: a long prompt is spread across steps so it never blocks the decode
tokens of the requests already running. It is on by default.

## 1.6 Paged KV cache: blocks, and why memory never fragments

The KV cache is not one array per request. vLLM carves the whole pool into fixed **blocks** of
16 tokens (`--block-size`) and gives each request a *list* of blocks, like pages of virtual
memory. A request that grows takes one more block from the free list; a request that finishes
gives all its blocks back. Any block can serve any request, so there is no fragmentation, and
two requests with the same prompt prefix can even *share* blocks (**prefix caching**).

For Llama-3.1-8B one block is 16 × 128 KiB = **2 MiB**. The picture is a small pool (200 blocks)
with three requests inside.
"""))

cells.append(code(r"""rng = np.random.default_rng(3)
pool = np.zeros(200, int)                         # 0 = free, k = owned by request k
owned = {1: 63, 2: 20, 3: 9}                        # 1000-token prompt, a 300-token request, a 140-token one
free_ids = list(range(200)); rng.shuffle(free_ids)
for k, n in owned.items():
    for _ in range(n): pool[free_ids.pop()] = k
fig, ax = plt.subplots(figsize=(12, 2.4))
cols = {0: "#eeeeee", 1: "#4e79a7", 2: "#f28e2b", 3: "#59a14f"}
for i, k in enumerate(pool):
    ax.add_patch(Rectangle((i % 40, 4 - i // 40), 0.92, 0.92, fc=cols[k], ec="white"))
ax.set_xlim(0, 40); ax.set_ylim(0, 5); ax.axis("off")
ax.set_title("A 200-block pool (each block = 16 tokens = 2 MiB). Blue: request 1 (63 blocks), orange: request 2 (20), green: request 3 (9). Grey: free.")
plt.show()
"""))

# =============================================================================
# 2. The flags
# =============================================================================
cells.append(md(r"""---
# Part 2 — Every flag, in plain words

`vllm serve meta-llama/Llama-3.1-8B-Instruct` starts with defaults. These are the flags that
decide memory and speed. Each row says what it *means* first, then what it *moves*.

| Flag | Plain meaning | Moves at startup | Moves at runtime |
|---|---|---|---|
| `--dtype` / `--quantization` | how many bytes each weight takes | weights size (and activation size) | speed of every step (fewer bytes to read) |
| `--gpu-memory-utilization` (0.92) | the share of the card vLLM may own, *for everything* | the ceiling; KV pool is what is left under it | nothing (fixed) |
| `--max-model-len` | the longest one request may ever be, prompt + answer | attention workspace; (older vLLM: profiling size) | KV a single request can consume → concurrency at full length; rejects longer prompts |
| `--max-num-seqs` | most requests allowed *in flight* at once | sampler spike in the profile (rows of logits) | concurrency cap; decode batch size → throughput |
| `--max-num-batched-tokens` | most tokens in one step | **the activation peak** the profile run measures | prefill throughput vs decode smoothness |
| `--block-size` (16) | tokens per KV block | pool granularity | waste at the tail of each request (≤ 15 tokens) |
| `--kv-cache-dtype` (auto → BF16; `fp8`) | bytes per KV number | KV per token halves with fp8 → pool holds 2× tokens | slightly less memory traffic per decode |
| `--enable-prefix-caching` (on) | reuse KV blocks for identical prompt prefixes | a hash table | fewer prefill tokens for repeated system prompts |
| `--tensor-parallel-size` | split every matrix across N GPUs | weights ÷ N, KV ÷ N, per-GPU activations ÷ N; NCCL buffers | every step needs GPU-to-GPU sync |
| `--enforce-eager` | do not capture CUDA graphs | frees the CUDA-graph memory | slower decode steps (launch overhead) |
| `--max-num-partial-prefills` etc. | fine control of chunking | – | latency trade-offs |

The rest of the notebook shows *how much* each of the important ones moves, with numbers.
"""))

# =============================================================================
# 3. Startup
# =============================================================================
cells.append(md(r"""---
# Part 3 — Startup, second by second

When you press enter on `vllm serve …`, memory on the GPU changes in a fixed sequence. The
figure below is memory-in-use over time during startup, with each phase labelled. Times are
typical for an 8 B model on a fast disk; the *order* is what matters.

| Phase | What happens | Memory effect |
|---|---|---|
| A. Init | Python imports, CUDA context created, NCCL if multi-GPU. vLLM takes a **snapshot** of free memory. | a few hundred MiB, **outside** the budget |
| B. Budget | `requested = total × util` = 73.2 GiB. If less than that is free right now, it **refuses to start**. | fixes the ceiling |
| C. Load weights | The checkpoint streams from disk to the GPU. | +14.96 GiB, permanent |
| D. **Profile run** | One *dummy* step at the worst-case shape. Memory spikes, then is freed. The spike height is **recorded**. | temporary spike; a number |
| E. KV allocation | `pool = requested − weights − spike − non-torch growth`. Allocated **all at once**, as blocks. | +54 GiB, permanent, even with zero requests |
| F. CUDA graphs | Decode kernels are recorded for a set of batch sizes so each step launches as one unit. | +0.5–2 GiB, taken **after** E |
| G. Ready | HTTP server listens. | memory now flat forever |
"""))

cells.append(code(r"""weights_GiB   = PARAMS * BYTES_W / GiB
requested_GiB = GPU_TOTAL_GiB * UTIL
non_torch_GiB = 0.3           # NCCL / attention workspaces, also measured
peak_act_GiB  = 4.0           # <-- MEASURED in phase D (placeholder; your log prints the real one)
graphs_GiB    = 1.0
kv_pool_GiB   = requested_GiB - weights_GiB - non_torch_GiB - peak_act_GiB
ctx_GiB       = 0.5

# a stylised memory-vs-time trace of startup
t = np.linspace(0, 60, 1200); mem = np.zeros_like(t)
mem[t >= 2]  = ctx_GiB                                           # A
load = (t >= 6) & (t < 26); mem[load] = ctx_GiB + weights_GiB * (t[load] - 6) / 20      # C
mem[t >= 26] = ctx_GiB + weights_GiB + non_torch_GiB
prof = (t >= 30) & (t < 33); mem[prof] += peak_act_GiB * np.sin(np.pi * (t[prof] - 30) / 3)   # D spike
mem[t >= 36] += kv_pool_GiB                                     # E
mem[t >= 42] += graphs_GiB                                       # F
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t, mem, color="#4e79a7", lw=2)
ax.axhline(requested_GiB, color="black", ls="--", lw=1); ax.text(0.5, requested_GiB + 1, "B. budget ceiling = total × util = 73.2 GiB", fontsize=8.5)
ax.axhline(GPU_TOTAL_GiB, color="#e15759", ls=":", lw=1); ax.text(0.5, GPU_TOTAL_GiB + 1, "physical 79.6 GiB", fontsize=8.5, color="#e15759")
for x, y, lab in [(2, 9, "A. CUDA context"), (6, 2, "C. weights stream in"), (30, 25, "D. profile run\n(spike measured)"), (36, 2, "E. KV pool allocated\n(all at once)"), (42, 9, "F. CUDA graphs"), (50, 2, "G. ready")]:
    ax.axvline(x, color="gray", lw=0.6, ls=":"); ax.text(x + 0.3, y, lab, fontsize=8.5, va="bottom")
ax.set_xlabel("seconds after launch (typical, 8 B model)"); ax.set_ylabel("GPU memory in use (GiB)"); ax.set_ylim(0, 86)
ax.set_title("Startup: memory climbs in steps and then never moves again — the profile spike (D) decides the size of E")
plt.tight_layout(); plt.show()
print(f"E. KV pool = {requested_GiB:.2f} − {weights_GiB:.2f} − {non_torch_GiB} − {peak_act_GiB} = {kv_pool_GiB:.2f} GiB")
"""))

cells.append(md(r"""## 3.1 Phase D in detail: the profile run

The question vLLM asks is not "how much will my users send?" but **"what is the largest single
step I could ever be asked to run?"** — because that is the only thing that can blow up memory
mid-flight. Then it runs exactly that step once, with garbage data, and watches the high-water
mark of memory during it.

The largest step is bounded by two flags:

- `--max-num-batched-tokens` = 8 192 tokens of work in the step, and
- `--max-num-seqs` = 256 requests contributing to it.

So the dummy step is **8 192 tokens over 256 fake requests, 32 tokens each**
(`vllm/v1/worker/gpu_model_runner.py`, `_dummy_run`). Then it runs a **dummy sampler**: picking
the next token for 256 requests means 256 × 128 256 scores in 32-bit floats (131 MiB) plus
temporary copies — a separate spike that must also be inside the budget.

What is inside the spike, and which flag each part scales with:

| Temporary tensor during the step | Size for our example | Scales with |
|---|---|---|
| hidden states of the tokens in the step | 8 192 × 4 096 × 2 B = 64 MiB per copy, several copies alive | batched tokens × hidden size |
| MLP intermediate (the widest layer) | 8 192 × 14 336 × 2 B = 224 MiB, ×2–3 alive | batched tokens × intermediate size |
| attention workspace (kernel-dependent) | tens to hundreds of MiB | batched tokens, sequence length, backend |
| logits for sampling | 256 × 128 256 × 4 B = 131 MiB, ×2–3 alive | max-num-seqs × vocabulary |
| vision encoder (VL models only, Part 7) | GiB-scale | largest image × items per step |

Add the CUDA-graph memory and, on multi-GPU, NCCL buffers. **None of this can be summed on
paper** with confidence: it depends on which attention kernel was chosen, how PyTorch's
allocator rounds, and the exact batch shape. So vLLM measures instead of estimating. Public logs
show the result spanning two orders of magnitude:

| Model | Setting | Measured activation peak per GPU |
|---|---|---|
| Qwen2.5-0.5B-AWQ | 2 048 batched tokens, 1 seq | 0.09 GiB |
| Qwen3-0.6B-FP8 | RTX 5090 defaults | 0.52 GiB |
| DeepSeek-R1, L40S | TP 8, PP 4, 16 K context | 1.52 GiB |
| Gemma-3-27B, A100 | 4 K context | 1.41 GiB |
| Gemma-3-27B, A100 | 131 K context (older vLLM profiled with max-model-len tokens) | 17.91 GiB |

**Why context length used to matter:** older vLLM profiled with `max_model_len` tokens in the
dummy step. Modern vLLM profiles with `max_num_batched_tokens`, because chunked prefill
guarantees no step is ever larger than that. That change alone is why the Gemma number fell
from 18 GiB to 1.4 GiB when the flag was lowered.
"""))

cells.append(code(r"""# Left: the measured spikes from the public logs in the table above. Right: what a spike of each
# size would leave for the KV pool on OUR server (same weights, same budget) — every GiB of
# spike is a GiB of pool lost, and with it a slice of concurrency.
measured = [("Qwen2.5-0.5B\n2 048 tok, 1 seq", 0.09), ("Qwen3-0.6B\ndefaults", 0.52), ("Gemma-3-27B\n4 K ctx", 1.41),
            ("DeepSeek-R1\nL40S, TP8", 1.52), ("our example\n(placeholder)", 4.0), ("Gemma-3-27B\n131 K ctx\nold profiler", 17.91)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 3.8))
names, vals = zip(*measured)
a1.bar(names, vals, color=["#76b7b2"]*4 + ["#f28e2b", "#e15759"]); a1.set_yscale("log"); a1.set_ylabel("measured activation peak per GPU (GiB, log)")
for i, v in enumerate(vals): a1.text(i, v * 1.15, f"{v:.2f}", ha="center", fontsize=8)
a1.tick_params(axis="x", labelsize=7); a1.set_title("The spike in real startup logs spans two orders of magnitude")
pools = [requested_GiB - weights_GiB - non_torch_GiB - v for v in vals]
concs = [p * GiB / kv_per_token / MAX_MODEL_LEN for p in pools]
a2.bar(names, pools, color="#59a14f")
for i, (p, c) in enumerate(zip(pools, concs)): a2.text(i, p + 0.5, f"{p:.1f} GiB\n{c:.1f}× at 32 K", ha="center", fontsize=8)
a2.tick_params(axis="x", labelsize=7); a2.set_ylim(0, 66); a2.set_ylabel("KV pool left on our H100 (GiB)")
a2.set_title("If our server measured that spike: pool left, and full-context concurrency")
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""## 3.2 Phase E: the KV pool becomes blocks, and the two log lines you will see

The remainder is turned into blocks and allocated as one tensor per layer. vLLM then prints two
numbers that summarise the whole startup:

- **`GPU KV cache size: N tokens`** — the pool ÷ bytes per token. This is the total number of
  tokens, across *all* requests together, that can be resident at once.
- **`Maximum concurrency for L tokens per request: X×`** — N ÷ `max-model-len`. The worst case:
  every request at its maximum length. Real traffic is shorter, so real concurrency is higher.
"""))

cells.append(code(r"""kv_blocks = int(kv_pool_GiB * GiB // (BLOCK * kv_per_token)); kv_tokens = kv_blocks * BLOCK
print(f"GPU KV cache size: {kv_tokens:,} tokens   ({kv_blocks:,} blocks of {BLOCK})")
print(f"Maximum concurrency for {MAX_MODEL_LEN:,} tokens per request: {kv_tokens/MAX_MODEL_LEN:.2f}x")

fig, ax = plt.subplots(figsize=(12, 2.8))
segs = [("weights", weights_GiB, "#4e79a7"), ("non-torch", non_torch_GiB, "#bab0ac"), ("activation peak (measured)", peak_act_GiB, "#f28e2b"),
        ("CUDA graphs", graphs_GiB, "#edc948"), ("KV cache pool", kv_pool_GiB - graphs_GiB, "#59a14f"), ("outside\n(1 − util)", GPU_TOTAL_GiB - requested_GiB, "#e15759")]
left = 0
above = {"non-torch": (0.45, -6), "CUDA graphs": (0.8, 6)}          # (height, x-offset in GiB) to keep thin labels apart
for name, w, c in segs:
    ax.barh(0, w, left=left, color=c, edgecolor="white", height=0.6)
    if w > 6: ax.text(left + w/2, 0, f"{name}\n{w:.1f} GiB", ha="center", va="center", fontsize=8.5, color="white")
    elif name in above:
        dy, dx = above[name]
        ax.annotate(f"{name} {w:.1f} GiB", xy=(left + w/2, 0.3), xytext=(left + w/2 + dx, dy), ha="center", va="bottom", fontsize=8.5, arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
    else:
        ax.annotate(f"{name} {w:.1f} GiB", xy=(left + w/2, -0.3), xytext=(left + w/2, -0.5), ha="center", va="top", fontsize=8.5, arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
    left += w
ax.axvline(requested_GiB, color="black", ls="--", lw=1); ax.text(requested_GiB - 0.5, 0.78, "budget ceiling 73.2 GiB", ha="right", fontsize=8.5)
ax.set_xlim(0, GPU_TOTAL_GiB); ax.set_ylim(-1.1, 1.1); ax.set_yticks([]); ax.set_xlabel("GiB on the H100")
ax.set_title("After phase G: who owns what. The green pool is allocated before the first request arrives.")
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""**Two things people find surprising, both visible above.** `nvidia-smi` shows ~73 GiB used on an
*idle* vLLM server, because the green pool is reserved up front. And the orange slice is not
wasted: during every step activations really are written there and freed; vLLM only refused to
let the KV cache grow into it.

## 3.3 How `--gpu-memory-utilization` and `--kv-cache-dtype` move the pool

Utilization moves the ceiling; the KV dtype moves how many tokens fit under it.
"""))

cells.append(code(r"""utils = [0.80, 0.85, 0.90, 0.92, 0.95, 0.98]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6))
pool_u = [u * GPU_TOTAL_GiB - weights_GiB - non_torch_GiB - peak_act_GiB for u in utils]
a1.bar([str(u) for u in utils], pool_u, color=["#59a14f"]*4 + ["#f28e2b", "#e15759"])
for i, p in enumerate(pool_u): a1.text(i, p + 0.4, f"{p:.1f}", ha="center", fontsize=8.5)
a1.set_xlabel("--gpu-memory-utilization"); a1.set_ylabel("KV pool (GiB)"); a1.set_title("Higher util → bigger pool, thinner cushion for CUDA graphs & other processes\n(0.98 is where public OOM reports cluster)")
toks_bf16 = kv_pool_GiB * GiB / kv_per_token; toks_fp8 = kv_pool_GiB * GiB / (kv_per_token / 2)
a2.bar(["BF16 KV (auto)", "FP8 KV"], [toks_bf16 / 1000, toks_fp8 / 1000], color=["#4e79a7", "#76b7b2"])
for i, v in enumerate([toks_bf16, toks_fp8]): a2.text(i, v/1000 + 15, f"{v/1000:.0f} k tokens\n= {v/MAX_MODEL_LEN:.1f} full-32K requests", ha="center", fontsize=8.5)
a2.set_ylabel("tokens the same pool can hold (thousands)"); a2.set_ylim(0, 1050); a2.set_title("--kv-cache-dtype fp8 halves bytes per token → 2× resident tokens")
plt.tight_layout(); plt.show()
"""))

# =============================================================================
# 4. Runtime: one request
# =============================================================================
cells.append(md(r"""---
# Part 4 — Runtime, one request: memory and time from arrival to last token

The server is idle: weights loaded, pool empty, memory flat. A request arrives over HTTP with a
1 000-token prompt and asks for up to 300 new tokens. Here is exactly what happens, in order.

1. **Tokenize** on the CPU: text → 1 000 ids. Check `1 000 + 300 ≤ max-model-len`; else reject.
2. **Prefix cache lookup.** Hash the prompt in 16-token blocks. If a previous request began with
   the same blocks (a shared system prompt, say), those blocks are reused and skipped.
3. **Queue.** The request enters the waiting list.
4. **Scheduling step.** The scheduler sees the pool has room and the step budget has room:
   it allocates `ceil(1000/16) = 63` blocks and puts all 1 000 tokens in this step.
5. **Forward pass (prefill).** 1 000 tokens flow through 32 layers. Activations occupy the
   orange slice briefly. Each layer writes its K and V for all 1 000 positions into the 63 blocks.
   The sampler picks output token 1. **TTFT** is the time to here.
6. **Decode steps.** 300 times: schedule 1 token, allocate a new block every 16 tokens, run one
   pass that reads all cached K/V, write one new K/V entry, sample. Each pass is one **ITL**.
7. **Finish.** End-of-text or 300 tokens reached. All 82 blocks return to the free list at once.
"""))

cells.append(code(r"""steps  = np.arange(0, ANSWER + 1)
tokens = PROMPT + steps
blocks = np.ceil(tokens / BLOCK).astype(int)

# Engineering-side time model (peak numbers; real kernels reach ~50-70% of peak, so scale by 0.6):
EFF = 0.6
def prefill_time_ms(n_tok):          # compute-bound: 2 FLOPs per parameter per token
    return 2 * PARAMS * n_tok / (GPU_TFLOPS * 1e12 * EFF) * 1e3
def decode_time_ms(kv_tokens_read, n_reqs=1):   # memory-bound: read all weights + all KV once
    bytes_moved = PARAMS * BYTES_W + kv_tokens_read * kv_per_token
    compute = 2 * PARAMS * n_reqs / (GPU_TFLOPS * 1e12 * EFF)
    return max(bytes_moved / (GPU_BW_TBs * 1e12 * EFF), compute) * 1e3
ttft = prefill_time_ms(PROMPT); itl = [decode_time_ms(PROMPT + s) for s in range(1, ANSWER + 1)]
print(f"TTFT ≈ prefill of {PROMPT} tokens = 2 × {PARAMS/1e9:.2f}e9 × {PROMPT} FLOP / ({GPU_TFLOPS} TFLOP/s × {EFF}) ≈ {ttft:.0f} ms")
print(f"ITL  ≈ (weights {PARAMS*BYTES_W/GiB:.1f} GiB + KV {PROMPT*kv_per_token/MiB:.0f} MiB) / ({GPU_BW_TBs} TB/s × {EFF}) ≈ {itl[0]:.1f} ms per token  → {1000/itl[0]:.0f} tokens/s for ONE request")
print(f"total ≈ {ttft + sum(itl):.0f} ms for 300 tokens; KV at the end: {blocks[-1]} blocks = {blocks[-1]*BLOCK*kv_per_token/MiB:.0f} MiB ({blocks[-1]/kv_blocks*100:.2f} % of the pool)")

fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
axs[0].step(steps, blocks, where="post", color="#59a14f"); axs[0].set_xlabel("step"); axs[0].set_ylabel("KV blocks held"); axs[0].set_title("Memory: +1 block / 16 decode tokens")
axs[0].annotate("prefill claims 63 blocks at once", xy=(0, blocks[0]), xytext=(60, blocks[0] - 7), arrowprops=dict(arrowstyle="->"), fontsize=8)
axs[1].bar([0], [ttft], color="#f28e2b"); axs[1].bar(steps[1:], itl, color="#4e79a7", width=1); axs[1].set_xlabel("step"); axs[1].set_ylabel("ms"); axs[1].set_title(f"Time: one {ttft:.0f} ms prefill, then {itl[0]:.0f} ms decodes")
cum = np.cumsum([ttft] + itl); axs[2].plot(cum / 1000, np.arange(1, ANSWER + 2), color="#4e79a7"); axs[2].set_xlabel("seconds since arrival"); axs[2].set_ylabel("output tokens delivered"); axs[2].set_title("User's view: pause (TTFT), then a stream")
axs[2].annotate("TTFT", xy=(ttft/1000, 1), xytext=(0.3, 60), arrowprops=dict(arrowstyle="->"), fontsize=9)
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""**The engineering point hiding in the middle panel.** A decode step for one request reads
~15 GiB of weights to produce a single token. The GPU's arithmetic units are idle almost the
whole time; the step is waiting on memory. That is why *batching* is nearly free: reading the
weights once serves 64 requests as easily as 1. Part 6 quantifies it.
"""))

# =============================================================================
# 5. Runtime: many requests
# =============================================================================
cells.append(md(r"""---
# Part 5 — Runtime, many requests: the scheduler loop, the block table, and preemption

Every step, the scheduler (`vllm/v1/core/sched/scheduler.py`) runs this exact loop:

```
budget = max_num_batched_tokens
1. for each RUNNING request (in order):            # decodes first — never starve a stream
       give it 1 token (or its next prompt chunk); allocate its next block if needed
       if no block is free: PREEMPT the lowest-priority running request (free its blocks,
                            push it to the front of WAITING) and retry
2. while WAITING is non-empty and budget > 0 and running < max_num_seqs:
       take the oldest waiting request; give it min(remaining prompt, budget) tokens
       (chunked prefill); allocate blocks; if no block is free, stop admitting
3. run ONE forward pass over everything scheduled; sample one token per request
4. finished requests free all their blocks
```

Below is that loop implemented in ~40 lines and run on 60 requests that arrive together, with
the real limits from our example. The simulator records, at every step, what was in the step,
how many requests were running and waiting, and which blocks each request owned.
"""))

cells.append(code(r"""def simulate(n_req, prompt_range, answer_range, pool_blocks, batch_tok=MAX_BATCH_TOK, max_seqs=MAX_NUM_SEQS,
             arrivals=None, seed=0, max_steps=100000, snapshot_steps=()):
    rng = np.random.default_rng(seed)
    prompts = rng.integers(*prompt_range, n_req); answers = rng.integers(*answer_range, n_req)
    arrive = np.zeros(n_req, int) if arrivals is None else arrivals
    computed = np.zeros(n_req, int); generated = np.zeros(n_req, int)
    owned = [[] for _ in range(n_req)]                       # block ids per request
    free = list(range(pool_blocks))[::-1]                    # free list; pop() hands out low ids first
    waiting, running, done = [], [], set()
    log, snaps, preemptions = [], {}, 0
    first_tok_step = np.full(n_req, -1)
    def need_blocks(r, extra): return int(np.ceil((computed[r] + generated[r] + extra) / BLOCK)) - len(owned[r])
    def take(r, k):
        for _ in range(k): owned[r].append(free.pop())
    def release(r):
        free.extend(owned[r]); owned[r] = []
    for step in range(max_steps):
        for r in np.where(arrive == step)[0]: waiting.append(int(r))
        budget = batch_tok; sched = {}
        # 1. running requests
        for r in list(running):
            n = min(prompts[r] - computed[r], budget) if computed[r] < prompts[r] else 1
            if n == 0: continue
            k = need_blocks(r, n)
            while k > len(free) and running:
                victim = running[-1]; running.remove(victim); release(victim)
                computed[victim] = 0; generated[victim] = 0; waiting.insert(0, victim); preemptions += 1
                if victim == r: k = -1; break
            if k < 0: continue
            take(r, k); budget -= n; sched[r] = n
        # 2. admit waiting (chunked prefill)
        while waiting and budget > 0 and len(running) < max_seqs:
            r = waiting[0]; n = min(prompts[r] - computed[r], budget); k = need_blocks(r, n)
            if k > len(free): break
            waiting.pop(0); running.append(r); take(r, k); budget -= n; sched[r] = n
        # 3. run the step
        pre = dec = 0
        for r, n in sched.items():
            if computed[r] < prompts[r]:
                computed[r] += n; pre += n
                if computed[r] == prompts[r]: generated[r] += 1; first_tok_step[r] = step if first_tok_step[r] < 0 else first_tok_step[r]
            else: generated[r] += 1; dec += 1
        # 4. finish
        for r in list(running):
            if computed[r] == prompts[r] and generated[r] >= answers[r]:
                running.remove(r); release(r); done.add(r)
        log.append((step, pre, dec, len(running), len(waiting), pool_blocks - len(free)))
        if step in snapshot_steps:
            table = np.zeros(pool_blocks, int)
            for r in running:
                for b in owned[r]: table[b] = r + 1
            snaps[step] = table
        if len(done) == n_req and step >= arrive.max(): break
    return np.array(log), preemptions, snaps, first_tok_step - arrive + 1, prompts   # +1: the step that produces the token must run

log, pre, snaps, ttft_steps, prompts = simulate(60, (500, 3000), (150, 400), kv_blocks, snapshot_steps=(2, 12, 200))
step, pre_tok, dec_tok, n_run, n_wait, used = log.T
print(f"60 requests: done in {len(log)} steps, preemptions {pre}, peak running {n_run.max()} (cap {MAX_NUM_SEQS}), peak pool use {used.max()/kv_blocks*100:.1f} %")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True)
a1.bar(step, pre_tok, color="#f28e2b", width=1, label="prefill tokens (prompts, chunked)"); a1.bar(step, dec_tok, bottom=pre_tok, color="#4e79a7", width=1, label="decode tokens (1 per running request)")
a1.set_yscale("symlog", linthresh=100); a1.axhline(MAX_BATCH_TOK, color="black", ls="--", lw=1); a1.text(len(step) * 0.99, MAX_BATCH_TOK * 1.15, "--max-num-batched-tokens = 8 192", ha="right", fontsize=8)
a1.set_ylabel("tokens in the step (symlog)"); a1.legend(loc="center right"); a1.set_title("60 requests arrive at once: the first ~10 steps are prompt-chewing at the cap; afterwards each step is 1 token per running request")
a2.plot(step, n_run, color="#59a14f", label="running"); a2.plot(step, n_wait, color="#e15759", label="waiting"); a2.plot(step, used / kv_blocks * 100, color="#4e79a7", ls="--", label="KV pool used (%)")
a2.set_xlabel("step"); a2.set_ylabel("requests / % pool"); a2.legend()
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""## 5.1 The block table, frozen at three moments

Below are three snapshots of the pool (drawn as a grid of blocks, one colour per request; the
pool is 27 635 blocks, so only the first 2 400 are drawn). Step 2: a few prompts have been
prefilled. Step 12: all 60 are in and decoding. Step 200: about half have finished and returned
their blocks, which the next admissions reuse — the pool never fragments.
"""))

cells.append(code(r"""fig, axs = plt.subplots(3, 1, figsize=(12, 6.2))
cmap = plt.get_cmap("tab20")
W, SHOW = 120, 2400
for ax, s in zip(axs, (2, 12, 200)):
    table = snaps[s][:SHOW]
    img = np.zeros((SHOW // W, W, 3))
    for i, v in enumerate(table): img[i // W, i % W] = (0.93, 0.93, 0.93) if v == 0 else cmap((v - 1) % 20)[:3]
    ax.imshow(img, aspect="auto", interpolation="nearest"); ax.set_yticks([]); ax.set_xticks([])
    ax.set_title(f"step {s}: {np.count_nonzero(snaps[s]):,} of {kv_blocks:,} blocks owned  ·  {len(set(snaps[s]) - {0})} requests resident", fontsize=9.5, loc="left")
plt.suptitle("The block table (first 2 400 blocks). Grey = free. Each request's blocks need not be contiguous.", fontsize=10)
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""## 5.2 The three concurrency limits — and which one you are hitting

"How many users can this server handle at once?" has three answers, and the smallest wins:

| Limit | Value in the example | It is the binding one when |
|---|---|---|
| `--max-num-seqs` | 256 | requests are short |
| KV pool ÷ tokens per request | 442 k ÷ 1 300 ≈ 340; at full 32 K: 13.5 | requests are long |
| `--max-num-batched-tokens` | 8 192 | *never* limits how many run — only how fast prompts get in |

The figure sweeps the average request length and shows which limit is binding.
"""))

cells.append(code(r"""lengths = np.array([256, 512, 1000, 2000, 4000, 8000, 16000, 32768])
by_kv = kv_tokens / lengths; by_seqs = np.full_like(lengths, MAX_NUM_SEQS, dtype=float)
fig, ax = plt.subplots(figsize=(12, 3.6))
ax.plot(lengths, by_kv, "o-", color="#59a14f", label="KV pool ÷ tokens per request"); ax.plot(lengths, by_seqs, "--", color="#e15759", label="--max-num-seqs = 256")
ax.plot(lengths, np.minimum(by_kv, by_seqs), lw=4, alpha=0.3, color="black", label="effective concurrency = min of the two")
ax.set_xscale("log", base=2); ax.set_xticks(lengths); ax.set_xticklabels([f"{l:,}" for l in lengths]); ax.set_yscale("log")
ax.set_xlabel("average tokens per request (prompt + answer)"); ax.set_ylabel("requests resident at once"); ax.legend()
ax.set_title("Short requests: capped by --max-num-seqs. Long requests: capped by the KV pool. The crossover here is ≈ 1 700 tokens.")
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""## 5.3 Requests arriving over time, not all at once

Real traffic trickles in. Here 300 requests arrive at random over 1 500 steps (≈ 7.5 s at 5 ms a
step), with prompts from 300 to 12 000 tokens. Watch the running count hover, the pool breathe,
and the TTFT of each request: a prompt under the 8 192 cap is usually scheduled in the step it
arrives (TTFT = 1 step), because there is always budget left after the running requests' one
token each; a longer prompt needs 2 steps (chunked), and a burst adds queueing on top.
"""))

cells.append(code(r"""rng = np.random.default_rng(1)
arr = np.sort(rng.integers(0, 1500, 300))
log3, pre3, _, ttft3, prompts3 = simulate(300, (300, 12000), (100, 500), kv_blocks, arrivals=arr, seed=1)
s3, p3, d3, r3, w3, u3 = log3.T
fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=False)
axs[0].bar(s3, p3, color="#f28e2b", width=1, label="prefill tokens"); axs[0].bar(s3, d3, bottom=p3, color="#4e79a7", width=1, label="decode tokens"); axs[0].set_yscale("symlog", linthresh=100)
axs[0].set_ylabel("tokens / step"); axs[0].legend(loc="upper right"); axs[0].set_title("Steady traffic: every step carries the running requests' decode tokens plus whichever new prompts arrived")
axs[1].plot(s3, r3, color="#59a14f", label="running"); axs[1].plot(s3, w3, color="#e15759", label="waiting"); axs[1].plot(s3, u3 / kv_blocks * 100, "--", color="#4e79a7", label="KV pool used (%)"); axs[1].set_ylabel("requests / %"); axs[1].legend(loc="upper right"); axs[1].set_xlabel("step")
axs[2].scatter(prompts3, ttft3 * 5, s=8, color="#4e79a7"); axs[2].set_xlabel("prompt length (tokens)"); axs[2].set_ylabel("TTFT (ms, at 5 ms/step)"); axs[2].set_title("Time to first token grows with prompt length (chunking) and with queueing when a burst lands")
plt.tight_layout(); plt.show()
print(f"preemptions: {pre3}; median TTFT {np.median(ttft3)*5:.0f} ms; p99 {np.percentile(ttft3, 99)*5:.0f} ms (scheduling steps only)")
"""))

cells.append(md(r"""## 5.4 When the pool runs out: preemption, never a crash

Now the failure case. 40 requests each want to grow to ~20 000 tokens, on a deliberately small
pool of 20 000 blocks (320 000 tokens). They need 800 000. The scheduler cannot fit them; what
does it do?
"""))

cells.append(code(r"""log4, pre4, _, _, _ = simulate(40, (4000, 6000), (14000, 15000), 20_000)
s4, _, _, r4, w4, u4 = log4.T
print(f"preemptions: {pre4}; all 40 finished in {len(log4):,} steps; the server never allocated beyond its pool")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
a1.plot(s4, u4 / 20_000 * 100, color="#59a14f"); a1.axhline(100, color="black", ls="--", lw=1); a1.set_ylabel("KV pool used (%)"); a1.set_title("Pool too small for the traffic: usage pins at 100 %, the scheduler evicts and later recomputes")
a2.plot(s4, r4, color="#59a14f", label="running"); a2.plot(s4, w4, color="#e15759", label="waiting (incl. evicted)"); a2.set_xlabel("step"); a2.set_ylabel("requests"); a2.legend()
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""When a running request needs one more block and none is free, vLLM **preempts** the
lowest-priority running request (by default the most recently arrived): frees *all* its blocks,
puts it at the front of the waiting queue, and later **recomputes** its prefill from scratch.
The request is delayed, not lost; the server never runs out of memory. In the log this appears
as a preemption warning; in `/metrics` as `vllm:num_preemptions_total`. Frequent preemption is
the signal that the pool is too small for the traffic: lower `--max-num-seqs`, lower
`--max-model-len`, switch to `--kv-cache-dtype fp8`, quantize the weights, or add a GPU.
"""))

# =============================================================================
# 6. Engineering: speed
# =============================================================================
cells.append(md(r"""---
# Part 6 — Engineering: what actually limits speed, and how the flags trade memory for it

A GPU has two ceilings. **Compute**: how many multiply-adds per second (H100 BF16 ≈ 989 TFLOP/s).
**Memory bandwidth**: how many bytes per second it can stream from its memory (H100 ≈ 3.35 TB/s).
A step is bound by whichever it exhausts first.

- A **decode** step for `n` requests does `2 × params × n` FLOPs but must read **all the weights
  plus every running request's KV cache** once. Reading 15 GiB at 3.35 TB/s takes ~4.5 ms;
  the arithmetic for n = 1 takes ~0.02 ms. Decode is **memory-bound** until n is in the hundreds.
- A **prefill** step over 8 192 tokens does `2 × params × 8 192` FLOPs = 131 TFLOP: ~130 ms of
  pure arithmetic. Prefill is **compute-bound**.

That asymmetry is the whole design rationale: batch decodes aggressively (near-free), and chunk
prefills so they do not stall the decodes behind them.
"""))

cells.append(code(r"""ns = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
avg_ctx = 1500
step_ms = np.array([decode_time_ms(n * avg_ctx, n) for n in ns])
tps = ns / step_ms * 1000
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6))
a1.plot(ns, step_ms, "o-", color="#4e79a7"); a1.set_xscale("log", base=2); a1.set_xlabel("requests decoding in the step"); a1.set_ylabel("step time (ms)")
a1.set_title("Decode step time is nearly flat: weights are read once for everyone\n(rises when KV traffic or compute finally dominates)")
a2.plot(ns, tps, "o-", color="#59a14f"); a2.set_xscale("log", base=2); a2.set_xlabel("requests decoding in the step"); a2.set_ylabel("output tokens / second (all users)")
a2.set_title("So throughput scales almost linearly with --max-num-seqs\n(each user still sees ~1 / step-time tokens per second)")
plt.tight_layout(); plt.show()
print("one user alone: %.0f tok/s;  256 users: %.0f tok/s total, %.0f tok/s each" % (tps[0], tps[8], tps[8] / 256))
"""))

cells.append(md(r"""## 6.1 `--max-num-batched-tokens`: prompt speed versus stream smoothness

A bigger token cap lets a long prompt finish prefill in fewer steps (lower TTFT for that
request) but makes each of those steps longer — and every *other* user's next token waits for
that step. A smaller cap keeps steps short and streams smooth, at the cost of more steps per
long prompt. And, from Part 3, the cap also sets the activation spike and therefore the KV pool.
"""))

cells.append(code(r"""caps = [1024, 2048, 4096, 8192, 16384]
long_prompt = 16_000
steps_needed = [int(np.ceil(long_prompt / c)) for c in caps]
step_len_ms  = [prefill_time_ms(c) + decode_time_ms(60 * 1500, 60) for c in caps]          # a prefill chunk sharing the step with 60 decoders
ttft_ms      = [s * l for s, l in zip(steps_needed, step_len_ms)]
worst_itl_ms = step_len_ms
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6))
a1.bar([str(c) for c in caps], ttft_ms, color="#f28e2b"); a1.set_ylabel("ms"); a1.set_xlabel("--max-num-batched-tokens")
for i, v in enumerate(ttft_ms): a1.text(i, v + 15, f"{v:.0f} ms\n{steps_needed[i]} steps", ha="center", fontsize=8)
a1.set_ylim(0, max(ttft_ms) * 1.25); a1.set_title("TTFT of a 16 000-token prompt: bigger cap → fewer, longer steps")
a2.bar([str(c) for c in caps], worst_itl_ms, color="#4e79a7"); a2.set_ylabel("ms"); a2.set_xlabel("--max-num-batched-tokens")
for i, v in enumerate(worst_itl_ms): a2.text(i, v + 3, f"{v:.0f} ms", ha="center", fontsize=8)
a2.set_title("Worst inter-token latency for everyone else while that prompt is chunking")
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""## 6.2 The full trade-off table

| Flag ↑ (increase it) | Startup memory | Runtime memory | Speed |
|---|---|---|---|
| `--gpu-memory-utilization` | bigger KV pool | more resident tokens | more concurrency; risk of OOM at CUDA-graph capture / from other processes |
| `--max-model-len` | (modern vLLM) little; attention workspace | each request may hold more KV → fewer fit at full length | longer prompts allowed; TTFT of long prompts |
| `--max-num-seqs` | bigger sampler spike (logits × seqs) | more requests resident (if pool allows) | higher total throughput; per-user speed flat until compute-bound |
| `--max-num-batched-tokens` | **bigger activation spike → smaller pool** | – | faster prefill of long prompts; rougher decode for others |
| `--kv-cache-dtype fp8` | – | 2× tokens per pool | slightly faster decode (less KV traffic); minor accuracy cost |
| `--quantization` / lower `--dtype` | weights ÷ 2–4 | pool grows by what weights freed | faster decode (fewer weight bytes) unless the format needs software dequant |
| `--tensor-parallel-size` | weights, KV, activations split N ways | pool = sum of N cards' remainders | per-step all-reduce cost; needed when one card cannot hold the weights |
| `--enforce-eager` | frees CUDA-graph GiB | – | slower decode (kernel launch overhead per step) |
| `--enable-prefix-caching` | hash tables | shared blocks for shared prefixes | skips prefill of repeated prefixes → much lower TTFT for chat with a system prompt |
"""))

# =============================================================================
# 7. Multimodal
# =============================================================================
cells.append(md(r"""---
# Part 7 — Models that take images and video

A vision-language model (Qwen2.5-VL, Llama-3.2-Vision, Gemma-3) is a text LLM with an
**encoder** bolted on the front: a separate network that turns pixels into a row of *image
tokens* the LLM reads as if they were text.

**How an image becomes tokens.** The image is cut into a grid of small squares (patches, e.g.
14×14 pixels), each patch becomes one vector, neighbouring patches are often merged (2×2), and
the result is a sequence. A 1280×720 image in Qwen2.5-VL is about 1 170 tokens. A video is a
stack of frames, each a smaller image: 16 frames × ~300 tokens ≈ 4 800 tokens. Once produced,
image tokens cost KV cache exactly like text tokens: 1 170 × 128 KiB ≈ 146 MiB for our model.
"""))

cells.append(code(r"""fig, axs = plt.subplots(1, 3, figsize=(13, 3.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1.6]})
ax = axs[0]; ax.set_title("1. image → 28×28-pixel merged patches")
for i in range(9):
    for j in range(16): ax.add_patch(Rectangle((j, i), 0.95, 0.95, fc=plt.get_cmap("viridis")((i * 16 + j) / 144), ec="white"))
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.set_aspect("equal"); ax.axis("off"); ax.text(8, -0.9, "1280×720 → 45×25 = 1 170 patches (drawn 16×9)", ha="center", fontsize=8)
ax = axs[1]; ax.set_title("2. vision encoder → 1 170 token vectors")
for k in range(24): ax.add_patch(Rectangle((k * 0.6, 0.3), 0.5, 0.5, fc=plt.get_cmap("viridis")(k / 24), ec="white"))
ax.text(7.2, 0.55, "…", fontsize=14, va="center"); ax.set_xlim(0, 15); ax.set_ylim(0, 1); ax.axis("off")
ax.text(7.5, -0.1, "a sequence, indistinguishable to the LLM from text tokens", ha="center", fontsize=8)
ax = axs[2]; ax.set_title("3. what each input costs in KV cache")
items = [("1 000-word text", 1300), ("one 1280×720 image", 1170), ("four images", 4680), ("16-frame video", 4800)]
ax.barh([n for n, _ in items], [t * kv_per_token / MiB for _, t in items], color=["#4e79a7", "#59a14f", "#59a14f", "#f28e2b"])
for i, (n, t) in enumerate(items): ax.text(t * kv_per_token / MiB + 10, i, f"{t:,} tok → {t*kv_per_token/MiB:.0f} MiB", va="center", fontsize=8)
ax.set_xlim(0, 900); ax.invert_yaxis(); ax.set_xlabel("MiB of KV cache")
plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""**What changes at profile time (phase D).** Two new memory consumers, both measured
(`profile_run`, the `supports_mm_inputs` branch):

1. **The encoder's own activations.** Running the vision tower on a large image is a big
   forward pass in itself. vLLM builds a dummy batch of the **largest allowed item** (largest
   image or longest video, from `--limit-mm-per-prompt` and the model's own limits), as many as
   could appear in one step, and runs the encoder on that.
2. **The encoder cache.** Encoder outputs are parked on the GPU until the LLM has consumed
   them, so a prompt with three images does not re-run the vision tower each time its prefill is
   chunked. Its size is a token budget (`min(encoder compute budget, encoder cache size)`); the
   dummy outputs sit in it during the profile so its footprint is inside the peak.

Then the ordinary text profile runs on top. Result: a larger measured spike, a smaller KV pool.

**What changes at runtime.** A request with an image is scheduled in two parts: the encoder
runs once (its output lands in the encoder cache), then the LLM prefill consumes those 1 170
embeddings like any other prompt tokens, writing 1 170 KV entries. The encoder cache entry is
freed once consumed. `--limit-mm-per-prompt image=N` caps N images per request, which caps both
the profile's worst case and the per-request KV. `--skip-mm-profiling` drops items 1–2 from the
profile — more KV pool, but safe only if you have also constrained image size and count.
"""))

cells.append(code(r"""vision_weights, vision_peak, encoder_cache = 1.3, 2.5, 1.0     # illustrative GiB for a ~600 M-param vision tower
rows = {"text-only LLM": [("weights", weights_GiB), ("non-torch", non_torch_GiB), ("LLM activation peak", peak_act_GiB)],
        "same LLM + vision encoder": [("weights", weights_GiB), ("vision weights", vision_weights), ("non-torch", non_torch_GiB),
                                      ("LLM activation peak", peak_act_GiB), ("vision encoder peak", vision_peak), ("encoder cache", encoder_cache)]}
colors = {"weights": "#4e79a7", "vision weights": "#76b7b2", "non-torch": "#bab0ac", "LLM activation peak": "#f28e2b",
          "vision encoder peak": "#e15759", "encoder cache": "#edc948", "KV cache pool": "#59a14f"}
fig, ax = plt.subplots(figsize=(12, 2.9))
for y, (label, parts) in enumerate(rows.items()):
    left = 0; parts = parts + [("KV cache pool", requested_GiB - sum(v for _, v in parts))]
    for name, w in parts:
        ax.barh(y, w, left=left, color=colors[name], edgecolor="white")
        if w > 2.5: ax.text(left + w/2, y, f"{w:.1f}", ha="center", va="center", fontsize=8, color="white")
        left += w
    ax.text(requested_GiB + 0.4, y, f"KV pool {parts[-1][1]:.1f} GiB", va="center", fontsize=9)
ax.set_yticks([0, 1]); ax.set_yticklabels(list(rows)); ax.set_xlim(0, GPU_TOTAL_GiB + 9); ax.axvline(requested_GiB, color="black", ls="--", lw=1)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()], list(colors), ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
ax.set_xlabel("GiB inside the utilization ceiling"); plt.tight_layout(); plt.show()
"""))

# =============================================================================
# 8. Reading the log
# =============================================================================
cells.append(md(r"""---
# Part 8 — Reading your own startup log

Every number in this notebook has a line in the real log. Here is a representative vLLM startup
log for our example, annotated with the part of the notebook that explains it.

```
INFO  Initializing a V1 LLM engine with config: model='meta-llama/Llama-3.1-8B-Instruct',
      dtype=torch.bfloat16, max_seq_len=32768, ...                     ← Part 2: dtype = 2 B/param; max_seq_len = --max-model-len
INFO  Chunked prefill is enabled with max_num_batched_tokens=8192.      ← Part 1.5 / 6.1
INFO  Loading weights took 18.4 seconds                                 ← Part 3, phase C
INFO  Model loading took 14.96 GiB memory                               ← Part 1.2: 8.03 B × 2 B
INFO  Memory profiling takes 1.31 seconds. Total non KV cache memory: 19.3GiB;
      torch peak memory increase: 4.0GiB; non-torch forward increase memory: 0.3GiB;
      weights memory: 14.96GiB.                                         ← Part 3.1, phase D: the MEASURED spike
INFO  Available KV cache memory: 53.9 GiB                               ← Part 3.2: 73.2 − 19.3
INFO  GPU KV cache size: 442,160 tokens                                 ← 53.9 GiB ÷ 128 KiB
INFO  Maximum concurrency for 32,768 tokens per request: 13.49x         ← Part 5.2, worst case
INFO  Graph capturing finished in 9 secs, took 0.93 GiB                 ← Part 3, phase F
INFO  Uvicorn running on http://0.0.0.0:8000                            ← phase G
```

And at runtime, the periodic stats line:

```
INFO  Avg prompt throughput: 3120.5 tokens/s, Avg generation throughput: 1874.2 tokens/s,
      Running: 58 reqs, Waiting: 0 reqs, GPU KV cache usage: 21.4%, Prefix cache hit rate: 37.2%
```
`Running` and `Waiting` are the two queues of Part 5; `GPU KV cache usage` is the green pool's
occupancy; `Prefix cache hit rate` is how much prefill was skipped by block sharing.

---
# Summary — the one picture to keep

```
GPU total 79.6 GiB
├── outside the budget (1 − util) ....... 6.4 GiB  CUDA context, other processes, safety
└── util × total ......................... 73.2 GiB  everything vLLM will ever touch
    ├── weights ........................... 15.0 GiB  fixed; set by params × bytes/param
    ├── non-torch (NCCL, workspaces) ......  0.3 GiB  measured
    ├── activation + sampler spike ........  4.0 GiB  MEASURED by one dummy worst-case step:
    │                                                 max-num-batched-tokens × max-num-seqs
    │                                                 (+ vision encoder for VL models)
    ├── CUDA graphs .......................  1.0 GiB  captured after the pool is sized
    └── KV cache pool ..................... 53.0 GiB  everything left; 16-token blocks;
                                                      claimed per request, freed on finish;
                                                      = concurrency; evicts, never OOMs
```

- **Startup:** measure the worst-case step; give the KV cache the remainder.
- **Runtime:** every step = 1 token for each running request + as much new prompt as fits under
  `max-num-batched-tokens`; blocks are claimed 16 tokens at a time and returned at the end.
- **Concurrency** = min(`max-num-seqs`, pool ÷ tokens per request); short traffic hits the
  first, long traffic the second; the pool evicts and recomputes rather than crash.
- **Speed:** decode is memory-bound (batching is nearly free), prefill is compute-bound (chunk
  it). `max-num-batched-tokens` is the knob that trades prompt speed, stream smoothness *and*
  KV pool size in one move.
"""))

nb["cells"] = cells
nbf.write(nb, "vllm_memory_explainer.ipynb")
print("wrote vllm_memory_explainer.ipynb with", len(cells), "cells")
