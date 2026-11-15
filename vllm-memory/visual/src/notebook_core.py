import numpy as np
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

kv_per_layer = 2 * KV_HEADS * HEAD_DIM * BYTES_KV              # K and V, 8 heads of 128 numbers, 2 bytes each
kv_per_token = LAYERS * kv_per_layer
print(f"per layer : 2 (K,V) × {KV_HEADS} heads × {HEAD_DIM} numbers × {BYTES_KV} B = {kv_per_layer:,} B")
print(f"per token : {LAYERS} layers × {kv_per_layer:,} B = {kv_per_token:,} B = {kv_per_token/KiB:.0f} KiB")
print(f"1 000-token prompt = {1000*kv_per_token/MiB:.0f} MiB;  32 768-token request = {MAX_MODEL_LEN*kv_per_token/GiB:.2f} GiB")

weights_GiB   = PARAMS * BYTES_W / GiB
requested_GiB = GPU_TOTAL_GiB * UTIL
non_torch_GiB = 0.3           # NCCL / attention workspaces, also measured
peak_act_GiB  = 4.0           # <-- MEASURED in phase D (placeholder; your log prints the real one)
graphs_GiB    = 1.0
kv_pool_GiB   = requested_GiB - weights_GiB - non_torch_GiB - peak_act_GiB
ctx_GiB       = 0.5
print(f"E. KV pool = {requested_GiB:.2f} − {weights_GiB:.2f} − {non_torch_GiB} − {peak_act_GiB} = {kv_pool_GiB:.2f} GiB")

kv_blocks = int(kv_pool_GiB * GiB // (BLOCK * kv_per_token)); kv_tokens = kv_blocks * BLOCK
print(f"GPU KV cache size: {kv_tokens:,} tokens   ({kv_blocks:,} blocks of {BLOCK})")
print(f"Maximum concurrency for {MAX_MODEL_LEN:,} tokens per request: {kv_tokens/MAX_MODEL_LEN:.2f}x")

PROMPT, ANSWER = 1000, 300
steps  = np.arange(0, ANSWER + 1)
tokens = PROMPT + steps
blocks = np.ceil(tokens / BLOCK).astype(int)

# A time model (peak numbers; real kernels reach ~50-70% of peak, so scale by 0.6):
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

def simulate(n_req, prompt_range, answer_range, pool_blocks, batch_tok=MAX_BATCH_TOK, max_seqs=MAX_NUM_SEQS,
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

ns = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], dtype=np.int64)   # int64: browser Python is 32-bit, so numpy's default int would overflow
avg_ctx = 1500
step_ms = np.array([decode_time_ms(n * avg_ctx, n) for n in ns])
tps = ns / step_ms * 1000
print("one user alone: %.0f tok/s;  256 users: %.0f tok/s total, %.0f tok/s each" % (tps[0], tps[8], tps[8] / 256))