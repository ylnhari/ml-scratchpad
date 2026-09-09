# How an LLM server uses a GPU: vLLM from power-on to the thousandth request

A visual course that assumes no prior knowledge. One worked example runs through it:
Llama-3.1-8B-Instruct in BF16 on a single H100 80 GB with vLLM's defaults.

| Part | What you learn |
|---|---|
| 1 | The concepts, each with a picture: token, parameter and precision, forward pass and attention, the KV cache and its per-token size, prefill vs decode, batching and the step, paged blocks |
| 2 | Every `vllm serve` flag in plain words, and what each one moves at startup and at runtime |
| 3 | Startup second by second: the memory-versus-time trace, the profile run and why its spike is measured rather than computed, how the KV pool is sized, what utilization and KV dtype change |
| 4 | One request from arrival to last token: blocks claimed, time per step, time to first token, inter-token latency |
| 5 | Many requests: the scheduler loop, the block table frozen at three moments, the three concurrency limits, steady traffic, preemption when the pool is full |
| 6 | Engineering: compute-bound prefill vs memory-bound decode, why batching is nearly free, the batched-tokens trade-off, the full flag trade-off table |
| 7 | Image and video models: pixels to tokens, what the encoder and encoder cache add at profile time and at runtime |
| 8 | Reading your own startup log line by line |

**Read it in a browser, no setup:** https://ylnhari.github.io/learning/vllm-memory/

Open the notebook with `make vllm-memory-notebook` from the `learning/` root. The notebook is pre-executed;
all 20 figures are embedded, so it reads without running anything. To change the example, edit
the constants in the first code cell and re-run.

Regenerate from source with `make vllm-memory-regen`, execute it (Kernel: Learning (shared)), then
`make vllm-memory-html` to refresh the published page (`index.html` is a generated snapshot).
