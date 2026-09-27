# Maintenance review — 2026-09-27

## Purpose and baseline

`ml-scratchpad` teaches machine-learning and inference concepts through small NumPy implementations and notebooks. The checkout was clean before this review. The continuous-batching scheduler and occupancy views did not validate batch sizes or generation lengths; zero-length work could be counted inconsistently, and the continuous occupancy loop could fail to drain.

## Changes

`continuous-batching/batching.py` now rejects non-positive or non-integer batch sizes and generation lengths, empty prompts, and duplicate request IDs before scheduling. Both occupancy functions return a consistent `(ticks, batch_size)` integer array, including an empty workload. The continuous occupancy queue uses `deque`, avoiding repeated list-front removals as requests arrive. Continuous scheduling materializes one-shot iterables before validation so generator workloads are not consumed prematurely.

## Verification

Ran `..\\.venv\\Scripts\\python.exe -m unittest -v test_batching` from `continuous-batching`: **4 passed**. The checked-in tests cover invalid zero/negative lengths, empty workload shapes, generator input, and output equivalence between both schedulers and standalone greedy decoding with a tiny untrained NumPy GPT. They do not train a model, read a dataset, or write files.

## Remaining gaps

The tests establish correctness on a small synthetic model; they do not measure scheduler throughput under a large workload.
