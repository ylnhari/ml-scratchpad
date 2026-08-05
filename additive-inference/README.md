# additive-inference

**How a 1.7-bit, multiplication-free model works** — and which part of the claim is
actually new.

Prompted by Syzygy Research's Mach-1 Additive announcement (3 Aug 2026): a 35B model
that "can inference without ever multiplying by a weight", at 1.7 bits per weight,
recovering 95% of the original model's quality.

## Open it

```bash
# from learning/ root
make additive-inference-notebook
```

The notebook ships **with outputs already embedded**, so you can read it end to end
without running anything. Re-run it to tweak the parameters.

## What it teaches

1. **Why the dot product is the only operation worth attacking** — traced by hand on
   four numbers.
2. **The trick**: restrict a weight to `{-1, 0, +1}` and `x × w` becomes
   *add / skip / subtract*. You build the multiply-free layer yourself and watch it
   execute with a single multiplication in the whole function.
3. **Where "1.7 bits" comes from** — `log2(3) = 1.585` plus amortised group scales
   lands on exactly 1.710, and 7.5 GB against FP16's 70 GB.
4. **Why smaller means faster** — decode is memory-bandwidth-bound. The same
   arithmetic shows the "120 tok/s" claim needs ~900 GB/s, which is not a typical
   consumer laptop.
5. **The trap**: the intuition that "errors cancel out over a long sum" is *wrong*,
   and the notebook measures it — relative error is flat at ~50% from n=4 to
   n=16,384, because error and signal both grow like √n.
6. **The actual answer**: a real (tiny) MLP is trained, ternarized until its error
   rate triples, then **retrained while ternarized** via a straight-through
   estimator — recovering ~100% of its original accuracy.

## The one-sentence version

Force every weight to −1/0/+1 so multiplication becomes addition and the model
shrinks 10×; this badly damages the network, so retrain it *in its damaged form*
until it learns to work anyway — and the real claim is that this retraining now costs
15 GPU-hours instead of a full pretraining run.

## Why that matters

Ternary weights are not new (BitNet, 2023/24). BitNet's problem was that it wanted
to be **pretrained from scratch** in ternary, which is why it never scaled. The
load-bearing claim in the announcement is post-training conversion in **under 15
GPU-hours** — the difference between "a way to build models" and "a pass you run on
anyone's existing checkpoint".

Deps: `numpy`, `matplotlib` (already in the shared `.venv`).
