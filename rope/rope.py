"""RoPE from scratch — rotary positional embeddings, and why they beat learned positions.

The context (from ../nanogpt-from-scratch and ../kv-cache): our GPT used *learned* positional
embeddings — a lookup table with one row per position. Two problems:
  1. It caps context at the table size (`block_size`); you can't run longer than you trained.
  2. It encodes *absolute* position, but language mostly cares about *relative* position
     ("the word two tokens back"), which the model must then learn indirectly.

RoPE (Rotary Position Embedding, Su et al. 2021 — used by Llama, Mistral, Qwen, …) fixes both.
The idea is elegant: instead of *adding* a position vector, **rotate** each query and key by an
angle proportional to its position. Because rotating by θ_i then measuring against a vector
rotated by θ_j leaves a dependence only on (i − j), the attention score q_i·k_j becomes a pure
function of the **relative** distance — and it's defined by a formula, so it works at any length.

This file implements RoPE in NumPy and *proves* the relative-position property numerically —
that proof is to RoPE what the gradient check was to nanogpt and the equivalence check to kv-cache.
"""
import numpy as np


def rope_tables(positions, dim, base=10000.0):
    """cos/sin tables for the given positions. Each adjacent pair of dims shares one angle.

    positions: (T,) integer or float positions (can exceed any training length).
    Returns cos, sin each (T, dim/2).
    """
    assert dim % 2 == 0, "RoPE needs an even head dimension"
    half = dim // 2
    m = np.arange(half)
    inv_freq = base ** (-2.0 * m / dim)              # (half,) — high dims rotate slowly
    ang = positions[:, None] * inv_freq[None, :]      # (T, half)
    return np.cos(ang), np.sin(ang)


def apply_rope(x, positions, base=10000.0):
    """Rotate each 2D pair (x[2m], x[2m+1]) of vector x by position*inv_freq[m].

    x: (..., T, dim). Returns the rotated array, same shape.
    """
    dim = x.shape[-1]
    cos, sin = rope_tables(positions, dim, base)       # (T, dim/2)
    x_even = x[..., 0::2]                               # (..., T, dim/2)
    x_odd = x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin         # standard 2D rotation
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def verify_relative_property(dim=16, trials=200, seed=0):
    """Prove <RoPE(q, i), RoPE(k, j)> depends ONLY on the offset (j - i).

    For a fixed offset we slide the pair (i, j=i+offset) along many absolute positions;
    the dot product must stay constant. Returns the worst deviation across offsets.
    """
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim)
    k = rng.standard_normal(dim)
    worst = 0.0
    for offset in range(0, 8):
        vals = []
        for i in range(trials):
            pos = np.array([i, i + offset], dtype=float)
            qk = apply_rope(np.stack([q, k]), pos)       # rotate q at i, k at i+offset
            vals.append(qk[0] @ qk[1])
        vals = np.array(vals)
        worst = max(worst, vals.max() - vals.min())      # spread at this offset (should be ~0)
    return worst


def score_vs_distance(dim=16, max_dist=64, seed=0):
    """How the RoPE attention score between two RANDOM vectors varies with their distance.

    Not a learned pattern — just shows RoPE injects a smooth, relative-position-dependent
    structure into the raw q·k score, and does so at distances far beyond any table size.
    """
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim)
    k = rng.standard_normal(dim)
    dists = np.arange(max_dist)
    scores = []
    for d in dists:
        qk = apply_rope(np.stack([q, k]), np.array([0.0, float(d)]))
        scores.append(qk[0] @ qk[1])
    return dists, np.array(scores)


if __name__ == "__main__":
    worst = verify_relative_property()
    print(f"relative-position check (max spread at fixed offset): {worst:.2e}")
    print("PASS — score depends only on (j - i)." if worst < 1e-9 else "FAIL")

    # RoPE is defined by a formula, so positions beyond any 'block_size' just work:
    x = np.random.default_rng(1).standard_normal((3, 16))
    _ = apply_rope(x, np.array([0.0, 100.0, 100000.0]))   # position 100000 — no table needed
    print("applied RoPE at position 100000 with no lookup table — OK")
