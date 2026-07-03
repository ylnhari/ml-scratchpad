"""KV-cache from scratch — make GPT generation fast, and prove it stays correct.

Builds directly on ../nanogpt-from-scratch: we take a trained nanoGPT and add the
single most important inference optimization — the key/value cache.

The problem: naive autoregressive generation recomputes attention over the ENTIRE
prefix at every step. Generating token t redoes all the work for tokens 0..t-1.
That's O(n^2) total work to produce n tokens.

The fix: the K and V projections of past tokens never change. Cache them. Each new
step then computes Q/K/V for ONLY the new token and attends against the cached K/V.
Per-step work drops from O(t) to O(1) in sequence length; total work O(n^2) -> O(n).

A subtle, beautiful consequence: with a cache you never even build the causal mask.
You only ever hold the past in the cache, so "can't see the future" is automatic.

We reuse the *verified* engine from nanogpt.py (same weights, same math) and assert
the cached path produces identical logits to the naive path — that equivalence check
is to this file what the gradient check was to nanogpt.py.
"""
import os
import sys
import numpy as np

# reuse the verified GPT engine from the sibling subfolder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanogpt-from-scratch"))
from nanogpt import GPT, Adam, CharTokenizer, get_batch, softmax   # noqa: E402


def _layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    std = np.sqrt((xc * xc).mean(-1, keepdims=True) + eps)
    return xc / std * g + b


def _block_weights(block):
    """Pull the raw parameter arrays out of a nanogpt Block (read-only use)."""
    attn = block.sub["attn"]
    mlp = block.sub["mlp"]
    return {
        "ln1_g": block.sub["ln1"].p["g"], "ln1_b": block.sub["ln1"].p["b"],
        "Wq": attn.sub["q"].p["W"], "Wk": attn.sub["k"].p["W"],
        "Wv": attn.sub["v"].p["W"], "Wo": attn.sub["proj"].p["W"],
        "nh": attn.nh, "hs": attn.hs, "C": attn.C,
        "ln2_g": block.sub["ln2"].p["g"], "ln2_b": block.sub["ln2"].p["b"],
        "fcW": mlp.sub["fc"].p["W"], "fcb": mlp.sub["fc"].p["b"],
        "prW": mlp.sub["proj"].p["W"], "prb": mlp.sub["proj"].p["b"],
    }


class KVCache:
    """Per-layer store of past K and V vectors, grown one token at a time."""
    def __init__(self, n_layer):
        self.K = [None] * n_layer   # each: (nh, t, hs)
        self.V = [None] * n_layer

    def append(self, layer, k_heads, v_heads):
        if self.K[layer] is None:
            self.K[layer], self.V[layer] = k_heads[:, None, :], v_heads[:, None, :]
        else:
            self.K[layer] = np.concatenate([self.K[layer], k_heads[:, None, :]], axis=1)
            self.V[layer] = np.concatenate([self.V[layer], v_heads[:, None, :]], axis=1)
        return self.K[layer], self.V[layer]

    def length(self):
        return 0 if self.K[0] is None else self.K[0].shape[1]


def decode_step(model, token_id, cache):
    """Advance one token using the cache. Returns logits (vocab,) for the NEXT token.

    Position is inferred from how many tokens are already cached — exactly the index
    this new token occupies. No causal mask is needed: the cache only holds the past.
    """
    blocks = model.sub["blocks"]
    n_layer = len(blocks._order)
    t = cache.length()                                  # this token's position
    h = model.p["tok_emb"][token_id] + model.p["pos_emb"][t]     # (C,)

    for li in range(n_layer):
        w = _block_weights(blocks.sub[str(li)])
        nh, hs = w["nh"], w["hs"]

        a = _layernorm(h, w["ln1_g"], w["ln1_b"])
        q = (a @ w["Wq"]).reshape(nh, hs)
        k = (a @ w["Wk"]).reshape(nh, hs)
        v = (a @ w["Wv"]).reshape(nh, hs)

        Kc, Vc = cache.append(li, k, v)                 # (nh, t+1, hs)
        scores = np.einsum("hd,htd->ht", q, Kc) / np.sqrt(hs)   # (nh, t+1)
        att = softmax(scores, axis=-1)
        ctx = np.einsum("ht,htd->hd", att, Vc).reshape(-1)      # (C,)
        h = h + ctx @ w["Wo"]                           # residual

        b = _layernorm(h, w["ln2_g"], w["ln2_b"])
        hidden = np.maximum(0, b @ w["fcW"] + w["fcb"])
        h = h + hidden @ w["prW"] + w["prb"]            # residual

    h = _layernorm(h, model.sub["ln_f"].p["g"], model.sub["ln_f"].p["b"])
    return h @ model.sub["head"].p["W"]                 # (vocab,)


def generate_cached(model, prompt_ids, n_new, rng, temperature=1.0):
    """Prefill the prompt into the cache, then decode n_new tokens, O(1) attention/step."""
    cache = KVCache(len(model.sub["blocks"]._order))
    ids = list(prompt_ids)
    logits = None
    for tid in ids:                                     # prefill
        logits = decode_step(model, tid, cache)
    for _ in range(n_new):                              # decode
        probs = softmax(logits / temperature, axis=-1)
        nxt = rng.choice(len(probs), p=probs)
        ids.append(nxt)
        logits = decode_step(model, nxt, cache)
    return np.array(ids)


def generate_naive(model, prompt_ids, n_new, rng, temperature=1.0):
    """Baseline: recompute the whole prefix every step (what nanogpt.generate does)."""
    idx = np.array(prompt_ids)[None, :]
    for _ in range(n_new):
        cond = idx[:, -model.block_size:]
        logits, _ = model.forward(cond)
        probs = softmax(logits[:, -1, :] / temperature, axis=-1)[0]
        idx = np.concatenate([idx, [[rng.choice(len(probs), p=probs)]]], axis=1)
    return idx[0]


def verify_equivalence(model, ids):
    """Assert cached decoding gives the SAME logits as a full forward, at every position.

    This is the KV-cache analogue of nanogpt's gradient_check: proof the optimization
    changed the speed, not the answer. Returns the max absolute logit difference.
    """
    full_logits, _ = model.forward(np.array(ids)[None, :])   # (1, T, V)
    cache = KVCache(len(model.sub["blocks"]._order))
    worst = 0.0
    for t, tid in enumerate(ids):
        step_logits = decode_step(model, tid, cache)         # predicts token t+1
        worst = max(worst, np.abs(step_logits - full_logits[0, t]).max())
    return worst


def _train_tiny():
    """Train a small GPT so the demo has real (not random) weights. Used by __main__."""
    text = "to be or not to be that is the question. " * 40
    tok = CharTokenizer(text)
    data = tok.encode(text)
    rng = np.random.default_rng(0)
    # block_size roomy enough that the whole demo (prompt + generated tokens) fits the
    # learned positional embeddings, whose size fixes the max context.
    model = GPT(tok.vocab_size, block_size=96, n_embd=64, n_head=4, n_layer=2, seed=1337)
    opt = Adam(model, lr=5e-3)
    for _ in range(300):
        xb, yb = get_batch(data, 32, 32, rng)
        _, _ = model.forward(xb, yb)
        model.backward()
        opt.step()
    return model, tok


if __name__ == "__main__":
    import time
    model, tok = _train_tiny()

    ids = tok.encode("to be or not to be")
    err = verify_equivalence(model, ids)
    print(f"equivalence check (max |logit diff| cached vs full): {err:.2e}")
    print("PASS — cache changed speed, not the answer." if err < 1e-6 else "FAIL")

    # same seed -> identical text from both paths, but different cost
    prompt = tok.encode("to be")
    for name, fn in (("naive", generate_naive), ("cached", generate_cached)):
        t0 = time.perf_counter()
        out = fn(model, prompt, 80, np.random.default_rng(7), temperature=0.8)
        dt = time.perf_counter() - t0
        print(f"{name:6s} {dt*1000:7.1f} ms  {tok.decode(out)[:40]!r}")
