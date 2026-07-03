"""A tiny inference engine from scratch — paged KV-cache + speculative decoding.

The capstone of the inference path (../nanogpt-from-scratch -> ../kv-cache -> ../rope).
Two ideas power modern LLM serving (vLLM, TGI, TensorRT-LLM); we build both in NumPy and
prove each keeps the output identical:

  1. PAGED KV-CACHE (vLLM's core). ../kv-cache stored K/V in one contiguous array that grows
     every step — wasteful and fragmentation-prone when serving many sequences. Paging stores
     the cache in fixed-size *blocks* (like OS virtual memory): allocate a block when the
     current one fills. The decoder sees a normal cache; underneath it's paged. We reuse the
     verified decode_step from ../kv-cache and just swap the cache implementation, then check
     the logits are byte-for-byte the same.

  2. SPECULATIVE DECODING. A big model needs one forward per token — memory-bound and slow.
     A small "draft" model cheaply proposes k tokens; the big "target" model verifies all k in
     ONE forward. With greedy decoding we accept every proposed token that matches the target's
     argmax and stop at the first miss — which yields EXACTLY the target's greedy output, but
     with far fewer target forwards. Correctness is provable: spec output == pure target greedy.

Reuses the verified engine (nanogpt.py) and cache/decoder (kvcache.py).
"""
import os
import sys
import numpy as np

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "..", "nanogpt-from-scratch"))
sys.path.insert(0, os.path.join(_here, "..", "kv-cache"))
from nanogpt import GPT, Adam, CharTokenizer, get_batch, softmax   # noqa: E402
from kvcache import KVCache, decode_step                            # noqa: E402


# ── 1 · Paged KV-cache ────────────────────────────────────────────────────────

class PagedKVCache:
    """Drop-in for KVCache, but stores K/V in fixed-size pages instead of one array.

    Same interface (`append` -> returns the full (nh, t+1, hs) view, `length`), so the
    verified `decode_step` from ../kv-cache works unchanged. Underneath, memory is handed
    out one fixed block at a time — no giant reallocation as the sequence grows, and blocks
    from finished sequences can be recycled. That block-table indirection is exactly what
    lets vLLM pack many sequences into GPU memory without fragmentation.
    """
    def __init__(self, n_layer, page_size=8):
        self.page_size = page_size
        self.pagesK = [[] for _ in range(n_layer)]   # per layer: list of (nh, page_size, hs)
        self.pagesV = [[] for _ in range(n_layer)]
        self.n = [0] * n_layer                        # tokens stored per layer

    def append(self, layer, k_heads, v_heads):
        pos = self.n[layer]
        if pos % self.page_size == 0:                 # current page full -> allocate a new block
            nh, hs = k_heads.shape
            self.pagesK[layer].append(np.zeros((nh, self.page_size, hs)))
            self.pagesV[layer].append(np.zeros((nh, self.page_size, hs)))
        slot = pos % self.page_size
        self.pagesK[layer][-1][:, slot, :] = k_heads
        self.pagesV[layer][-1][:, slot, :] = v_heads
        self.n[layer] = pos + 1
        # reassemble the logical (nh, t+1, hs) view the attention step expects
        Kc = np.concatenate(self.pagesK[layer], axis=1)[:, :self.n[layer], :]
        Vc = np.concatenate(self.pagesV[layer], axis=1)[:, :self.n[layer], :]
        return Kc, Vc

    def length(self):
        return self.n[0]

    def n_blocks(self):
        return sum(len(p) for p in self.pagesK)


def verify_paging(model, ids, page_size=8):
    """Assert paged and contiguous caches produce identical logits at every position."""
    n_layer = len(model.sub["blocks"]._order)
    contig, paged = KVCache(n_layer), PagedKVCache(n_layer, page_size)
    worst = 0.0
    for tid in ids:
        lc = decode_step(model, tid, contig)
        lp = decode_step(model, tid, paged)
        worst = max(worst, np.abs(lc - lp).max())
    return worst


# ── 2 · Speculative decoding (greedy) ────────────────────────────────────────

def greedy_target(model, prompt_ids, n_new):
    """Reference: pure greedy decoding with the target model (one forward per token)."""
    ids = list(prompt_ids)
    forwards = 0
    for _ in range(n_new):
        cond = np.array(ids[-model.block_size:])[None, :]
        logits, _ = model.forward(cond); forwards += 1
        ids.append(int(logits[0, -1].argmax()))
    return np.array(ids), forwards


def _greedy_propose(draft, ids, k):
    """Draft cheaply proposes k next tokens greedily."""
    out = []
    cur = list(ids)
    for _ in range(k):
        cond = np.array(cur[-draft.block_size:])[None, :]
        logits, _ = draft.forward(cond)
        nxt = int(logits[0, -1].argmax())
        out.append(nxt); cur.append(nxt)
    return out


def speculative_greedy(target, draft, prompt_ids, n_new, k=4):
    """Draft proposes k, target verifies all k in ONE forward, accept matching prefix.

    Returns (ids, target_forwards). Output is identical to greedy_target; target_forwards
    is far below n_new when the draft agrees often — that's the whole speedup.
    """
    ids = list(prompt_ids)
    target_forwards = 0
    while len(ids) - len(prompt_ids) < n_new:
        proposal = _greedy_propose(draft, ids, k)
        # ONE target forward over context + all proposed tokens verifies every position
        cond = np.array((ids + proposal)[-target.block_size:])[None, :]
        logits, _ = target.forward(cond); target_forwards += 1
        # target's argmax for the position after each of [last ctx token, proposal[0..k-2]]
        start = cond.shape[1] - len(proposal) - 1
        target_next = [int(logits[0, start + j].argmax()) for j in range(len(proposal) + 1)]

        accepted = 0
        for j in range(len(proposal)):
            if proposal[j] == target_next[j]:
                ids.append(proposal[j]); accepted += 1
                if len(ids) - len(prompt_ids) >= n_new:
                    break
            else:
                break
        # on a mismatch (or all accepted) we also get one free correct token from the target
        if accepted < len(proposal) and len(ids) - len(prompt_ids) < n_new:
            ids.append(target_next[accepted])
    return np.array(ids[:len(prompt_ids) + n_new]), target_forwards


def verify_speculative(target, draft, prompt_ids, n_new, k=4):
    """Prove speculative greedy output == pure target greedy output."""
    ref, _ = greedy_target(target, prompt_ids, n_new)
    spec, tf = speculative_greedy(target, draft, prompt_ids, n_new, k)
    return np.array_equal(ref, spec), tf


def _train_pair():
    """Train a good target (2-layer) and a cheaper draft (1-layer, fewer steps)."""
    text = "to be or not to be that is the question. " * 40
    tok = CharTokenizer(text)
    data = tok.encode(text)
    rng = np.random.default_rng(0)

    def train(n_layer, steps):
        m = GPT(tok.vocab_size, block_size=64, n_embd=64, n_head=4, n_layer=n_layer, seed=1337)
        opt = Adam(m, lr=5e-3)
        for _ in range(steps):
            xb, yb = get_batch(data, 32, 32, rng)
            m.forward(xb, yb); m.backward(); opt.step()
        return m

    return train(2, 400), train(1, 150), tok   # target, draft, tokenizer


if __name__ == "__main__":
    target, draft, tok = _train_pair()
    ids = tok.encode("to be or not to be")

    err = verify_paging(target, ids)
    print(f"paged-vs-contiguous cache (max |logit diff|): {err:.2e}",
          "PASS" if err < 1e-6 else "FAIL")

    prompt = tok.encode("to be")
    ok, tf = verify_speculative(target, draft, prompt, 60, k=4)
    _, greedy_forwards = greedy_target(target, prompt, 60)
    print(f"speculative output == target greedy: {ok}")
    print(f"target forwards: greedy={greedy_forwards}  speculative={tf} "
          f"({greedy_forwards / tf:.2f}x fewer)")
