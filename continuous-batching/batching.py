"""Continuous batching from scratch — keep the GPU busy when requests vary in length.

The last step of the inference path (../nanogpt-from-scratch -> ../kv-cache -> ../rope ->
../inference-engine -> here). It answers: given many generation requests of DIFFERENT
lengths arriving over time, how do you schedule them so the batch stays full?

STATIC batching (the naive way): grab B requests, run them together until the LONGEST one
finishes, then load the next B. Short requests finish early and their slots sit idle while
the batch waits — wasted compute, and new requests wait for the whole batch to drain.

CONTINUOUS batching (vLLM/TGI): the batch is a rolling set of B slots. The moment any
sequence finishes, it leaves and a waiting request is admitted into that slot immediately.
The batch stays full, utilization approaches 100%, and latency drops.

The scheduling changes; the math per sequence does not. So each request's output must be
identical to generating it alone — that equivalence is this file's correctness proof.

Reuses the verified model (nanogpt.py) and cache/decoder (kvcache.py).
"""
import os
import sys
from dataclasses import dataclass, field
import numpy as np

_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, "..", "nanogpt-from-scratch"))
sys.path.insert(0, os.path.join(_here, "..", "kv-cache"))
from nanogpt import GPT, Adam, CharTokenizer, get_batch   # noqa: E402
from kvcache import KVCache, decode_step                   # noqa: E402


@dataclass
class Request:
    rid: int
    prompt_ids: list
    gen_len: int                       # how many tokens this request wants
    out: list = field(default_factory=list)


class _Seq:
    """A live sequence occupying a batch slot: its own KV-cache and next-token logits."""
    def __init__(self, model, req):
        self.req = req
        self.cache = KVCache(len(model.sub["blocks"]._order))
        self.logits = None
        for t in req.prompt_ids:                    # prefill
            self.logits = decode_step(model, t, self.cache)
        self.left = req.gen_len

    def step(self, model):
        """Emit one greedy token. Returns True while the request still wants more."""
        nxt = int(self.logits.argmax())
        self.req.out.append(nxt)
        self.left -= 1
        if self.left > 0:
            self.logits = decode_step(model, nxt, self.cache)
            return True
        return False


def standalone_greedy(model, req):
    """Ground truth: generate this one request by itself, greedily."""
    seq = _Seq(model, Request(req.rid, req.prompt_ids, req.gen_len))
    while seq.step(model):
        pass
    return seq.req.out


def run_static(model, requests, B):
    """Static batching: fixed groups of B, each run until its LONGEST member finishes.

    Returns (outputs, slot_steps, useful_steps). slot_steps counts every occupied batch slot
    per tick (incl. idle ones waiting for the longest); useful_steps counts real decode work.
    """
    outputs, slot_steps, useful_steps = {}, 0, 0
    for i in range(0, len(requests), B):
        group = [Request(r.rid, r.prompt_ids, r.gen_len) for r in requests[i:i + B]]
        seqs = [_Seq(model, r) for r in group]
        active = list(seqs)
        ticks = max(s.left for s in seqs)           # batch runs until the longest is done
        for _ in range(ticks):
            for s in seqs:
                slot_steps += 1                     # slot occupied whether or not still needed
                if s in active:
                    useful_steps += 1
                    if not s.step(model):
                        active.remove(s)            # finished, but the slot stays reserved
        for r in group:
            outputs[r.rid] = r.out
    return outputs, slot_steps, useful_steps


def run_continuous(model, requests, B):
    """Continuous batching: a rolling set of B slots; finished sequences are replaced at once.

    Returns (outputs, slot_steps, useful_steps). A finished slot is refilled from the queue in
    the same tick, so slots rarely sit idle.
    """
    outputs, slot_steps, useful_steps = {}, 0, 0
    queue = [Request(r.rid, r.prompt_ids, r.gen_len) for r in requests]
    waiting = iter(queue)
    slots = []
    for _ in range(B):                              # admit the first B
        r = next(waiting, None)
        if r is not None:
            slots.append(_Seq(model, r))

    while slots:
        for i, s in enumerate(slots):
            slot_steps += 1
            useful_steps += 1
            if not s.step(model):                   # finished this tick
                outputs[s.req.rid] = s.req.out
                r = next(waiting, None)
                slots[i] = _Seq(model, r) if r is not None else None   # admit next / vacate
        slots = [s for s in slots if s is not None]
    return outputs, slot_steps, useful_steps


def verify_equivalence(model, requests, B):
    """Prove both schedulers produce the SAME tokens as generating each request alone."""
    ref = {r.rid: standalone_greedy(model, r) for r in requests}
    stat, _, _ = run_static(model, requests, B)
    cont, _, _ = run_continuous(model, requests, B)
    ok_static = all(ref[k] == stat[k] for k in ref)
    ok_cont = all(ref[k] == cont[k] for k in ref)
    return ok_static and ok_cont


def occupancy_static(gen_lens, B):
    """Slot-occupancy grid (ticks x B) for static batching; -1 = idle slot. Uses only lengths.

    Faithful to run_static's schedule (a group runs until its longest member finishes), but
    needs no model — occupancy depends only on the generation lengths, B, and the policy.
    """
    rows = []
    for i in range(0, len(gen_lens), B):
        group = list(enumerate(gen_lens))[i:i + B]     # (rid, length)
        ticks = max(l for _, l in group)
        for t in range(ticks):
            row = [-1] * B
            for slot, (rid, l) in enumerate(group):
                row[slot] = rid if t < l else -1        # idle once this request is done
            rows.append(row)
    return np.array(rows)


def occupancy_continuous(gen_lens, B):
    """Slot-occupancy grid (ticks x B) for continuous batching; -1 = idle (only at the drain)."""
    remaining = {rid: l for rid, l in enumerate(gen_lens)}
    waiting = list(range(len(gen_lens)))
    slots = [waiting.pop(0) if waiting else -1 for _ in range(B)]
    rows = []
    while any(s != -1 for s in slots):
        rows.append(list(slots))
        for i, rid in enumerate(slots):
            if rid == -1:
                continue
            remaining[rid] -= 1
            if remaining[rid] == 0:                     # finished -> admit next immediately
                slots[i] = waiting.pop(0) if waiting else -1
    return np.array(rows)


def make_requests(tok, n=12, seed=0):
    """A workload with deliberately VARIED generation lengths (where scheduling matters)."""
    rng = np.random.default_rng(seed)
    reqs = []
    for i in range(n):
        prompt = list(tok.encode("to be"))
        reqs.append(Request(i, prompt, int(rng.integers(8, 48))))
    return reqs


def _train_tiny():
    text = "to be or not to be that is the question. " * 40
    tok = CharTokenizer(text)
    data = tok.encode(text)
    rng = np.random.default_rng(0)
    model = GPT(tok.vocab_size, block_size=96, n_embd=64, n_head=4, n_layer=2, seed=1337)
    opt = Adam(model, lr=5e-3)
    for _ in range(300):
        xb, yb = get_batch(data, 32, 32, rng)
        model.forward(xb, yb); model.backward(); opt.step()
    return model, tok


if __name__ == "__main__":
    model, tok = _train_tiny()
    reqs = make_requests(tok, n=12)

    print("outputs identical to standalone greedy:", verify_equivalence(model, reqs, B=4))

    _, s_slots, s_useful = run_static(model, reqs, B=4)
    _, c_slots, c_useful = run_continuous(model, reqs, B=4)
    print(f"static     : {s_useful} useful / {s_slots} slot-steps = {s_useful/s_slots:5.1%} utilization")
    print(f"continuous : {c_useful} useful / {c_slots} slot-steps = {c_useful/c_slots:5.1%} utilization")
    print(f"throughput gain: {(s_slots/c_slots):.2f}x fewer slot-steps for the same work")
