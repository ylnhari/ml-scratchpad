"""nanoGPT from scratch — a tiny char-level GPT you can train in pure NumPy.

Where this sits relative to ../attention-transformer:
  attention.py shows the transformer FORWARD pass — the mechanism.
  This file adds the two things that turn a mechanism into a model that LEARNS:
    1. A full training-capable GPT (token+position embeddings → blocks → head).
    2. Manual BACKPROPAGATION through every layer, so you can watch loss fall
       and generation become coherent — no autograd, every gradient visible.

Read top-to-bottom before opening the notebook. Focus on the shape comments and
on how each layer's `backward` is the mirror image of its `forward`.

Design choices (clarity > performance, like the rest of learning/):
  - Pure NumPy. Every gradient is written out by hand.
  - Modern GPT layout: pre-LayerNorm blocks, learned positional embeddings.
  - ReLU in the MLP (exact, pure-NumPy) where real GPT-2 uses GELU — noted in the
    notebook. Everything else matches a real decoder-only transformer.
  - Batched shapes (B, T, C) throughout, exactly as real training uses.

Every layer is a tiny Module with `forward` / `backward` and its own parameter
grads, so each piece is independently gradient-checkable (see `gradient_check`).
"""

import numpy as np


# ── Primitives ────────────────────────────────────────────────────────────────

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)      # subtract max for numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class Module:
    """Base class: holds learnable params `p` and their grads `g` (same keys)."""
    def __init__(self):
        self.p = {}      # name -> ndarray (parameter)
        self.g = {}      # name -> ndarray (gradient, filled by backward)
        self.sub = {}    # name -> child Module

    def params(self):
        """Yield (unique_key, module, param_name) for every learnable array in the tree."""
        for name in self.p:
            yield (id(self), name), self, name
        for child in self.sub.values():
            yield from child.params()


# ── Linear: y = x W + b ─────────────────────────────────────────────────────────

class Linear(Module):
    def __init__(self, n_in, n_out, rng, bias=True, scale=0.02):
        super().__init__()
        self.p["W"] = rng.standard_normal((n_in, n_out)) * scale
        self.bias = bias
        if bias:
            self.p["b"] = np.zeros(n_out)

    def forward(self, x):
        self._x = x                                   # cache input for backward
        y = x @ self.p["W"]
        if self.bias:
            y = y + self.p["b"]
        return y

    def backward(self, dy):
        xr = self._x.reshape(-1, self._x.shape[-1])   # (N, n_in)
        dyr = dy.reshape(-1, dy.shape[-1])            # (N, n_out)
        self.g["W"] = xr.T @ dyr                      # dL/dW = xᵀ dy
        if self.bias:
            self.g["b"] = dyr.sum(0)                  # dL/db = sum over rows
        return dy @ self.p["W"].T                     # dL/dx = dy Wᵀ


# ── LayerNorm: normalise each token vector, then scale+shift ─────────────────────

class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.p["g"] = np.ones(dim)                    # gamma (scale)
        self.p["b"] = np.zeros(dim)                   # beta  (shift)
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        self._xn = xc / std                           # normalised (cache)
        self._std = std
        return self._xn * self.p["g"] + self.p["b"]

    def backward(self, dy):
        xn, std, g = self._xn, self._std, self.p["g"]
        axes = tuple(range(dy.ndim - 1))
        self.g["g"] = (dy * xn).sum(axes)
        self.g["b"] = dy.sum(axes)
        dxn = dy * g
        C = xn.shape[-1]
        # standard LayerNorm input gradient
        dx = (dxn - dxn.mean(-1, keepdims=True)
              - xn * (dxn * xn).mean(-1, keepdims=True)) / std
        return dx


# ── ReLU ─────────────────────────────────────────────────────────────────────

class ReLU(Module):
    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dy):
        return dy * self._mask


# ── Causal (masked) multi-head self-attention ────────────────────────────────

class CausalSelfAttention(Module):
    """Each position attends only to itself and earlier positions (the causal mask).

    That mask is the whole reason a decoder can be trained on every position at
    once yet still only "see the past" when predicting the next token.
    """
    def __init__(self, C, n_head, rng):
        super().__init__()
        assert C % n_head == 0
        self.C, self.nh, self.hs = C, n_head, C // n_head
        self.sub["q"] = Linear(C, C, rng, bias=False)
        self.sub["k"] = Linear(C, C, rng, bias=False)
        self.sub["v"] = Linear(C, C, rng, bias=False)
        self.sub["proj"] = Linear(C, C, rng, bias=False)

    def _split(self, t, B, T):
        return t.reshape(B, T, self.nh, self.hs).transpose(0, 2, 1, 3)   # (B,nh,T,hs)

    def _merge(self, t, B, T):
        return t.transpose(0, 2, 1, 3).reshape(B, T, self.C)

    def forward(self, x):
        B, T, C = x.shape
        self._BT = (B, T)
        q = self._split(self.sub["q"].forward(x), B, T)
        k = self._split(self.sub["k"].forward(x), B, T)
        v = self._split(self.sub["v"].forward(x), B, T)

        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.hs)         # (B,nh,T,T)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)                  # True above diagonal = future
        scores = np.where(mask, -1e9, scores)
        att = softmax(scores, axis=-1)
        y = att @ v                                                       # (B,nh,T,hs)

        self._cache = (q, k, v, att, mask)
        ymerged = self._merge(y, B, T)
        self._ymerged = ymerged
        return self.sub["proj"].forward(ymerged)

    def backward(self, dout):
        B, T = self._BT
        q, k, v, att, mask = self._cache
        dymerged = self.sub["proj"].backward(dout)                       # (B,T,C)
        dy = self._split(dymerged, B, T)                                 # (B,nh,T,hs)

        # y = att @ v
        datt = dy @ v.transpose(0, 1, 3, 2)                              # (B,nh,T,T)
        dv = att.transpose(0, 1, 3, 2) @ dy                             # (B,nh,T,hs)

        # softmax backward (per row), then kill masked (future) entries
        dscores = att * (datt - (att * datt).sum(-1, keepdims=True))
        dscores = np.where(mask, 0.0, dscores)
        dscores /= np.sqrt(self.hs)

        # scores = q @ kᵀ
        dq = dscores @ k                                                 # (B,nh,T,hs)
        dk = dscores.transpose(0, 1, 3, 2) @ q                          # (B,nh,T,hs)

        dxq = self.sub["q"].backward(self._merge(dq, B, T))
        dxk = self.sub["k"].backward(self._merge(dk, B, T))
        dxv = self.sub["v"].backward(self._merge(dv, B, T))
        return dxq + dxk + dxv


# ── MLP (position-wise feed-forward) ─────────────────────────────────────────

class MLP(Module):
    def __init__(self, C, mult, rng):
        super().__init__()
        self.sub["fc"] = Linear(C, mult * C, rng)
        self.sub["act"] = ReLU()
        self.sub["proj"] = Linear(mult * C, C, rng)

    def forward(self, x):
        return self.sub["proj"].forward(self.sub["act"].forward(self.sub["fc"].forward(x)))

    def backward(self, dy):
        return self.sub["fc"].backward(self.sub["act"].backward(self.sub["proj"].backward(dy)))


# ── Transformer block: pre-norm residuals ────────────────────────────────────

class Block(Module):
    """x = x + attn(ln1(x));  x = x + mlp(ln2(x)).

    The residual '+' is what lets gradients reach early layers — remove it and a
    deep stack won't train. Pre-norm (LN inside the residual) is the modern GPT
    choice; it makes training stable without learning-rate warmup tricks.
    """
    def __init__(self, C, n_head, mult, rng):
        super().__init__()
        self.sub["ln1"] = LayerNorm(C)
        self.sub["attn"] = CausalSelfAttention(C, n_head, rng)
        self.sub["ln2"] = LayerNorm(C)
        self.sub["mlp"] = MLP(C, mult, rng)

    def forward(self, x):
        x = x + self.sub["attn"].forward(self.sub["ln1"].forward(x))
        x = x + self.sub["mlp"].forward(self.sub["ln2"].forward(x))
        return x

    def backward(self, dx):
        # mirror the forward residuals in reverse
        dx = dx + self.sub["ln2"].backward(self.sub["mlp"].backward(dx))
        dx = dx + self.sub["ln1"].backward(self.sub["attn"].backward(dx))
        return dx


# ── The GPT ──────────────────────────────────────────────────────────────────

class GPT(Module):
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4, n_layer=2,
                 mlp_mult=4, seed=1337):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.block_size = block_size
        self.p["tok_emb"] = rng.standard_normal((vocab_size, n_embd)) * 0.02
        self.p["pos_emb"] = rng.standard_normal((block_size, n_embd)) * 0.02
        self.sub["blocks"] = _Sequential([Block(n_embd, n_head, mlp_mult, rng)
                                          for _ in range(n_layer)])
        self.sub["ln_f"] = LayerNorm(n_embd)
        self.sub["head"] = Linear(n_embd, vocab_size, rng, bias=False)

    def forward(self, idx, targets=None):
        """idx: (B,T) int tokens. Returns (logits, loss). loss is None if no targets."""
        B, T = idx.shape
        self._idx = idx
        x = self.p["tok_emb"][idx] + self.p["pos_emb"][:T]               # (B,T,C)
        x = self.sub["blocks"].forward(x)
        x = self.sub["ln_f"].forward(x)
        logits = self.sub["head"].forward(x)                            # (B,T,V)
        if targets is None:
            return logits, None
        # softmax cross-entropy over the vocab
        probs = softmax(logits, axis=-1)
        self._probs = probs
        self._targets = targets
        Bt = B * T
        pr = probs.reshape(Bt, -1)
        loss = -np.log(pr[np.arange(Bt), targets.reshape(-1)] + 1e-12).mean()
        return logits, loss

    def backward(self):
        B, T = self._idx.shape
        V = self._probs.shape[-1]
        # dL/dlogits for softmax-CE = (probs - onehot) / N
        dlogits = self._probs.copy()
        Bt = B * T
        dl = dlogits.reshape(Bt, V)
        dl[np.arange(Bt), self._targets.reshape(-1)] -= 1.0
        dl /= Bt
        dlogits = dl.reshape(B, T, V)

        dx = self.sub["head"].backward(dlogits)
        dx = self.sub["ln_f"].backward(dx)
        dx = self.sub["blocks"].backward(dx)                            # (B,T,C)

        # embeddings: scatter gradient back to the rows that were looked up
        self.g["tok_emb"] = np.zeros_like(self.p["tok_emb"])
        np.add.at(self.g["tok_emb"], self._idx, dx)
        self.g["pos_emb"] = np.zeros_like(self.p["pos_emb"])
        self.g["pos_emb"][:T] = dx.sum(axis=0)

    def generate(self, idx, n_new, rng, temperature=1.0, top_k=None):
        """Autoregressively sample n_new tokens. idx: (1,T0) seed."""
        for _ in range(n_new):
            idx_cond = idx[:, -self.block_size:]                        # crop to context
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature                    # last position
            if top_k is not None:
                kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = np.where(logits < kth, -np.inf, logits)
            probs = softmax(logits, axis=-1)[0]
            nxt = rng.choice(len(probs), p=probs)
            idx = np.concatenate([idx, np.array([[nxt]])], axis=1)
        return idx


class _Sequential(Module):
    """A list of blocks applied in order (forward) / reverse (backward)."""
    def __init__(self, blocks):
        super().__init__()
        for i, b in enumerate(blocks):
            self.sub[str(i)] = b
        self._order = [str(i) for i in range(len(blocks))]

    def forward(self, x):
        for name in self._order:
            x = self.sub[name].forward(x)
        return x

    def backward(self, dx):
        for name in reversed(self._order):
            dx = self.sub[name].backward(dx)
        return dx


# ── Adam optimizer ───────────────────────────────────────────────────────────

class Adam:
    def __init__(self, model, lr=1e-2, betas=(0.9, 0.99), eps=1e-8):
        self.model, self.lr, self.b1, self.b2, self.eps = model, lr, *betas, eps
        self.t = 0
        self.m, self.v = {}, {}
        for key, mod, name in model.params():
            self.m[key] = np.zeros_like(mod.p[name])
            self.v[key] = np.zeros_like(mod.p[name])

    def step(self):
        self.t += 1
        for key, mod, name in self.model.params():
            grad = mod.g[name]
            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * grad
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * (grad * grad)
            mhat = self.m[key] / (1 - self.b1 ** self.t)
            vhat = self.v[key] / (1 - self.b2 ** self.t)
            mod.p[name] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ── Char tokenizer + batching ────────────────────────────────────────────────

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def get_batch(data, block_size, batch_size, rng):
    """Sample batch_size random (x, y) windows. y is x shifted by one (next char)."""
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


# ── Gradient check: proof the manual backprop is correct ─────────────────────

def gradient_check(seed=0, eps=1e-5):
    """Compare analytic grads to numerical ones on a tiny model. Returns max rel err.

    This is how you TRUST hand-written backprop: perturb each parameter by ±eps,
    measure the loss change, and confirm it matches the analytic gradient.
    """
    rng = np.random.default_rng(seed)
    V, T, B = 7, 4, 2
    model = GPT(V, T, n_embd=8, n_head=2, n_layer=2, mlp_mult=2, seed=seed)
    idx = rng.integers(0, V, size=(B, T))
    targets = rng.integers(0, V, size=(B, T))

    _, loss = model.forward(idx, targets)
    model.backward()

    worst = 0.0
    for key, mod, name in model.params():
        P = mod.p[name]
        flat = P.ravel()
        gflat = mod.g[name].ravel()
        for _ in range(min(6, flat.size)):           # spot-check a few entries
            i = rng.integers(0, flat.size)
            orig = flat[i]
            flat[i] = orig + eps
            _, lp = model.forward(idx, targets)
            flat[i] = orig - eps
            _, lm = model.forward(idx, targets)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = gflat[i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
    return worst


# ── Self-test / tiny demo ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("gradient check (max relative error):", f"{gradient_check():.2e}")

    text = "to be or not to be that is the question. " * 20
    tok = CharTokenizer(text)
    data = tok.encode(text)
    rng = np.random.default_rng(0)

    model = GPT(tok.vocab_size, block_size=16, n_embd=64, n_head=4, n_layer=2, seed=1337)
    opt = Adam(model, lr=5e-3)

    for step in range(400):
        xb, yb = get_batch(data, 16, 32, rng)
        _, loss = model.forward(xb, yb)
        model.backward()
        opt.step()
        if step % 100 == 0 or step == 399:
            print(f"step {step:4d}  loss {loss:.4f}")

    seed_ids = tok.encode("to be")[None, :]
    out = model.generate(seed_ids, 60, rng, temperature=0.8)
    print("sample:", repr(tok.decode(out[0])))
