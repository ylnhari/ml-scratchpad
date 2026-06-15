"""Attention Is All You Need — minimal, readable implementation.

Paper: https://arxiv.org/abs/1706.03762

Read top-to-bottom before opening the notebook.
Focus on what each function's inputs/outputs are, not every line.
"""

import numpy as np


# ── Primitives ────────────────────────────────────────────────────────────────

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)   # subtract max for numerical stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):
    """Normalise each token's embedding to zero mean, unit variance."""
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


# ── Core attention operation ──────────────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V, mask=None):
    """The fundamental operation of the transformer.

    Every query attends to every key. The result is a weighted sum of values.

    Q : (..., seq_q, d_k)
    K : (..., seq_k, d_k)
    V : (..., seq_k, d_v)
    Returns (output, weights) — output: (..., seq_q, d_v), weights: (..., seq_q, seq_k)

    Why divide by sqrt(d_k)?
    Dot products grow with dimension. Without scaling, softmax saturates
    (one near-1, rest near-0) → gradients vanish during training.
    """
    d_k = Q.shape[-1]
    scores  = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)   # (..., seq_q, seq_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)           # masked positions get -inf
    weights = softmax(scores, axis=-1)                  # (..., seq_q, seq_k)
    output  = weights @ V                               # (..., seq_q, d_v)
    return output, weights


# ── Multi-Head Attention ──────────────────────────────────────────────────────

class MultiHeadAttention:
    """Split d_model into n_heads parallel attention heads, run each independently.

    Why multiple heads?
    One head can only learn one type of relationship at a time (e.g. syntactic).
    Multiple heads learn multiple relationship types simultaneously.
    Their outputs are concatenated and projected back to d_model.
    """

    def __init__(self, d_model, n_heads, seed=42):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads   # dimension per head

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2 / d_model)
        self.W_Q = rng.standard_normal((d_model, d_model)) * scale
        self.W_K = rng.standard_normal((d_model, d_model)) * scale
        self.W_V = rng.standard_normal((d_model, d_model)) * scale
        self.W_O = rng.standard_normal((d_model, d_model)) * scale   # output projection

    def forward(self, X, mask=None):
        """X: (seq_len, d_model) → output: (seq_len, d_model), weights: (n_heads, seq_len, seq_len)"""
        seq_len = X.shape[0]

        # Project input to Q, K, V
        Q = X @ self.W_Q   # (seq_len, d_model)
        K = X @ self.W_K
        V = X @ self.W_V

        # Split into heads → (n_heads, seq_len, d_k)
        def split_heads(t):
            return t.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Parallel attention across all heads
        attn_out, weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out: (n_heads, seq_len, d_k)

        # Concatenate heads and project
        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        output   = attn_out @ self.W_O

        return output, weights   # weights shape: (n_heads, seq_len, seq_len)


# ── Positional Encoding ───────────────────────────────────────────────────────

def positional_encoding(seq_len, d_model):
    """Inject position information using sine/cosine waves at different frequencies.

    Without this, the model sees a bag of words — order doesn't matter.
    With this, position i and i+1 have similar encodings, i and i+100 differ more.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """
    pos    = np.arange(seq_len)[:, None]           # (seq_len, 1)
    i      = np.arange(d_model)[None, :]           # (1, d_model)
    angles = pos / 10000 ** (2 * (i // 2) / d_model)
    pe     = np.where(i % 2 == 0, np.sin(angles), np.cos(angles))
    return pe   # (seq_len, d_model)


# ── Feed-Forward Network ──────────────────────────────────────────────────────

class FeedForward:
    """Position-wise FFN: applied identically to each token.

    FFN(x) = max(0, x W1 + b1) W2 + b2

    d_ff is typically 4 × d_model. This is where most parameters live.
    Attention finds relationships; FFN processes each token's representation.
    """

    def __init__(self, d_model, d_ff, seed=42):
        rng    = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((d_model, d_ff)) * np.sqrt(2 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.standard_normal((d_ff, d_model)) * np.sqrt(2 / d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        return np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2


# ── Encoder Block ─────────────────────────────────────────────────────────────

class EncoderBlock:
    """One transformer encoder layer.

    Flow: X → [Multi-Head Self-Attention] → Add&Norm → [FFN] → Add&Norm → output

    The residual connections (Add) ensure gradients flow during training
    and let the block learn incremental refinements rather than full transforms.
    """

    def __init__(self, d_model, n_heads, d_ff, seed=42):
        self.attn = MultiHeadAttention(d_model, n_heads, seed)
        self.ff   = FeedForward(d_model, d_ff, seed + 1)

    def forward(self, X, mask=None):
        attn_out, weights = self.attn.forward(X, mask)
        X = layer_norm(X + attn_out)          # residual + norm
        ff_out = self.ff.forward(X)
        X = layer_norm(X + ff_out)            # residual + norm
        return X, weights
