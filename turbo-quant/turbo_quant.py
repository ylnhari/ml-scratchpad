"""TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.

Paper: https://arxiv.org/abs/2504.19874 (ICLR 2026)

Two variants:
  TurboQuant.quantize / dequantize         -> MSE-optimal
  TurboQuant.quantize_prod / dequantize_prod -> inner-product-optimal (for vector search / KV cache)
"""

import numpy as np
from scipy.special import erfinv

# ---------------------------------------------------------------------------
# Lloyd-Max centroid computation
# ---------------------------------------------------------------------------

def _lloyd_max_centroids_gaussian(n_levels: int, n_samples: int = 200_000, n_iter: int = 300) -> np.ndarray:
    """Compute Lloyd-Max optimal centroids for N(0,1) via Lloyd's algorithm."""
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(n_samples)

    # Initialize at quantile midpoints
    q = np.linspace(1 / (2 * n_levels), 1 - 1 / (2 * n_levels), n_levels)
    centroids = np.sqrt(2) * erfinv(2 * q - 1)

    for _ in range(n_iter):
        dists = np.abs(samples[:, None] - centroids[None, :])   # (n_samples, n_levels)
        assign = np.argmin(dists, axis=1)                        # (n_samples,)
        new_c = np.array([
            samples[assign == k].mean() if (assign == k).any() else centroids[k]
            for k in range(n_levels)
        ])
        if np.max(np.abs(new_c - centroids)) < 1e-10:
            break
        centroids = new_c

    return np.sort(centroids)


_CENTROID_CACHE: dict[int, np.ndarray] = {}


def _get_centroids(bits: int, d: int) -> np.ndarray:
    """Lloyd-Max centroids for N(0, 1/d), i.e., N(0,1) scaled by 1/sqrt(d)."""
    if bits not in _CENTROID_CACHE:
        _CENTROID_CACHE[bits] = _lloyd_max_centroids_gaussian(2 ** bits)
    return _CENTROID_CACHE[bits] / np.sqrt(d)


# ---------------------------------------------------------------------------
# TurboQuant
# ---------------------------------------------------------------------------

class TurboQuant:
    """TurboQuant vector quantizer.

    Args:
        d:    Vector dimension.
        bits: Default bit-width per coordinate.
        seed: RNG seed (fixes rotation and QJL matrices).
    """

    def __init__(self, d: int, bits: int = 2, seed: int = 42):
        self.d = d
        self.bits = bits

        rng = np.random.default_rng(seed)

        # Rotation matrix Pi via QR of random Gaussian matrix
        M = rng.standard_normal((d, d))
        self.Pi, _ = np.linalg.qr(M)   # Pi is orthogonal: Pi @ Pi.T = I

        # S matrix for QJL (d x d Gaussian, fixed per instance)
        self.S = rng.standard_normal((d, d))

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def rotate(self, x: np.ndarray) -> np.ndarray:
        """Apply Pi to x.  x: (..., d) -> (..., d)."""
        return x @ self.Pi.T     # row-vector convention: (... x d) @ (d x d)

    def _unrotate(self, y: np.ndarray) -> np.ndarray:
        """Inverse: Pi^T (orthogonal, so Pi^T = Pi^{-1})."""
        return y @ self.Pi

    # ------------------------------------------------------------------
    # MSE-optimal variant (Algorithm 1)
    # ------------------------------------------------------------------

    def quantize(self, x: np.ndarray, bits: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """MSE-optimal quantization.

        Steps: normalize -> rotate -> nearest-centroid per coordinate.

        Returns:
            indices: int array (..., d), indices into centroid table.
            norms:   float array (...,), original L2 norms (stored in fp32).
        """
        bits = bits if bits is not None else self.bits
        x = np.asarray(x, dtype=np.float64)

        norms = np.linalg.norm(x, axis=-1, keepdims=True)                 # (..., 1)
        x_hat = x / np.where(norms > 0, norms, 1.0)                       # unit norm

        y = self.rotate(x_hat)                                             # (..., d)
        centroids = _get_centroids(bits, self.d)                           # (2^b,)

        # nearest centroid index per coordinate
        diff = np.abs(y[..., :, None] - centroids[None, :])               # (..., d, 2^b)
        indices = np.argmin(diff, axis=-1).astype(np.uint8)               # (..., d)

        return indices, norms.squeeze(-1)

    def dequantize(self, indices: np.ndarray, norms: np.ndarray, bits: int | None = None) -> np.ndarray:
        """Reconstruct vectors from MSE-quantized indices."""
        bits = bits if bits is not None else self.bits
        centroids = _get_centroids(bits, self.d)
        y_hat = centroids[indices]               # (..., d)
        x_hat = self._unrotate(y_hat)            # (..., d), unit-norm approximation
        return x_hat * norms[..., None]          # rescale to original norm

    # ------------------------------------------------------------------
    # Inner-product-optimal variant (Algorithm 2)
    # ------------------------------------------------------------------

    def quantize_prod(self, x: np.ndarray, bits: int | None = None) -> tuple:
        """Inner-product-optimized quantization.

        Uses (bits-1) bits for MSE stage + 1-bit QJL on residual.
        The QJL term corrects the bias in inner-product estimation.

        Returns:
            indices: (..., d) int — (bits-1)-bit MSE indices
            qjl:     (..., d) float in {-1, +1} — residual sign bits
            gamma:   (...,) float — residual L2 norm (stored in fp32)
            norms:   (...,) float — input L2 norm
        """
        bits = bits if bits is not None else self.bits
        if bits < 2:
            raise ValueError("prod variant requires bits >= 2 (1 for MSE + 1 for QJL)")

        x = np.asarray(x, dtype=np.float64)
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        x_hat = x / np.where(norms > 0, norms, 1.0)

        # Stage 1: (bits-1)-bit MSE quantization
        idx_mse, _ = self.quantize(x_hat, bits=bits - 1)
        x_mse = self.dequantize(idx_mse, np.ones(x_hat.shape[:-1]), bits=bits - 1)

        # Stage 2: 1-bit QJL on residual r = x_hat - x_mse
        r = x_hat - x_mse                                           # (..., d)
        gamma = np.linalg.norm(r, axis=-1)                          # (...,)

        # QJL: sign(S · r).  Note sign(S·r) = sign(S·r_hat) since gamma > 0
        qjl = np.sign(r @ self.S.T)                                 # (..., d)
        qjl = np.where(qjl == 0, 1.0, qjl)                         # avoid exact 0

        return idx_mse, qjl, gamma, norms.squeeze(-1)

    def dequantize_prod(
        self,
        indices: np.ndarray,
        qjl: np.ndarray,
        gamma: np.ndarray,
        norms: np.ndarray,
        bits: int | None = None,
    ) -> np.ndarray:
        """Reconstruct from inner-product-optimized quantization.

        x̃ = x̃_mse + (√(π/2)/d) · γ · S^T · qjl
        E[⟨q, x̃⟩] = ⟨q, x⟩  (unbiased inner product estimator).
        """
        bits = bits if bits is not None else self.bits

        x_mse = self.dequantize(indices, np.ones(indices.shape[:-1]), bits=bits - 1)

        # QJL correction: sqrt(π/2)/d * gamma * S^T * qjl
        # qjl @ self.S  ==  S^T · qjl  (row-vector convention)
        correction = (np.sqrt(np.pi / 2) / self.d) * gamma[..., None] * (qjl @ self.S)

        x_hat = x_mse + correction
        return x_hat * norms[..., None]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def bits_per_vector(self, bits: int | None = None, variant: str = "mse") -> float:
        """Total bits used per vector (indices + stored fp32 values)."""
        bits = bits if bits is not None else self.bits
        index_bits = self.d * bits
        norm_bits = 32                          # one fp32 for the L2 norm
        if variant == "prod":
            qjl_bits = self.d * 1              # 1 bit per coordinate
            gamma_bits = 32                    # one fp32 for residual norm
            return index_bits + qjl_bits + norm_bits + gamma_bits
        return index_bits + norm_bits

    def compression_ratio(self, bits: int | None = None, variant: str = "mse") -> float:
        """Compression vs float32 (d * 32 bits)."""
        original = self.d * 32
        return original / self.bits_per_vector(bits, variant)

    def mse_upper_bound(self, bits: int | None = None) -> float:
        """Theoretical MSE upper bound: √(3π)/2 · 1/4^b."""
        bits = bits if bits is not None else self.bits
        return (np.sqrt(3 * np.pi) / 2) / (4 ** bits)

    def mse_lower_bound(self, bits: int | None = None) -> float:
        """Information-theoretic MSE lower bound: 1/4^b."""
        bits = bits if bits is not None else self.bits
        return 1.0 / (4 ** bits)
