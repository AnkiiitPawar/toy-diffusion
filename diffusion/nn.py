"""The noise-prediction network ε_θ(xₜ, t): an MLP on [xₜ, sinusoidal embedding of t].

Hidden layers use SiLU, s(z) = z σ(z), whose derivative is σ(z) (1 + z (1 - σ(z))).
The backward pass is written out; ``tests/test_nn.py`` checks it with finite differences.
"""

from __future__ import annotations

import numpy as np


def time_embedding(t: np.ndarray, dim: int = 32, max_period: float = 10_000.0) -> np.ndarray:
    """Transformer-style features [sin(t ωᵢ), cos(t ωᵢ)] with ωᵢ = max_period^(-i / (dim/2))."""
    half = dim // 2
    freqs = max_period ** (-np.arange(half) / half)
    ang = np.asarray(t, dtype=float)[:, None] * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # overflow-free form of 1 / (1 + e^-z)


class EpsNet:
    def __init__(self, data_dim: int = 2, hidden: int = 128, depth: int = 3, emb_dim: int = 32,
                 seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        sizes = [data_dim + emb_dim] + [hidden] * depth + [data_dim]
        self.emb_dim = emb_dim
        # He-style scaling for the SiLU layers; a small last layer so ε̂ starts near zero.
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.W[-1] *= 0.1
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.gW = [np.zeros_like(w) for w in self.W]
        self.gb = [np.zeros_like(b) for b in self.b]

    def params(self):
        return [p for pair in zip(self.W, self.b) for p in pair]

    def grads(self):
        return [g for pair in zip(self.gW, self.gb) for g in pair]

    def zero_grad(self) -> None:
        for g in self.grads():
            g.fill(0.0)

    def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        h = np.concatenate([x, time_embedding(t, self.emb_dim)], axis=1)
        self.cache = []
        n = len(self.W)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i < n - 1:
                s = sigmoid(z)
                self.cache.append((h, z, s))
                h = z * s
            else:
                self.cache.append((h, None, None))
                h = z
        return h

    def backward(self, dout: np.ndarray) -> None:
        d = dout
        for i in reversed(range(len(self.W))):
            h_in, z, s = self.cache[i]
            if z is not None:
                d = d * (s * (1.0 + z * (1.0 - s)))
            self.gW[i] += h_in.T @ d
            self.gb[i] += d.sum(axis=0)
            d = d @ self.W[i].T

    def state(self):
        return [p.copy() for p in self.params()]

    def load(self, state) -> None:
        for p, v in zip(self.params(), state):
            p[...] = v


class Adam:
    def __init__(self, params, grads, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8) -> None:
        self.params, self.grads = list(params), list(grads)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        self.t += 1
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        for p, g, m, v in zip(self.params, self.grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
