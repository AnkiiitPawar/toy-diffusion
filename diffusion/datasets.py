"""2-D data distributions, each scaled to roughly unit variance per coordinate."""

from __future__ import annotations

import numpy as np


def moons(n: int, rng: np.random.Generator, noise: float = 0.05) -> np.ndarray:
    k = n // 2
    a = rng.uniform(0, np.pi, k)
    b = rng.uniform(0, np.pi, n - k)
    upper = np.stack([np.cos(a), np.sin(a)], 1)
    lower = np.stack([1 - np.cos(b), 0.5 - np.sin(b)], 1)
    x = np.concatenate([upper, lower]) + rng.normal(0, noise, (n, 2))
    return (x - [0.5, 0.25]) / [0.87, 0.5]


def spiral(n: int, rng: np.random.Generator, noise: float = 0.03) -> np.ndarray:
    """Two-dimensional swiss roll: radius grows linearly with angle over 1.5 turns."""
    u = np.sqrt(rng.uniform(0, 1, n))  # density uniform along the arc
    ang = 1.5 * np.pi * (1 + 2 * u)
    r = ang / (4.5 * np.pi)
    x = np.stack([r * np.cos(ang), r * np.sin(ang)], 1) + rng.normal(0, noise, (n, 2))
    return (x - [0.21, 0.04]) / 0.55


def checkerboard(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform on the 8 black squares of a 4×4 board on [-2, 2]²."""
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(0, 1, n) + rng.integers(0, 2, n) * 2 - 2  # one of two unit bands
    y = y + (np.floor(x) % 2)  # shift by one row on alternate columns
    return np.stack([x, y], 1) / 1.15


class GaussianMixture:
    """Equal-weight isotropic Gaussians at the vertices of a regular polygon.

    Its noised marginals are mixtures too, so the exact score is available (see ``score``).
    """

    def __init__(self, k: int = 8, radius: float = 2.0, std: float = 0.15) -> None:
        ang = 2 * np.pi * np.arange(k) / k
        self.means = radius * np.stack([np.cos(ang), np.sin(ang)], 1)
        self.std = std
        self.weights = np.full(k, 1.0 / k)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        comp = rng.choice(len(self.weights), n, p=self.weights)
        return self.means[comp] + self.std * rng.standard_normal((n, 2))

    def noised_log_terms(self, x: np.ndarray, alpha_bar: float):
        """Per-component log-weights and residuals of pₜ(x) = Σₖ wₖ N(x; √ᾱ μₖ, (ᾱs² + 1 - ᾱ) I)."""
        var = alpha_bar * self.std**2 + (1 - alpha_bar)
        diff = x[:, None, :] - np.sqrt(alpha_bar) * self.means[None, :, :]
        logw = np.log(self.weights)[None, :] - 0.5 * (diff**2).sum(-1) / var - np.log(2 * np.pi * var)
        return logw, diff, var

    def log_density(self, x: np.ndarray, alpha_bar: float = 1.0) -> np.ndarray:
        logw, _, _ = self.noised_log_terms(x, alpha_bar)
        m = logw.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(logw - m).sum(axis=1, keepdims=True)))[:, 0]

    def score(self, x: np.ndarray, alpha_bar: float) -> np.ndarray:
        """∇ₓ log pₜ(x) = Σₖ γₖ(x) · (-(x - √ᾱ μₖ) / σₜ²), with responsibilities γₖ."""
        logw, diff, var = self.noised_log_terms(x, alpha_bar)
        resp = np.exp(logw - logw.max(axis=1, keepdims=True))
        resp /= resp.sum(axis=1, keepdims=True)
        return -(resp[:, :, None] * diff).sum(axis=1) / var


def eight_gaussians() -> GaussianMixture:
    """The mixture used as the "gaussians" dataset (unit variance per coordinate, near enough)."""
    return GaussianMixture(k=8, radius=1.4, std=0.1)


def get(name: str):
    """Return a sampler f(n, rng) for the named dataset."""
    if name == "gaussians":
        return eight_gaussians().sample
    return {"moons": moons, "spiral": spiral, "checkerboard": checkerboard}[name]


DATASETS = ["moons", "spiral", "checkerboard", "gaussians"]
