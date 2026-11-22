"""Training objective and samplers for denoising diffusion (DDPM and DDIM).

The network predicts the noise ε in xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε. Because
∇ₓ log q(xₜ | x₀) = -ε / √(1-ᾱₜ), the minimiser of E‖ε̂(xₜ, t) - ε‖² satisfies
    ε̂*(x, t) = -√(1-ᾱₜ) ∇ₓ log pₜ(x),
so a noise predictor *is* a score model (Vincent, 2011; Song & Ermon, 2019).
"""

from __future__ import annotations

import numpy as np

from .schedules import Schedule


def q_sample(x0: np.ndarray, t: np.ndarray, eps: np.ndarray, sched: Schedule) -> np.ndarray:
    ab = sched.alpha_bar[t][:, None]
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps


def loss_and_grad(net, x0: np.ndarray, sched: Schedule, rng: np.random.Generator) -> float:
    """L = mean over batch and coordinates of (ε̂ - ε)²; fills the network's gradients."""
    n, d = x0.shape
    t = rng.integers(1, sched.T + 1, n)
    eps = rng.standard_normal(x0.shape)
    pred = net.forward(q_sample(x0, t, eps, sched), t)
    err = pred - eps
    net.backward(2.0 * err / (n * d))
    return float((err**2).mean())


def eps_from_score(score: np.ndarray, t: int, sched: Schedule) -> np.ndarray:
    return -np.sqrt(1 - sched.alpha_bar[t]) * score


def score_from_eps(eps: np.ndarray, t: int, sched: Schedule) -> np.ndarray:
    return -eps / np.sqrt(1 - sched.alpha_bar[t])


def ddpm_sample(eps_fn, n: int, sched: Schedule, rng: np.random.Generator, dim: int = 2, keep=()):
    """Ancestral sampling (Ho et al., 2020, Algorithm 2) with variance β̃ₜ.

    xₜ₋₁ = (xₜ - βₜ/√(1-ᾱₜ) ε̂(xₜ, t)) / √αₜ + √β̃ₜ z.
    ``eps_fn(x, t)`` returns ε̂ for a batch at integer step t. Returns (x₀, {t: xₜ for t in keep}).
    """
    x = rng.standard_normal((n, dim))
    snaps = {}
    for t in range(sched.T, 0, -1):
        if t in keep:
            snaps[t] = x.copy()
        e = eps_fn(x, t)
        b, a, ab = sched.beta[t], sched.alpha[t], sched.alpha_bar[t]
        x = (x - b / np.sqrt(1 - ab) * e) / np.sqrt(a)
        if t > 1:
            _, _, var = sched.posterior(t)
            x = x + np.sqrt(var) * rng.standard_normal(x.shape)
    snaps[0] = x.copy()
    return x, snaps


def ddim_sample(eps_fn, n: int, sched: Schedule, steps: int, rng: np.random.Generator, dim: int = 2,
                clip_x0: float | None = 3.0):
    """Deterministic DDIM (Song, Meng & Ermon, 2021, η = 0) on an evenly spaced sub-sequence.

    x̂₀ = (xₜ - √(1-ᾱₜ) ε̂) / √ᾱₜ,   xₛ = √ᾱₛ x̂₀ + √(1-ᾱₛ) ε̂   for s < t.
    The first step divides by √ᾱ_T ≈ 2.5e-4 (the cosine schedule ends at ᾱ_T ≈ 6e-8), so any
    error in ε̂ there is amplified thousands of times. Like most implementations, we clip x̂₀
    to the data range and re-derive ε̂ from it; ``clip_x0=None`` turns this off.
    """
    ts = np.unique(np.linspace(sched.T, 1, steps).round().astype(int))[::-1]
    x = rng.standard_normal((n, dim))
    for i, t in enumerate(ts):
        e = eps_fn(x, t)
        ab = sched.alpha_bar[t]
        x0 = (x - np.sqrt(1 - ab) * e) / np.sqrt(ab)
        if clip_x0 is not None:
            x0 = np.clip(x0, -clip_x0, clip_x0)
            e = (x - np.sqrt(ab) * x0) / np.sqrt(1 - ab)
        s = ts[i + 1] if i + 1 < len(ts) else 0
        ab_s = sched.alpha_bar[s]
        x = np.sqrt(ab_s) * x0 + np.sqrt(1 - ab_s) * e
    return x


def network_eps(net):
    """Adapter: the trained network as an eps_fn for the samplers."""
    return lambda x, t: net.forward(x, np.full(len(x), t))
