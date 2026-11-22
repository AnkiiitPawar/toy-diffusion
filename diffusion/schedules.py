"""Noise schedules and the closed-form quantities of the forward process.

Forward (noising) process, for t = 1..T:
    q(xₜ | xₜ₋₁) = N(√(1-βₜ) xₜ₋₁, βₜ I)
which composes to
    q(xₜ | x₀) = N(√ᾱₜ x₀, (1 - ᾱₜ) I),   αₜ = 1 - βₜ,  ᾱₜ = Πₛ≤ₜ αₛ.
The reverse posterior given x₀ is Gaussian too (Ho et al., 2020, eq. 7):
    q(xₜ₋₁ | xₜ, x₀) = N(μ̃ₜ, β̃ₜ I)
    μ̃ₜ = √ᾱₜ₋₁ βₜ / (1-ᾱₜ) · x₀ + √αₜ (1-ᾱₜ₋₁) / (1-ᾱₜ) · xₜ
    β̃ₜ = (1-ᾱₜ₋₁) / (1-ᾱₜ) · βₜ
Arrays are indexed by t = 0..T, with ᾱ₀ = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Schedule:
    beta: np.ndarray       # (T+1,), beta[0] = 0
    alpha: np.ndarray
    alpha_bar: np.ndarray

    @property
    def T(self) -> int:
        return len(self.beta) - 1

    @classmethod
    def from_betas(cls, betas: np.ndarray) -> "Schedule":
        beta = np.concatenate([[0.0], betas])
        alpha = 1.0 - beta
        return cls(beta, alpha, np.cumprod(alpha))

    @classmethod
    def linear(cls, T: int, beta_1: float = 1e-4, beta_T: float = 0.02) -> "Schedule":
        """Ho et al.'s linear schedule, rescaled by 1000/T so short chains still end in noise."""
        scale = 1000.0 / T
        return cls.from_betas(np.linspace(beta_1 * scale, beta_T * scale, T))

    @classmethod
    def cosine(cls, T: int, s: float = 0.008) -> "Schedule":
        """Nichol & Dhariwal (2021): ᾱ(t) = f(t)/f(0), f(t) = cos²((t/T + s)/(1 + s) · π/2)."""
        t = np.arange(T + 1)
        f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
        ab = f / f[0]
        betas = np.clip(1.0 - ab[1:] / ab[:-1], 0.0, 0.999)
        return cls.from_betas(betas)

    def posterior(self, t: int):
        """Coefficients (c₀, cₜ, variance) with μ̃ₜ = c₀ x₀ + cₜ xₜ."""
        ab, ab_prev, b, a = self.alpha_bar[t], self.alpha_bar[t - 1], self.beta[t], self.alpha[t]
        c0 = np.sqrt(ab_prev) * b / (1 - ab)
        ct = np.sqrt(a) * (1 - ab_prev) / (1 - ab)
        var = (1 - ab_prev) / (1 - ab) * b
        return c0, ct, var
