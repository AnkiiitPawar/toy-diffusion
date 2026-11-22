import numpy as np
import pytest

from diffusion import EpsNet, Schedule, ddim_sample, ddpm_sample, loss_and_grad, q_sample
from diffusion.datasets import DATASETS, GaussianMixture, eight_gaussians, get
from diffusion.ddpm import eps_from_score, score_from_eps
from diffusion.metrics import energy_distance
from diffusion.nn import time_embedding
from diffusion.train import train


@pytest.mark.parametrize("make", [Schedule.cosine, Schedule.linear])
def test_schedule_ends_in_noise(make):
    s = make(200)
    assert s.alpha_bar[0] == 1.0
    assert np.all(np.diff(s.alpha_bar) < 0)
    assert s.alpha_bar[-1] < 1e-4
    assert np.all((s.beta[1:] > 0) & (s.beta[1:] < 1))


def test_closed_form_marginal_matches_the_step_by_step_chain():
    """Applying q(xₜ | xₜ₋₁) fifty times must give N(√ᾱ₅₀ x₀, 1 - ᾱ₅₀)."""
    s = Schedule.cosine(200)
    rng = np.random.default_rng(0)
    x0 = 1.7
    x = np.full(200_000, x0)
    for t in range(1, 51):
        x = np.sqrt(1 - s.beta[t]) * x + np.sqrt(s.beta[t]) * rng.standard_normal(x.shape)
    assert x.mean() == pytest.approx(np.sqrt(s.alpha_bar[50]) * x0, abs=0.01)
    assert x.var() == pytest.approx(1 - s.alpha_bar[50], abs=0.01)


def test_posterior_matches_bayes_rule():
    """q(xₜ₋₁ | xₜ, x₀) ∝ q(xₜ | xₜ₋₁) q(xₜ₋₁ | x₀): multiply the two Gaussians directly."""
    s = Schedule.cosine(100)
    x0, xt = 0.8, -0.3
    for t in (2, 10, 50, 99):
        prior_mean, prior_var = np.sqrt(s.alpha_bar[t - 1]) * x0, 1 - s.alpha_bar[t - 1]
        a, b = s.alpha[t], s.beta[t]
        prec = 1 / prior_var + a / b
        mean = (prior_mean / prior_var + np.sqrt(a) * xt / b) / prec
        c0, ct, var = s.posterior(t)
        assert c0 * x0 + ct * xt == pytest.approx(mean)
        assert var == pytest.approx(1 / prec)


def test_network_gradients():
    rng = np.random.default_rng(1)
    net = EpsNet(hidden=16, depth=2, emb_dim=8, seed=1)
    for p in net.params():  # sizeable weights so every path matters
        p[...] = rng.standard_normal(p.shape) * 0.5
    x, t = rng.standard_normal((5, 2)), rng.integers(1, 100, 5)
    r = rng.standard_normal((5, 2))
    net.zero_grad()
    net.forward(x, t)
    net.backward(r)
    for p, g in zip(net.params(), net.grads()):
        num = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + 1e-6
            fp = (net.forward(x, t) * r).sum()
            p[i] = old - 1e-6
            fm = (net.forward(x, t) * r).sum()
            p[i] = old
            num[i] = (fp - fm) / 2e-6
        assert np.linalg.norm(g - num) <= 1e-6 * (np.linalg.norm(g) + np.linalg.norm(num))


def test_time_embedding_shape_and_range():
    e = time_embedding(np.array([0, 1, 199]), 32)
    assert e.shape == (3, 32) and np.all(np.abs(e) <= 1)
    assert np.allclose(e[0], np.r_[np.zeros(16), np.ones(16)])


def test_exact_mixture_score_is_the_gradient_of_the_log_density():
    gm = GaussianMixture(k=5, radius=1.2, std=0.2)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((20, 2))
    for ab in (1.0, 0.6, 0.05):
        h = 1e-6
        num = np.stack([(gm.log_density(x + h * e, ab) - gm.log_density(x - h * e, ab)) / (2 * h)
                        for e in np.eye(2)], 1)
        assert np.allclose(gm.score(x, ab), num, atol=1e-6)


def test_eps_and_score_are_the_same_thing():
    s = Schedule.cosine(200)
    e = np.random.default_rng(3).standard_normal((4, 2))
    assert np.allclose(eps_from_score(score_from_eps(e, 37, s), 37, s), e)


@pytest.mark.parametrize("sampler", ["ddpm", "ddim20"])
def test_samplers_recover_the_mixture_given_the_exact_score(sampler):
    """With the true ε* plugged in, sampling must reproduce the data distribution: this checks
    the samplers independently of any training."""
    gm = eight_gaussians()
    s = Schedule.cosine(200)
    rng = np.random.default_rng(4)
    exact = lambda x, t: eps_from_score(gm.score(x, s.alpha_bar[t]), t, s)  # noqa: E731
    if sampler == "ddpm":
        x = ddpm_sample(exact, 2000, s, rng)[0]
    else:
        x = ddim_sample(exact, 2000, s, 20, rng)
    ref = gm.sample(2000, rng)
    assert energy_distance(x, ref) < 0.005
    nearest = np.argmin(((x[:, None, :] - gm.means[None]) ** 2).sum(-1), axis=1)
    counts = np.bincount(nearest, minlength=8) / len(x)
    assert np.all(np.abs(counts - 1 / 8) < 0.03)
    assert np.mean(np.min(np.linalg.norm(x[:, None, :] - gm.means[None], axis=-1), axis=1)) < 0.2


def test_q_sample_moments():
    s = Schedule.cosine(200)
    rng = np.random.default_rng(5)
    x0 = np.tile([[1.0, -2.0]], (100_000, 1))
    t = np.full(len(x0), 80)
    xt = q_sample(x0, t, rng.standard_normal(x0.shape), s)
    assert np.allclose(xt.mean(0), np.sqrt(s.alpha_bar[80]) * x0[0], atol=0.01)
    assert np.allclose(xt.var(0), 1 - s.alpha_bar[80], atol=0.01)


def test_energy_distance_detects_a_shift():
    rng = np.random.default_rng(6)
    a, b = rng.standard_normal((800, 2)), rng.standard_normal((800, 2))
    assert abs(energy_distance(a, b)) < 0.01
    assert energy_distance(a, b + [0.5, 0.0]) > 0.1


@pytest.mark.parametrize("name", DATASETS)
def test_datasets_are_roughly_standardised(name):
    x = get(name)(20000, np.random.default_rng(7))
    assert x.shape == (20000, 2)
    assert np.all(np.abs(x.mean(0)) < 0.25) and np.all((x.std(0) > 0.7) & (x.std(0) < 1.3))


def test_short_training_approaches_the_bayes_optimal_loss():
    gm = eight_gaussians()
    s = Schedule.cosine(100)
    net = EpsNet(hidden=64, depth=3, seed=0)
    _, losses = train(net, gm.sample, s, 1500, batch=256, seed=0)
    assert losses[:3].mean() > 0.85          # starts near E[ε²] = 1
    assert losses[-200:].mean() < 0.45       # the optimum is about 0.34 (see README)


def test_one_gradient_step_lowers_the_loss_on_a_fixed_batch():
    s = Schedule.cosine(50)
    net = EpsNet(hidden=32, depth=2, seed=2)
    x0 = eight_gaussians().sample(256, np.random.default_rng(8))
    net.zero_grad()
    before = loss_and_grad(net, x0, s, np.random.default_rng(9))
    for p, g in zip(net.params(), net.grads()):
        p -= 0.05 * g
    after = loss_and_grad(net, x0, s, np.random.default_rng(9))  # same t and ε draws
    assert after < before
