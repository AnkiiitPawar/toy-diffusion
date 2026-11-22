"""README figures from the trained models in runs/ and the numbers in docs/results.json.

usage: python scripts/figures.py     (needs matplotlib)
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion import EpsNet, Schedule, ddpm_sample, network_eps  # noqa: E402
from diffusion.datasets import DATASETS, eight_gaussians, get  # noqa: E402
from diffusion.ddpm import eps_from_score, score_from_eps  # noqa: E402

DOCS = Path("docs")
SCHED = Schedule.cosine(200)
TITLES = {"moons": "two moons", "spiral": "spiral", "checkerboard": "checkerboard", "gaussians": "8 Gaussians"}


def load(name):
    net = EpsNet(seed=0)
    data = np.load(f"runs/{name}.npz")
    net.load([data[f"arr_{i}"] for i in range(len(data.files))])
    return net


def style(ax, lim=2.6):
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xticks=[], yticks=[])


def samples():
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    rng = np.random.default_rng(0)
    for j, name in enumerate(DATASETS):
        x = get(name)(3000, rng)
        y = ddpm_sample(network_eps(load(name)), 3000, SCHED, rng)[0]
        axes[0, j].scatter(*x.T, s=1, color="#555", alpha=0.6)
        axes[1, j].scatter(*y.T, s=1, color="#1f5f99", alpha=0.6)
        axes[0, j].set_title(TITLES[name])
        for ax in axes[:, j]:
            style(ax)
    axes[0, 0].set_ylabel("training data")
    axes[1, 0].set_ylabel("generated (DDPM, 200 steps)")
    fig.tight_layout()
    fig.savefig(DOCS / "samples.png", dpi=120)


def reverse_process():
    keep = [200, 120, 70, 40, 20, 8]
    _, snaps = ddpm_sample(network_eps(load("spiral")), 3000, SCHED, np.random.default_rng(1), keep=keep)
    order = keep + [0]
    fig, axes = plt.subplots(1, len(order), figsize=(15, 2.5))
    for ax, t in zip(axes, order):
        ax.scatter(*snaps[t].T, s=0.6, color=plt.cm.viridis(1 - t / 200), alpha=0.7)
        style(ax, 3.2)
        ax.set_title(f"t = {t}", fontsize=10)
    fig.suptitle("Reverse process: noise to spiral", y=1.02)
    fig.tight_layout()
    fig.savefig(DOCS / "reverse_process.png", dpi=120, bbox_inches="tight")


def score_field(results):
    gm = eight_gaussians()
    net = load("gaussians")
    errs = results["datasets"]["gaussians"]["score_rel_error"]
    ts = [5, 50, 100]
    g = np.linspace(-2.2, 2.2, 15)
    X, Y = np.meshgrid(g, g)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    fine = np.linspace(-2.4, 2.4, 200)
    FX, FY = np.meshgrid(fine, fine)
    fpts = np.stack([FX.ravel(), FY.ravel()], 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, t in zip(axes, ts):
        ab = SCHED.alpha_bar[t]
        dens = gm.log_density(fpts, ab).reshape(FX.shape)
        ax.contourf(FX, FY, np.exp(dens), levels=20, cmap="Greys")
        exact = gm.score(pts, ab)
        learned = score_from_eps(net.forward(pts, np.full(len(pts), t)), t, SCHED)
        scale = np.percentile(np.linalg.norm(exact, axis=1), 90) * 9
        ax.quiver(*pts.T, *exact.T, color="black", scale=scale, width=0.006, label="exact score")
        ax.quiver(*pts.T, *learned.T, color="#e07b00", scale=scale, width=0.004, label="learned score")
        style(ax, 2.4)
        ax.set_title(f"t = {t}  (ᾱ = {ab:.2f}), relative error {errs[str(t)]:.1%}", fontsize=10)
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle("Learned score −ε̂/√(1−ᾱ) vs the exact score of the noised mixture, over pₜ", y=0.98)
    fig.tight_layout()
    fig.savefig(DOCS / "score_field.png", dpi=120)


def loss_and_steps(results):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9))
    g = results["datasets"]["gaussians"]
    curve = np.array(g["loss_curve"])
    a.plot(np.arange(len(curve)) * 200, curve, color="#1f5f99", label="training loss (8 Gaussians)")
    a.axhline(g["bayes_loss"], color="#c0504d", ls="--", lw=1, label=f"Bayes-optimal loss {g['bayes_loss']:.3f}")
    a.set(xlabel="step", ylabel="mean squared error of ε̂", ylim=(0.3, 0.6), title="The loss can't reach zero")
    a.legend(frameon=False)
    colors = {"moons": "#1f5f99", "spiral": "#2e8b57", "checkerboard": "#8064a2", "gaussians": "#c0504d"}
    for name in DATASETS:
        r = results["datasets"][name]
        steps = sorted(int(k) for k in r["ddim"])
        b.plot(steps, [max(r["ddim"][str(s)], 1e-4) for s in steps], "o-", color=colors[name], ms=3,
               label=TITLES[name])
        b.plot(steps, [max(r["ddim_unclipped"][str(s)], 1e-4) for s in steps], "--", color=colors[name], lw=0.9,
               alpha=0.6)
        b.plot([200], [max(r["ddpm"], 1e-4)], "s", color=colors[name], ms=6, mfc="none")
    floor = np.mean([abs(results["datasets"][n]["noise_floor"]) for n in DATASETS])
    b.axhspan(0, floor * 2, color="#eee", zorder=0)
    b.text(5, floor * 2.4, "noise floor (two independent true samples)", fontsize=8, color="#666")
    b.set(xscale="log", yscale="log", xlabel="sampling steps", ylabel="energy distance to true data",
          title="DDIM: solid with x̂₀ clipping, dashed without; squares: DDPM")
    b.legend(frameon=False, fontsize=8)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "loss_and_steps.png", dpi=120)


if __name__ == "__main__":
    res = json.loads((DOCS / "results.json").read_text())
    samples()
    reverse_process()
    score_field(res)
    loss_and_steps(res)
    print("figures written to docs/")
