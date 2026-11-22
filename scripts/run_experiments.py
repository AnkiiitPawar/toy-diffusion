"""Train one model per dataset, sample with DDPM and DDIM, and score the samples.

usage: python scripts/run_experiments.py [--steps 20000]
Writes docs/results.json and runs/<dataset>.npz (EMA weights).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion import EpsNet, Schedule, ddim_sample, ddpm_sample, network_eps, q_sample  # noqa: E402
from diffusion.datasets import DATASETS, eight_gaussians, get  # noqa: E402
from diffusion.ddpm import eps_from_score  # noqa: E402
from diffusion.metrics import energy_distance  # noqa: E402
from diffusion.train import train  # noqa: E402

N_EVAL = 3000
DDIM_STEPS = [5, 10, 20, 50, 100, 200]


def bayes_optimal_loss(gm, sched, rng, n=20000):
    """E‖ε*(xₜ, t) - ε‖² / 2 with the exact ε*: the lowest loss any network could reach."""
    x0 = gm.sample(n, rng)
    t = rng.integers(1, sched.T + 1, n)
    eps = rng.standard_normal(x0.shape)
    xt = q_sample(x0, t, eps, sched)
    err = 0.0
    for tt in np.unique(t):
        sel = t == tt
        e_star = eps_from_score(gm.score(xt[sel], sched.alpha_bar[tt]), tt, sched)
        err += float(((e_star - eps[sel]) ** 2).sum())
    return err / (2 * n)


def score_errors(net, gm, sched, rng, ts=(5, 20, 50, 100, 150), n=4000):
    """Relative error ‖ε̂ - ε*‖ / ‖ε*‖ on points drawn from pₜ, per noise level."""
    out = {}
    for t in ts:
        xt = q_sample(gm.sample(n, rng), np.full(n, t), rng.standard_normal((n, 2)), sched)
        exact = eps_from_score(gm.score(xt, sched.alpha_bar[t]), t, sched)
        learned = net.forward(xt, np.full(n, t))
        out[int(t)] = float(np.linalg.norm(learned - exact) / np.linalg.norm(exact))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval-only", action="store_true", help="reuse runs/*.npz and the saved training curves")
    args = ap.parse_args()
    old = json.loads(Path("docs/results.json").read_text()) if args.eval_only else None
    sched = Schedule.cosine(200)
    Path("runs").mkdir(exist_ok=True)
    results = {"steps": args.steps, "T": sched.T, "n_eval": N_EVAL, "datasets": {}}
    for name in DATASETS:
        data_fn = get(name)
        net = EpsNet(seed=0)
        if old is None:
            t0 = time.perf_counter()
            ema, losses = train(net, data_fn, sched, args.steps, seed=0)
            seconds = time.perf_counter() - t0
            np.savez(f"runs/{name}.npz", *ema)
        else:
            saved = np.load(f"runs/{name}.npz")
            ema = [saved[f"arr_{i}"] for i in range(len(saved.files))]
            prev = old["datasets"][name]
            seconds, losses = prev["train_seconds"], None
        net.load(ema)
        rng = np.random.default_rng(1)
        ref, ref2 = data_fn(N_EVAL, rng), data_fn(N_EVAL, rng)
        eps_fn = network_eps(net)
        if losses is not None:
            final_loss, curve = float(losses[-500:].mean()), [float(losses[i:i + 200].mean())
                                                               for i in range(0, len(losses), 200)]
        else:
            final_loss, curve = prev["final_loss"], prev["loss_curve"]
        r = {"train_seconds": seconds, "final_loss": final_loss, "loss_curve": curve,
             "noise_floor": energy_distance(ref, ref2),
             "ddpm": energy_distance(ddpm_sample(eps_fn, N_EVAL, sched, rng)[0], ref),
             "ddim": {s: energy_distance(ddim_sample(eps_fn, N_EVAL, sched, s, rng), ref) for s in DDIM_STEPS},
             "ddim_unclipped": {s: energy_distance(ddim_sample(eps_fn, N_EVAL, sched, s, rng, clip_x0=None), ref)
                                for s in DDIM_STEPS}}
        if name == "gaussians":
            gm = eight_gaussians()
            exact = lambda x, t: eps_from_score(gm.score(x, sched.alpha_bar[t]), t, sched)  # noqa: E731
            r["bayes_loss"] = bayes_optimal_loss(gm, sched, rng)
            r["score_rel_error"] = score_errors(net, gm, sched, rng)
            r["exact_ddpm"] = energy_distance(ddpm_sample(exact, N_EVAL, sched, rng)[0], ref)
            r["exact_ddim"] = {s: energy_distance(ddim_sample(exact, N_EVAL, sched, s, rng, clip_x0=None), ref)
                               for s in DDIM_STEPS}
        results["datasets"][name] = r
        print(f"{name}: loss {r['final_loss']:.4f}, DDPM energy distance {r['ddpm']:.5f} "
              f"(floor {r['noise_floor']:.5f}), {seconds:.0f} s", flush=True)
    Path("docs/results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
