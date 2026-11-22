"""Training loop: fresh data every step, Adam with cosine learning-rate decay, and an
exponential moving average (EMA) of the weights, which is what gets sampled from."""

from __future__ import annotations

import math
import time

import numpy as np

from .ddpm import loss_and_grad
from .nn import Adam, EpsNet
from .schedules import Schedule


def train(net: EpsNet, data_fn, sched: Schedule, steps: int, batch: int = 512, lr: float = 2e-3,
          ema_decay: float = 0.999, seed: int = 0, log_every: int = 0, log=print):
    """data_fn(n, rng) draws training points. Returns (EMA weights, loss history)."""
    rng = np.random.default_rng(seed)
    opt = Adam(net.params(), net.grads(), lr)
    ema = net.state()
    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        net.zero_grad()
        losses.append(loss_and_grad(net, data_fn(batch, rng), sched, rng))
        opt.step(lr * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for e, p in zip(ema, net.params()):
            e *= ema_decay
            e += (1 - ema_decay) * p
        if log_every and (step % log_every == 0 or step == steps - 1):
            log(f"step {step:6d}  loss {np.mean(losses[-log_every:]):.4f}  ({time.perf_counter() - t0:.0f} s)")
    return ema, np.array(losses)
