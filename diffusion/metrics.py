"""Sample-quality metric: the energy distance between two point clouds.

    D²(P, Q) = 2 E‖X - Y‖ - E‖X - X'‖ - E‖Y - Y'‖,   X, X' ~ P,  Y, Y' ~ Q
It is zero exactly when P = Q (Székely & Rizzo, 2013) and needs no bandwidth or binning.
Comparing against two independent draws from the true distribution gives the noise floor.
"""

from __future__ import annotations

import numpy as np


def _mean_dist(a: np.ndarray, b: np.ndarray, same: bool, chunk: int = 500) -> float:
    """Mean pairwise Euclidean distance, in row chunks to bound memory."""
    total = 0.0
    for i in range(0, len(a), chunk):
        d = a[i:i + chunk, None, :] - b[None, :, :]
        total += float(np.sqrt((d * d).sum(-1)).sum())
    n, m = len(a), len(b)
    return total / (n * (n - 1)) if same else total / (n * m)


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased estimate of D² (it can be slightly negative when P = Q)."""
    return 2 * _mean_dist(x, y, False) - _mean_dist(x, x, True) - _mean_dist(y, y, True)
