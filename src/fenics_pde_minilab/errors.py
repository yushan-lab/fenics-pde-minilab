"""Manufactured solutions and error-rate helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def poisson_exact(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Return sin(pi*x) sin(pi*y) at scalar or array coordinates."""
    return np.sin(math.pi * np.asarray(x)) * np.sin(math.pi * np.asarray(y))


def poisson_rhs(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Return f = 2*pi^2*sin(pi*x)*sin(pi*y) for the Poisson problem."""
    return 2.0 * math.pi**2 * poisson_exact(x, y)


def heat_exact(x: ArrayLike, y: ArrayLike, *, t: float, kappa: float) -> np.ndarray:
    """Return the exact heat-equation solution at time t."""
    decay = math.exp(-2.0 * math.pi**2 * kappa * t)
    return decay * poisson_exact(x, y)


def convergence_rates(mesh_sizes: Sequence[float], errors: Sequence[float]) -> list[float]:
    """Estimate adjacent-grid rates log(e_i/e_j) / log(h_i/h_j)."""
    h = np.asarray(mesh_sizes, dtype=float)
    e = np.asarray(errors, dtype=float)
    if h.shape != e.shape:
        raise ValueError("mesh_sizes and errors must have the same length")
    if h.ndim != 1:
        raise ValueError("mesh_sizes and errors must be one-dimensional")

    rates = [float("nan")]
    for i in range(1, len(h)):
        if h[i] <= 0.0 or h[i - 1] <= 0.0 or e[i] <= 0.0 or e[i - 1] <= 0.0:
            rates.append(float("nan"))
        else:
            rates.append(float(math.log(e[i - 1] / e[i]) / math.log(h[i - 1] / h[i])))
    return rates
