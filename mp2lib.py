"""Reusable numerical functions for ECE 3715 Mini-Project 2."""

import numpy as np
from scipy.special import erfc

__all__ = [
    "inv_cdf_exponential",
    "inv_cdf_rayleigh",
    "sample_by_inversion",
    "ks_statistic",
    "q_function",
]


def inv_cdf_exponential(u, lam):
    values = np.asarray(u, dtype=float)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be finite and positive")
    if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values >= 1):
        raise ValueError("u must contain values in [0, 1)")
    return -np.log1p(-values) / lam


def inv_cdf_rayleigh(u, sigma):
    values = np.asarray(u, dtype=float)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values >= 1):
        raise ValueError("u must contain values in [0, 1)")
    return sigma * np.sqrt(-2 * np.log1p(-values))


def sample_by_inversion(inv_cdf, n, rng, **kw):
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return np.asarray(inv_cdf(rng.random(n), **kw), dtype=float)


def ks_statistic(samples, cdf):
    values = np.sort(np.asarray(samples, dtype=float).ravel())
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("samples must contain at least one finite value")
    theory = np.asarray(cdf(values), dtype=float)
    if theory.shape != values.shape or np.any(~np.isfinite(theory)) or np.any(theory < 0) or np.any(theory > 1):
        raise ValueError("cdf must return values in [0, 1] with the sample shape")
    n = values.size
    upper = np.arange(1, n + 1) / n - theory
    lower = theory - np.arange(0, n) / n
    return float(max(np.max(upper), np.max(lower)))


def q_function(x):
    values = np.asarray(x, dtype=float)
    return 0.5 * erfc(values / np.sqrt(2))
