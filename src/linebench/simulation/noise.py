"""Noise generator for task-time variability (Master Plan §A8, Phase 5.1).

Generates mean-one multiplicative noise factors for each (replication, unit, task)
or (replication, task) combination.

Distributions supported:
  - Lognormal: X ~ Lognormal(mu_ln, sigma_ln) where
      sigma_ln^2 = ln(1 + CV^2)
      mu_ln = -0.5 * sigma_ln^2
    which guarantees E[X] = exp(mu_ln + 0.5 * sigma_ln^2) = 1.0
    and Var[X] / (E[X])^2 = CV^2.
  - Gamma: X ~ Gamma(shape=k, scale=theta) where
      k = 1 / CV^2
      theta = CV^2 = 1 / k
    which guarantees E[X] = k * theta = 1.0
    and Var[X] / (E[X])^2 = (k * theta^2) / 1.0 = CV^2.

Common Random Numbers (CRN):
  A seeded NumPy Generator produces reproducible arrays so the exact same
  stochastic sequence can be reused across competing balances.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gamma_factors",
    "generate_noise_factors",
    "lognormal_factors",
]


def lognormal_factors(
    shape: tuple[int, ...],
    cv: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate mean-one lognormal multiplicative noise factors.

    Parameters
    ----------
    shape:
        Desired array shape, e.g. (n_replications, n_units, n_tasks).
    cv:
        Target coefficient of variation (sigma / mu). Must be >= 0.
        If cv == 0, returns an array of ones.
    rng:
        Seeded numpy Generator.

    Returns
    -------
    np.ndarray:
        Array of multiplicative noise factors with theoretical E[X] = 1.0.
    """
    if cv < 0.0:
        raise ValueError(f"CV must be non-negative, got {cv}")
    if cv == 0.0:
        return np.ones(shape, dtype=np.float64)

    sigma_ln = np.sqrt(np.log1p(cv * cv))
    mu_ln = -0.5 * (sigma_ln * sigma_ln)
    return rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=shape)


def gamma_factors(
    shape: tuple[int, ...],
    cv: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate mean-one Gamma multiplicative noise factors.

    Parameters
    ----------
    shape:
        Desired array shape, e.g. (n_replications, n_units, n_tasks).
    cv:
        Target coefficient of variation (sigma / mu). Must be >= 0.
        If cv == 0, returns an array of ones.
    rng:
        Seeded numpy Generator.

    Returns
    -------
    np.ndarray:
        Array of multiplicative noise factors with theoretical E[X] = 1.0.
    """
    if cv < 0.0:
        raise ValueError(f"CV must be non-negative, got {cv}")
    if cv == 0.0:
        return np.ones(shape, dtype=np.float64)

    k = 1.0 / (cv * cv)
    scale = cv * cv
    return rng.gamma(shape=k, scale=scale, size=shape)


def generate_noise_factors(
    shape: tuple[int, ...],
    cv: float,
    seed: int | np.random.Generator,
    distribution: str = "lognormal",
) -> np.ndarray:
    """Convenience function to generate noise factors.

    Parameters
    ----------
    shape:
        Target array shape, e.g. (n_replications, n_units, n_tasks).
    cv:
        Target coefficient of variation.
    seed:
        Integer seed or an existing numpy Generator instance.
    distribution:
        'lognormal' (default) or 'gamma'.

    Returns
    -------
    np.ndarray:
        Multiplicative noise factors.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    dist = distribution.lower().strip()
    if dist == "lognormal":
        return lognormal_factors(shape, cv, rng)
    elif dist == "gamma":
        return gamma_factors(shape, cv, rng)
    else:
        raise ValueError(f"Unknown distribution '{distribution}'. Expected 'lognormal' or 'gamma'.")
