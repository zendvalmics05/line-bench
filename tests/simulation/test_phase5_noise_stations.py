"""Unit tests for Phase 5.1 (noise generator) and Phase 5.2 (station-time builder)."""

from __future__ import annotations

import numpy as np
import pytest

from linebench.simulation.noise import (
    gamma_factors,
    generate_noise_factors,
    lognormal_factors,
)
from linebench.simulation.stations import assignment_to_matrix, build_station_times


def test_lognormal_noise_mean_and_cv():
    """Verify lognormal noise generates empirical mean ~ 1.0 and empirical CV ~ target."""
    rng = np.random.default_rng(42)
    shape = (50000, 1)  # Large sample size for law of large numbers
    target_cv = 0.25

    factors = lognormal_factors(shape, target_cv, rng)
    empirical_mean = float(np.mean(factors))
    empirical_std = float(np.std(factors))
    empirical_cv = empirical_std / empirical_mean

    # Mean must be 1.0 within 0.5% tolerance
    assert np.isclose(empirical_mean, 1.0, atol=0.005)
    # CV must be 0.25 within 2% relative tolerance
    assert np.isclose(empirical_cv, target_cv, rtol=0.02)
    # Strictly positive
    assert np.all(factors > 0.0)


def test_gamma_noise_mean_and_cv():
    """Verify gamma noise generates empirical mean ~ 1.0 and empirical CV ~ target."""
    rng = np.random.default_rng(42)
    shape = (50000, 1)
    target_cv = 0.20

    factors = gamma_factors(shape, target_cv, rng)
    empirical_mean = float(np.mean(factors))
    empirical_std = float(np.std(factors))
    empirical_cv = empirical_std / empirical_mean

    assert np.isclose(empirical_mean, 1.0, atol=0.005)
    assert np.isclose(empirical_cv, target_cv, rtol=0.02)
    assert np.all(factors > 0.0)


def test_zero_cv_deterministic():
    """CV = 0 should return strictly ones."""
    rng = np.random.default_rng(123)
    shape = (10, 5, 4)
    ln_ones = lognormal_factors(shape, 0.0, rng)
    ga_ones = gamma_factors(shape, 0.0, rng)

    assert np.all(ln_ones == 1.0)
    assert np.all(ga_ones == 1.0)


def test_generate_noise_factors_reproducibility():
    """Identical seed must yield bit-identical arrays (CRN)."""
    shape = (20, 100, 8)
    cv = 0.30
    arr1 = generate_noise_factors(shape, cv, seed=999, distribution="lognormal")
    arr2 = generate_noise_factors(shape, cv, seed=999, distribution="lognormal")
    arr3 = generate_noise_factors(shape, cv, seed=1000, distribution="lognormal")

    assert np.array_equal(arr1, arr2)
    assert not np.array_equal(arr1, arr3)


def test_invalid_noise_parameters():
    """Ensure invalid CV or distribution raises ValueError."""
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="non-negative"):
        lognormal_factors((5, 5), -0.1, rng)

    with pytest.raises(ValueError, match="non-negative"):
        gamma_factors((5, 5), -0.1, rng)

    with pytest.raises(ValueError, match="Unknown distribution"):
        generate_noise_factors((5, 5), 0.2, seed=1, distribution="cauchy")


def test_assignment_to_matrix_toy_example():
    """Verify conversion of station assignment dict to binary matrix."""
    # Toy example: 5 tasks, 2 stations (1-based indices)
    # Station 1: tasks 1, 2, 4
    # Station 2: tasks 3, 5
    assignments = {
        1: [1, 2, 4],
        2: [3, 5],
    }
    n_tasks = 5
    A = assignment_to_matrix(assignments, n_tasks)

    expected = np.array(
        [
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
        ]
    )
    assert np.array_equal(A, expected)


def test_build_station_times_toy_example():
    """Test build_station_times against hand-computed toy calculation."""
    # 5 tasks: nominal times = [10, 20, 30, 15, 25]
    # Station 1: tasks 1, 2, 4 -> nominal load = 10 + 20 + 15 = 45
    # Station 2: tasks 3, 5    -> nominal load = 30 + 25 = 55
    assignments = {1: [1, 2, 4], 2: [3, 5]}
    base_times = np.array([10, 20, 30, 15, 25], dtype=float)

    # 1 replication, 2 units, 5 tasks
    # Unit 0: noise factors all 1.0 -> station loads exactly 45, 55
    # Unit 1: noise factors [1.1, 0.9, 1.2, 1.0, 0.8]
    # Unit 1 Station 1: 10*1.1 + 20*0.9 + 15*1.0 = 11 + 18 + 15 = 44.0
    # Unit 1 Station 2: 30*1.2 + 25*0.8 = 36 + 20 = 56.0
    noise_factors = np.array(
        [
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.1, 0.9, 1.2, 1.0, 0.8],
            ]
        ]
    )  # shape (1, 2, 5)

    st = build_station_times(assignments, base_times, noise_factors)
    assert st.shape == (1, 2, 2)

    # Unit 0
    assert np.isclose(st[0, 0, 0], 45.0)
    assert np.isclose(st[0, 0, 1], 55.0)

    # Unit 1
    assert np.isclose(st[0, 1, 0], 44.0)
    assert np.isclose(st[0, 1, 1], 56.0)


def test_crn_balance_comparison_reproducibility():
    """Verify that Common Random Numbers isolates balance effects cleanly."""
    n_tasks = 4
    base_times = np.array([10.0, 10.0, 10.0, 10.0])
    # Balance 1: [1, 2], [3, 4]
    b1 = {1: [1, 2], 2: [3, 4]}
    # Balance 2: [1, 3], [2, 4]
    b2 = {1: [1, 3], 2: [2, 4]}

    shape = (5, 50, n_tasks)
    noise = generate_noise_factors(shape, cv=0.2, seed=777)

    st1 = build_station_times(b1, base_times, noise)
    st2 = build_station_times(b2, base_times, noise)

    assert st1.shape == (5, 50, 2)
    assert st2.shape == (5, 50, 2)

    # Station times should differ because assignment grouping differs,
    # but the total line work per unit (sum across stations) must be strictly identical!
    total_line_work_1 = st1.sum(axis=-1)
    total_line_work_2 = st2.sum(axis=-1)
    assert np.allclose(total_line_work_1, total_line_work_2)
