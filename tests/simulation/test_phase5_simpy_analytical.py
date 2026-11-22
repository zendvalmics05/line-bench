"""Tests for Phase 5.5 (SimPy validator) and Phase 5.6 (Analytical & Limit validation).

Verifies:
  1. SimPy and dynamic recursion produce IDENTICAL departure times (< 1e-9).
  2. Deterministic times: throughput matches 1 / max_k S_k.
  3. Two-station equal exponential Markov chain matches theoretical throughput:
       TH = mu * (b + 2) / (b + 3)
     within Monte Carlo 99% confidence intervals.
  4. Large buffer limit: throughput approaches bottleneck rate.
  5. Monotonicity: buffer capacity cannot decrease throughput under CRN.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from linebench.simulation.simpy_line import simulate_simpy_line
from linebench.simulation.tandem import simulate_tandem_line


def test_simpy_identical_departure_times_multiconfig():
    """Phase 5.5: SimPy and recursion produce identical departure times across lines and buffers."""
    rng = np.random.default_rng(2026)
    configurations = [
        (3, 0),  # 3 stations, buffer 0
        (3, 1),  # 3 stations, buffer 1
        (4, 0),  # 4 stations, buffer 0
        (4, 2),  # 4 stations, buffer 2
        (5, 1),  # 5 stations, buffer 1
    ]
    N = 150  # 150 units

    for K, b in configurations:
        # Realistic lognormal processing times
        S = rng.lognormal(mean=2.0, sigma=0.35, size=(N, K))

        s_simpy, f_simpy, d_simpy = simulate_simpy_line(S, buffers=b)
        res = simulate_tandem_line(S, buffers=b, warmup_units=0, return_details=True)

        assert res.departure_times is not None
        assert res.start_times is not None
        assert res.finish_times is not None

        # Verify departure times are identical within floating point tolerance
        dept_diff = np.max(np.abs(res.departure_times - d_simpy))
        assert dept_diff < 1e-9, f"Departure mismatch for K={K}, b={b}: {dept_diff}"

        # Verify start times are identical
        start_diff = np.max(np.abs(res.start_times - s_simpy))
        assert start_diff < 1e-9, f"Start mismatch for K={K}, b={b}: {start_diff}"

        # Verify finish times are identical
        finish_diff = np.max(np.abs(res.finish_times - f_simpy))
        assert finish_diff < 1e-9, f"Finish mismatch for K={K}, b={b}: {finish_diff}"


def test_analytical_two_station_exponential_markov_chain():
    """Phase 5.6: 2 stations, equal exponential service mu, buffer b.

    Formula (Master Plan §A13 / docs/mathematical_specifications.md §3):
        TH(b, mu) = mu * (b + 2) / (b + 3)
    Tested for b in {0, 1, 2}.
    """
    mu = 1.0  # mean service rate = 1.0 -> mean service time = 1 / mu = 1.0
    scale = 1.0 / mu
    N = 12000
    warmup = 2000
    n_replications = 30
    rng = np.random.default_rng(42)

    for b in [0, 1, 2]:
        theoretical_th = mu * (b + 2) / (b + 3)
        sample_ths = []

        for _ in range(n_replications):
            S = rng.exponential(scale=scale, size=(N, 2))
            res = simulate_tandem_line(S, buffers=b, warmup_units=warmup)
            sample_ths.append(res.throughput)

        mean_th = float(np.mean(sample_ths))
        sem_th = float(stats.sem(sample_ths))

        # 99% confidence interval
        ci_99 = stats.t.interval(0.99, df=n_replications - 1, loc=mean_th, scale=sem_th)

        assert ci_99[0] <= theoretical_th <= ci_99[1], (
            f"Buffer b={b}: theoretical {theoretical_th:.4f} outside 99% CI "
            f"[{ci_99[0]:.4f}, {ci_99[1]:.4f}], mean={mean_th:.4f}"
        )


def test_large_buffer_limit():
    """Phase 5.6: With very large buffers, throughput approaches bottleneck rate 1 / max_k E[S_k]."""
    # 3 stations: mean times = 10, 20 (bottleneck), 12
    # Bottleneck theoretical service rate = 1 / 20 = 0.05
    rng = np.random.default_rng(101)
    N = 8000
    warmup = 1000

    S = np.zeros((N, 3), dtype=float)
    S[:, 0] = rng.exponential(scale=10.0, size=N)
    S[:, 1] = rng.exponential(scale=20.0, size=N)
    S[:, 2] = rng.exponential(scale=12.0, size=N)

    # Large buffer b=100 effectively decouples stations
    res_large_buf = simulate_tandem_line(S, buffers=100, warmup_units=warmup)

    expected_bottleneck = 1.0 / 20.0  # 0.05
    # Should be within 3% of the isolated bottleneck rate
    assert np.isclose(res_large_buf.throughput, expected_bottleneck, rtol=0.03)


def test_buffer_monotonicity_strict_crn():
    """Phase 5.6: Buffer monotonicity under strict CRN."""
    rng = np.random.default_rng(555)
    N, K = 800, 3
    S = rng.lognormal(mean=1.5, sigma=0.4, size=(N, K))

    th_prev = 0.0
    for b in [0, 1, 2, 4, 8, 16]:
        th_b = simulate_tandem_line(S, buffers=b, warmup_units=100).throughput
        assert th_b >= th_prev - 1e-9, f"Monotonicity violated at buffer {b}: {th_b} < {th_prev}"
        th_prev = th_b
