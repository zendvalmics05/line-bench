"""Unit tests for Phase 5.3 (unpaced tandem recursion) and Phase 5.4 (paced model)."""

from __future__ import annotations

import numpy as np

from linebench.simulation.paced import (
    normal_approx_overload_probability,
    simulate_paced_line,
    simulate_paced_replications,
)
from linebench.simulation.tandem import (
    simulate_tandem_line,
    simulate_tandem_replications,
)


def test_deterministic_unpaced_matches_bottleneck():
    """Deterministic times: throughput equals 1 / max_k S_k (Master Plan §5.6 limit)."""
    # 3 stations with deterministic times: 10, 25 (bottleneck), 15
    N = 100
    S = np.zeros((N, 3), dtype=float)
    S[:, 0] = 10.0
    S[:, 1] = 25.0
    S[:, 2] = 15.0

    res = simulate_tandem_line(S, buffers=1, warmup_units=10)
    expected_th = 1.0 / 25.0  # 0.040

    assert np.isclose(res.throughput, expected_th, rtol=1e-3)
    # Bottleneck station 1 should have utilization ~ 1.0
    assert np.isclose(res.utilization[1], 1.0, atol=0.02)
    # Upstream station 0 should experience some blocking
    assert res.blocked_fraction[0] > 0.0
    # Downstream station 2 should experience starvation
    assert res.starved_fraction[2] > 0.0


def test_unpaced_hand_calculated_trace():
    """Hand-calculated departure times for 2 stations, 2 units, buffer b=0."""
    # S = [[10, 20], [15, 10]]
    # Unit 0:
    #   s(0,0)=0, f(0,0)=10, s(1,0)=10, f(1,0)=30, d(1,0)=30
    #   d(0,0) with b=0: unblock is d(1, -1)=0 -> d(0,0)=10
    # Unit 1:
    #   s(0,1)=max(0, d(0,0))=10, f(0,1)=10+15=25
    #   b=0: unblock for station 0 is d(1, 1 - 0 - 1) = d(1, 0) = 30
    #   d(0,1) = max(f(0,1)=25, d(1,0)=30) = 30  (station 0 blocked for 5 time units!)
    #   s(1,1)=max(d(0,1)=30, d(1,0)=30)=30, f(1,1)=30+10=40, d(1,1)=40
    S = np.array(
        [
            [10.0, 20.0],
            [15.0, 10.0],
        ]
    )
    res = simulate_tandem_line(S, buffers=0, warmup_units=0, return_details=True)

    assert res.departure_times is not None
    assert np.isclose(res.departure_times[0, 0], 10.0)
    assert np.isclose(res.departure_times[0, 1], 30.0)
    assert np.isclose(res.departure_times[1, 0], 30.0)  # blocked until unit 0 left station 1!
    assert np.isclose(res.departure_times[1, 1], 40.0)

    # Station 0 blocked time: for unit 1, finish was 25, departure was 30 -> 5 units blocked
    assert res.blocked_fraction[0] > 0.0


def test_buffer_monotonicity():
    """Buffer monotonicity: larger buffer never lowers throughput under identical service times."""
    rng = np.random.default_rng(42)
    # 4 stations, 500 units, lognormal noise
    N, K = 500, 4
    S = rng.lognormal(mean=2.0, sigma=0.4, size=(N, K))

    th_b0 = simulate_tandem_line(S, buffers=0, warmup_units=50).throughput
    th_b1 = simulate_tandem_line(S, buffers=1, warmup_units=50).throughput
    th_b3 = simulate_tandem_line(S, buffers=3, warmup_units=50).throughput
    th_b10 = simulate_tandem_line(S, buffers=10, warmup_units=50).throughput

    assert th_b1 >= th_b0 - 1e-9
    assert th_b3 >= th_b1 - 1e-9
    assert th_b10 >= th_b3 - 1e-9


def test_batch_replications_tandem():
    """Verify batch runner matches individual single-replication calls."""
    rng = np.random.default_rng(123)
    n_reps, N, K = 3, 200, 3
    S = rng.exponential(scale=10.0, size=(n_reps, N, K))

    batch_res = simulate_tandem_replications(S, buffers=2, warmup_units=20)

    for r in range(n_reps):
        single_res = simulate_tandem_line(S[r], buffers=2, warmup_units=20)
        assert np.isclose(batch_res["throughput"][r], single_res.throughput)
        assert np.allclose(batch_res["utilization"][r], single_res.utilization)


def test_deterministic_paced_matches_max_station():
    """Deterministic times: paced throughput equals 1 / max_k S_k."""
    N = 50
    cycle_time = 30.0
    S = np.zeros((N, 3), dtype=float)
    S[:, 0] = 12.0
    S[:, 1] = 28.0
    S[:, 2] = 20.0

    res = simulate_paced_line(S, cycle_time=cycle_time)
    expected_th = 1.0 / 28.0

    assert np.isclose(res.mean_cycle_length, 28.0)
    assert np.isclose(res.throughput, expected_th)
    # No station exceeds cycle_time=30
    assert np.all(res.station_overload_prob == 0.0)
    assert res.line_overload_prob == 0.0


def test_paced_overload_probability():
    """Verify overload probability when station loads exceed cycle time."""
    N = 100
    cycle_time = 25.0
    # Station 0 has 40 units at 30.0 (> 25) and 60 units at 20.0 (<= 25)
    S = np.full((N, 2), 20.0)
    S[:40, 0] = 30.0

    res = simulate_paced_line(S, cycle_time=cycle_time)

    assert np.isclose(res.station_overload_prob[0], 0.40)
    assert np.isclose(res.station_overload_prob[1], 0.0)
    assert np.isclose(res.line_overload_prob, 0.40)


def test_normal_approx_overload_probability():
    """Normal approximation agrees roughly with empirical monte carlo."""
    tasks = [10.0, 15.0, 12.0, 8.0, 5.0]  # mean sum = 50
    cycle_time = 55.0
    cv = 0.25

    p_approx = normal_approx_overload_probability(tasks, cycle_time, cv)
    assert 0.0 < p_approx < 0.5  # Mean 50 < cycle time 55

    # With higher mean load exceeding cycle time, p > 0.5
    p_high = normal_approx_overload_probability(tasks, cycle_time=45.0, cv=cv)
    assert p_high > 0.5

    # Boundary cases
    assert normal_approx_overload_probability([], 50.0, 0.2) == 0.0
    assert normal_approx_overload_probability([60.0], 50.0, 0.0) == 1.0
    assert normal_approx_overload_probability([40.0], 50.0, 0.0) == 0.0


def test_batch_replications_paced():
    """Batch paced simulation matches individual run."""
    rng = np.random.default_rng(99)
    n_reps, N, K = 4, 150, 3
    S = rng.lognormal(mean=2.5, sigma=0.3, size=(n_reps, N, K))
    c = 15.0

    batch = simulate_paced_replications(S, cycle_time=c)
    for r in range(n_reps):
        single = simulate_paced_line(S[r], cycle_time=c)
        assert np.isclose(batch["throughput"][r], single.throughput)
        assert np.isclose(batch["line_overload_prob"][r], single.line_overload_prob)
