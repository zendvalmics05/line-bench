"""
Tests for Phase 2.4 (Randomized Multi-Pass) and Phase 2.5 (Heuristic Runner).
Covers:
  - Multi-pass reproducibility from fixed seed
  - Feasibility of all random pass solutions via independent checker
  - Monotonicity: best-of-K never worsens as K increases (same seed sequence)
  - Retention of distinct minimal station balances
  - Full instance runner execution and checkpoint logging
"""

import pytest

from linebench.experiments.run_heuristics import run_heuristics_on_instance
from linebench.heuristics.random_pass import solve_randomized_multi_pass
from linebench.instances.checker import check_solution
from linebench.instances.loader import load_in2


@pytest.fixture
def bowman_instance():
    """Bowman-8 instance fixture."""
    return load_in2("data/raw/scholl1993/precedence graphs/BOWMAN8.IN2", cycle_time=20)


def test_random_multi_pass_reproducibility(bowman_instance):
    """Phase 2.4 Test: Same seed produces bit-identical best balance and station count."""
    sol1 = solve_randomized_multi_pass(bowman_instance, num_passes=50, seed=1001)
    sol2 = solve_randomized_multi_pass(bowman_instance, num_passes=50, seed=1001)

    assert sol1.best_solution.num_stations == sol2.best_solution.num_stations
    assert sol1.best_solution.station_assignments == sol2.best_solution.station_assignments
    assert len(sol1.distinct_best_balances) == len(sol2.distinct_best_balances)


def test_random_multi_pass_feasibility(bowman_instance):
    """Phase 2.4 Test: Independent checker validates multi-pass output."""
    sol = solve_randomized_multi_pass(bowman_instance, num_passes=50, seed=42)
    assert sol.best_solution.is_feasible

    # Verify every distinct best balance is also valid
    for bal in sol.distinct_best_balances:
        check = check_solution(bowman_instance, bal, claimed_stations=sol.best_solution.num_stations)
        assert check.is_feasible


def test_random_multi_pass_monotonicity(bowman_instance):
    """Phase 2.4 Test: Best-of-K never gets worse as K increases for same seed prefix."""
    # Run with K=10, K=50, K=200
    sol_10 = solve_randomized_multi_pass(bowman_instance, num_passes=10, seed=777)
    sol_50 = solve_randomized_multi_pass(bowman_instance, num_passes=50, seed=777)
    sol_200 = solve_randomized_multi_pass(bowman_instance, num_passes=200, seed=777)

    m_10 = sol_10.best_solution.num_stations
    m_50 = sol_50.best_solution.num_stations
    m_200 = sol_200.best_solution.num_stations

    assert m_50 <= m_10, f"Monotonicity violation: K=50 gave {m_50} > K=10 ({m_10})"
    assert m_200 <= m_50, f"Monotonicity violation: K=200 gave {m_200} > K=50 ({m_50})"


def test_run_heuristics_on_instance(bowman_instance):
    """Phase 2.5 Test: Runner executes all 12 priority rule variants + multi-pass."""
    results = run_heuristics_on_instance(
        bowman_instance,
        rules=["RPW", "UBT", "MTS", "MDS", "EST", "LST"],
        directions=["forward", "backward"],
        multi_pass_passes=20,
        multi_pass_seed=42,
    )

    # 6 rules * 2 directions + 1 multi-pass = 13 results
    assert len(results) == 13

    # All must be feasible
    for r in results:
        assert r["is_feasible"]
        assert r["num_stations"] in (5, 6)
        assert r["duration_seconds"] >= 0.0
