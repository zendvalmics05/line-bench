"""
Tests for Phase 1.3 (Structure Measures) and Phase 1.4 (Independent Checker).
Includes verification on hand examples, measure calculations, valid solutions,
and deliberately corrupted solutions (overload, precedence violation, missing/duplicate tasks).
"""

import pytest

from linebench.instances.checker import check_solution
from linebench.instances.loader import Instance, load_in2
from linebench.instances.measures import compute_instance_measures


@pytest.fixture
def bowman_instance() -> Instance:
    """Fixture providing Bowman-8 instance (c=20)."""
    return load_in2("data/raw/scholl1993/precedence graphs/BOWMAN8.IN2", cycle_time=20)


@pytest.fixture
def small_diamond_instance() -> Instance:
    """Diamond DAG with 4 tasks:

      1 (t=4) -> 2 (t=3) -> 4 (t=2)
      1 (t=4) -> 3 (t=3) -> 4 (t=2)
    Cycle time c = 7.
    """
    times = {1: 4, 2: 3, 3: 3, 4: 2}
    precedences = {(1, 2), (1, 3), (2, 4), (3, 4)}
    return Instance(
        name="diamond",
        num_tasks=4,
        cycle_time=7,
        task_times=times,
        precedences=precedences,
    )


# -------------------------------------------------------------
# Phase 1.3: Structure Measures Tests
# -------------------------------------------------------------


def test_structure_measures_diamond(small_diamond_instance):
    """Verify structural measures on diamond graph."""
    m = compute_instance_measures(small_diamond_instance)
    assert m.num_tasks == 4
    assert m.cycle_time == 7
    assert m.total_work == 12
    # Transitive relations: (1,2), (1,3), (1,4), (2,4), (3,4) = 5 pairs
    # Max possible pairs = 4 * 3 / 2 = 6
    # Order strength = 5 / 6 = 0.833333
    assert m.order_strength == pytest.approx(5 / 6, abs=1e-4)
    assert m.num_sources == 1
    assert m.num_sinks == 1
    assert m.t_min == 2
    assert m.t_max == 4
    assert m.tightness_ratio == pytest.approx(7 / 4, abs=1e-4)
    # Bounds: sum(t)=12, c=7 -> lb1 = ceil(12/7) = 2
    # tasks > 3.5: {1} (count 1), tasks == 3.5: 0 -> lb2 = 1
    assert m.lb1 == 2
    assert m.lb2 == 1
    assert m.lb == 2


def test_structure_measures_bowman(bowman_instance):
    """Verify structural measures on Bowman-8 instance."""
    m = compute_instance_measures(bowman_instance)
    assert m.num_tasks == 8
    assert m.cycle_time == 20
    assert m.total_work == 75
    # lb1 = ceil(75 / 20) = 4
    # tasks with t > 10: task 2 (17), task 6 (12), task 1 (11) -> 3 tasks
    # tasks with t == 10: task 7 (10) -> 1 task
    # lb2 = 3 + ceil(1 / 2) = 4
    assert m.lb1 == 4
    assert m.lb2 == 4
    assert m.lb == 4


# -------------------------------------------------------------
# Phase 1.4: Independent Checker Tests
# -------------------------------------------------------------


def test_checker_valid_solution_bowman(bowman_instance):
    """Verify that a known optimal 5-station solution passes the checker."""
    # Optimal Bowman-8 balance:
    # S1: {1} (load 11)
    # S2: {2} (load 17)
    # S3: {3, 5} (load 9 + 8 = 17)
    # S4: {4, 7} (load 5 + 10 = 15)
    # S5: {6, 8} (load 12 + 3 = 15)
    valid_sol = {
        1: [1],
        2: [2],
        3: [3, 5],
        4: [4, 7],
        5: [6, 8],
    }
    res = check_solution(bowman_instance, valid_sol, claimed_stations=5)
    assert res.is_feasible
    assert res.num_stations == 5
    assert len(res.errors) == 0
    assert res.station_loads[1] == 11
    assert res.station_loads[2] == 17
    assert res.station_loads[3] == 17
    assert res.station_loads[4] == 15
    assert res.station_loads[5] == 15


def test_checker_task_to_station_format(bowman_instance):
    """Verify checker accepts task-to-station dictionary representation."""
    task_sol = {1: 1, 2: 2, 3: 3, 4: 4, 5: 3, 6: 5, 7: 4, 8: 5}
    res = check_solution(bowman_instance, task_sol)
    assert res.is_feasible
    assert res.num_stations == 5


def test_checker_rejects_missing_task(bowman_instance):
    """Deliberately omit task 8."""
    bad_sol = {
        1: [1],
        2: [2],
        3: [3, 5],
        4: [4, 7],
        5: [6],  # Missing task 8
    }
    res = check_solution(bowman_instance, bad_sol)
    assert not res.is_feasible
    assert any("Missing task assignments" in e for e in res.errors)


def test_checker_rejects_duplicate_task(bowman_instance):
    """Deliberately assign task 3 to two different stations."""
    bad_sol = {
        1: [1],
        2: [2, 3],  # Duplicate task 3
        3: [3, 5],
        4: [4, 7],
        5: [6, 8],
    }
    res = check_solution(bowman_instance, bad_sol)
    assert not res.is_feasible
    assert any("Duplicate assignment" in e for e in res.errors)


def test_checker_rejects_overloaded_station(bowman_instance):
    """Overload station 1 by assigning tasks 1 (11) and 2 (17) -> load 28 > 20."""
    bad_sol = {
        1: [1, 2],  # Load = 28 > c=20
        2: [3, 5],
        3: [4, 7],
        4: [6, 8],
    }
    res = check_solution(bowman_instance, bad_sol)
    assert not res.is_feasible
    assert any("capacity violated" in e for e in res.errors)


def test_checker_rejects_precedence_violation(bowman_instance):
    """Precedence pair 1 -> 2: Assign task 2 to station 1, and task 1 to station 2."""
    bad_sol = {
        1: [2],  # Task 2 assigned before task 1
        2: [1],  # Task 1 is predecessor of 2
        3: [3, 5],
        4: [4, 7],
        5: [6, 8],
    }
    res = check_solution(bowman_instance, bad_sol)
    assert not res.is_feasible
    assert any("Precedence violated" in e for e in res.errors)


def test_checker_rejects_wrong_claimed_stations(bowman_instance):
    """Verify claimed stations check."""
    valid_sol = {
        1: [1],
        2: [2],
        3: [3, 5],
        4: [4, 7],
        5: [6, 8],
    }
    # Actual stations = 5, but claim 4
    res = check_solution(bowman_instance, valid_sol, claimed_stations=4)
    assert not res.is_feasible
    assert any("Claimed station count mismatch" in e for e in res.errors)


def test_checker_rejects_non_contiguous_stations(bowman_instance):
    """Stations defined as 1, 2, 4, 5 (missing station 3)."""
    bad_sol = {
        1: [1],
        2: [2],
        4: [3, 5, 4, 7],
        5: [6, 8],
    }
    res = check_solution(bowman_instance, bad_sol)
    assert not res.is_feasible
    assert any("Non-contiguous" in e for e in res.errors)
