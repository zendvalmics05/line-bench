"""
Tests for Phase 2.1 (Constructor) and Phase 2.2 (Priority Rules).
Covers:
  - Six priority rules: RPW, UBT, MTS, MDS, EST, LST.
  - Hand-solved instance with verified step-by-step assignments.
  - Deterministic tie-breaking verification (task index ascending).
  - Validation of all heuristic solutions by independent checker.
"""

import pytest

from linebench.heuristics.construct import solve_station_oriented
from linebench.heuristics.priorities import (
    PRIORITY_RULE_REGISTRY,
    compute_est_priorities,
    compute_lst_priorities,
    compute_mds_priorities,
    compute_mts_priorities,
    compute_rpw_priorities,
    compute_ubt_priorities,
)
from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance, load_in2


@pytest.fixture
def hand_solved_instance() -> Instance:
    """Hand-crafted test instance.

    Precedence graph:
      1 (t=5) -> 2 (t=3) -> 4 (t=4)
      3 (t=6) -> 4 (t=4)
    Cycle time c = 8.
    Total tasks n = 4.
    Precedences: (1, 2), (2, 4), (3, 4).
    Task times: 1:5, 2:3, 3:6, 4:4.

    Priorities by rule:
      RPW:
        1: 5 + 3 + 4 = 12
        2: 3 + 4 = 7
        3: 6 + 4 = 10
        4: 4
      UBT (durations):
        1: 5, 2: 3, 3: 6, 4: 4
      MTS (all successors):
        1: |{2, 4}| = 2
        2: |{4}| = 1
        3: |{4}| = 1
        4: |{}| = 0
      MDS (immediate successors):
        1: |{2}| = 1
        2: |{4}| = 1
        3: |{4}| = 1
        4: |{}| = 0
    """
    times = {1: 5, 2: 3, 3: 6, 4: 4}
    precedences = {(1, 2), (2, 4), (3, 4)}
    return Instance(
        name="hand_example",
        num_tasks=4,
        cycle_time=8,
        task_times=times,
        precedences=precedences,
    )


def test_priority_value_calculations(hand_solved_instance):
    """Verify exact numerical values computed for each priority rule."""
    graph = PrecedenceGraph(hand_solved_instance)

    # RPW: 1:12, 2:7, 3:10, 4:4
    rpw = compute_rpw_priorities(hand_solved_instance, graph)
    assert rpw == {1: 12.0, 2: 7.0, 3: 10.0, 4: 4.0}

    # UBT: 1:5, 2:3, 3:6, 4:4
    ubt = compute_ubt_priorities(hand_solved_instance, graph)
    assert ubt == {1: 5.0, 2: 3.0, 3: 6.0, 4: 4.0}

    # MTS: 1:2, 2:1, 3:1, 4:0
    mts = compute_mts_priorities(hand_solved_instance, graph)
    assert mts == {1: 2.0, 2: 1.0, 3: 1.0, 4: 0.0}

    # MDS: 1:1, 2:1, 3:1, 4:0
    mds = compute_mds_priorities(hand_solved_instance, graph)
    assert mds == {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0}

    # EST & LST check keys exist and are finite
    est = compute_est_priorities(hand_solved_instance, graph)
    assert len(est) == 4
    lst = compute_lst_priorities(hand_solved_instance, graph)
    assert len(lst) == 4


def test_rpw_hand_solved_trace(hand_solved_instance):
    """Trace RPW step-by-step on hand-solved instance.

    c = 8.
    Step 1 (Station 1, remaining = 8):
      Available tasks: 1 (t=5, RPW=12), 3 (t=6, RPW=10).
      Both fit. Max RPW is task 1.
      Assign 1 to Station 1. Load = 5, remaining = 3.

    Step 2 (Station 1, remaining = 3):
      Eligible: task 2 (predecessor 1 is done, t=3, RPW=7).
      Task 3 (t=6) does not fit (6 > 3).
      Assign 2 to Station 1. Load = 5 + 3 = 8, remaining = 0.
      Station 1 full: S1 = [1, 2].

    Step 3 (Station 2, remaining = 8):
      Eligible: task 3 (predecessors done, t=6, RPW=10).
      Task 4 is not eligible because predecessor 3 is not yet assigned.
      Assign 3 to Station 2. Load = 6, remaining = 2.

    Step 4 (Station 2, remaining = 2):
      Eligible: task 4 (predecessors 2 and 3 are done, t=4).
      Does not fit (4 > 2).
      Station 2 closed: S2 = [3].

    Step 5 (Station 3, remaining = 8):
      Assign 4 to Station 3. Load = 4.
      Station 3 closed: S3 = [4].

    Total stations: 3.
    """
    sol = solve_station_oriented(hand_solved_instance, priority_rule="RPW")
    assert sol.num_stations == 3
    assert sol.station_assignments[1] == [1, 2]
    assert sol.station_assignments[2] == [3]
    assert sol.station_assignments[3] == [4]
    assert sol.is_feasible


def test_ubt_hand_solved_trace(hand_solved_instance):
    """Trace UBT (longest task time) step-by-step.

    c = 8.
    Step 1 (Station 1, remaining = 8):
      Available: task 1 (t=5), task 3 (t=6).
      Max duration is task 3 (t=6).
      Assign 3 to Station 1. Load = 6, remaining = 2.

    Step 2 (Station 1, remaining = 2):
      Available: task 1 (t=5). Does not fit (5 > 2).
      Station 1 closed: S1 = [3].

    Step 3 (Station 2, remaining = 8):
      Available: task 1 (t=5).
      Assign 1 to Station 2. Load = 5, remaining = 3.
      Now task 2 is eligible (t=3). Fits! (5 + 3 = 8).
      Assign 2 to Station 2.
      Station 2 closed: S2 = [1, 2].

    Step 4 (Station 3, remaining = 8):
      Assign task 4 (t=4) to Station 3: S3 = [4].

    Total stations: 3.
    """
    sol = solve_station_oriented(hand_solved_instance, priority_rule="UBT")
    assert sol.num_stations == 3
    assert sol.station_assignments[1] == [3]
    assert sol.station_assignments[2] == [1, 2]
    assert sol.station_assignments[3] == [4]
    assert sol.is_feasible


def test_deterministic_tie_breaking():
    """Verify tie-breaking rule: smallest task index j is selected on equal priority."""
    # Two tasks with no precedence, identical times, identical priorities
    times = {1: 4, 2: 4}
    inst = Instance(name="tie_test", num_tasks=2, cycle_time=10, task_times=times)

    # When both fit, task 1 must be picked before task 2
    sol = solve_station_oriented(inst, priority_rule="UBT")
    assert sol.station_assignments[1] == [1, 2]


def test_all_priority_rules_on_bowman():
    """Verify that all six priority rules run successfully on Bowman-8 and pass independent checker."""
    bowman = load_in2("data/raw/scholl1993/precedence graphs/BOWMAN8.IN2", cycle_time=20)

    for rule_name in PRIORITY_RULE_REGISTRY:
        sol = solve_station_oriented(bowman, priority_rule=rule_name, validate_with_checker=True)
        assert sol.is_feasible
        # Bowman-8 optimal is 5; heuristics should find 5 or 6 stations
        assert sol.num_stations in (5, 6), f"Rule {rule_name} got {sol.num_stations}"
        assert sum(sol.station_loads.values()) == 75  # Total work of Bowman-8


def test_backward_direction_on_bowman():
    """Phase 2.3: Verify backward solving direction on Bowman-8 for all six rules.

    Checks:
      1. Every backward solution is valid and feasible according to independent checker on original graph.
      2. Station loads match total work content.
    """
    bowman = load_in2("data/raw/scholl1993/precedence graphs/BOWMAN8.IN2", cycle_time=20)

    for rule_name in PRIORITY_RULE_REGISTRY:
        sol_back = solve_station_oriented(
            bowman,
            priority_rule=rule_name,
            direction="backward",
            validate_with_checker=True,
        )
        assert sol_back.direction == "backward"
        assert sol_back.is_feasible
        assert sol_back.num_stations in (5, 6)
        assert sum(sol_back.station_loads.values()) == 75


def test_backward_hand_solved_trace(hand_solved_instance):
    """Phase 2.3: Trace backward direction step-by-step on hand-solved instance.

    Original instance:
      1 (t=5) -> 2 (t=3) -> 4 (t=4)
      3 (t=6) -> 4 (t=4)
      c = 8.
    Reversed instance:
      4 (t=4) -> 2 (t=3) -> 1 (t=5)
      4 (t=4) -> 3 (t=6)
    Solving RPW on reversed graph:
      RPW values on reversed graph:
        4: 4 + 3 + 5 + 6 = 18
        2: 3 + 5 = 8
        3: 6
        1: 5
      Step 1: Assign task 4 (t=4) to Station 1. Remaining capacity = 4.
      Step 2: Tasks 2 (t=3, RPW=8) and 3 (t=6, RPW=6) become eligible.
              Task 2 fits (3 <= 4), task 3 does not fit. Assign task 2.
              Station 1 load = 4 + 3 = 7. (S1 = [4, 2]).
      Step 3: Station 2 opened: tasks 1 (t=5) and 3 (t=6) eligible.
              Task 3 has higher RPW (6 > 5). Assign task 3. Load = 6. (S2 = [3]).
      Step 4: Station 3 opened: task 1 assigned. (S3 = [1]).
      Total stations m = 3.

    Mapping back to original forward line (station s -> m - s + 1):
      Station 1 = S3 = [1]
      Station 2 = S2 = [3]
      Station 3 = S1 = [4, 2] (or [2, 4])
    """
    sol_back = solve_station_oriented(
        hand_solved_instance, priority_rule="RPW", direction="backward"
    )
    assert sol_back.direction == "backward"
    assert sol_back.is_feasible
    assert sol_back.num_stations == 3
    # Check that task 1 is in station 1, task 3 in station 2, and tasks 2, 4 in station 3
    assert sol_back.station_assignments[1] == [1]
    assert sol_back.station_assignments[2] == [3]
    assert set(sol_back.station_assignments[3]) == {2, 4}

