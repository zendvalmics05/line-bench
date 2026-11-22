"""
Tests for Phase 1.5 (Lower Bounds) and Phase 1.6 (Station Bounds).
Covers:
  - Theoretical bounds LB1, LB2, and LB
  - Hand examples where LB2 > LB1
  - Earliest station E(j) and latest station L(j) bounds
  - Consistency checks on real benchmark instance (Bowman-8)
"""

import pytest

from linebench.instances.bounds import compute_lb1, compute_lb2, compute_lower_bound
from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance, load_in2
from linebench.instances.station_bounds import compute_station_bounds


def test_lower_bound_lb2_exceeds_lb1():
    """Hand-crafted instance where LB2 > LB1.

    Let c = 10.
    Suppose we have 4 tasks, each with duration 6.
    sum(t_j) = 24.
    LB1 = ceil(24 / 10) = 3 stations.

    Tasks with t_j > c / 2 (i.e. > 5): all 4 tasks!
    Because each task > 5, no two tasks can fit in any station of capacity 10.
    Therefore, at least 4 stations are required.
    LB2 = 4 + ceil(0 / 2) = 4 stations.
    Combined LB = max(3, 4) = 4.
    """
    times = {1: 6, 2: 6, 3: 6, 4: 6}
    inst = Instance(name="lb2_dominant", num_tasks=4, cycle_time=10, task_times=times)

    lb1 = compute_lb1(inst)
    lb2 = compute_lb2(inst)
    lb = compute_lower_bound(inst)

    assert lb1 == 3
    assert lb2 == 4
    assert lb == 4
    assert lb2 > lb1


def test_lower_bound_half_cycle_pairs():
    """Verify exact half-cycle task counting in LB2.

    Let c = 10 (c / 2 = 5).
    Tasks:
      - 2 tasks with t = 7 (> 5)
      - 3 tasks with t = 5 (== 5)
      - 1 task with t = 2 (< 5)
    Total work = 14 + 15 + 2 = 31.
    LB1 = ceil(31 / 10) = 4.

    LB2 calculation:
      strictly_greater (> 5): 2 tasks
      equal_half (== 5): 3 tasks -> ceil(3 / 2) = 2 stations
      LB2 = 2 + 2 = 4 stations.
    """
    times = {1: 7, 2: 7, 3: 5, 4: 5, 5: 5, 6: 2}
    inst = Instance(name="half_cycle_pairs", num_tasks=6, cycle_time=10, task_times=times)

    assert compute_lb1(inst) == 4
    assert compute_lb2(inst) == 4
    assert compute_lower_bound(inst) == 4


def test_station_bounds_hand_example():
    """Verify E(j) and L(j) calculations on a small chain DAG.

    Graph:
      1 (t=4) -> 2 (t=3) -> 3 (t=5)
    Cycle time c = 6.
    m_ub = 3.

    Calculations:
      Task 1:
        pred_work = t_1 = 4.  E(1) = ceil(4 / 6) = 1.
        succ_work = t_1 + t_2 + t_3 = 4 + 3 + 5 = 12.  L(1) = 3 + 1 - ceil(12 / 6) = 2.
        Station range for 1: [1, 2].

      Task 2:
        pred_work = t_1 + t_2 = 4 + 3 = 7.  E(2) = ceil(7 / 6) = 2.
        succ_work = t_2 + t_3 = 3 + 5 = 8.   L(2) = 3 + 1 - ceil(8 / 6) = 2.
        Station range for 2: [2, 2].

      Task 3:
        pred_work = t_1 + t_2 + t_3 = 12.   E(3) = ceil(12 / 6) = 2.
        succ_work = t_3 = 5.                L(3) = 3 + 1 - ceil(5 / 6) = 3.
        Station range for 3: [2, 3].
    """
    times = {1: 4, 2: 3, 3: 5}
    precedences = {(1, 2), (2, 3)}
    inst = Instance(name="chain_3", num_tasks=3, cycle_time=6, task_times=times, precedences=precedences)

    sb = compute_station_bounds(inst, upper_bound_stations=3)

    assert sb.earliest[1] == 1
    assert sb.latest[1] == 2

    assert sb.earliest[2] == 2
    assert sb.latest[2] == 2

    assert sb.earliest[3] == 2
    assert sb.latest[3] == 3


def test_station_bounds_bowman():
    """Verify station bounds on Bowman-8 with known optimum m_ub = 5."""
    bowman = load_in2("data/raw/scholl1993/precedence graphs/BOWMAN8.IN2", cycle_time=20)
    graph = PrecedenceGraph(bowman)
    sb = compute_station_bounds(bowman, graph=graph, upper_bound_stations=5)

    # Every task must satisfy E(j) <= L(j) when m_ub is a feasible station count
    for j in range(1, bowman.num_tasks + 1):
        assert 1 <= sb.earliest[j] <= sb.latest[j] <= 5

    # Task 1 is first source (no preds, pred_work = 11 <= 20) -> E(1) = 1
    assert sb.earliest[1] == 1
    # Task 8 is last sink (no succs, succ_work = 3 <= 20) -> L(8) = 5 + 1 - 1 = 5
    assert sb.latest[8] == 5

    # Check precedence ordering consistency: if u -> v, then E(u) <= E(v) and L(u) <= L(v)
    for u, v in bowman.precedences:
        assert sb.earliest[u] <= sb.earliest[v]
        assert sb.latest[u] <= sb.latest[v]


def test_station_bounds_invalid_m_ub():
    """Verify error on invalid m_ub."""
    times = {1: 2}
    inst = Instance(name="single", num_tasks=1, cycle_time=5, task_times=times)
    with pytest.raises(ValueError, match="upper_bound_stations must be positive"):
        compute_station_bounds(inst, upper_bound_stations=0)
