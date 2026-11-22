"""Tests for Phase 1.1 (Loader) and Phase 1.2 (Graph closure)."""

from pathlib import Path

import pytest

from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance, load_alb, load_in2


def test_instance_validation():
    """Verify validation logic in Instance dataclass."""
    # Negative cycle time
    with pytest.raises(ValueError, match="cycle_time must be positive"):
        Instance(name="test", num_tasks=2, cycle_time=-10, task_times={1: 5, 2: 5})

    # Task duration exceeds cycle time
    with pytest.raises(ValueError, match="exceeds cycle time"):
        Instance(name="test", num_tasks=2, cycle_time=10, task_times={1: 5, 2: 12})

    # Task ID indexing mismatch
    with pytest.raises(ValueError, match="Task index mismatch"):
        Instance(name="test", num_tasks=2, cycle_time=10, task_times={0: 5, 1: 5})

    # Self loop precedence
    with pytest.raises(ValueError, match="Self-loop"):
        Instance(
            name="test",
            num_tasks=2,
            cycle_time=10,
            task_times={1: 5, 2: 5},
            precedences={(1, 1)},
        )


def test_load_in2_bowman():
    """Verify loading Bowman-8 IN2 file."""
    in2_path = Path("data/raw/scholl1993/precedence graphs/BOWMAN8.IN2")
    inst = load_in2(in2_path, cycle_time=20)
    assert inst.num_tasks == 8
    assert inst.cycle_time == 20
    assert inst.task_times[1] == 11
    assert inst.task_times[2] == 17
    assert (1, 2) in inst.precedences
    assert (6, 8) in inst.precedences
    assert len(inst.precedences) == 8


def test_load_alb_sample():
    """Verify loading small ALB file from Otto et al. 2013."""
    alb_path = Path("data/raw/otto2013/small data set_n=20/small data set_n=20/instance_n=20_1.alb")
    inst = load_alb(alb_path)
    assert inst.num_tasks == 20
    assert inst.cycle_time == 1000
    assert inst.order_strength == pytest.approx(0.268, abs=1e-4)
    assert inst.task_times[1] == 142
    assert (1, 6) in inst.precedences


def test_graph_closure_hand_example():
    """Verify transitive closure and topological sort on hand-crafted DAG.

    Graph:
      1 -> 2 -> 4
      1 -> 3 -> 4
    Transitive closure relations:
      (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)
    """
    times = {1: 2, 2: 3, 3: 4, 4: 1}
    precedences = {(1, 2), (1, 3), (2, 4), (3, 4)}
    inst = Instance(name="diamond", num_tasks=4, cycle_time=10, task_times=times, precedences=precedences)
    pg = PrecedenceGraph(inst)

    # Immediate queries
    assert pg.immediate_successors(1) == {2, 3}
    assert pg.immediate_predecessors(4) == {2, 3}

    # Transitive closure queries
    assert pg.all_successors(1) == {2, 3, 4}
    assert pg.all_predecessors(4) == {1, 2, 3}
    assert pg.all_predecessors(2) == {1}
    assert pg.all_successors(2) == {4}
    assert pg.total_precedence_pairs() == 5

    # Sources & Sinks
    assert pg.sources() == [1]
    assert pg.sinks() == [4]

    # Topological order
    top_order = pg.topological_order()
    assert top_order == [1, 2, 3, 4]


def test_graph_cycle_detection():
    """Verify that cycles raise a ValueError in PrecedenceGraph."""
    times = {1: 2, 2: 3}
    precedences = {(1, 2), (2, 1)}
    inst = Instance(name="cyclic", num_tasks=2, cycle_time=10, task_times=times, precedences=precedences)
    with pytest.raises(ValueError, match="contains cycles"):
        PrecedenceGraph(inst)


def test_reversed_instance():
    """Verify graph reversal for backward heuristic direction."""
    times = {1: 2, 2: 3, 3: 4}
    precedences = {(1, 2), (2, 3)}
    inst = Instance(name="chain", num_tasks=3, cycle_time=10, task_times=times, precedences=precedences)
    pg = PrecedenceGraph(inst)

    rev_inst = pg.reversed_instance()
    assert rev_inst.precedences == {(2, 1), (3, 2)}
    rev_pg = PrecedenceGraph(rev_inst)
    assert rev_pg.sources() == [3]
    assert rev_pg.sinks() == [1]
    assert rev_pg.topological_order() == [3, 2, 1]
