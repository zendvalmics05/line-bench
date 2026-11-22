"""Tests for Phase 3.3 verification of CP-SAT exact SALBP-1 model against published optima."""

import pytest
import yaml

from linebench.exact.cpsat_salbp1 import solve_salbp1_cpsat
from linebench.instances.checker import check_solution
from linebench.instances.loader import Instance
from linebench.instances.subset import load_subset_instance


def create_bowman8_instance() -> Instance:
    """Bowman (1960) 8-task benchmark instance with cycle time c=20.

    Optimal number of stations is known to be m* = 5.
    """
    task_times = {
        1: 11,
        2: 17,
        3: 9,
        4: 5,
        5: 8,
        6: 12,
        7: 10,
        8: 3,
    }
    precedences = [
        (1, 2),
        (2, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (7, 8),
    ]
    return Instance(
        name="bowman8",
        num_tasks=8,
        cycle_time=20,
        task_times=task_times,
        precedences=precedences,
    )


def test_cpsat_bowman8_optimal():
    """Verify that CP-SAT finds proven optimal m*=5 for Bowman-8 with c=20."""
    inst = create_bowman8_instance()
    sol = solve_salbp1_cpsat(
        inst,
        time_limit_seconds=10.0,
        num_workers=2,
        random_seed=42,
        validate_with_checker=True,
    )

    assert sol.is_feasible is True
    assert sol.num_stations == 5
    assert sol.is_proven_optimal is True
    assert sol.status == "OPTIMAL"

    # Direct validation with checker
    chk = check_solution(inst, sol.station_assignments, claimed_stations=5)
    assert chk.is_feasible is True


def test_cpsat_thread_and_seed_control():
    """Verify solver runs cleanly with specified seeds and worker threads."""
    inst = create_bowman8_instance()
    sol1 = solve_salbp1_cpsat(inst, time_limit_seconds=5.0, num_workers=1, random_seed=123)
    sol2 = solve_salbp1_cpsat(inst, time_limit_seconds=5.0, num_workers=2, random_seed=123)

    assert sol1.is_feasible is True
    assert sol2.is_feasible is True
    assert sol1.num_stations == 5
    assert sol2.num_stations == 5


def test_cpsat_with_heuristic_hint():
    """Verify solver accepts and respects heuristic hint and upper bounds."""
    inst = create_bowman8_instance()
    # Provide a feasible hint
    hint = {1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 5, 8: 5}  # 5 stations
    sol = solve_salbp1_cpsat(
        inst,
        heuristic_hint=hint,
        upper_bound_stations=5,
        time_limit_seconds=5.0,
    )
    assert sol.is_feasible is True
    assert sol.num_stations == 5
    assert sol.is_proven_optimal is True


@pytest.mark.parametrize(
    "instance_id,expected_m",
    [
        ("scholl_bowman_c20", 5),
        ("scholl_jaeschke_c6", 8),
        ("scholl_jaeschke_c7", 7),
        ("scholl_jaeschke_c8", 6),
        ("scholl_jackson_c7", 8),
        ("scholl_jackson_c9", 6),
        ("scholl_jackson_c10", 5),
    ],
)
def test_cpsat_reproduces_published_optima_classical(instance_id: str, expected_m: int):
    """Phase 3.3: Verify CP-SAT reproduces published optima on classical benchmark instances."""
    with open("configs/instances.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bench_dict = {x["id"]: x for x in cfg["benchmark_instances"]}
    assert instance_id in bench_dict, f"Instance {instance_id} not found in benchmark_instances"

    entry = bench_dict[instance_id]
    inst = load_subset_instance(entry)

    sol = solve_salbp1_cpsat(
        inst,
        time_limit_seconds=10.0,
        num_workers=2,
        random_seed=42,
        validate_with_checker=True,
    )

    assert sol.is_feasible is True
    assert sol.num_stations == expected_m
    assert sol.is_proven_optimal is True
    assert sol.status == "OPTIMAL"

    # Independent check
    chk = check_solution(inst, sol.station_assignments, claimed_stations=expected_m)
    assert chk.is_feasible is True
    assert len(chk.errors) == 0


def test_secondary_objectives_so1_and_so2_bowman8():
    """Phase 3.4: Verify SO1 and SO2 secondary objectives on Bowman-8."""
    from linebench.exact.secondary import (
        solve_m_plus_1_min_max_load,
        solve_secondary_objective,
    )

    inst = create_bowman8_instance()

    # SO1: MinMaxLoad with m=5
    so1 = solve_secondary_objective(
        inst,
        target_stations=5,
        objective_type="SO1",
        time_limit_seconds=10.0,
        validate_with_checker=True,
    )
    assert so1.is_feasible is True
    assert so1.target_stations == 5
    assert so1.status == "OPTIMAL"
    assert so1.max_load == 17  # Task 2 has length 17, so max_load cannot be lower than 17
    chk1 = check_solution(inst, so1.station_assignments, claimed_stations=5)
    assert chk1.is_feasible is True

    # SO2: Smoothness (minimize sum of squared idle times) with m=5
    so2 = solve_secondary_objective(
        inst,
        target_stations=5,
        objective_type="SO2",
        time_limit_seconds=10.0,
        validate_with_checker=True,
    )
    assert so2.is_feasible is True
    assert so2.target_stations == 5
    assert so2.status == "OPTIMAL"
    assert so2.sum_squared_idle > 0
    chk2 = check_solution(inst, so2.station_assignments, claimed_stations=5)
    assert chk2.is_feasible is True

    # m+1 variant: SO1 with m=6
    m_plus_1 = solve_m_plus_1_min_max_load(
        inst,
        optimal_stations=5,
        time_limit_seconds=10.0,
        validate_with_checker=True,
    )
    assert m_plus_1.is_feasible is True
    assert m_plus_1.target_stations == 6
    assert m_plus_1.status == "OPTIMAL"
    chk3 = check_solution(inst, m_plus_1.station_assignments, claimed_stations=6)
    assert chk3.is_feasible is True


def test_run_exact_benchmark_small_limit(tmp_path):
    """Phase 3.5: Verify run_exact_benchmark with resumption, checkpointing and metadata."""
    from linebench.experiments.run_exact import run_exact_benchmark

    out_file = str(tmp_path / "exact_runs_test.jsonl")

    # Run limit of 2 instances
    res1 = run_exact_benchmark(
        instances_yaml="configs/instances.yaml",
        solver_yaml="configs/solver.yaml",
        results_path=out_file,
        instance_limit=2,
        time_limit_override=5.0,
        log_progress=False,
    )
    assert res1["status"] == "COMPLETED"
    assert res1["processed_now"] == 2
    assert "python_version" in res1["environment"]
    assert "ortools_version" in res1["environment"]

    # Verify resumption: running again should skip already processed
    res2 = run_exact_benchmark(
        instances_yaml="configs/instances.yaml",
        solver_yaml="configs/solver.yaml",
        results_path=out_file,
        instance_limit=2,
        resume=True,
        log_progress=False,
    )
    assert res2["processed_now"] == 0
    assert res2["skipped_resumed"] == 2


def test_build_balance_population_bowman8():
    """Phase 3.6: Verify balance population assembly for a simulation instance."""
    from linebench.experiments.balances import build_balance_population_for_instance

    inst = create_bowman8_instance()
    pop = build_balance_population_for_instance(
        inst,
        optimal_or_best_stations=5,
        max_random_distinct=3,
        random_passes=100,
        solver_time_limit=5.0,
    )

    assert pop.instance_id == "bowman8"
    assert pop.optimal_or_best_stations == 5
    assert len(pop.balances) >= 4  # Heur_best, Exact, SO1, SO2, m+1, plus randoms

    types = {b.balance_type for b in pop.balances}
    assert "HEUR_BEST" in types
    assert "EXACT" in types
    assert "SO1" in types
    assert "SO2" in types
    assert "M_PLUS_1" in types

    # Check that all balances are feasible and have stable IDs
    for b in pop.balances:
        assert b.is_feasible is True
        assert b.balance_id.startswith("bowman8__")
        chk = check_solution(inst, b.station_assignments, claimed_stations=b.num_stations)
        assert chk.is_feasible is True
