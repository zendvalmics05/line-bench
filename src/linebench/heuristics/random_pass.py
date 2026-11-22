"""
Randomized multi-pass constructive heuristic for SALBP-1.
Performs priority-biased random selection of eligible tasks over K passes.
Maintains the best solution (minimal stations) and collects distinct
equal-station-count solutions for the balance population.
Ref: Master Plan Appendix A3, Implementation Plan Phase 2.4.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from linebench.heuristics.construct import HeuristicSolution
from linebench.heuristics.priorities import PRIORITY_RULE_REGISTRY
from linebench.instances.checker import check_solution
from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance


@dataclass(frozen=True)
class MultiPassSolution:
    """Outcome of a randomized multi-pass heuristic run."""

    best_solution: HeuristicSolution
    num_passes: int
    seed: int
    duration_seconds: float
    # Distinct balances achieving the minimal station count (tuple representation for hashing)
    distinct_best_balances: list[dict[int, list[int]]] = field(default_factory=list)


def _canonical_balance_key(station_assignments: dict[int, list[int]]) -> tuple[tuple[int, ...], ...]:
    """Generates a hashable tuple representation of station task assignments."""
    return tuple(tuple(sorted(tasks)) for s, tasks in sorted(station_assignments.items()))


def solve_single_random_pass(
    instance: Instance,
    priorities: dict[int, float],
    rng: np.random.Generator,
    graph: PrecedenceGraph,
    temperature: float = 1.0,
) -> dict[int, list[int]]:
    """Generates a single feasible solution using priority-biased softmax sampling.

    For eligible candidates E, sampling probability is:
      p(j) = exp(priority(j) / (T * scale)) / sum_k(exp(priority(k) / (T * scale)))
    where scale normalizes the priority range of candidate tasks.
    """
    n = instance.num_tasks
    c = instance.cycle_time

    assigned_tasks: set[int] = set()
    station_assignments: dict[int, list[int]] = {}
    current_station = 1
    current_load = 0
    station_tasks: list[int] = []

    in_degrees = {j: len(graph.immediate_predecessors(j)) for j in range(1, n + 1)}

    while len(assigned_tasks) < n:
        eligible = [
            j
            for j in range(1, n + 1)
            if j not in assigned_tasks
            and in_degrees[j] == 0
            and (current_load + instance.task_times[j] <= c)
        ]

        if eligible:
            if len(eligible) == 1:
                chosen_task = eligible[0]
            else:
                # Softmax probabilities over eligible task priorities
                cand_priorities = np.array([priorities[j] for j in eligible], dtype=np.float64)
                p_span = np.ptp(cand_priorities)
                scale = p_span if p_span > 1e-6 else 1.0
                logits = (cand_priorities - np.max(cand_priorities)) / (temperature * scale)
                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits)

                chosen_idx = rng.choice(len(eligible), p=probs)
                chosen_task = eligible[chosen_idx]

            station_tasks.append(chosen_task)
            current_load += instance.task_times[chosen_task]
            assigned_tasks.add(chosen_task)

            for succ in graph.immediate_successors(chosen_task):
                in_degrees[succ] -= 1
        else:
            if station_tasks:
                station_assignments[current_station] = station_tasks
                current_station += 1
                station_tasks = []
                current_load = 0
            else:
                raise RuntimeError("Deadlock in random pass: no task fits in empty station")

    if station_tasks:
        station_assignments[current_station] = station_tasks

    return station_assignments


def solve_randomized_multi_pass(
    instance: Instance,
    num_passes: int = 500,
    seed: int = 1001,
    priority_rule: str = "RPW",
    max_distinct_best: int = 20,
    graph: PrecedenceGraph | None = None,
    validate_with_checker: bool = True,
) -> MultiPassSolution:
    """Executes K randomized passes and collects the best and distinct optimal balances.

    Args:
        instance: The SALBP-1 instance.
        num_passes: Total random iterations (default 500 from configs/solver.yaml).
        seed: Random seed for NumPy Generator.
        priority_rule: Base priority rule for sampling bias (default 'RPW').
        max_distinct_best: Maximum number of distinct best balances to retain.
        graph: Optional PrecedenceGraph.
        validate_with_checker: Whether to validate the best solution with the independent checker.

    Returns:
        MultiPassSolution dataclass.
    """
    start_time = time.perf_counter()

    if graph is None:
        graph = PrecedenceGraph(instance)

    rule_name = priority_rule.upper()
    if rule_name not in PRIORITY_RULE_REGISTRY:
        raise ValueError(f"Unknown priority rule '{priority_rule}'")
    priorities = PRIORITY_RULE_REGISTRY[rule_name](instance, graph)

    rng = np.random.default_rng(seed)

    best_m = float("inf")
    best_assignment: dict[int, list[int]] | None = None
    best_loads: dict[int, int] | None = None

    seen_best_keys: set[tuple[tuple[int, ...], ...]] = set()
    distinct_best_balances: list[dict[int, list[int]]] = []

    for _ in range(num_passes):
        assignment = solve_single_random_pass(instance, priorities, rng, graph)
        m = len(assignment)

        if m < best_m:
            best_m = m
            best_assignment = assignment
            best_loads = {s: sum(instance.task_times[t] for t in tasks) for s, tasks in assignment.items()}

            # New best found -> reset distinct pool
            seen_best_keys = {_canonical_balance_key(assignment)}
            distinct_best_balances = [assignment]
        elif m == best_m:
            canon_key = _canonical_balance_key(assignment)
            if canon_key not in seen_best_keys and len(distinct_best_balances) < max_distinct_best:
                seen_best_keys.add(canon_key)
                distinct_best_balances.append(assignment)

    if best_assignment is None or best_loads is None:
        raise RuntimeError("No solution found during randomized multi-pass")

    duration = time.perf_counter() - start_time

    # Validate best solution with independent checker
    is_feasible = True
    if validate_with_checker:
        check_res = check_solution(instance, best_assignment, claimed_stations=int(best_m))
        is_feasible = check_res.is_feasible
        if not is_feasible:
            raise RuntimeError(f"Random multi-pass produced invalid solution: {check_res.errors}")

    best_solution = HeuristicSolution(
        method=f"RANDOM_MULTI_PASS_{rule_name}",
        direction="forward",
        num_stations=int(best_m),
        station_assignments=best_assignment,
        station_loads=best_loads,
        duration_seconds=duration,
        is_feasible=is_feasible,
    )

    return MultiPassSolution(
        best_solution=best_solution,
        num_passes=num_passes,
        seed=seed,
        duration_seconds=duration,
        distinct_best_balances=distinct_best_balances,
    )
