"""
Station-oriented constructive heuristic scheme for SALBP-1.
Supports pluggable priority functions, forward and backward solving directions,
and deterministic tie-breaking (task ID ascending).
Ref: Master Plan Appendix A3, Implementation Plan Phase 2.1 & 2.3.
"""

import time
from dataclasses import dataclass

from linebench.heuristics.priorities import PRIORITY_RULE_REGISTRY
from linebench.instances.checker import check_solution
from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance


@dataclass(frozen=True)
class HeuristicSolution:
    """Represents the outcome of a constructive heuristic run."""

    method: str
    direction: str  # 'forward' or 'backward'
    num_stations: int
    station_assignments: dict[int, list[int]]  # station_id (1..m) -> [task_ids]
    station_loads: dict[int, int]  # station_id -> total_time
    duration_seconds: float
    is_feasible: bool


def solve_station_oriented(
    instance: Instance,
    priority_rule: str | dict[int, float],
    direction: str = "forward",
    graph: PrecedenceGraph | None = None,
    validate_with_checker: bool = True,
) -> HeuristicSolution:
    """Solves SALBP-1 using the station-oriented heuristic scheme in forward or backward direction.

    Directions (Phase 2.3):
      - 'forward': Assign tasks from station 1 to m using original graph precedences.
      - 'backward': Solve on the reversed precedence graph (edges u -> v become v -> u),
        then reverse the station order (station s in reversed graph maps to station m - s + 1).

    Args:
        instance: The SALBP-1 instance.
        priority_rule: Name of registered priority rule (e.g. 'RPW', 'UBT')
                       or precomputed dictionary mapping task_id -> priority float.
        direction: 'forward' or 'backward'.
        graph: Optional PrecedenceGraph for the forward instance (constructed if None).
        validate_with_checker: Whether to verify feasibility on the original instance.

    Returns:
        HeuristicSolution object with stations 1..m feasible for the original instance.
    """
    start_time = time.perf_counter()

    direction_norm = direction.strip().lower()
    if direction_norm not in ("forward", "backward"):
        raise ValueError(f"direction must be 'forward' or 'backward', got '{direction}'")

    if graph is None:
        graph = PrecedenceGraph(instance)

    # If solving backward, operate on reversed graph
    if direction_norm == "backward":
        solve_instance = graph.reversed_instance()
        solve_graph = PrecedenceGraph(solve_instance)
    else:
        solve_instance = instance
        solve_graph = graph

    n = solve_instance.num_tasks
    c = solve_instance.cycle_time

    # Resolve priority values
    if isinstance(priority_rule, str):
        rule_name = priority_rule.upper()
        if rule_name not in PRIORITY_RULE_REGISTRY:
            raise ValueError(
                f"Unknown priority rule '{priority_rule}'. Available: {list(PRIORITY_RULE_REGISTRY.keys())}"
            )
        priorities = PRIORITY_RULE_REGISTRY[rule_name](solve_instance, solve_graph)
    elif isinstance(priority_rule, dict):
        rule_name = "CUSTOM"
        priorities = priority_rule
    else:
        raise TypeError(f"priority_rule must be str or dict, got {type(priority_rule)}")

    assigned_tasks: set[int] = set()
    raw_station_assignments: dict[int, list[int]] = {}
    raw_station_loads: dict[int, int] = {}

    current_station = 1
    current_load = 0
    station_tasks: list[int] = []

    # In-degree tracking on the working graph
    in_degrees = {j: len(solve_graph.immediate_predecessors(j)) for j in range(1, n + 1)}

    while len(assigned_tasks) < n:
        # Eligible tasks: predecessors satisfied (in_degree == 0) and fits in station
        eligible = [
            j
            for j in range(1, n + 1)
            if j not in assigned_tasks
            and in_degrees[j] == 0
            and (current_load + solve_instance.task_times[j] <= c)
        ]

        if eligible:
            # Maximizing priority; deterministic tie-break: smallest task ID j
            best_task = max(eligible, key=lambda j: (priorities[j], -j))

            station_tasks.append(best_task)
            current_load += solve_instance.task_times[best_task]
            assigned_tasks.add(best_task)

            # Update in-degrees for immediate successors in the working graph
            for succ in solve_graph.immediate_successors(best_task):
                in_degrees[succ] -= 1
        else:
            # Close station and open next
            if station_tasks:
                raw_station_assignments[current_station] = station_tasks
                raw_station_loads[current_station] = current_load
                current_station += 1
                station_tasks = []
                current_load = 0
            else:
                raise RuntimeError("Deadlock in heuristic: no eligible task fits in an empty station")

    if station_tasks:
        raw_station_assignments[current_station] = station_tasks
        raw_station_loads[current_station] = current_load

    m = len(raw_station_assignments)

    # In backward mode, reverse the station order:
    # Station s in reversed graph maps to station (m - s + 1) in original line
    if direction_norm == "backward":
        final_assignments: dict[int, list[int]] = {}
        final_loads: dict[int, int] = {}
        for s in range(1, m + 1):
            target_station = m - s + 1
            final_assignments[target_station] = raw_station_assignments[s]
            final_loads[target_station] = raw_station_loads[s]
        # Sort keys 1..m
        final_assignments = {s: final_assignments[s] for s in sorted(final_assignments.keys())}
        final_loads = {s: final_loads[s] for s in sorted(final_loads.keys())}
    else:
        final_assignments = raw_station_assignments
        final_loads = raw_station_loads

    duration = time.perf_counter() - start_time

    # Validate against original instance using independent checker
    is_feasible = True
    if validate_with_checker:
        check_res = check_solution(instance, final_assignments, claimed_stations=m)
        is_feasible = check_res.is_feasible
        if not is_feasible:
            raise RuntimeError(
                f"Heuristic ({rule_name}, direction={direction_norm}) produced invalid solution: {check_res.errors}"
            )

    return HeuristicSolution(
        method=rule_name,
        direction=direction_norm,
        num_stations=m,
        station_assignments=final_assignments,
        station_loads=final_loads,
        duration_seconds=duration,
        is_feasible=is_feasible,
    )
