"""
CP-SAT exact formulation for SALBP-1.

Implements the standard constraint programming model for Simple Assembly Line
Balancing Problem 1 (SALBP-1):
  - Integer station variable per task within [E(j), L(j)] station bounds
  - Precedence constraints: station(u) <= station(v) for all (u, v) in precedences
  - Boolean assignment indicators x[j, s] channeling station(j) == s
  - Station capacity constraints: sum(t_j * x[j, s]) <= c for all stations s
  - Objective: minimize the number of stations m
  - Upper bound and warm-start hint from best heuristic solution
  - Parameters (time limit, worker threads, random seed) from solver config
  - Validated by independent checker

Ref: Master Plan §5.2, Implementation Plan Phase 3.1 & 3.2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ortools.sat.python import cp_model

from linebench.heuristics.construct import HeuristicSolution, solve_station_oriented
from linebench.instances.bounds import compute_lower_bound
from linebench.instances.checker import check_solution
from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance
from linebench.instances.station_bounds import compute_station_bounds


@dataclass(frozen=True)
class ExactSolution:
    """Outcome of solving SALBP-1 using CP-SAT."""

    instance_id: str
    status: str  # "OPTIMAL", "FEASIBLE", "INFEASIBLE", "TIMEOUT_FEASIBLE", "TIMEOUT_UNKNOWN", "UNKNOWN"
    is_proven_optimal: bool
    num_stations: int | None
    lower_bound: int
    best_bound: float | None
    duration_seconds: float
    is_feasible: bool
    station_assignments: dict[int, list[int]] = field(default_factory=dict)
    station_loads: dict[int, int] = field(default_factory=dict)
    task_to_station: dict[int, int] = field(default_factory=dict)
    extra_stats: dict[str, Any] = field(default_factory=dict)


def solve_salbp1_cpsat(
    instance: Instance,
    time_limit_seconds: float = 60.0,
    num_workers: int = 2,
    random_seed: int = 42,
    log_search_progress: bool = False,
    heuristic_hint: HeuristicSolution | dict[int, int] | None = None,
    upper_bound_stations: int | None = None,
    graph: PrecedenceGraph | None = None,
    validate_with_checker: bool = True,
) -> ExactSolution:
    """Solves SALBP-1 to optimality (or within time limit) using OR-Tools CP-SAT.

    Formulation:
      Variables:
        station[j] in [E(j), L(j)] for each task j in 1..n
        x[j, s] in {0, 1} indicating task j is in station s
        m in [LB, m_ub]: maximum station index used (objective)

      Constraints:
        1. Channeling: station[j] == s <=> x[j, s] == 1
        2. Exactly one station per task: sum_s x[j, s] == 1
        3. Capacity: sum_j (t_j * x[j, s]) <= c for each station s
        4. Precedence: station[u] <= station[v] for all (u, v) in precedences
        5. Objective bounding: station[j] <= m for all j

    Args:
        instance: The SALBP-1 problem instance.
        time_limit_seconds: Maximum solver runtime in seconds.
        num_workers: Number of parallel CP-SAT worker threads.
        random_seed: Random seed for deterministic search.
        log_search_progress: Whether CP-SAT prints search log to stdout.
        heuristic_hint: Optional initial feasible assignment to seed upper bound and warm start.
        upper_bound_stations: Optional known upper bound on stations.
        graph: Optional PrecedenceGraph (constructed if None).
        validate_with_checker: Verify final solution with independent checker.

    Returns:
        ExactSolution containing status, station count, assignments, and bounds.
    """
    start_time = time.perf_counter()
    n = instance.num_tasks
    c = instance.cycle_time

    if graph is None:
        graph = PrecedenceGraph(instance)

    # Theoretical lower bound
    lb = compute_lower_bound(instance)

    # Determine upper bound & warm start hint
    hint_assignment: dict[int, int] | None = None
    if isinstance(heuristic_hint, HeuristicSolution):
        hint_assignment = {}
        for s, tasks in heuristic_hint.station_assignments.items():
            for t in tasks:
                hint_assignment[t] = s
        if upper_bound_stations is None:
            upper_bound_stations = heuristic_hint.num_stations
    elif isinstance(heuristic_hint, dict):
        hint_assignment = heuristic_hint
        if upper_bound_stations is None and hint_assignment:
            upper_bound_stations = max(hint_assignment.values())

    # If no valid upper bound given, run a fast RPW forward heuristic
    if upper_bound_stations is None or hint_assignment is None:
        fast_heur = solve_station_oriented(
            instance,
            priority_rule="RPW",
            direction="forward",
            graph=graph,
            validate_with_checker=False,
        )
        if upper_bound_stations is None:
            upper_bound_stations = fast_heur.num_stations
        if hint_assignment is None:
            hint_assignment = {}
            for s, tasks in fast_heur.station_assignments.items():
                for t in tasks:
                    hint_assignment[t] = s

    m_ub = max(lb, upper_bound_stations)

    # If heuristic upper bound already matches theoretical lower bound,
    # the heuristic solution is already mathematically optimal!
    if m_ub == lb and hint_assignment is not None:
        # Check and return immediately as proven optimal
        chk = check_solution(instance, hint_assignment, claimed_stations=lb)
        duration = time.perf_counter() - start_time
        station_to_tasks: dict[int, list[int]] = {s: [] for s in range(1, lb + 1)}
        for task_id, st in hint_assignment.items():
            station_to_tasks[st].append(task_id)

        return ExactSolution(
            instance_id=instance.name,
            status="OPTIMAL",
            is_proven_optimal=True,
            num_stations=lb,
            lower_bound=lb,
            best_bound=float(lb),
            duration_seconds=duration,
            is_feasible=chk.is_feasible,
            station_assignments=station_to_tasks,
            station_loads=chk.station_loads,
            task_to_station=hint_assignment,
            extra_stats={"proven_by": "LB_match", "wall_time": duration},
        )

    # Compute earliest and latest station bounds [E(j), L(j)]
    bounds = compute_station_bounds(instance, graph=graph, upper_bound_stations=m_ub)

    # Build CP-SAT Model
    model = cp_model.CpModel()

    # 1. Variables:
    # station[j] is the station index (1..m_ub) for task j
    station_var: dict[int, cp_model.IntVar] = {}
    for j in range(1, n + 1):
        e_j = max(1, bounds.earliest[j])
        l_j = min(m_ub, bounds.latest[j])
        if e_j > l_j:
            # Overconstrained bound fallback
            e_j = 1
            l_j = m_ub
        station_var[j] = model.NewIntVar(e_j, l_j, f"s_{j}")

    # Objective variable m: number of stations
    m_var = model.NewIntVar(lb, m_ub, "num_stations")

    # Boolean assignment variables x[j, s] for station capacity
    # x[j, s] == 1 iff station[j] == s
    x_var: dict[tuple[int, int], cp_model.BoolVar] = {}
    for j in range(1, n + 1):
        e_j = max(1, bounds.earliest[j])
        l_j = min(m_ub, bounds.latest[j])
        if e_j > l_j:
            e_j = 1
            l_j = m_ub
        for s in range(e_j, l_j + 1):
            x_var[(j, s)] = model.NewBoolVar(f"x_{j}_{s}")
            # Channeling: x[j, s] <=> (station[j] == s)
            model.Add(station_var[j] == s).OnlyEnforceIf(x_var[(j, s)])
            model.Add(station_var[j] != s).OnlyEnforceIf(~x_var[(j, s)])

        # Each task assigned to exactly one station in its valid range
        model.AddExactlyOne(x_var[(j, s)] for s in range(e_j, l_j + 1))

    # 2. Objective bounding: station[j] <= m_var for all j
    for j in range(1, n + 1):
        model.Add(station_var[j] <= m_var)

    # 3. Precedence constraints: station[u] <= station[v]
    for u, v in instance.precedences:
        if u in station_var and v in station_var:
            model.Add(station_var[u] <= station_var[v])

    # 4. Capacity constraints per station s: sum(t_j * x[j, s]) <= c
    for s in range(1, m_ub + 1):
        tasks_can_use_s = [j for j in range(1, n + 1) if (j, s) in x_var]
        if tasks_can_use_s:
            model.Add(
                sum(instance.task_times[j] * x_var[(j, s)] for j in tasks_can_use_s) <= c
            )

    # 5. Objective: minimize m_var
    model.Minimize(m_var)

    # 6. Warm-start solution hint
    if hint_assignment:
        for j, s in hint_assignment.items():
            if j in station_var and s <= m_ub:
                model.AddHint(station_var[j], s)
                if (j, s) in x_var:
                    model.AddHint(x_var[(j, s)], 1)
        model.AddHint(m_var, max(hint_assignment.values()))

    # Configure Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_workers = int(num_workers)
    solver.parameters.random_seed = int(random_seed)
    solver.parameters.log_search_progress = bool(log_search_progress)

    # Solve
    sat_status = solver.Solve(model)
    duration = time.perf_counter() - start_time

    # Map solver status
    if sat_status == cp_model.OPTIMAL:
        solved_m = int(solver.ObjectiveValue())
        status = "OPTIMAL"
        is_optimal = True
    elif sat_status == cp_model.FEASIBLE:
        solved_m = int(solver.ObjectiveValue())
        status = "TIMEOUT_FEASIBLE" if duration >= time_limit_seconds * 0.95 else "FEASIBLE"
        is_optimal = (solved_m == lb)
    elif sat_status == cp_model.INFEASIBLE:
        return ExactSolution(
            instance_id=instance.name,
            status="INFEASIBLE",
            is_proven_optimal=False,
            num_stations=None,
            lower_bound=lb,
            best_bound=None,
            duration_seconds=duration,
            is_feasible=False,
            extra_stats={"sat_status": "INFEASIBLE", "wall_time": duration},
        )
    else:  # UNKNOWN / TIMEOUT with no solution
        return ExactSolution(
            instance_id=instance.name,
            status="TIMEOUT_UNKNOWN",
            is_proven_optimal=False,
            num_stations=None,
            lower_bound=lb,
            best_bound=solver.BestObjectiveBound() if solver.BestObjectiveBound() else float(lb),
            duration_seconds=duration,
            is_feasible=False,
            extra_stats={"sat_status": "UNKNOWN", "wall_time": duration},
        )

    # Extract assignment
    task_to_station: dict[int, int] = {}
    station_to_tasks: dict[int, list[int]] = {s: [] for s in range(1, solved_m + 1)}

    for j in range(1, n + 1):
        s = int(solver.Value(station_var[j]))
        task_to_station[j] = s
        if s not in station_to_tasks:
            station_to_tasks[s] = []
        station_to_tasks[s].append(j)

    # Compact station numbering if any empty middle station exists (safety check)
    used_stations = sorted(s for s in station_to_tasks if station_to_tasks[s])
    reindexed_station_to_tasks: dict[int, list[int]] = {}
    reindexed_task_to_station: dict[int, int] = {}
    for new_idx, old_idx in enumerate(used_stations, start=1):
        reindexed_station_to_tasks[new_idx] = sorted(station_to_tasks[old_idx])
        for t in station_to_tasks[old_idx]:
            reindexed_task_to_station[t] = new_idx

    final_m = len(used_stations)

    # Compute loads and validate
    chk = None
    is_feasible = True
    loads: dict[int, int] = {}
    if validate_with_checker:
        chk = check_solution(instance, reindexed_station_to_tasks, claimed_stations=final_m)
        is_feasible = chk.is_feasible
        loads = chk.station_loads
    else:
        for s, tasks in reindexed_station_to_tasks.items():
            loads[s] = sum(instance.task_times[t] for t in tasks)

    best_bound = solver.BestObjectiveBound() if solver.BestObjectiveBound() > 0 else float(lb)

    return ExactSolution(
        instance_id=instance.name,
        status="OPTIMAL" if (is_optimal or final_m == lb) else status,
        is_proven_optimal=(is_optimal or final_m == lb),
        num_stations=final_m,
        lower_bound=lb,
        best_bound=best_bound,
        duration_seconds=duration,
        is_feasible=is_feasible,
        station_assignments=reindexed_station_to_tasks,
        station_loads=loads,
        task_to_station=reindexed_task_to_station,
        extra_stats={
            "sat_status": solver.StatusName(sat_status),
            "user_time": solver.UserTime(),
            "wall_time": solver.WallTime(),
            "num_branches": solver.NumBranches(),
            "num_conflicts": solver.NumConflicts(),
            "checker_errors": chk.errors if chk and not chk.is_feasible else [],
        },
    )
