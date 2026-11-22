"""Secondary objectives for SALBP-1 using CP-SAT.

Once an optimal (or target) station count m is determined:
  - SO1 (MinMaxLoad): Fix station count at m, minimize largest station load L_max:
        L_max = max_{s=1..m} (load_s)
        Minimize L_max
  - SO2 (Smoothness): Fix station count at m, minimize sum of squared idle times:
        idle_s = c - load_s
        Minimize sum_{s=1..m} (idle_s)^2
  - m+1 Variant: Fix station count at m + 1, minimize largest station load (SO1 with m+1).

Ref: Master Plan §5.2, Implementation Plan Phase 3.4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ortools.sat.python import cp_model

from linebench.instances.checker import check_solution
from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance
from linebench.instances.station_bounds import compute_station_bounds


@dataclass(frozen=True)
class SecondarySolution:
    """Outcome of optimizing a secondary objective with fixed station count."""

    instance_id: str
    objective_type: str  # "SO1_MIN_MAX_LOAD", "SO2_SMOOTHNESS", "M_PLUS_1_MIN_MAX_LOAD"
    target_stations: int
    status: str  # "OPTIMAL", "FEASIBLE", "TIMEOUT_FEASIBLE", "TIMEOUT_UNKNOWN", "INFEASIBLE"
    is_optimal: bool
    objective_value: float | None
    max_load: int
    smoothness_index: float  # sqrt(sum((c - load_s)^2) / m) or sum((c - load_s)^2)
    sum_squared_idle: int
    duration_seconds: float
    is_feasible: bool
    station_assignments: dict[int, list[int]] = field(default_factory=dict)
    station_loads: dict[int, int] = field(default_factory=dict)
    task_to_station: dict[int, int] = field(default_factory=dict)
    extra_stats: dict[str, Any] = field(default_factory=dict)


def solve_secondary_objective(
    instance: Instance,
    target_stations: int,
    objective_type: str = "SO1",  # "SO1" (MinMaxLoad) or "SO2" (Smoothness)
    time_limit_seconds: float = 30.0,
    num_workers: int = 2,
    random_seed: int = 42,
    initial_assignment: dict[int, int] | None = None,
    graph: PrecedenceGraph | None = None,
    validate_with_checker: bool = True,
) -> SecondarySolution:
    """Solves SALBP with fixed station count under a secondary objective.

    Objectives:
      - 'SO1': Minimize max_{s=1..m} load(s)
      - 'SO2': Minimize sum_{s=1..m} (c - load(s))^2

    Args:
        instance: The SALBP-1 problem instance.
        target_stations: Exactly m stations to use (contiguous 1..m).
        objective_type: 'SO1' or 'SO2'.
        time_limit_seconds: Maximum solver runtime.
        num_workers: Worker threads.
        random_seed: Deterministic seed.
        initial_assignment: Optional task->station hint mapping for warm start.
        graph: Optional PrecedenceGraph.
        validate_with_checker: Whether to verify feasibility via checker.

    Returns:
        SecondarySolution dataclass with assignments and objective metrics.
    """
    start_time = time.perf_counter()
    n = instance.num_tasks
    c = instance.cycle_time
    m = int(target_stations)

    if m <= 0:
        raise ValueError(f"target_stations must be positive, got {m}")

    if graph is None:
        graph = PrecedenceGraph(instance)

    obj_key = objective_type.upper().strip()
    if obj_key not in ("SO1", "SO2"):
        raise ValueError(f"Unknown objective_type '{objective_type}'. Must be 'SO1' or 'SO2'.")

    # Station bounds for each task within [1, m]
    bounds = compute_station_bounds(instance, graph=graph, upper_bound_stations=m)

    model = cp_model.CpModel()

    # 1. Variables
    # station_var[j] in [e_j, l_j]
    station_var: dict[int, cp_model.IntVar] = {}
    for j in range(1, n + 1):
        e_j = max(1, bounds.earliest[j])
        l_j = min(m, bounds.latest[j])
        if e_j > l_j:
            e_j = 1
            l_j = m
        station_var[j] = model.NewIntVar(e_j, l_j, f"s_{j}")

    # x_var[(j, s)] boolean indicator: task j in station s
    x_var: dict[tuple[int, int], cp_model.BoolVar] = {}
    for j in range(1, n + 1):
        e_j = max(1, bounds.earliest[j])
        l_j = min(m, bounds.latest[j])
        if e_j > l_j:
            e_j = 1
            l_j = m
        for s in range(e_j, l_j + 1):
            x_var[(j, s)] = model.NewBoolVar(f"x_{j}_{s}")
            model.Add(station_var[j] == s).OnlyEnforceIf(x_var[(j, s)])
            model.Add(station_var[j] != s).OnlyEnforceIf(~x_var[(j, s)])

        # Each task assigned to exactly one station
        model.AddExactlyOne(x_var[(j, s)] for s in range(e_j, l_j + 1))

    # Precedence constraints: station_var[u] <= station_var[v]
    for u, v in instance.precedences:
        if u in station_var and v in station_var:
            model.Add(station_var[u] <= station_var[v])

    # Station load variables: load_var[s] in [0, c]
    load_var: dict[int, cp_model.IntVar] = {}
    for s in range(1, m + 1):
        tasks_can_use_s = [j for j in range(1, n + 1) if (j, s) in x_var]
        load_var[s] = model.NewIntVar(0, c, f"load_{s}")
        if tasks_can_use_s:
            model.Add(
                load_var[s] == sum(instance.task_times[j] * x_var[(j, s)] for j in tasks_can_use_s)
            )
        else:
            model.Add(load_var[s] == 0)

        # Enforce non-empty stations to ensure all m stations are utilized
        # (each station must have at least one task)
        if tasks_can_use_s:
            model.Add(sum(x_var[(j, s)] for j in tasks_can_use_s) >= 1)

    # 2. Objective Formulation
    if obj_key == "SO1":
        # MinMaxLoad: minimize L_max where load_var[s] <= L_max for all s
        # Theoretical lower bound on max load is ceil(sum(t_j) / m)
        tot_work = sum(instance.task_times.values())
        min_possible_max_load = (tot_work + m - 1) // m
        max_load_var = model.NewIntVar(min_possible_max_load, c, "max_load")
        for s in range(1, m + 1):
            model.Add(load_var[s] <= max_load_var)
        model.Minimize(max_load_var)
    else:  # SO2: Smoothness - minimize sum of squared idle times
        # idle_var[s] = c - load_var[s]
        # idle_sq_var[s] = idle_var[s] ** 2
        idle_sq_vars: list[cp_model.IntVar] = []
        for s in range(1, m + 1):
            idle_var = model.NewIntVar(0, c, f"idle_{s}")
            model.Add(idle_var == c - load_var[s])
            idle_sq = model.NewIntVar(0, c * c, f"idle_sq_{s}")
            model.AddMultiplicationEquality(idle_sq, [idle_var, idle_var])
            idle_sq_vars.append(idle_sq)

        model.Minimize(sum(idle_sq_vars))

    # 3. Warm start hint if provided
    if initial_assignment:
        for j, s in initial_assignment.items():
            if j in station_var and 1 <= s <= m:
                model.AddHint(station_var[j], s)
                if (j, s) in x_var:
                    model.AddHint(x_var[(j, s)], 1)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_workers = int(num_workers)
    solver.parameters.random_seed = int(random_seed)

    sat_status = solver.Solve(model)
    duration = time.perf_counter() - start_time

    status_str = solver.StatusName(sat_status)
    if sat_status == cp_model.OPTIMAL:
        status = "OPTIMAL"
        is_opt = True
    elif sat_status == cp_model.FEASIBLE:
        status = "TIMEOUT_FEASIBLE" if duration >= time_limit_seconds * 0.95 else "FEASIBLE"
        is_opt = False
    elif sat_status == cp_model.INFEASIBLE:
        return SecondarySolution(
            instance_id=instance.name,
            objective_type=obj_key,
            target_stations=m,
            status="INFEASIBLE",
            is_optimal=False,
            objective_value=None,
            max_load=c,
            smoothness_index=float("inf"),
            sum_squared_idle=0,
            duration_seconds=duration,
            is_feasible=False,
            extra_stats={"sat_status": "INFEASIBLE", "wall_time": duration},
        )
    else:
        return SecondarySolution(
            instance_id=instance.name,
            objective_type=obj_key,
            target_stations=m,
            status="TIMEOUT_UNKNOWN",
            is_optimal=False,
            objective_value=None,
            max_load=c,
            smoothness_index=float("inf"),
            sum_squared_idle=0,
            duration_seconds=duration,
            is_feasible=False,
            extra_stats={"sat_status": status_str, "wall_time": duration},
        )

    # Extract solution
    task_to_station: dict[int, int] = {}
    station_to_tasks: dict[int, list[int]] = {s: [] for s in range(1, m + 1)}
    for j in range(1, n + 1):
        s = int(solver.Value(station_var[j]))
        task_to_station[j] = s
        station_to_tasks[s].append(j)

    station_loads: dict[int, int] = {}
    for s in range(1, m + 1):
        station_loads[s] = int(solver.Value(load_var[s]))

    max_l = max(station_loads.values()) if station_loads else 0
    sum_sq_idle = sum((c - l_val) ** 2 for l_val in station_loads.values())
    smoothness_idx = (sum_sq_idle / m) ** 0.5

    chk = None
    is_feas = True
    if validate_with_checker:
        chk = check_solution(instance, station_to_tasks, claimed_stations=m)
        is_feas = chk.is_feasible

    obj_val = float(solver.ObjectiveValue())

    return SecondarySolution(
        instance_id=instance.name,
        objective_type=obj_key,
        target_stations=m,
        status=status,
        is_optimal=is_opt,
        objective_value=obj_val,
        max_load=max_l,
        smoothness_index=smoothness_idx,
        sum_squared_idle=sum_sq_idle,
        duration_seconds=duration,
        is_feasible=is_feas,
        station_assignments=station_to_tasks,
        station_loads=station_loads,
        task_to_station=task_to_station,
        extra_stats={
            "sat_status": status_str,
            "wall_time": duration,
            "checker_errors": chk.errors if chk and not chk.is_feasible else [],
        },
    )


def solve_m_plus_1_min_max_load(
    instance: Instance,
    optimal_stations: int,
    time_limit_seconds: float = 30.0,
    num_workers: int = 2,
    random_seed: int = 42,
    initial_assignment: dict[int, int] | None = None,
    graph: PrecedenceGraph | None = None,
    validate_with_checker: bool = True,
) -> SecondarySolution:
    """Solves the m+1 variant of SALBP minimizing the largest station load (SO1).

    Allocates exactly optimal_stations + 1 stations and balances the workload
    to minimize peak station utilization.

    Args:
        instance: The SALBP-1 problem instance.
        optimal_stations: The base optimal/best station count m.
        time_limit_seconds: Maximum solve time.
        num_workers: Worker threads.
        random_seed: Deterministic seed.
        initial_assignment: Optional initial warm start hint.
        graph: Optional PrecedenceGraph.
        validate_with_checker: Independent feasibility check.

    Returns:
        SecondarySolution for the (m + 1) station line.
    """
    m_plus_1 = optimal_stations + 1
    return solve_secondary_objective(
        instance=instance,
        target_stations=m_plus_1,
        objective_type="SO1",
        time_limit_seconds=time_limit_seconds,
        num_workers=num_workers,
        random_seed=random_seed,
        initial_assignment=initial_assignment,
        graph=graph,
        validate_with_checker=validate_with_checker,
    )
