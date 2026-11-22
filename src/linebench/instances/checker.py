"""
Independent solution checker for assembly line balancing problems.
Written completely independently from any heuristic or exact solver method.
Verifies feasibility against:
  1. Complete and unique task coverage (every task assigned exactly once).
  2. Station capacity constraints (sum of task times <= cycle time for all stations).
  3. Precedence constraints (station(pred) <= station(succ) for all relations).
  4. Station numbering consistency (1 to m contiguous with non-empty stations).
"""

from dataclasses import dataclass, field

from linebench.instances.loader import Instance


@dataclass(frozen=True)
class CheckResult:
    """Outcome of validating a balancing solution."""

    is_feasible: bool
    num_stations: int
    station_loads: dict[int, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        """Convenience method to raise ValueError if the solution is invalid."""
        if not self.is_feasible:
            err_msg = "; ".join(self.errors)
            raise ValueError(f"Solution validation failed: {err_msg}")


def check_solution(
    instance: Instance,
    solution: dict[int, list[int]] | dict[int, int],
    claimed_stations: int | None = None,
) -> CheckResult:
    """Verifies a line balancing solution independently.

    Accepts solution in either of two standard representations:
      - Station-to-tasks mapping: {station_id (1..m): [task_ids]}
      - Task-to-station mapping: {task_id (1..n): station_id (1..m)}

    Args:
        instance: The Instance being solved.
        solution: Mapping of station->tasks or task->station.
        claimed_stations: Optional expected station count m to verify.

    Returns:
        CheckResult containing feasibility boolean, station loads, and error details.
    """
    errors: list[str] = []
    n = instance.num_tasks
    c = instance.cycle_time

    # Normalize solution format to:
    #   task_to_station: dict[task_id, station_id]
    #   station_to_tasks: dict[station_id, list[task_ids]]
    task_to_station: dict[int, int] = {}
    station_to_tasks: dict[int, list[int]] = {}

    if not solution:
        return CheckResult(
            is_feasible=False,
            num_stations=0,
            station_loads={},
            errors=["Empty solution provided"],
        )

    # Detect representation
    first_key = next(iter(solution.keys()))
    first_val = solution[first_key]

    if isinstance(first_val, (list, tuple, set)):
        # Format: {station_id: [task_ids]}
        for s, tasks in solution.items():
            s_int = int(s)
            station_to_tasks[s_int] = list(tasks)
            for t in tasks:
                t_int = int(t)
                if t_int in task_to_station:
                    errors.append(
                        f"Duplicate assignment: Task {t_int} assigned to multiple stations "
                        f"({task_to_station[t_int]} and {s_int})"
                    )
                task_to_station[t_int] = s_int
    elif isinstance(first_val, (int, float)):
        # Format: {task_id: station_id}
        for t, s in solution.items():
            t_int = int(t)
            s_int = int(s)
            task_to_station[t_int] = s_int
            station_to_tasks.setdefault(s_int, []).append(t_int)
    else:
        return CheckResult(
            is_feasible=False,
            num_stations=0,
            station_loads={},
            errors=[f"Unrecognized solution format: values are {type(first_val)}"],
        )

    # 1. Check Task Coverage (Every task 1..n assigned exactly once)
    expected_tasks = set(range(1, n + 1))
    assigned_tasks = set(task_to_station.keys())

    missing_tasks = expected_tasks - assigned_tasks
    if missing_tasks:
        errors.append(f"Missing task assignments: {sorted(missing_tasks)}")

    invalid_tasks = assigned_tasks - expected_tasks
    if invalid_tasks:
        errors.append(f"Unknown tasks assigned: {sorted(invalid_tasks)}")

    # 2. Check Station Numbering
    # Stations should be contiguous integers 1..m
    station_indices = sorted(station_to_tasks.keys())
    if station_indices:
        min_s = station_indices[0]
        max_s = station_indices[-1]
        if min_s != 1:
            errors.append(f"Station indices must start at 1, found {min_s}")
        expected_stations = set(range(1, max_s + 1))
        actual_stations = set(station_indices)
        if expected_stations != actual_stations:
            gaps = expected_stations - actual_stations
            errors.append(f"Non-contiguous station numbering (missing stations {sorted(gaps)})")
        m = max_s
    else:
        m = 0

    # Verify claimed station count if provided
    if claimed_stations is not None and m != claimed_stations:
        errors.append(
            f"Claimed station count mismatch: claimed {claimed_stations}, actual used {m}"
        )

    # 3. Check Station Loads (Capacity Constraint)
    station_loads: dict[int, int] = {}
    for s in range(1, m + 1):
        tasks = station_to_tasks.get(s, [])
        if not tasks:
            errors.append(f"Empty station: Station {s} has no assigned tasks")
        load = sum(instance.task_times[t] for t in tasks if t in instance.task_times)
        station_loads[s] = load
        if load > c:
            errors.append(
                f"Station capacity violated: Station {s} load ({load}) exceeds cycle time ({c})"
            )

    # 4. Check Precedence Constraints
    for u, v in instance.precedences:
        if u in task_to_station and v in task_to_station:
            s_u = task_to_station[u]
            s_v = task_to_station[v]
            if s_u > s_v:
                errors.append(
                    f"Precedence violated: Predecessor task {u} (station {s_u}) "
                    f"assigned after successor task {v} (station {s_v})"
                )

    is_feasible = len(errors) == 0
    return CheckResult(
        is_feasible=is_feasible,
        num_stations=m,
        station_loads=station_loads,
        errors=errors,
    )
