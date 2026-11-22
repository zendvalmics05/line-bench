"""
Prototype CP-SAT Model for SALBP-1.
Validates OR-Tools installation, time limiting, worker thread controls,
and objective minimization on simple SALBP instances.
"""

import sys
import time

import ortools
from ortools.sat.python import cp_model


def solve_salbp1_prototype(
    task_times: list[int],
    precedences: list[tuple[int, int]],
    cycle_time: int,
    time_limit_seconds: float = 10.0,
    num_workers: int = 2,
    log_search: bool = False,
) -> dict[str, object]:
    """Solves SALBP-1 on a given instance using OR-Tools CP-SAT."""
    n = len(task_times)
    # Upper bound on stations: n stations
    max_stations = n

    model = cp_model.CpModel()

    # Decision variables: x[i, s] == 1 if task i is assigned to station s
    x = {}
    for i in range(n):
        for s in range(max_stations):
            x[i, s] = model.NewBoolVar(f"x_{i}_{s}")

    # Station indicator: y[s] == 1 if station s is opened
    y = [model.NewBoolVar(f"y_{s}") for s in range(max_stations)]

    # 1. Assignment constraint: Each task assigned to exactly one station
    for i in range(n):
        model.AddExactlyOne(x[i, s] for s in range(max_stations))

    # 2. Cycle time constraint: Station load <= cycle_time
    for s in range(max_stations):
        model.Add(sum(task_times[i] * x[i, s] for i in range(n)) <= cycle_time)

    # 3. Precedence constraints: If i -> j, station(i) <= station(j)
    for i, j in precedences:
        station_i = sum(s * x[i, s] for s in range(max_stations))
        station_j = sum(s * x[j, s] for s in range(max_stations))
        model.Add(station_i <= station_j)

    # 4. Linking x and y: x[i, s] <= y[s]
    for s in range(max_stations):
        for i in range(n):
            model.Add(x[i, s] <= y[s])
        # Symmetry breaking: y[s] <= y[s-1]
        if s > 0:
            model.Add(y[s] <= y[s - 1])

    # Objective: Minimize total stations used
    model.Minimize(sum(y))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_workers = num_workers
    solver.parameters.log_search_progress = log_search

    start_time = time.time()
    status = solver.Solve(model)
    solve_duration = time.time() - start_time

    status_name = solver.StatusName(status)
    stations_used = int(solver.ObjectiveValue()) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None

    assignments = {}
    if stations_used is not None:
        for s in range(stations_used):
            assigned_tasks = [i for i in range(n) if solver.Value(x[i, s]) == 1]
            assignments[s + 1] = assigned_tasks

    return {
        "status": status_name,
        "stations_used": stations_used,
        "duration_seconds": solve_duration,
        "assignments": assignments,
        "python_version": sys.version,
        "ortools_version": ortools.__version__,
    }


if __name__ == "__main__":
    # Test on Bowman-8 instance from classical benchmark
    # 8 tasks with times: 11, 17, 9, 5, 8, 12, 10, 3
    # c = 20 (published optimum m* = 5)
    bowman_times = [11, 17, 9, 5, 8, 12, 10, 3]
    # 1-based precedences converted to 0-based:
    bowman_prec = [
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
        (2, 5),
        (3, 5),
        (4, 6),
        (5, 7),
    ]
    cycle_time = 20

    print("Running Bowman-8 (c=20) with CP-SAT prototype...")
    res = solve_salbp1_prototype(bowman_times, bowman_prec, cycle_time, time_limit_seconds=5.0, num_workers=2)
    print(f"Status: {res['status']}")
    print(f"Stations used: {res['stations_used']} (Expected: 5)")
    print(f"Duration: {res['duration_seconds']:.4f}s")
    for s, tasks in res["assignments"].items():
        load = sum(bowman_times[t] for t in tasks)
        print(f"  Station {s}: tasks {[t+1 for t in tasks]}, load = {load}/{cycle_time}")
