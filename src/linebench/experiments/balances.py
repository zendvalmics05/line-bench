"""Balance population builder for LINEBENCH simulation experiments.

For each simulation instance in `configs/instances.yaml`:
  1. Best heuristic balance (from portfolio / fast heuristics)
  2. Exact CP-SAT balance
  3. SO1 balance (min largest load)
  4. SO2 balance (min sum of squared idle times / smoothness)
  5. m+1 balance (one extra station with min largest load)
  6. Up to N distinct equal-station-count random balances (from multi-pass RPW)

All balances:
  - Verified by the independent checker
  - Assigned stable, unique IDs: e.g.
    `{instance_id}__heur_best`, `{instance_id}__exact`, `{instance_id}__so1`,
    `{instance_id}__so2`, `{instance_id}__m_plus_1`, `{instance_id}__rand_{i}`
  - Exported to structured JSON for the discrete-event simulation experiments.

Ref: Master Plan §6.3, Implementation Plan Phase 3.6.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from linebench.exact.cpsat_salbp1 import solve_salbp1_cpsat
from linebench.exact.secondary import (
    solve_m_plus_1_min_max_load,
    solve_secondary_objective,
)
from linebench.heuristics.construct import solve_station_oriented
from linebench.heuristics.random_pass import solve_randomized_multi_pass
from linebench.instances.checker import check_solution
from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance
from linebench.instances.subset import load_subset_instance


@dataclass(frozen=True)
class BalanceRecord:
    """A distinct feasible assembly line balance ready for simulation."""

    balance_id: str
    instance_id: str
    balance_type: str  # 'HEUR_BEST', 'EXACT', 'SO1', 'SO2', 'M_PLUS_1', 'RANDOM_DISTINCT'
    num_stations: int
    station_assignments: dict[int, list[int]]  # station_id -> [task_ids]
    station_loads: dict[int, int]  # station_id -> load
    max_load: int
    sum_squared_idle: int
    is_feasible: bool
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InstanceBalancePopulation:
    """Complete balance population for a single simulation instance."""

    instance_id: str
    num_tasks: int
    cycle_time: int
    optimal_or_best_stations: int
    balances: list[BalanceRecord]
    num_random_balances_found: int


def build_balance_population_for_instance(
    instance: Instance,
    optimal_or_best_stations: int | None = None,
    max_random_distinct: int = 10,
    random_passes: int = 300,
    random_seed: int = 2026,
    solver_time_limit: float = 30.0,
    graph: PrecedenceGraph | None = None,
) -> InstanceBalancePopulation:
    """Builds the comprehensive balance population for a simulation instance."""
    inst_id = instance.name
    n = instance.num_tasks
    c = instance.cycle_time

    if graph is None:
        graph = PrecedenceGraph(instance)

    balances: list[BalanceRecord] = []

    # 1. Best Heuristic Balance
    # Evaluate RPW, UBT, MTS in forward & backward to find best heuristic balance
    heur_candidates = []
    for rule in ("RPW", "UBT", "MTS"):
        for direction in ("forward", "backward"):
            sol = solve_station_oriented(
                instance,
                priority_rule=rule,
                direction=direction,
                graph=graph,
                validate_with_checker=True,
            )
            heur_candidates.append(sol)

    heur_candidates.sort(key=lambda s: (s.num_stations, max(s.station_loads.values())))
    best_heur = heur_candidates[0]

    max_l = max(best_heur.station_loads.values())
    sum_sq_idle = sum((c - l_val) ** 2 for l_val in best_heur.station_loads.values())
    balances.append(
        BalanceRecord(
            balance_id=f"{inst_id}__heur_best",
            instance_id=inst_id,
            balance_type="HEUR_BEST",
            num_stations=best_heur.num_stations,
            station_assignments=best_heur.station_assignments,
            station_loads=best_heur.station_loads,
            max_load=max_l,
            sum_squared_idle=sum_sq_idle,
            is_feasible=best_heur.is_feasible,
            extra={"method": f"{best_heur.method}_{best_heur.direction}"},
        )
    )

    # 2. Exact CP-SAT Balance
    exact_sol = solve_salbp1_cpsat(
        instance,
        time_limit_seconds=solver_time_limit,
        num_workers=2,
        random_seed=42,
        graph=graph,
        heuristic_hint=best_heur,
        validate_with_checker=True,
    )

    best_m = exact_sol.num_stations if exact_sol.num_stations is not None else best_heur.num_stations
    if optimal_or_best_stations is not None:
        best_m = optimal_or_best_stations

    if exact_sol.is_feasible and exact_sol.station_assignments:
        max_l = max(exact_sol.station_loads.values())
        sum_sq_idle = sum((c - l_val) ** 2 for l_val in exact_sol.station_loads.values())
        balances.append(
            BalanceRecord(
                balance_id=f"{inst_id}__exact",
                instance_id=inst_id,
                balance_type="EXACT",
                num_stations=exact_sol.num_stations if exact_sol.num_stations else best_m,
                station_assignments=exact_sol.station_assignments,
                station_loads=exact_sol.station_loads,
                max_load=max_l,
                sum_squared_idle=sum_sq_idle,
                is_feasible=exact_sol.is_feasible,
                extra={"status": exact_sol.status, "duration": exact_sol.duration_seconds},
            )
        )

    # 3. SO1: MinMaxLoad Balance (Fixed at best_m stations)
    so1_sol = solve_secondary_objective(
        instance,
        target_stations=best_m,
        objective_type="SO1",
        time_limit_seconds=solver_time_limit,
        num_workers=2,
        random_seed=42,
        initial_assignment=exact_sol.task_to_station or None,
        graph=graph,
        validate_with_checker=True,
    )
    if so1_sol.is_feasible:
        balances.append(
            BalanceRecord(
                balance_id=f"{inst_id}__so1",
                instance_id=inst_id,
                balance_type="SO1",
                num_stations=so1_sol.target_stations,
                station_assignments=so1_sol.station_assignments,
                station_loads=so1_sol.station_loads,
                max_load=so1_sol.max_load,
                sum_squared_idle=so1_sol.sum_squared_idle,
                is_feasible=so1_sol.is_feasible,
                extra={"status": so1_sol.status, "duration": so1_sol.duration_seconds},
            )
        )

    # 4. SO2: Smoothness Balance (Fixed at best_m stations)
    so2_sol = solve_secondary_objective(
        instance,
        target_stations=best_m,
        objective_type="SO2",
        time_limit_seconds=solver_time_limit,
        num_workers=2,
        random_seed=42,
        initial_assignment=exact_sol.task_to_station or None,
        graph=graph,
        validate_with_checker=True,
    )
    if so2_sol.is_feasible:
        balances.append(
            BalanceRecord(
                balance_id=f"{inst_id}__so2",
                instance_id=inst_id,
                balance_type="SO2",
                num_stations=so2_sol.target_stations,
                station_assignments=so2_sol.station_assignments,
                station_loads=so2_sol.station_loads,
                max_load=so2_sol.max_load,
                sum_squared_idle=so2_sol.sum_squared_idle,
                is_feasible=so2_sol.is_feasible,
                extra={"status": so2_sol.status, "duration": so2_sol.duration_seconds},
            )
        )

    # 5. m+1 Variant Balance (SO1 with best_m + 1 stations)
    m_plus_1_sol = solve_m_plus_1_min_max_load(
        instance,
        optimal_stations=best_m,
        time_limit_seconds=solver_time_limit,
        num_workers=2,
        random_seed=42,
        graph=graph,
        validate_with_checker=True,
    )
    if m_plus_1_sol.is_feasible:
        balances.append(
            BalanceRecord(
                balance_id=f"{inst_id}__m_plus_1",
                instance_id=inst_id,
                balance_type="M_PLUS_1",
                num_stations=m_plus_1_sol.target_stations,
                station_assignments=m_plus_1_sol.station_assignments,
                station_loads=m_plus_1_sol.station_loads,
                max_load=m_plus_1_sol.max_load,
                sum_squared_idle=m_plus_1_sol.sum_squared_idle,
                is_feasible=m_plus_1_sol.is_feasible,
                extra={"status": m_plus_1_sol.status, "duration": m_plus_1_sol.duration_seconds},
            )
        )

    # 6. Distinct Equal-Station-Count Random Balances
    # Use randomized multi-pass heuristic to sample distinct feasible balances with station count <= best_m
    mp_sol = solve_randomized_multi_pass(
        instance,
        num_passes=random_passes,
        seed=random_seed,
        priority_rule="RPW",
        graph=graph,
        max_distinct_best=max_random_distinct,
        validate_with_checker=True,
    )

    random_count = 0
    # Collect distinct balances that match the best station count
    seen_signatures: set[tuple[tuple[int, ...], ...]] = set()

    # Add existing balance signatures so we don't duplicate
    for b in balances:
        sig = tuple(tuple(sorted(b.station_assignments.get(s, []))) for s in sorted(b.station_assignments.keys()))
        seen_signatures.add(sig)

    for distinct_assignments in mp_sol.distinct_best_balances:
        m_stations = len(distinct_assignments)
        if m_stations == best_m and random_count < max_random_distinct:
            sig = tuple(
                tuple(sorted(tasks))
                for s, tasks in sorted(distinct_assignments.items())
            )
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                random_count += 1
                loads = {
                    s: sum(instance.task_times[t] for t in tasks)
                    for s, tasks in distinct_assignments.items()
                }
                max_l = max(loads.values()) if loads else 0
                sum_sq_idle = sum((c - l_val) ** 2 for l_val in loads.values())
                balances.append(
                    BalanceRecord(
                        balance_id=f"{inst_id}__rand_{random_count}",
                        instance_id=inst_id,
                        balance_type="RANDOM_DISTINCT",
                        num_stations=m_stations,
                        station_assignments=distinct_assignments,
                        station_loads=loads,
                        max_load=max_l,
                        sum_squared_idle=sum_sq_idle,
                        is_feasible=True,
                        extra={"random_index": random_count},
                    )
                )

    # Independent checker validation pass on every record
    for b in balances:
        chk = check_solution(instance, b.station_assignments, claimed_stations=b.num_stations)
        chk.raise_if_invalid()

    return InstanceBalancePopulation(
        instance_id=inst_id,
        num_tasks=n,
        cycle_time=c,
        optimal_or_best_stations=best_m,
        balances=balances,
        num_random_balances_found=random_count,
    )


def build_simulation_balance_population(
    instances_yaml: str = "configs/instances.yaml",
    output_path: str = "results/balance_population.json",
    max_random_distinct: int = 5,
    instance_limit: int | None = None,
    log_progress: bool = True,
) -> dict[str, Any]:
    """Builds and serializes balance populations for all pre-declared simulation instances."""
    with open(instances_yaml, "r", encoding="utf-8") as f:
        inst_cfg = yaml.safe_load(f)

    sim_instances = inst_cfg["simulation_instances"]
    if instance_limit is not None:
        sim_instances = sim_instances[:instance_limit]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    populations: dict[str, Any] = {}
    summary_counts: dict[str, int] = {}

    for idx, entry in enumerate(sim_instances, start=1):
        inst_id = entry["id"]
        instance = load_subset_instance(entry)
        best_opt = entry.get("optimal_stations") or entry.get("best_known_stations")

        pop = build_balance_population_for_instance(
            instance,
            optimal_or_best_stations=best_opt,
            max_random_distinct=max_random_distinct,
            random_passes=200,
            solver_time_limit=15.0,
        )

        populations[inst_id] = asdict(pop)
        summary_counts[inst_id] = len(pop.balances)

        if log_progress:
            print(
                f"[{idx}/{len(sim_instances)}] Built balances for {inst_id}: "
                f"{len(pop.balances)} balances (random_distinct={pop.num_random_balances_found})",
                flush=True,
            )

    # Save to JSON
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(populations, f, indent=2)

    return {
        "status": "COMPLETED",
        "num_instances": len(populations),
        "total_balances_built": sum(summary_counts.values()),
        "output_path": output_path,
        "summary_counts": summary_counts,
    }
