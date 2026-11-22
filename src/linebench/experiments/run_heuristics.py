"""
Heuristic experiment runner for LINEBENCH.
Runs all six priority rules in both directions (Forward and Backward),
plus the randomized multi-pass heuristic across the benchmark subset.
Outputs structured JSONL results with resumption support.
Ref: Master Plan §9, Implementation Plan Phase 2.5.
"""

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from linebench.evaluation.logging import ResultsLogger, get_config_hash
from linebench.heuristics.construct import solve_station_oriented
from linebench.heuristics.random_pass import solve_randomized_multi_pass
from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance
from linebench.instances.subset import load_subset_instance


def run_heuristics_on_instance(
    instance: Instance,
    rules: Sequence[str] = ("RPW", "UBT", "MTS", "MDS", "EST", "LST"),
    directions: Sequence[str] = ("forward", "backward"),
    multi_pass_passes: int = 500,
    multi_pass_seed: int = 1001,
) -> list[dict[str, Any]]:
    """Runs all declared deterministic heuristics and randomized multi-pass on a single instance."""
    results: list[dict[str, Any]] = []
    graph = PrecedenceGraph(instance)

    # 1. Deterministic Priority Rules (Forward & Backward)
    for rule in rules:
        for direction in directions:
            sol = solve_station_oriented(
                instance,
                priority_rule=rule,
                direction=direction,
                graph=graph,
                validate_with_checker=True,
            )
            results.append(
                {
                    "method": f"{rule}_{direction.upper()}",
                    "rule": rule,
                    "direction": direction,
                    "num_stations": sol.num_stations,
                    "duration_seconds": sol.duration_seconds,
                    "is_feasible": sol.is_feasible,
                    "station_loads": sol.station_loads,
                    "station_assignments": sol.station_assignments,
                }
            )

    # 2. Randomized Multi-Pass
    mp_sol = solve_randomized_multi_pass(
        instance,
        num_passes=multi_pass_passes,
        seed=multi_pass_seed,
        priority_rule="RPW",
        graph=graph,
        validate_with_checker=True,
    )
    best_sol = mp_sol.best_solution
    results.append(
        {
            "method": f"MULTI_PASS_RPW_K{multi_pass_passes}",
            "rule": "RPW",
            "direction": "random_multi_pass",
            "num_stations": best_sol.num_stations,
            "duration_seconds": mp_sol.duration_seconds,
            "is_feasible": best_sol.is_feasible,
            "station_loads": best_sol.station_loads,
            "station_assignments": best_sol.station_assignments,
            "num_distinct_best": len(mp_sol.distinct_best_balances),
            "seed": multi_pass_seed,
        }
    )

    return results


def run_heuristic_benchmark(
    instances_yaml: str = "configs/instances.yaml",
    solver_yaml: str = "configs/solver.yaml",
    results_path: str = "results/heuristic_runs.jsonl",
    instance_limit: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Runs the full heuristic benchmark across the pre-declared instance subset.

    Supports resumption by checking already logged instance IDs in results_path.
    """
    with open(instances_yaml, "r", encoding="utf-8") as f:
        inst_cfg = yaml.safe_load(f)

    with open(solver_yaml, "r", encoding="utf-8") as f:
        solv_cfg = yaml.safe_load(f)

    rules = solv_cfg["heuristics"]["priority_rules"]
    directions = solv_cfg["heuristics"]["directions"]
    multi_pass_passes = solv_cfg["heuristics"]["multi_pass"]["num_passes"]
    multi_pass_seed = solv_cfg["heuristics"]["multi_pass"]["seed"]

    config_hash = get_config_hash(solv_cfg)
    logger = ResultsLogger(results_path)

    # Check already completed instances for resumption
    completed_instances: set[str] = set()
    res_path = Path(results_path)
    if resume and res_path.exists():
        with open(res_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_instances.add(record.get("instance_id", ""))
                    except (json.JSONDecodeError, KeyError):
                        continue

    benchmark_list = inst_cfg["benchmark_instances"]
    if instance_limit is not None:
        benchmark_list = benchmark_list[:instance_limit]

    start_bench = time.perf_counter()
    processed_count = 0
    skipped_count = 0

    for entry in benchmark_list:
        inst_id = entry["id"]
        if resume and inst_id in completed_instances:
            skipped_count += 1
            continue

        instance = load_subset_instance(entry)
        inst_results = run_heuristics_on_instance(
            instance,
            rules=rules,
            directions=directions,
            multi_pass_passes=multi_pass_passes,
            multi_pass_seed=multi_pass_seed,
        )

        for res in inst_results:
            logger.log_run(
                run_id="heuristics_batch",
                config_hash=config_hash,
                seed=res.get("seed", 42),
                instance_id=inst_id,
                method=res["method"],
                balance_id=None,
                factor_settings={
                    "rule": res["rule"],
                    "direction": res["direction"],
                    "stratum": entry["stratum"],
                    "num_tasks": entry["num_tasks"],
                    "cycle_time": entry["cycle_time"],
                },
                metric="stations",
                value=float(res["num_stations"]),
                extra={
                    "duration_seconds": res["duration_seconds"],
                    "is_feasible": res["is_feasible"],
                    "optimal_stations": entry["optimal_stations"],
                    "best_known_stations": entry["best_known_stations"],
                },
            )

        processed_count += 1
        print(f"[{processed_count + skipped_count}/{len(benchmark_list)}] Completed instance {inst_id} (stations={res['num_stations']})", flush=True)

    total_duration = time.perf_counter() - start_bench
    return {
        "status": "COMPLETED",
        "total_instances_subset": len(benchmark_list),
        "processed_now": processed_count,
        "skipped_resumed": skipped_count,
        "total_duration_seconds": total_duration,
        "results_path": results_path,
    }
