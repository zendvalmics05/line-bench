"""Exact experiment runner for LINEBENCH.

Runs CP-SAT SALBP-1 across the pre-declared benchmark subset.
Features:
  - Resumable JSONL logging (skips already completed instances).
  - Checkpointing after every instance.
  - Logs machine metadata, OS, Python version, OR-Tools version, worker threads, time limits.
  - Records status (OPTIMAL, FEASIBLE, TIMEOUT, etc.), station count, runtime, bounds.
  - Validates every solution with the independent checker.

Ref: Master Plan §5.2, Implementation Plan Phase 3.5.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import ortools
import yaml

from linebench.evaluation.logging import ResultsLogger, get_config_hash
from linebench.exact.cpsat_salbp1 import solve_salbp1_cpsat
from linebench.instances.subset import load_subset_instance


def get_system_environment_metadata() -> dict[str, Any]:
    """Collects hardware and runtime environment metadata for provenance."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "ortools_version": ortools.__version__,
    }


def run_exact_benchmark(
    instances_yaml: str = "configs/instances.yaml",
    solver_yaml: str = "configs/solver.yaml",
    results_path: str = "results/exact_runs.jsonl",
    instance_limit: int | None = None,
    resume: bool = True,
    time_limit_override: float | None = None,
    log_progress: bool = True,
) -> dict[str, Any]:
    """Runs the CP-SAT exact solver across the pre-declared benchmark subset.

    Args:
        instances_yaml: Path to frozen instances configuration.
        solver_yaml: Path to solver configuration.
        results_path: Path to output JSONL log file.
        instance_limit: Optional cap on instances to process (for testing/partial runs).
        resume: If True, skips instances already recorded in results_path.
        time_limit_override: Optional time limit override in seconds.
        log_progress: Whether to print progress to stdout after each instance.

    Returns:
        Summary dictionary of the benchmark run.
    """
    with open(instances_yaml, "r", encoding="utf-8") as f:
        inst_cfg = yaml.safe_load(f)

    with open(solver_yaml, "r", encoding="utf-8") as f:
        solv_cfg = yaml.safe_load(f)

    cp_cfg = solv_cfg["cp_sat"]
    time_limit = float(time_limit_override if time_limit_override is not None else cp_cfg["time_limit_seconds"])
    num_workers = int(cp_cfg["num_workers"])
    seed = int(cp_cfg["random_seed"])

    config_hash = get_config_hash(solv_cfg)
    env_metadata = get_system_environment_metadata()

    # Load completed instances for resumption
    completed_instances: set[str] = set()
    res_path = Path(results_path)
    res_path.parent.mkdir(parents=True, exist_ok=True)

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

    logger = ResultsLogger(results_path)

    start_bench = time.perf_counter()
    processed_count = 0
    skipped_count = 0

    for entry in benchmark_list:
        inst_id = entry["id"]
        if resume and inst_id in completed_instances:
            skipped_count += 1
            continue

        instance = load_subset_instance(entry)

        # Solve with CP-SAT
        sol = solve_salbp1_cpsat(
            instance,
            time_limit_seconds=time_limit,
            num_workers=num_workers,
            random_seed=seed,
            validate_with_checker=True,
        )

        # Log result
        logger.log_run(
            run_id="exact_batch",
            config_hash=config_hash,
            seed=seed,
            instance_id=inst_id,
            method="CPSAT",
            balance_id=None,
            factor_settings={
                "stratum": entry["stratum"],
                "num_tasks": entry["num_tasks"],
                "cycle_time": entry["cycle_time"],
                "time_limit_seconds": time_limit,
                "num_workers": num_workers,
            },
            metric="stations",
            value=float(sol.num_stations) if sol.num_stations is not None else -1.0,
            extra={
                "status": sol.status,
                "is_proven_optimal": sol.is_proven_optimal,
                "duration_seconds": sol.duration_seconds,
                "is_feasible": sol.is_feasible,
                "lower_bound": sol.lower_bound,
                "best_bound": sol.best_bound,
                "optimal_stations": entry["optimal_stations"],
                "best_known_stations": entry["best_known_stations"],
                "environment": env_metadata,
            },
        )

        processed_count += 1
        if log_progress:
            total_target = len(benchmark_list)
            cur_pos = processed_count + skipped_count
            print(
                f"[{cur_pos}/{total_target}] Exact CP-SAT instance {inst_id}: "
                f"status={sol.status}, stations={sol.num_stations}, "
                f"time={sol.duration_seconds:.2f}s",
                flush=True,
            )

    total_duration = time.perf_counter() - start_bench
    return {
        "status": "COMPLETED",
        "total_instances_subset": len(benchmark_list),
        "processed_now": processed_count,
        "skipped_resumed": skipped_count,
        "total_duration_seconds": total_duration,
        "results_path": results_path,
        "environment": env_metadata,
    }
