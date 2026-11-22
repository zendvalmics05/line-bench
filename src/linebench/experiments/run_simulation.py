"""Simulation sweep experiment runner (Master Plan §6.3, Phase 6.1).

Runs full factorial simulation across:
  - 30 pre-declared simulation instances in `configs/instances.yaml`
  - Balance population from `results/balance_population.json`
  - CV levels: [0.10, 0.25, 0.50]
  - Buffer capacities: [0, 1, 3]
  - Line types: 'unpaced' (blocking-after-service) and 'paced' (wait-for-slowest)
  - Replications: 30 reps, 5500 total units (500 warmup)

Common Random Numbers (CRN):
  For each instance and CV level, ONE master noise array of shape
  `(n_replications, total_units, n_tasks)` is drawn and reused across all
  balances and buffer sizes. This guarantees variance reduction and isolates
  the effect of line balance structure.

Resumability & Checkpointing:
  Appends replication-aggregated results to `results/simulation_runs.jsonl`.
  If interrupted, re-running skips already-simulated runs seamlessly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from linebench.instances.subset import load_subset_instance
from linebench.simulation.noise import generate_noise_factors
from linebench.simulation.paced import simulate_paced_replications
from linebench.simulation.stations import build_station_times
from linebench.simulation.tandem import simulate_tandem_replications

__all__ = [
    "SimulationRunRecord",
    "run_simulation_sweep",
]


@dataclass(frozen=True)
class SimulationRunRecord:
    """Aggregated output record for a single simulation configuration."""

    instance_id: str
    balance_id: str
    balance_type: str
    num_stations: int
    cycle_time: float
    line_type: str  # 'unpaced' or 'paced'
    cv: float
    distribution: str  # 'lognormal' or 'gamma'
    buffer_capacity: int  # for unpaced (0 if paced)
    n_replications: int
    total_units: int
    warmup_units: int
    measured_units: int
    # Performance summary across replications
    mean_throughput: float
    std_throughput: float
    sem_throughput: float
    ci_95_lower: float
    ci_95_upper: float
    # Relative throughput metrics (Master Plan §A14)
    rel_throughput_design: float  # TH / (1 / c) = TH * c
    rel_throughput_bottleneck: float  # TH / (1 / max_k S_k)
    # Line metrics
    mean_utilization_max_station: float
    mean_utilization_avg_station: float
    mean_blocked_fraction: float  # for unpaced
    mean_starved_fraction: float  # for unpaced
    mean_wip: float  # for unpaced
    mean_line_overload_prob: float  # for paced
    max_station_overload_prob: float  # for paced


def _load_completed_keys(output_file: Path) -> set[tuple[str, str, str, float, int]]:
    """Load set of already-completed run signatures from checkpointed jsonl."""
    completed = set()
    if not output_file.exists():
        return completed
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (
                    rec["instance_id"],
                    rec["balance_id"],
                    rec["line_type"],
                    float(rec["cv"]),
                    int(rec["buffer_capacity"]),
                )
                completed.add(key)
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    return completed


def run_simulation_sweep(
    instances_yaml: str = "configs/instances.yaml",
    balance_pop_path: str = "results/balance_population.json",
    simulation_yaml: str = "configs/simulation.yaml",
    output_path: str = "results/simulation_runs.jsonl",
    instance_limit: int | None = None,
    log_every: int = 10,
) -> dict[str, Any]:
    """Execute full factorial simulation sweep across instances, balances, CVs, and buffers."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load instances config
    with open(instances_yaml, "r", encoding="utf-8") as f:
        inst_cfg = yaml.safe_load(f)
    sim_instances = inst_cfg["simulation_instances"]
    if instance_limit is not None:
        sim_instances = sim_instances[:instance_limit]

    # 2. Load simulation parameters
    with open(simulation_yaml, "r", encoding="utf-8") as f:
        sim_cfg = yaml.safe_load(f)

    cv_levels: list[float] = [float(x) for x in sim_cfg["variability"]["cv_levels"]]
    primary_dist: str = sim_cfg["variability"]["distributions"]["primary"]
    buffers: list[int] = [int(b) for b in sim_cfg["lines"]["unpaced"]["buffers"]]
    reps: int = int(sim_cfg["run_parameters"]["replications"])
    total_units: int = int(sim_cfg["run_parameters"]["total_units"])
    warmup_units: int = int(sim_cfg["run_parameters"]["warmup_units"])
    measured_units: int = total_units - warmup_units

    # 3. Load balance populations
    with open(balance_pop_path, "r", encoding="utf-8") as f:
        balance_data = json.load(f)

    # 4. Load checkpointed keys
    completed_keys = _load_completed_keys(out_file)
    print(f"Loaded {len(completed_keys)} previously completed simulation runs.", flush=True)

    new_runs_count = 0
    total_evals = 0

    with open(out_file, "a", encoding="utf-8") as f_out:
        for inst_idx, entry in enumerate(sim_instances, start=1):
            inst_id = entry["id"]
            if inst_id not in balance_data:
                continue

            instance = load_subset_instance(entry)
            cycle_time = float(instance.cycle_time)
            # instance.task_times is a dict {1: t1, 2: t2, ...} -> convert to 0-indexed array
            n_tasks = len(instance.task_times)
            base_times = np.array([float(instance.task_times[j]) for j in range(1, n_tasks + 1)], dtype=np.float64)

            pop_entry = balance_data[inst_id]
            balances = pop_entry["balances"]

            n_balances = len(balances)
            print(
                f"  → [{inst_idx}/{len(sim_instances)}] {inst_id}  "
                f"({n_balances} balances, {n_tasks} tasks)",
                flush=True,
            )

            # For each CV level, generate CRN noise array once for all balances of this instance
            for cv_idx, cv in enumerate(cv_levels, start=1):
                seed_for_crn = int(abs(hash((inst_id, cv, primary_dist))) % (2**31 - 1))
                noise_tensor = generate_noise_factors(
                    shape=(reps, total_units, n_tasks),
                    cv=cv,
                    seed=seed_for_crn,
                    distribution=primary_dist,
                )

                for bal in balances:
                    bal_id = bal["balance_id"]
                    bal_type = bal["balance_type"]
                    m_stations = bal["num_stations"]
                    station_assignments = {int(s): [int(t) for t in ts] for s, ts in bal["station_assignments"].items()}

                    # Build station times: shape (reps, total_units, m_stations)
                    S_tensor = build_station_times(station_assignments, base_times, noise_tensor)
                    mean_station_loads = np.mean(S_tensor, axis=(0, 1))
                    det_bottleneck = float(np.max(mean_station_loads))

                    # A. Unpaced line runs (for each buffer capacity)
                    for b in buffers:
                        key_unpaced = (inst_id, bal_id, "unpaced", cv, b)
                        total_evals += 1
                        if key_unpaced in completed_keys:
                            continue

                        unpaced_res = simulate_tandem_replications(
                            S_tensor,
                            buffers=b,
                            warmup_units=warmup_units,
                        )
                        ths = unpaced_res["throughput"]
                        m_th = float(np.mean(ths))
                        s_th = float(np.std(ths, ddof=1)) if reps > 1 else 0.0
                        sem_th = s_th / np.sqrt(reps)
                        ci_half = 1.96 * sem_th

                        # Metrics
                        util_mat = unpaced_res["utilization"]  # (reps, m)
                        mean_max_util = float(np.mean(np.max(util_mat, axis=1)))
                        mean_avg_util = float(np.mean(util_mat))
                        mean_blocked = float(np.mean(unpaced_res["blocked_fraction"]))
                        mean_starved = float(np.mean(unpaced_res["starved_fraction"]))
                        mean_wip = float(np.mean(unpaced_res["mean_wip"]))

                        rec = SimulationRunRecord(
                            instance_id=inst_id,
                            balance_id=bal_id,
                            balance_type=bal_type,
                            num_stations=m_stations,
                            cycle_time=cycle_time,
                            line_type="unpaced",
                            cv=cv,
                            distribution=primary_dist,
                            buffer_capacity=b,
                            n_replications=reps,
                            total_units=total_units,
                            warmup_units=warmup_units,
                            measured_units=measured_units,
                            mean_throughput=m_th,
                            std_throughput=s_th,
                            sem_throughput=sem_th,
                            ci_95_lower=m_th - ci_half,
                            ci_95_upper=m_th + ci_half,
                            rel_throughput_design=m_th * cycle_time,
                            rel_throughput_bottleneck=m_th * det_bottleneck,
                            mean_utilization_max_station=mean_max_util,
                            mean_utilization_avg_station=mean_avg_util,
                            mean_blocked_fraction=mean_blocked,
                            mean_starved_fraction=mean_starved,
                            mean_wip=mean_wip,
                            mean_line_overload_prob=0.0,
                            max_station_overload_prob=0.0,
                        )
                        f_out.write(json.dumps(asdict(rec)) + "\n")
                        f_out.flush()
                        completed_keys.add(key_unpaced)
                        new_runs_count += 1

                    # B. Paced line run (buffer capacity = 0)
                    key_paced = (inst_id, bal_id, "paced", cv, 0)
                    total_evals += 1
                    if key_paced not in completed_keys:
                        paced_res = simulate_paced_replications(
                            S_tensor,
                            cycle_time=cycle_time,
                            warmup_units=warmup_units,
                        )
                        p_ths = paced_res["throughput"]
                        m_th_p = float(np.mean(p_ths))
                        s_th_p = float(np.std(p_ths, ddof=1)) if reps > 1 else 0.0
                        sem_th_p = s_th_p / np.sqrt(reps)
                        ci_half_p = 1.96 * sem_th_p

                        mean_line_ovld = float(np.mean(paced_res["line_overload_prob"]))
                        mean_max_stn_ovld = float(np.mean(np.max(paced_res["station_overload_prob"], axis=1)))

                        rec_p = SimulationRunRecord(
                            instance_id=inst_id,
                            balance_id=bal_id,
                            balance_type=bal_type,
                            num_stations=m_stations,
                            cycle_time=cycle_time,
                            line_type="paced",
                            cv=cv,
                            distribution=primary_dist,
                            buffer_capacity=0,
                            n_replications=reps,
                            total_units=total_units,
                            warmup_units=warmup_units,
                            measured_units=measured_units,
                            mean_throughput=m_th_p,
                            std_throughput=s_th_p,
                            sem_throughput=sem_th_p,
                            ci_95_lower=m_th_p - ci_half_p,
                            ci_95_upper=m_th_p + ci_half_p,
                            rel_throughput_design=m_th_p * cycle_time,
                            rel_throughput_bottleneck=m_th_p * det_bottleneck,
                            mean_utilization_max_station=0.0,
                            mean_utilization_avg_station=0.0,
                            mean_blocked_fraction=0.0,
                            mean_starved_fraction=0.0,
                            mean_wip=0.0,
                            mean_line_overload_prob=mean_line_ovld,
                            max_station_overload_prob=mean_max_stn_ovld,
                        )
                        f_out.write(json.dumps(asdict(rec_p)) + "\n")
                        f_out.flush()
                        completed_keys.add(key_paced)
                        new_runs_count += 1

                print(
                    f"     CV={cv:.2f} done  |  total runs so far: {len(completed_keys)}",
                    flush=True,
                )

            if inst_idx % log_every == 0 or inst_idx == len(sim_instances):
                print(
                    f"[{inst_idx}/{len(sim_instances)}] Instances processed. "
                    f"Total runs logged: {len(completed_keys)}",
                    flush=True,
                )

    return {
        "status": "COMPLETED",
        "output_path": output_path,
        "new_runs_executed": new_runs_count,
        "total_runs_completed": len(completed_keys),
    }
