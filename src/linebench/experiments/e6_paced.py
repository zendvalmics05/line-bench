"""Phase 6.6: Paced Line Analysis (Master Plan §6.3, Phase 6.6).

Analyzes the performance of paced lines:
1. Overload probability per balance and its relation to largest station load.
2. Station-level and line-level overload probabilities.
3. Empirical overload probability vs analytical normal approximation (§A11).
4. Error metrics (MAE, RMSE) of the normal approximation across CV levels and balance types.

Outputs
-------
- ``results/e6_paced_analysis.csv`` — per-balance detailed pacing and normal approx results.
- ``results/e6_paced_summary.csv`` — aggregated summary by CV and balance type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from linebench.instances.subset import load_subset_instance
from linebench.simulation.paced import normal_approx_overload_probability

__all__ = ["run_paced_analysis"]


def run_paced_analysis(
    simulation_jsonl: str = "results/simulation_runs.jsonl",
    instances_yaml: str = "configs/instances.yaml",
    balance_pop_path: str = "results/balance_population.json",
    output_detailed: str = "results/e6_paced_analysis.csv",
    output_summary: str = "results/e6_paced_summary.csv",
) -> dict[str, Any]:
    """Run Phase 6.6 paced line evaluation and output summary CSVs."""
    # 1. Load instances config
    with open(instances_yaml, "r", encoding="utf-8") as f:
        inst_cfg = yaml.safe_load(f)
    sim_instances = {entry["id"]: entry for entry in inst_cfg["simulation_instances"]}

    # 2. Load balance population
    with open(balance_pop_path, "r", encoding="utf-8") as f:
        balance_data = json.load(f)

    # Pre-compute task times dict for each instance
    instance_task_times: dict[str, dict[int, float]] = {}
    for inst_id, entry in sim_instances.items():
        inst = load_subset_instance(entry)
        instance_task_times[inst_id] = {int(k): float(v) for k, v in inst.task_times.items()}

    # Map (instance_id, balance_id) -> assignments
    balance_assignments: dict[tuple[str, str], dict[int, list[int]]] = {}
    for inst_id, p_entry in balance_data.items():
        for b in p_entry["balances"]:
            b_id = b["balance_id"]
            station_assignments = {int(s): [int(t) for t in ts] for s, ts in b["station_assignments"].items()}
            balance_assignments[(inst_id, b_id)] = station_assignments

    # 3. Load simulation records (paced only)
    jsonl_path = Path(simulation_jsonl)
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    df = pd.DataFrame(records)
    df_paced = df[df["line_type"] == "paced"].copy()

    detailed_rows: list[dict] = []

    for _, row in df_paced.iterrows():
        inst_id = row["instance_id"]
        b_id = row["balance_id"]
        b_type = row["balance_type"]
        cv = float(row["cv"])
        cycle_time = float(row["cycle_time"])
        empirical_line_ovld = float(row["mean_line_overload_prob"])
        empirical_max_stn_ovld = float(row["max_station_overload_prob"])
        mean_th = float(row["mean_throughput"])

        key = (inst_id, b_id)
        if key not in balance_assignments or inst_id not in instance_task_times:
            continue

        assignments = balance_assignments[key]
        t_times = instance_task_times[inst_id]

        # Calculate normal approximation for each station
        stn_norm_ovlds: list[float] = []
        stn_nominal_loads: list[float] = []

        for _, task_ids in sorted(assignments.items()):
            task_list = [t_times[t] for t in task_ids if t in t_times]
            stn_nominal_loads.append(sum(task_list))
            p_ovld = normal_approx_overload_probability(task_list, cycle_time=cycle_time, cv=cv)
            stn_norm_ovlds.append(p_ovld)

        nominal_max_load = max(stn_nominal_loads) if stn_nominal_loads else 0.0
        norm_max_stn_ovld = max(stn_norm_ovlds) if stn_norm_ovlds else 0.0
        # Line overload approx: 1 - prod(1 - p_k) assuming station independence as upper/approx proxy
        norm_line_ovld_proxy = 1.0 - float(np.prod([1.0 - p for p in stn_norm_ovlds]))

        # Approximation error for max station overload prob
        error_max_stn = norm_max_stn_ovld - empirical_max_stn_ovld
        abs_error_max_stn = abs(error_max_stn)

        detailed_rows.append({
            "instance_id": inst_id,
            "balance_id": b_id,
            "balance_type": b_type,
            "cv": cv,
            "cycle_time": cycle_time,
            "num_stations": row["num_stations"],
            "nominal_max_station_load": nominal_max_load,
            "bottleneck_load_ratio": nominal_max_load / cycle_time if cycle_time > 0 else float("nan"),
            "mean_throughput": mean_th,
            "empirical_line_overload_prob": empirical_line_ovld,
            "empirical_max_stn_overload_prob": empirical_max_stn_ovld,
            "norm_approx_max_stn_overload_prob": norm_max_stn_ovld,
            "norm_approx_line_overload_proxy": norm_line_ovld_proxy,
            "error_max_stn_overload": error_max_stn,
            "abs_error_max_stn_overload": abs_error_max_stn,
        })

    df_detail = pd.DataFrame(detailed_rows)

    # 4. Summary table by CV and balance_type
    summary_rows: list[dict] = []
    for (cv, b_type), grp in df_detail.groupby(["cv", "balance_type"]):
        n_runs = len(grp)
        mean_emp_max = float(grp["empirical_max_stn_overload_prob"].mean())
        mean_norm_max = float(grp["norm_approx_max_stn_overload_prob"].mean())
        mae_max = float(grp["abs_error_max_stn_overload"].mean())
        rmse_max = float(np.sqrt((grp["error_max_stn_overload"] ** 2).mean()))
        mean_emp_line = float(grp["empirical_line_overload_prob"].mean())

        # Correlation between bottleneck load ratio and empirical max stn overload
        corr_val = float(grp["bottleneck_load_ratio"].corr(grp["empirical_max_stn_overload_prob"]))

        summary_rows.append({
            "cv": cv,
            "balance_type": b_type,
            "n_runs": n_runs,
            "mean_empirical_line_ovld": mean_emp_line,
            "mean_empirical_max_stn_ovld": mean_emp_max,
            "mean_norm_approx_max_stn_ovld": mean_norm_max,
            "mae_normal_approx": mae_max,
            "rmse_normal_approx": rmse_max,
            "corr_load_ratio_vs_overload": corr_val,
        })

    df_summary = pd.DataFrame(summary_rows).sort_values(["cv", "balance_type"])

    Path(output_detailed).parent.mkdir(parents=True, exist_ok=True)
    df_detail.to_csv(output_detailed, index=False)
    df_summary.to_csv(output_summary, index=False)

    print(f"Paced detailed rows written : {len(df_detail):>5}  → {output_detailed}")
    print(f"Paced summary rows written  : {len(df_summary):>5}  → {output_summary}")

    return {
        "status": "COMPLETED",
        "output_detailed": output_detailed,
        "output_summary": output_summary,
        "n_detailed": len(df_detail),
        "n_summary": len(df_summary),
    }
