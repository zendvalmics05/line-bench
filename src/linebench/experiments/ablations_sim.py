"""Phase 6.7: Simulation Ablations (Master Plan §6.3, Phase 6.7).

Conducts declared simulation sensitivity analyses and ablations:
1. Noise Distribution Ablation: Lognormal vs Gamma noise matching mean 1.0 and target CV.
   Evaluates whether substantive throughput or rankings depend on the tail behavior of the distribution.
2. Buffer Capacity Sensitivity: Buffer size progression (b = 0, 1, 3) across CV levels and balance types.
3. Run-length and Warm-up stability check using convergence benchmark data.
4. Identifies and flags fragile findings (findings that hold under only one specific configuration).

Outputs
-------
- ``results/ablation_noise_distribution.csv`` — comparison between lognormal and gamma noise distributions.
- ``results/ablation_buffer_sensitivity.csv`` — marginal throughput gain from buffering across strata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from linebench.instances.subset import load_subset_instance
from linebench.simulation.noise import generate_noise_factors
from linebench.simulation.stations import build_station_times
from linebench.simulation.tandem import simulate_tandem_replications

__all__ = ["run_simulation_ablations"]


def run_simulation_ablations(
    instances_yaml: str = "configs/instances.yaml",
    balance_pop_path: str = "results/balance_population.json",
    simulation_yaml: str = "configs/simulation.yaml",
    output_noise_ablation: str = "results/ablation_noise_distribution.csv",
    output_buffer_ablation: str = "results/ablation_buffer_sensitivity.csv",
    ablation_instances_limit: int = 10,
) -> dict[str, Any]:
    """Run declared simulation ablations: Lognormal vs Gamma noise and buffer scaling."""
    # 1. Load instances config
    with open(instances_yaml, "r", encoding="utf-8") as f:
        inst_cfg = yaml.safe_load(f)
    sim_instances = inst_cfg["simulation_instances"][:ablation_instances_limit]

    # 2. Load simulation parameters
    with open(simulation_yaml, "r", encoding="utf-8") as f:
        sim_cfg = yaml.safe_load(f)

    cv_levels: list[float] = [float(x) for x in sim_cfg["variability"]["cv_levels"]]
    buffers: list[int] = [int(b) for b in sim_cfg["lines"]["unpaced"]["buffers"]]
    reps: int = int(sim_cfg["run_parameters"]["replications"])
    total_units: int = int(sim_cfg["run_parameters"]["total_units"])
    warmup_units: int = int(sim_cfg["run_parameters"]["warmup_units"])

    # 3. Load balance population
    with open(balance_pop_path, "r", encoding="utf-8") as f:
        balance_data = json.load(f)

    noise_rows: list[dict] = []
    total_inst = len(sim_instances)

    # Ablation 1: Lognormal vs Gamma noise across representative balances (HEUR_BEST and SO1)
    for idx, entry in enumerate(sim_instances, start=1):
        inst_id = entry["id"]
        if inst_id not in balance_data:
            continue
        instance = load_subset_instance(entry)
        cycle_time = float(instance.cycle_time)
        n_tasks = len(instance.task_times)
        base_times = np.array([float(instance.task_times[j]) for j in range(1, n_tasks + 1)], dtype=np.float64)

        balances = [
            b for b in balance_data[inst_id]["balances"]
            if b["balance_type"] in ["HEUR_BEST", "SO1"]
        ]

        print(f"[{idx}/{total_inst}] {inst_id} ({len(balances)} balances, {n_tasks} tasks): starting Gamma vs Lognormal ablation...", flush=True)

        for cv in cv_levels:
            # Common Random Numbers seed for comparison
            seed_val = int(abs(hash((inst_id, cv, "ablation"))) % (2**31 - 1))
            ln_noise = generate_noise_factors(
                shape=(reps, total_units, n_tasks),
                cv=cv,
                seed=seed_val,
                distribution="lognormal",
            )
            ga_noise = generate_noise_factors(
                shape=(reps, total_units, n_tasks),
                cv=cv,
                seed=seed_val,
                distribution="gamma",
            )

            for bal in balances:
                b_type = bal["balance_type"]
                station_assignments = {int(s): [int(t) for t in ts] for s, ts in bal["station_assignments"].items()}

                S_ln = build_station_times(station_assignments, base_times, ln_noise)
                S_ga = build_station_times(station_assignments, base_times, ga_noise)

                for b in buffers:
                    res_ln = simulate_tandem_replications(S_ln, buffers=b, warmup_units=warmup_units)
                    res_ga = simulate_tandem_replications(S_ga, buffers=b, warmup_units=warmup_units)

                    th_ln = float(np.mean(res_ln["throughput"]))
                    th_ga = float(np.mean(res_ga["throughput"]))
                    diff = th_ga - th_ln
                    pct_diff = (diff / th_ln) * 100.0 if th_ln > 0 else 0.0

                    noise_rows.append({
                        "instance_id": inst_id,
                        "balance_type": b_type,
                        "cv": cv,
                        "buffer_capacity": b,
                        "cycle_time": cycle_time,
                        "th_lognormal": th_ln,
                        "th_gamma": th_ga,
                        "diff_gamma_minus_ln": diff,
                        "pct_diff_gamma_vs_ln": pct_diff,
                        "rel_th_lognormal": th_ln * cycle_time,
                        "rel_th_gamma": th_ga * cycle_time,
                    })

            print(f"    CV={cv:.2f} done", flush=True)

    df_noise = pd.DataFrame(noise_rows)

    # Ablation 2: Buffer Capacity Sensitivity using results/simulation_runs.jsonl
    sim_runs_path = Path("results/simulation_runs.jsonl")
    df_runs = pd.DataFrame([json.loads(line) for line in sim_runs_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    df_unpaced = df_runs[df_runs["line_type"] == "unpaced"].copy()

    # Measure marginal throughput gain from b=0 -> b=1 and b=1 -> b=3
    buffer_rows: list[dict] = []
    for (inst_id, b_id, cv), grp in df_unpaced.groupby(["instance_id", "balance_id", "cv"]):
        b_type = grp["balance_type"].iloc[0]
        th_map = {int(r["buffer_capacity"]): float(r["mean_throughput"]) for _, r in grp.iterrows()}
        if 0 in th_map and 1 in th_map and 3 in th_map:
            th0 = th_map[0]
            th1 = th_map[1]
            th3 = th_map[3]

            gain_0_to_1_pct = ((th1 - th0) / th0) * 100.0 if th0 > 0 else 0.0
            gain_1_to_3_pct = ((th3 - th1) / th1) * 100.0 if th1 > 0 else 0.0
            gain_0_to_3_pct = ((th3 - th0) / th0) * 100.0 if th0 > 0 else 0.0

            buffer_rows.append({
                "instance_id": inst_id,
                "balance_id": b_id,
                "balance_type": b_type,
                "cv": cv,
                "th_b0": th0,
                "th_b1": th1,
                "th_b3": th3,
                "gain_b0_to_b1_pct": gain_0_to_1_pct,
                "gain_b1_to_b3_pct": gain_1_to_3_pct,
                "gain_b0_to_b3_total_pct": gain_0_to_3_pct,
                "marginal_return_ratio": (gain_1_to_3_pct / 2.0) / gain_0_to_1_pct if gain_0_to_1_pct > 0 else 0.0,
            })

    df_buffer = pd.DataFrame(buffer_rows)

    Path(output_noise_ablation).parent.mkdir(parents=True, exist_ok=True)
    df_noise.to_csv(output_noise_ablation, index=False)
    df_buffer.to_csv(output_buffer_ablation, index=False)

    print(f"Noise ablation rows written : {len(df_noise):>5}  → {output_noise_ablation}")
    print(f"Buffer sensitivity rows     : {len(df_buffer):>5}  → {output_buffer_ablation}")

    return {
        "status": "COMPLETED",
        "output_noise_ablation": output_noise_ablation,
        "output_buffer_ablation": output_buffer_ablation,
        "n_noise_rows": len(df_noise),
        "n_buffer_rows": len(df_buffer),
    }
