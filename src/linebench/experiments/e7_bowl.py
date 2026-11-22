"""Phase 6.5: E7 Bowl Experiment (Master Plan §6.3, Phase 6.5).

Evaluates whether deliberately unbalancing the workload in an unpaced line—the "bowl"
phenomenon where end stations have higher nominal workload than interior stations—improves
throughput compared to the even-load SO1 balance.

Model & Design:
1. Target profile: Bowl load shape where end stations receive slightly more work than interior stations.
   For station s in 1..m:
       target_ratio_s = 1.0 + amplitude * ( ((s - (m+1)/2) / ((m-1)/2))^2 - 0.5 )
   Normalized so the average equals nominal average load.
2. Built with CP-SAT by minimizing sum of absolute deviations from the bowl target loads,
   subject to cycle time and precedence constraints, with station count fixed at optimal m.
3. Compare simulated unpaced-line throughput of BOWL vs SO1 across CV levels and buffer sizes under CRN.
4. Report empirical differences and confidence intervals, evaluating literature claims.

Outputs
-------
- ``results/e7_bowl_summary.csv`` — per-(instance, CV, buffer) detailed comparison between BOWL and SO1.
- ``results/e7_bowl_paired.csv`` — aggregated paired comparison across instances by CV and buffer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from ortools.sat.python import cp_model

from linebench.instances.checker import check_solution
from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance
from linebench.instances.station_bounds import compute_station_bounds
from linebench.instances.subset import load_subset_instance
from linebench.simulation.noise import generate_noise_factors
from linebench.simulation.stations import build_station_times
from linebench.simulation.tandem import simulate_tandem_replications

__all__ = ["run_bowl_experiment"]

_BOOTSTRAP_REPS = 5000
_RNG_SEED = 2026


def solve_bowl_balance(
    instance: Instance,
    target_stations: int,
    amplitude: float = 0.15,
    time_limit_seconds: float = 15.0,
    graph: PrecedenceGraph | None = None,
) -> dict[int, list[int]] | None:
    """Solve for a bowl-shaped balance with fixed station count using CP-SAT."""
    m = target_stations
    if m < 3:
        # Bowl pattern requires at least 3 stations (start, interior, end)
        return None

    if graph is None:
        graph = PrecedenceGraph(instance)

    c = int(instance.cycle_time)
    tasks = sorted(instance.task_times.keys())
    t_times = {t: int(instance.task_times[t]) for t in tasks}
    total_work = sum(t_times.values())
    avg_load = total_work / m

    # Calculate bowl target loads for each station s in 1..m
    # Convex quadratic curve centered at (m+1)/2
    s_coords = np.array([float(s) for s in range(1, m + 1)])
    mid = (m + 1.0) / 2.0
    half_span = (m - 1.0) / 2.0
    u = (s_coords - mid) / half_span  # in [-1, 1]
    raw_weights = 1.0 + amplitude * (u**2 - 1.0 / 3.0)
    norm_weights = raw_weights / np.mean(raw_weights)
    target_loads = [round(w * avg_load) for w in norm_weights]

    station_bounds = compute_station_bounds(instance, graph, upper_bound_stations=m)

    model = cp_model.CpModel()
    x = {}
    for j in tasks:
        e_j = max(1, station_bounds.earliest[j])
        l_j = min(m, station_bounds.latest[j])
        if e_j > l_j:
            return None
        for s in range(e_j, l_j + 1):
            x[j, s] = model.NewBoolVar(f"x_{j}_{s}")

    for j in tasks:
        e_j = max(1, station_bounds.earliest[j])
        l_j = min(m, station_bounds.latest[j])
        model.Add(sum(x[j, s] for s in range(e_j, l_j + 1)) == 1)

    for i, j in instance.precedences:
        e_i = max(1, station_bounds.earliest[i])
        l_i = min(m, station_bounds.latest[i])
        e_j = max(1, station_bounds.earliest[j])
        l_j = min(m, station_bounds.latest[j])
        pos_i = sum(s * x[i, s] for s in range(e_i, l_i + 1))
        pos_j = sum(s * x[j, s] for s in range(e_j, l_j + 1))
        model.Add(pos_i <= pos_j)

    load_vars = []
    dev_vars = []
    for s in range(1, m + 1):
        stn_tasks = [j for j in tasks if (j, s) in x]
        load_s = model.NewIntVar(0, c, f"load_{s}")
        model.Add(load_s == sum(t_times[j] * x[j, s] for j in stn_tasks))
        load_vars.append(load_s)

        # Deviation from bowl target
        dev_s = model.NewIntVar(0, c, f"dev_{s}")
        t_load = min(c, max(0, target_loads[s - 1]))
        model.Add(dev_s >= load_s - t_load)
        model.Add(dev_s >= t_load - load_s)
        dev_vars.append(dev_s)

    # Minimize total absolute deviation from bowl profile
    model.Minimize(sum(dev_vars))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = 2
    solver.parameters.random_seed = 42

    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignments: dict[int, list[int]] = {s: [] for s in range(1, m + 1)}
        for j in tasks:
            for s in range(1, m + 1):
                if (j, s) in x and solver.Value(x[j, s]) == 1:
                    assignments[s].append(j)
                    break
        chk = check_solution(instance, assignments)
        if chk.is_feasible:
            return assignments

    return None


def _bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    n_boot: int = _BOOTSTRAP_REPS,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap percentile CI."""
    if rng is None:
        rng = np.random.default_rng(_RNG_SEED)
    valid_vals = values[~np.isnan(values)]
    n = len(valid_vals)
    if n < 2:
        v = float(valid_vals[0]) if n == 1 else float("nan")
        return v, v
    boot = rng.choice(valid_vals, size=(n_boot, n), replace=True)
    if statistic == "mean":
        boot_stats = boot.mean(axis=1)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi


def run_bowl_experiment(
    instances_yaml: str = "configs/instances.yaml",
    balance_pop_path: str = "results/balance_population.json",
    simulation_yaml: str = "configs/simulation.yaml",
    simulation_jsonl: str = "results/simulation_runs.jsonl",
    output_summary: str = "results/e7_bowl_summary.csv",
    output_paired: str = "results/e7_bowl_paired.csv",
    amplitude: float = 0.15,
) -> dict[str, Any]:
    """Run E7 bowl experiment comparing bowl vs SO1 balance under variability."""
    # 1. Load instances config
    with open(instances_yaml, "r", encoding="utf-8") as f:
        inst_cfg = yaml.safe_load(f)
    sim_instances = {entry["id"]: entry for entry in inst_cfg["simulation_instances"]}

    # 2. Load simulation config
    with open(simulation_yaml, "r", encoding="utf-8") as f:
        sim_cfg = yaml.safe_load(f)
    cv_levels: list[float] = [float(x) for x in sim_cfg["variability"]["cv_levels"]]
    buffers: list[int] = [int(b) for b in sim_cfg["lines"]["unpaced"]["buffers"]]
    reps: int = int(sim_cfg["run_parameters"]["replications"])
    total_units: int = int(sim_cfg["run_parameters"]["total_units"])
    warmup_units: int = int(sim_cfg["run_parameters"]["warmup_units"])
    primary_dist: str = sim_cfg["variability"]["distributions"]["primary"]

    # 3. Load balance population
    with open(balance_pop_path, "r", encoding="utf-8") as f:
        balance_data = json.load(f)

    # 4. Load existing simulation runs for SO1 lookup
    jsonl_path = Path(simulation_jsonl)
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    df_runs = pd.DataFrame(records)
    df_so1 = df_runs[(df_runs["line_type"] == "unpaced") & (df_runs["balance_type"] == "SO1")]
    so1_th_map = {
        (r["instance_id"], float(r["cv"]), int(r["buffer_capacity"])): float(r["mean_throughput"])
        for _, r in df_so1.iterrows()
    }

    summary_rows: list[dict] = []
    total_inst = len(sim_instances)

    for idx, (inst_id, entry) in enumerate(sim_instances.items(), start=1):
        if inst_id not in balance_data:
            continue
        p_entry = balance_data[inst_id]
        m_stations = int(p_entry["optimal_or_best_stations"])
        if m_stations < 3:
            print(f"[{idx}/{total_inst}] {inst_id}: skipped (m={m_stations} < 3)", flush=True)
            continue

        instance = load_subset_instance(entry)
        cycle_time = float(instance.cycle_time)
        n_tasks = len(instance.task_times)
        base_times = np.array([float(instance.task_times[j]) for j in range(1, n_tasks + 1)], dtype=np.float64)

        print(f"[{idx}/{total_inst}] {inst_id}: solving bowl balance (m={m_stations}, tasks={n_tasks})...", end="", flush=True)

        # Solve bowl balance
        bowl_assignments = solve_bowl_balance(instance, target_stations=m_stations, amplitude=amplitude)
        if bowl_assignments is None:
            print(" no bowl balance found", flush=True)
            continue
        print(" found! Simulating...", flush=True)

        # Simulate bowl balance across CVs and buffers using exact same CRN seeds as sweep runner
        for cv in cv_levels:
            seed_for_crn = int(abs(hash((inst_id, cv, primary_dist))) % (2**31 - 1))
            noise_tensor = generate_noise_factors(
                shape=(reps, total_units, n_tasks),
                cv=cv,
                seed=seed_for_crn,
                distribution=primary_dist,
            )
            S_tensor = build_station_times(bowl_assignments, base_times, noise_tensor)

            for b in buffers:
                unpaced_res = simulate_tandem_replications(
                    S_tensor,
                    buffers=b,
                    warmup_units=warmup_units,
                )
                th_bowl = float(np.mean(unpaced_res["throughput"]))
                rel_th_bowl = th_bowl * cycle_time

                so1_key = (inst_id, cv, b)
                th_so1 = so1_th_map.get(so1_key, float("nan"))
                rel_th_so1 = th_so1 * cycle_time if not np.isnan(th_so1) else float("nan")

                diff_th = th_bowl - th_so1 if not np.isnan(th_so1) else float("nan")
                pct_diff = (diff_th / th_so1) * 100.0 if not np.isnan(th_so1) and th_so1 > 0 else float("nan")

                summary_rows.append({
                    "instance_id": inst_id,
                    "cv": cv,
                    "buffer_capacity": b,
                    "m_stations": m_stations,
                    "cycle_time": cycle_time,
                    "th_bowl": th_bowl,
                    "th_so1": th_so1,
                    "rel_th_bowl": rel_th_bowl,
                    "rel_th_so1": rel_th_so1,
                    "diff_th_bowl_vs_so1": diff_th,
                    "pct_gain_bowl_vs_so1": pct_diff,
                })

            print(f"    CV={cv:.2f} simulated across all buffers", flush=True)

    df_summary = pd.DataFrame(summary_rows)

    # 5. Bootstrap Paired Analysis
    paired_rows: list[dict] = []
    rng = np.random.default_rng(_RNG_SEED)

    for (cv, b), grp in df_summary.groupby(["cv", "buffer_capacity"]):
        n_inst = len(grp)
        pct_gains = grp["pct_gain_bowl_vs_so1"].dropna().values
        m_gain = float(np.mean(pct_gains)) if len(pct_gains) > 0 else float("nan")
        lo, hi = _bootstrap_ci(pct_gains, statistic="mean", rng=rng) if len(pct_gains) >= 2 else (float("nan"), float("nan"))

        bowl_rel_vals = grp["rel_th_bowl"].dropna().values
        bowl_rel_m = float(np.mean(bowl_rel_vals)) if len(bowl_rel_vals) > 0 else float("nan")

        so1_rel_vals = grp["rel_th_so1"].dropna().values
        so1_rel_m = float(np.mean(so1_rel_vals)) if len(so1_rel_vals) > 0 else float("nan")

        # Win rate: proportion of instances where bowl strictly beats SO1
        win_rate = float(np.mean(pct_gains > 0.0)) * 100.0 if len(pct_gains) > 0 else float("nan")

        paired_rows.append({
            "cv": cv,
            "buffer_capacity": b,
            "n_instances": n_inst,
            "rel_th_bowl_mean": bowl_rel_m,
            "rel_th_so1_mean": so1_rel_m,
            "pct_gain_bowl_vs_so1_mean": m_gain,
            "pct_gain_bowl_boot_lo": lo,
            "pct_gain_bowl_boot_hi": hi,
            "bowl_win_rate_pct": win_rate,
        })

    df_paired = pd.DataFrame(paired_rows).sort_values(["cv", "buffer_capacity"])

    Path(output_summary).parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(output_summary, index=False)
    df_paired.to_csv(output_paired, index=False)

    print(f"Bowl summary rows written : {len(df_summary):>5}  → {output_summary}")
    print(f"Bowl paired rows written  : {len(df_paired):>5}  → {output_paired}")

    return {
        "status": "COMPLETED",
        "output_summary": output_summary,
        "output_paired": output_paired,
        "n_summary_rows": len(df_summary),
        "n_paired_rows": len(df_paired),
    }
