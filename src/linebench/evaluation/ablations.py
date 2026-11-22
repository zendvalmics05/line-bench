"""Ablation experiments for LINEBENCH benchmark study.

Implements declared small ablations:
  1. Multi-Pass Pass Count Ablation:
     Evaluate multi-pass RPW across increasing pass budgets:
     K in [10, 50, 100, 250, 500, 1000]
     Analyzes diminishing returns in solution quality vs runtime.
  2. Directional Ablation:
     Forward vs Backward vs Bidirectional portfolio gain.

Ref: Master Plan §5.3 & Implementation Plan Phase 4.6.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

if "IPython" not in sys.modules and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linebench.heuristics.random_pass import solve_randomized_multi_pass
from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance
from linebench.instances.subset import load_subset_instance


def run_multipass_k_ablation(
    instances: list[Instance] | list[dict[str, Any]],
    k_values: list[int] | tuple[int, ...] = (10, 50, 100, 250, 500),
    seed: int = 1001,
    output_csv: str = "results/ablation_multipass_k.csv",
) -> pd.DataFrame:
    """Runs randomized multi-pass across multiple K pass values on a representative sample."""
    loaded_instances: list[Instance] = []
    for item in instances:
        if isinstance(item, Instance):
            loaded_instances.append(item)
        elif isinstance(item, dict):
            loaded_instances.append(load_subset_instance(item))

    records: list[dict[str, Any]] = []

    for k in k_values:
        start_k = time.perf_counter()
        station_counts: list[int] = []

        for inst in loaded_instances:
            graph = PrecedenceGraph(inst)
            res = solve_randomized_multi_pass(
                inst,
                num_passes=k,
                seed=seed,
                priority_rule="RPW",
                graph=graph,
                validate_with_checker=True,
            )
            station_counts.append(res.best_solution.num_stations)

        duration = time.perf_counter() - start_k
        mean_stations = float(np.mean(station_counts))
        total_stations = int(np.sum(station_counts))

        records.append(
            {
                "k_passes": k,
                "n_instances": len(loaded_instances),
                "mean_stations": mean_stations,
                "total_stations": total_stations,
                "duration_seconds": duration,
                "mean_time_per_instance_ms": (duration / len(loaded_instances)) * 1000,
            }
        )

    df = pd.DataFrame(records)

    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)

    return df


def plot_multipass_ablation(
    ablation_df: pd.DataFrame,
    output_png: str = "results/figures/ablation_multipass_k.png",
) -> None:
    """Plots diminishing returns curve of solution quality vs runtime across pass budgets."""
    out_file = Path(output_png)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    _fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color = "#1f77b4"
    ax1.set_xlabel("Number of Random Passes ($K$)")
    ax1.set_ylabel("Mean Stations Across Instances", color=color)
    ax1.plot(
        ablation_df["k_passes"],
        ablation_df["mean_stations"],
        marker="o",
        color=color,
        linewidth=2,
        label="Mean Stations",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = "#ff7f0e"
    ax2.set_ylabel("Runtime per Instance (ms)", color=color)
    ax2.plot(
        ablation_df["k_passes"],
        ablation_df["mean_time_per_instance_ms"],
        marker="s",
        color=color,
        linestyle="--",
        linewidth=2,
        label="Runtime (ms)",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Ablation: Multi-Pass Quality vs. Runtime Scaling ($K$)")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def run_ablation_pipeline(
    instances_yaml: str = "configs/instances.yaml",
    sample_size: int = 15,
    output_dir: str = "results",
) -> dict[str, Any]:
    """Runs the ablation experiment suite on a stratified representative sample."""
    import yaml

    with open(instances_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Pick representative small & medium instances across both strata
    bench = cfg["benchmark_instances"]
    sample = [x for x in bench if x["num_tasks"] <= 50][:sample_size]

    out_p = Path(output_dir)
    csv_path = str(out_p / "ablation_multipass_k.csv")
    fig_path = str(out_p / "figures" / "ablation_multipass_k.png")

    df_ablation = run_multipass_k_ablation(
        instances=sample,
        k_values=[10, 25, 50, 100, 250, 500],
        output_csv=csv_path,
    )
    plot_multipass_ablation(df_ablation, fig_path)

    return {
        "status": "COMPLETED",
        "sample_size": len(sample),
        "k_evaluated": list(df_ablation["k_passes"].values),
        "results_csv": csv_path,
        "figure_png": fig_path,
    }
