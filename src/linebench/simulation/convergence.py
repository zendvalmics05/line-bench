"""Timing benchmark and convergence study for simulation engines (Phase 5.7 & 5.8).

Phase 5.7:
  - Benchmark execution speed of the unpaced recursion and SimPy validator across
    representative configurations (number of stations, units, replications).
  - Quantifies speedup of the vectorized recursion vs discrete-event SimPy.

Phase 5.8:
  - Convergence study: examines how throughput point estimates and standard errors
    evolve as a function of warm-up length (W) and total units (N).
  - Confirms and validates default protocol run parameters:
      warmup_units: 500, total_units: 5500, replications: 30.
  - Exports summary tables and plots to `results/` and `results/figures/`.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

if "IPython" not in sys.modules and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linebench.simulation.noise import generate_noise_factors
from linebench.simulation.simpy_line import simulate_simpy_line
from linebench.simulation.stations import build_station_times
from linebench.simulation.tandem import (
    simulate_tandem_replications,
)

__all__ = [
    "run_convergence_study",
    "run_phase5_study_pipeline",
    "run_timing_benchmark",
]


def run_timing_benchmark(
    stations_list: Sequence[int] = (3, 5, 10, 20),
    units_list: Sequence[int] = (1000, 3000, 5500),
    n_replications: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Benchmark runtime of unpaced recursion vs SimPy.

    Returns a DataFrame of runtimes and throughput execution speed (units simulated / sec).
    """
    rng = np.random.default_rng(seed)
    records = []

    for k in stations_list:
        for n_units in units_list:
            # Generate random station times
            S_batch = rng.lognormal(mean=2.0, sigma=0.3, size=(n_replications, n_units, k))

            # Time unpaced recursion
            t0 = time.perf_counter()
            simulate_tandem_replications(S_batch, buffers=1, warmup_units=min(500, n_units // 5))
            t1 = time.perf_counter()
            time_recursion = t1 - t0

            # Time SimPy on 1 replication for smaller configurations (SimPy is event-driven and slower)
            time_simpy_rep = np.nan
            speedup = np.nan
            if k <= 5 and n_units <= 3000:
                t0_sim = time.perf_counter()
                simulate_simpy_line(S_batch[0], buffers=1)
                t1_sim = time.perf_counter()
                time_simpy_rep = t1_sim - t0_sim
                speedup = (time_simpy_rep * n_replications) / time_recursion

            units_total = n_replications * n_units
            throughput_units_per_sec = units_total / time_recursion if time_recursion > 0 else 0.0

            records.append(
                {
                    "n_stations": k,
                    "n_units": n_units,
                    "n_replications": n_replications,
                    "total_units_simulated": units_total,
                    "time_recursion_sec": round(time_recursion, 5),
                    "time_simpy_single_rep_sec": round(time_simpy_rep, 5) if not np.isnan(time_simpy_rep) else None,
                    "speedup_vs_simpy": round(speedup, 1) if not np.isnan(speedup) else None,
                    "recursion_rate_units_per_sec": round(throughput_units_per_sec, 1),
                }
            )

    return pd.DataFrame(records)


def run_convergence_study(
    warmup_grid: Sequence[int] = (100, 250, 500, 1000),
    run_length_grid: Sequence[int] = (1000, 2000, 3500, 5500, 8000),
    n_replications: int = 30,
    seed: int = 12345,
) -> pd.DataFrame:
    """Evaluate throughput stability across warmup and total unit lengths.

    Tests on a representative line (Bowman-like 5 stations, CV=0.25, buffer b=1).
    """
    # 5 stations with heterogeneous base loads (e.g. 15, 18, 20, 19, 16)
    base_loads = np.array([15.0, 18.0, 20.0, 19.0, 16.0])
    k_stations = len(base_loads)
    cv = 0.25
    max_units = max(run_length_grid)

    # Pre-generate noise using CRN for max_units
    # We assign 1 pseudo-task per station for this structural convergence test
    dummy_assignments = {s + 1: [s + 1] for s in range(k_stations)}
    noise = generate_noise_factors(
        shape=(n_replications, max_units, k_stations),
        cv=cv,
        seed=seed,
        distribution="lognormal",
    )
    # Shape: (n_replications, max_units, k_stations)
    S_full = build_station_times(dummy_assignments, base_loads, noise)

    records = []
    for w in warmup_grid:
        for n_total in run_length_grid:
            if n_total <= w:
                continue

            # Slice prefix
            S_sub = S_full[:, :n_total, :]

            # Run batch tandem simulation with buffer b=1
            res = simulate_tandem_replications(S_sub, buffers=1, warmup_units=w)
            ths = res["throughput"]

            mean_th = float(np.mean(ths))
            std_th = float(np.std(ths, ddof=1))
            sem_th = float(stats.sem(ths))
            ci_half_width = float(stats.t.ppf(0.975, df=n_replications - 1) * sem_th)

            records.append(
                {
                    "warmup_units": w,
                    "total_units": n_total,
                    "measured_units": n_total - w,
                    "n_replications": n_replications,
                    "mean_throughput": round(mean_th, 6),
                    "std_throughput": round(std_th, 6),
                    "sem_throughput": round(sem_th, 6),
                    "ci_95_half_width": round(ci_half_width, 6),
                    "relative_ci_width_pct": round((ci_half_width / mean_th) * 100.0, 3) if mean_th > 0 else 0.0,
                }
            )

    return pd.DataFrame(records)


def plot_convergence_curves(
    df_convergence: pd.DataFrame,
    output_path: str | Path = "results/figures/simulation_convergence.png",
) -> None:
    """Plot convergence of mean throughput and 95% CI as total units increase."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    warmups = df_convergence["warmup_units"].unique()
    for w in warmups:
        sub = df_convergence[df_convergence["warmup_units"] == w].sort_values("total_units")
        ax1.plot(
            sub["total_units"],
            sub["mean_throughput"],
            marker="o",
            label=f"Warmup W = {w}",
        )
        ax1.fill_between(
            sub["total_units"],
            sub["mean_throughput"] - sub["ci_95_half_width"],
            sub["mean_throughput"] + sub["ci_95_half_width"],
            alpha=0.15,
        )

        ax2.plot(
            sub["total_units"],
            sub["relative_ci_width_pct"],
            marker="s",
            label=f"Warmup W = {w}",
        )

    ax1.set_xlabel("Total Units (N)")
    ax1.set_ylabel("Mean Throughput (units / time)")
    ax1.set_title("Throughput Estimate vs Total Units with 95% CI")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.set_xlabel("Total Units (N)")
    ax2.set_ylabel("95% CI Half-Width (% of Mean)")
    ax2.set_title("Relative Precision vs Total Units")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def run_phase5_study_pipeline(
    results_dir: str | Path = "results",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete Phase 5.7 and 5.8 study and save all outputs."""
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)
    fig_path = res_path / "figures"
    fig_path.mkdir(parents=True, exist_ok=True)

    # 1. Timing Benchmark
    df_timing = run_timing_benchmark()
    df_timing.to_csv(res_path / "simulation_timing_benchmark.csv", index=False)

    # 2. Convergence Study
    df_convergence = run_convergence_study()
    df_convergence.to_csv(res_path / "simulation_convergence_study.csv", index=False)

    # 3. Plot Convergence Curves
    plot_convergence_curves(df_convergence, fig_path / "simulation_convergence.png")

    return df_timing, df_convergence
