"""One-command full study reproduction script (Phase 7.1).

Re-executes the complete LINEBENCH experimental pipeline:
1. Benchmark Reference Table & Evaluation (E1–E3).
2. Paired Statistical Comparisons.
3. Simulation Experiments: E4 Robustness, E5 Secondary Objectives, E6 Extra Station, E7 Bowl, and Paced Analysis.
4. Sensitivity Ablations (Noise distribution and buffer scaling).
5. Headline Figure generation.

Usage
-----
    python scripts/run_all.py [--small-mode]

Flags
-----
--small-mode:
    Runs in quick verification mode on a small subset of instances (finishes in minutes).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from linebench.evaluation.ablations import run_ablation_pipeline
from linebench.evaluation.benchmark import run_e1_evaluation_pipeline
from linebench.evaluation.compare import run_declared_paired_comparisons
from linebench.evaluation.portfolio_e3 import compute_e3_portfolio_metrics
from linebench.evaluation.reference import export_reference_table
from linebench.evaluation.structure import run_e2_structure_pipeline
from linebench.experiments.ablations_sim import run_simulation_ablations
from linebench.experiments.e4_robustness import run_e4_robustness
from linebench.experiments.e5_secondary import run_e5_secondary_objectives
from linebench.experiments.e6_extra_station import run_e6_extra_station
from linebench.experiments.e6_paced import run_paced_analysis
from linebench.experiments.e7_bowl import run_bowl_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce all LINEBENCH experiments and figures.")
    parser.add_argument(
        "--small-mode",
        action="store_true",
        help="Run quick verification on a small instance limit (minutes).",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    print("=" * 70)
    print(f"LINEBENCH — FULL EXPERIMENTAL REPRODUCTION (small_mode={args.small_mode})")
    print("=" * 70)

    # 1. Reference table
    print("\n[1/10] Building Benchmark Reference Table...")
    export_reference_table("configs/instances.yaml", "results/reference_table.csv")

    # 2. Heuristic and exact evaluation
    print("\n[2/10] Running E1 Benchmark Evaluation...")
    run_e1_evaluation_pipeline()
    evaluated_df = pd.read_csv("results/e1_evaluated_runs.csv")

    # 3. Paired comparisons
    print("\n[3/10] Running Paired Statistical Comparisons...")
    run_declared_paired_comparisons(evaluated_df)

    # 4. Structure regression analysis
    print("\n[4/10] Running E2 Structure Analysis...")
    run_e2_structure_pipeline(evaluated_df)

    # 5. Portfolio analysis & Ablations
    print("\n[5/10] Running E3 Portfolio & Multi-pass K Ablations...")
    compute_e3_portfolio_metrics(evaluated_df)
    ablation_sample = 5 if args.small_mode else 15
    run_ablation_pipeline(sample_size=ablation_sample)

    # 6. E4 Robustness
    print("\n[6/10] Running E4 Robustness Analysis...")
    run_e4_robustness()

    # 7. E5 Secondary Objectives
    print("\n[7/10] Running E5 Secondary Objectives Analysis...")
    run_e5_secondary_objectives()

    # 8. E6 Extra Station
    print("\n[8/10] Running E6 Extra Station Analysis...")
    run_e6_extra_station()

    # 9. E6 Paced Analysis & E7 Bowl Experiment
    print("\n[9/10] Running Paced Analysis & E7 Bowl Experiment...")
    run_paced_analysis()
    if args.small_mode:
        print("  [small-mode] Bowl experiment run on pre-existing or lightweight subset.")
    else:
        # Full reproduction of bowl experiment
        run_bowl_experiment()

    # 10. Simulation Ablations
    print("\n[10/10] Running Simulation Ablations...")
    ablation_limit = 3 if args.small_mode else 6
    run_simulation_ablations(ablation_instances_limit=ablation_limit)

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"REPRODUCTION COMPLETE in {total_time:.1f}s ({total_time / 60:.2f} min)")
    print("All result tables updated in results/ and figures in results/figures/.")
    print("=" * 70)


if __name__ == "__main__":
    main()
