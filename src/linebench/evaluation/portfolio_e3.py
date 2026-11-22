"""E3 portfolio analysis module for LINEBENCH.

Analyzes the performance of heuristic portfolios against the exact CP-SAT solver:
  - Portfolio definitions:
      * Fast Deterministic Portfolio: Best across all 12 deterministic priority rules
        (RPW, UBT, MTS, MDS, EST, LST in Forward and Backward)
      * Full Heuristic Portfolio: Best across all 12 deterministic rules + Multi-Pass (K=500)
  - Evaluates:
      * Portfolio hit rate against reference optima
      * Station gap and relative gap
      * Total portfolio CPU runtime vs Exact solver runtime (runtime ratio)
      * Coverage: percentage of instances where heuristic portfolio equals or beats exact solver
  - Stratified reporting by size tier and stratum

Ref: Master Plan §5.3 Module D & RQ3, Implementation Plan Phase 4.5.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

if "IPython" not in sys.modules and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioSummary:
    """Performance summary of a heuristic portfolio."""

    portfolio_name: str
    n_instances: int
    hit_rate_pct: float
    mean_gap: float
    median_gap: float
    mean_relative_gap_pct: float
    total_runtime_seconds: float
    mean_runtime_per_instance_ms: float
    runtime_ratio_vs_exact: float | None = None
    match_or_beat_exact_pct: float | None = None


def compute_e3_portfolio_metrics(
    evaluated_df: pd.DataFrame,
    exact_df: pd.DataFrame | None = None,
    output_dir: str = "results",
) -> dict[str, Any]:
    """Computes comprehensive E3 portfolio metrics and runtime comparisons."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Isolate Heuristic Runs
    heur_df = evaluated_df[evaluated_df["method_family"] == "heuristic"].copy()

    # Define portfolios
    det_methods = [
        "RPW_FORWARD", "RPW_BACKWARD",
        "UBT_FORWARD", "UBT_BACKWARD",
        "MTS_FORWARD", "MTS_BACKWARD",
        "MDS_FORWARD", "MDS_BACKWARD",
        "EST_FORWARD", "EST_BACKWARD",
        "LST_FORWARD", "LST_BACKWARD",
    ]

    # Fast Deterministic Portfolio
    det_sub = heur_df[heur_df["method"].isin(det_methods)]
    det_portfolio = (
        det_sub.groupby("instance_id")
        .agg(
            best_stations=("stations", "min"),
            total_duration=("duration_seconds", "sum"),
            reference_stations=("reference_stations", "first"),
            lower_bound=("lower_bound", "first"),
            stratum=("stratum", "first"),
            size_tier=("size_tier", "first"),
            num_tasks=("num_tasks", "first"),
        )
        .reset_index()
    )
    det_portfolio["station_gap"] = det_portfolio["best_stations"] - det_portfolio["reference_stations"]
    det_portfolio["relative_gap"] = det_portfolio["station_gap"] / det_portfolio["reference_stations"]
    det_portfolio["hit_reference"] = det_portfolio["best_stations"] == det_portfolio["reference_stations"]

    # Full Portfolio (Deterministic + Multi-Pass)
    full_portfolio = (
        heur_df.groupby("instance_id")
        .agg(
            best_stations=("stations", "min"),
            total_duration=("duration_seconds", "sum"),
            reference_stations=("reference_stations", "first"),
            lower_bound=("lower_bound", "first"),
            stratum=("stratum", "first"),
            size_tier=("size_tier", "first"),
            num_tasks=("num_tasks", "first"),
        )
        .reset_index()
    )
    full_portfolio["station_gap"] = full_portfolio["best_stations"] - full_portfolio["reference_stations"]
    full_portfolio["relative_gap"] = full_portfolio["station_gap"] / full_portfolio["reference_stations"]
    full_portfolio["hit_reference"] = full_portfolio["best_stations"] == full_portfolio["reference_stations"]

    # Compute Summaries
    summaries: list[dict[str, Any]] = []

    # 1. Deterministic Portfolio Summary
    det_sum = PortfolioSummary(
        portfolio_name="PORTFOLIO_DETERMINISTIC_12_RULES",
        n_instances=len(det_portfolio),
        hit_rate_pct=float(det_portfolio["hit_reference"].mean() * 100),
        mean_gap=float(det_portfolio["station_gap"].mean()),
        median_gap=float(det_portfolio["station_gap"].median()),
        mean_relative_gap_pct=float(det_portfolio["relative_gap"].mean() * 100),
        total_runtime_seconds=float(det_portfolio["total_duration"].sum()),
        mean_runtime_per_instance_ms=float(det_portfolio["total_duration"].mean() * 1000),
    )
    summaries.append(asdict(det_sum))

    # 2. Full Portfolio Summary
    full_sum = PortfolioSummary(
        portfolio_name="PORTFOLIO_FULL_WITH_MULTIPASS",
        n_instances=len(full_portfolio),
        hit_rate_pct=float(full_portfolio["hit_reference"].mean() * 100),
        mean_gap=float(full_portfolio["station_gap"].mean()),
        median_gap=float(full_portfolio["station_gap"].median()),
        mean_relative_gap_pct=float(full_portfolio["relative_gap"].mean() * 100),
        total_runtime_seconds=float(full_portfolio["total_duration"].sum()),
        mean_runtime_per_instance_ms=float(full_portfolio["total_duration"].mean() * 1000),
    )
    summaries.append(asdict(full_sum))

    # Compare against Exact Solver if available
    exact_comparison = None
    if exact_df is not None and not exact_df.empty:
        exact_sub = exact_df[exact_df["method"] == "CPSAT"].copy()
        merged = full_portfolio.merge(
            exact_sub[["instance_id", "stations", "duration_seconds", "is_proven_optimal"]],
            on="instance_id",
            how="inner",
            suffixes=("_portfolio", "_exact"),
        )
        if not merged.empty:
            match_or_beat = float((merged["best_stations"] <= merged["stations"]).mean() * 100)
            exact_runtime = float(merged["duration_seconds"].sum())
            ratio = float(full_sum.total_runtime_seconds / exact_runtime) if exact_runtime > 0 else 0.0

            exact_comparison = {
                "n_paired_instances": len(merged),
                "portfolio_matches_or_beats_exact_pct": match_or_beat,
                "exact_total_runtime_seconds": exact_runtime,
                "portfolio_total_runtime_seconds": full_sum.total_runtime_seconds,
                "runtime_ratio_heur_to_exact": ratio,
            }

    df_portfolios = pd.DataFrame(summaries)
    df_portfolios.to_csv(out_p / "e3_portfolio_summary.csv", index=False)

    # Stratified portfolio performance by size tier
    stratified = (
        full_portfolio.groupby(["stratum", "size_tier"])
        .agg(
            n_instances=("instance_id", "count"),
            hit_rate_pct=("hit_reference", lambda x: np.mean(x) * 100),
            mean_gap=("station_gap", "mean"),
            mean_relative_gap_pct=("relative_gap", lambda x: np.mean(x) * 100),
            mean_runtime_ms=("total_duration", lambda x: np.mean(x) * 1000),
        )
        .reset_index()
    )
    stratified.to_csv(out_p / "e3_portfolio_stratified.csv", index=False)

    # Plot portfolio comparison
    _fig, ax = plt.subplots(figsize=(8, 5))
    names = ["Deterministic Portfolio", "Full Portfolio (w/ Multi-Pass)"]
    hit_rates = [det_sum.hit_rate_pct, full_sum.hit_rate_pct]
    gaps = [det_sum.mean_relative_gap_pct, full_sum.mean_relative_gap_pct]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width / 2, hit_rates, width, label="Hit Rate (%)", color="#1f77b4")
    ax.bar(x + width / 2, gaps, width, label="Mean Relative Gap (%)", color="#ff7f0e")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("E3: Heuristic Portfolio Benchmark Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig_path = str(out_p / "figures" / "e3_portfolio_performance.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return {
        "status": "COMPLETED",
        "portfolio_summary": summaries,
        "exact_comparison": exact_comparison,
        "summary_csv": str(out_p / "e3_portfolio_summary.csv"),
        "stratified_csv": str(out_p / "e3_portfolio_stratified.csv"),
        "figure_path": fig_path,
    }
