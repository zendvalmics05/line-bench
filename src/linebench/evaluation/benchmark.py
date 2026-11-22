"""E1 benchmark evaluation module for LINEBENCH.

Computes benchmark evaluation metrics for each heuristic and exact method:
  - Station gap: m - m_ref
  - Relative station gap: (m - m_ref) / m_ref
  - Hit rate: I(m == m_ref)
  - Lower bound gap: (m - LB) / m
  - Runtime: duration in seconds
  - Share of exact-solver results proven optimal
  - Stratified aggregations across:
      * Method (priority rules, multi-pass, CP-SAT)
      * Stratum (classical vs generated)
      * Size tier (small, medium, large, very large)
      * Order strength / graph type / task-time distribution / tightness

Generates standardized summary tables and evaluation figures.

Ref: Master Plan §5.3, Implementation Plan Phase 4.2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

if "IPython" not in sys.modules and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from linebench.evaluation.reference import build_reference_table


def load_benchmark_run_results(
    heuristic_results_path: str = "results/heuristic_runs.jsonl",
    exact_results_path: str | None = "results/exact_runs.jsonl",
) -> pd.DataFrame:
    """Loads all logged benchmark runs from JSONL files into a consolidated DataFrame."""
    records: list[dict[str, Any]] = []

    # 1. Load Heuristic Runs
    heur_p = Path(heuristic_results_path)
    if heur_p.exists():
        with open(heur_p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                extra = rec.get("extra", {})
                factors = rec.get("factor_settings", {})
                records.append(
                    {
                        "instance_id": rec["instance_id"],
                        "method": rec["method"],
                        "method_family": "heuristic",
                        "rule": factors.get("rule"),
                        "direction": factors.get("direction"),
                        "stations": int(rec["value"]),
                        "duration_seconds": float(extra.get("duration_seconds", 0.0)),
                        "is_feasible": bool(extra.get("is_feasible", False)),
                        "is_proven_optimal": False,
                        "status": "FEASIBLE" if extra.get("is_feasible") else "INFEASIBLE",
                    }
                )

    # 2. Load Exact Runs (if available)
    if exact_results_path:
        exact_p = Path(exact_results_path)
        if exact_p.exists():
            with open(exact_p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    extra = rec.get("extra", {})
                    factors = rec.get("factor_settings", {})
                    st_val = int(rec["value"])
                    records.append(
                        {
                            "instance_id": rec["instance_id"],
                            "method": rec["method"],
                            "method_family": "exact",
                            "rule": "CPSAT",
                            "direction": "exact",
                            "stations": st_val if st_val > 0 else None,
                            "duration_seconds": float(extra.get("duration_seconds", 0.0)),
                            "is_feasible": bool(extra.get("is_feasible", False)),
                            "is_proven_optimal": bool(extra.get("is_proven_optimal", False)),
                            "status": extra.get("status", "UNKNOWN"),
                        }
                    )

    df = pd.DataFrame(records)
    return df


def compute_e1_evaluation_metrics(
    runs_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merges benchmark runs with reference values and computes standard E1 metrics."""
    # Merge runs with reference data
    merged = runs_df.merge(
        reference_df,
        on="instance_id",
        how="inner",
        suffixes=("", "_ref_table"),
    )

    # Compute metrics
    # Gap = m - m_ref
    # Hit = I(m == m_ref)
    # Rel Gap = (m - m_ref) / m_ref
    merged["station_gap"] = merged["stations"] - merged["reference_stations"]
    merged["relative_gap"] = merged["station_gap"] / merged["reference_stations"]
    # Handle possible negative gaps if heuristic beat an unproven reference
    merged["hit_reference"] = merged["stations"] == merged["reference_stations"]
    merged["beat_or_hit_reference"] = merged["stations"] <= merged["reference_stations"]

    merged["lb_gap"] = merged["stations"] - merged["lower_bound"]
    merged["relative_lb_gap"] = merged["lb_gap"] / merged["lower_bound"]

    return merged


def summarize_e1_by_method(evaluated_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates E1 evaluation metrics across all benchmark methods."""
    summary = (
        evaluated_df.groupby("method")
        .agg(
            instances_evaluated=("instance_id", "count"),
            feasible_rate=("is_feasible", "mean"),
            mean_gap=("station_gap", "mean"),
            median_gap=("station_gap", "median"),
            max_gap=("station_gap", "max"),
            mean_relative_gap_pct=("relative_gap", lambda x: np.mean(x) * 100),
            hit_rate_pct=("hit_reference", lambda x: np.mean(x) * 100),
            beat_or_hit_pct=("beat_or_hit_reference", lambda x: np.mean(x) * 100),
            mean_runtime_ms=("duration_seconds", lambda x: np.mean(x) * 1000),
            median_runtime_ms=("duration_seconds", lambda x: np.median(x) * 1000),
            proven_optimal_share_pct=("is_proven_optimal", lambda x: np.mean(x) * 100),
        )
        .reset_index()
    )

    summary.sort_values(by="mean_relative_gap_pct", inplace=True)
    return summary


def summarize_e1_by_stratum(
    evaluated_df: pd.DataFrame,
    group_cols: list[str] | tuple[str, ...] = ("stratum", "size_tier", "method"),
) -> pd.DataFrame:
    """Aggregates E1 evaluation metrics by experimental strata (e.g. stratum, size tier)."""
    valid_cols = [c for c in group_cols if c in evaluated_df.columns]
    summary = (
        evaluated_df.groupby(valid_cols)
        .agg(
            instance_count=("instance_id", "count"),
            mean_gap=("station_gap", "mean"),
            mean_relative_gap_pct=("relative_gap", lambda x: np.mean(x) * 100),
            hit_rate_pct=("hit_reference", lambda x: np.mean(x) * 100),
            mean_runtime_ms=("duration_seconds", lambda x: np.mean(x) * 1000),
        )
        .reset_index()
    )
    return summary


def plot_e1_gap_distributions(
    evaluated_df: pd.DataFrame,
    output_png: str = "results/figures/e1_gap_distributions.png",
) -> None:
    """Generates boxplot figure of station gap distributions by heuristic method."""
    out_file = Path(output_png)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    methods = sorted(evaluated_df["method"].unique())
    data = [evaluated_df[evaluated_df["method"] == m]["station_gap"].dropna() for m in methods]

    plt.boxplot(data, tick_labels=methods, orientation="vertical", patch_artist=True)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Method")
    plt.ylabel("Station Gap $(m - m^*)$")
    plt.title("E1: Optimality Gap Distributions Across Benchmark Subset")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_e1_hit_rate_by_size(
    evaluated_df: pd.DataFrame,
    output_png: str = "results/figures/e1_hit_rate_by_size.png",
) -> None:
    """Generates bar chart of hit rate (%) stratified by instance size tier."""
    out_file = Path(output_png)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Pick key representative methods for clarity
    key_methods = [
        "RPW_FORWARD",
        "RPW_BACKWARD",
        "UBT_FORWARD",
        "MULTI_PASS_RPW_K500",
    ]
    sub = evaluated_df[evaluated_df["method"].isin(key_methods)].copy()
    if sub.empty:
        sub = evaluated_df.copy()

    agg = (
        sub.groupby(["size_tier", "method"])["hit_reference"]
        .mean()
        .unstack()
        .fillna(0.0)
        * 100
    )

    agg.plot(kind="bar", figsize=(10, 5), width=0.8)
    plt.xlabel("Instance Size Tier")
    plt.ylabel("Hit Rate (%)")
    plt.title("E1: Hit Rate by Problem Size Tier")
    plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def run_e1_evaluation_pipeline(
    instances_yaml: str = "configs/instances.yaml",
    heuristic_results_path: str = "results/heuristic_runs.jsonl",
    exact_results_path: str | None = "results/exact_runs.jsonl",
    output_dir: str = "results",
) -> dict[str, Any]:
    """Executes the full E1 benchmark evaluation pipeline and exports tables and figures."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Reference Table
    ref_df = build_reference_table(instances_yaml)
    ref_df.to_csv(out_p / "reference_table.csv", index=False)

    # 2. Runs & Metric Evaluation
    runs_df = load_benchmark_run_results(heuristic_results_path, exact_results_path)
    if runs_df.empty:
        raise ValueError(f"No run results found at {heuristic_results_path}")

    eval_df = compute_e1_evaluation_metrics(runs_df, ref_df)
    eval_df.to_csv(out_p / "e1_evaluated_runs.csv", index=False)

    # 3. Aggregation Tables
    method_summary = summarize_e1_by_method(eval_df)
    method_summary.to_csv(out_p / "e1_method_summary.csv", index=False)

    stratum_summary = summarize_e1_by_stratum(eval_df)
    stratum_summary.to_csv(out_p / "e1_stratum_summary.csv", index=False)

    # 4. Figures
    fig_dir = out_p / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_e1_gap_distributions(eval_df, str(fig_dir / "e1_gap_distributions.png"))
    plot_e1_hit_rate_by_size(eval_df, str(fig_dir / "e1_hit_rate_by_size.png"))

    return {
        "status": "COMPLETED",
        "total_evaluated_records": len(eval_df),
        "methods_evaluated": list(method_summary["method"].values),
        "reference_table_path": str(out_p / "reference_table.csv"),
        "method_summary_path": str(out_p / "e1_method_summary.csv"),
        "stratum_summary_path": str(out_p / "e1_stratum_summary.csv"),
        "figures_generated": [
            str(fig_dir / "e1_gap_distributions.png"),
            str(fig_dir / "e1_hit_rate_by_size.png"),
        ],
    }
