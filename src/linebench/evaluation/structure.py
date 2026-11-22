"""E2 instance structural analysis for LINEBENCH.

Analyzes how structural instance measures influence heuristic performance:
  - Factors: Order Strength (OS), task count n, cycle time c, tightness ratio, graph density
  - Multiple Linear Regression of heuristic relative gap on structural predictors
  - Logistic Regression of hit-reference (I(m == m*)) on structural predictors
  - Variance Inflation Factors (VIF) to detect and warn about predictor multicollinearity
  - Stratified tabulations and publication figures

Ref: Master Plan §5.3 Module D, Implementation Plan Phase 4.4.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

if "IPython" not in sys.modules and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from linebench.instances.measures import compute_instance_measures
from linebench.instances.subset import load_subset_instance


def augment_with_instance_measures(
    evaluated_df: pd.DataFrame,
    instances_yaml: str = "configs/instances.yaml",
) -> pd.DataFrame:
    """Computes order strength and graph structure measures for each evaluated instance."""
    import yaml

    with open(instances_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bench_dict = {x["id"]: x for x in cfg["benchmark_instances"]}

    # Cache measures by instance_id
    measures_cache: dict[str, dict[str, Any]] = {}
    unique_ids = evaluated_df["instance_id"].unique()

    for inst_id in unique_ids:
        if inst_id in bench_dict:
            entry = bench_dict[inst_id]
            inst = load_subset_instance(entry)
            m = compute_instance_measures(inst)
            measures_cache[inst_id] = {
                "order_strength": m.order_strength,
                "tightness_ratio": m.tightness_ratio,
                "task_time_mean": m.t_mean,
                "task_time_std": m.t_std,
                "task_time_cv": (m.t_std / m.t_mean) if m.t_mean > 0 else 0.0,
            }

    measures_df = pd.DataFrame.from_dict(measures_cache, orient="index")
    measures_df.index.name = "instance_id"
    measures_df.reset_index(inplace=True)

    augmented = evaluated_df.merge(measures_df, on="instance_id", how="left")
    return augmented


def fit_structural_regression(
    augmented_df: pd.DataFrame,
    method: str = "RPW_FORWARD",
) -> dict[str, Any]:
    """Fits an OLS regression model predicting heuristic relative gap from structural measures.

    Predictors:
      - order_strength: DAG precedence constraint density [0, 1]
      - num_tasks: problem size n
      - tightness_ratio: sum(t_j) / (n * c)
      - task_time_cv: coefficient of variation of deterministic task durations

    Also computes Variance Inflation Factors (VIF) to report multicollinearity warnings.
    """
    sub = augmented_df[augmented_df["method"] == method].dropna(
        subset=["relative_gap", "order_strength", "num_tasks", "tightness_ratio", "task_time_cv"]
    ).copy()

    if len(sub) < 20:
        raise ValueError(f"Insufficient samples for regression on method {method}: {len(sub)}")

    predictors = ["order_strength", "num_tasks", "tightness_ratio", "task_time_cv"]
    X = sub[predictors]
    X_const = sm.add_constant(X)
    y = sub["relative_gap"] * 100.0  # Percentage relative gap

    ols_model = sm.OLS(y, X_const).fit()

    # Compute Variance Inflation Factors (VIF)
    vif_data = []
    for i, col in enumerate(predictors):
        vif_val = variance_inflation_factor(X.values, i)
        vif_data.append({"predictor": col, "vif": float(vif_val)})

    vif_df = pd.DataFrame(vif_data)
    multicollinearity_warning = bool((vif_df["vif"] > 10.0).any())

    # Format summary table
    coef_table = pd.DataFrame({
        "predictor": ols_model.params.index,
        "coefficient": ols_model.params.values,
        "std_error": ols_model.bse.values,
        "t_stat": ols_model.tvalues.values,
        "p_value": ols_model.pvalues.values,
    })

    return {
        "method": method,
        "nobs": int(ols_model.nobs),
        "r_squared": float(ols_model.rsquared),
        "adj_r_squared": float(ols_model.rsquared_adj),
        "f_pvalue": float(ols_model.f_pvalue),
        "coefficients": coef_table.to_dict(orient="records"),
        "vif": vif_df.to_dict(orient="records"),
        "multicollinearity_warning": multicollinearity_warning,
        "summary_text": str(ols_model.summary()),
    }


def plot_e2_gap_vs_order_strength(
    augmented_df: pd.DataFrame,
    output_png: str = "results/figures/e2_gap_vs_order_strength.png",
) -> None:
    """Plots scatter and trend of optimality gap against order strength."""
    out_file = Path(output_png)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    _fig, ax = plt.subplots(figsize=(9, 5))
    key_methods = ["RPW_FORWARD", "MULTI_PASS_RPW_K500"]

    colors = {"RPW_FORWARD": "#1f77b4", "MULTI_PASS_RPW_K500": "#2ca02c"}

    for m in key_methods:
        sub = augmented_df[augmented_df["method"] == m].dropna(subset=["order_strength", "relative_gap"])
        if not sub.empty:
            ax.scatter(
                sub["order_strength"],
                sub["relative_gap"] * 100,
                alpha=0.4,
                label=m,
                color=colors.get(m, "gray"),
                s=25,
            )
            # Add simple linear trendline
            z = np.polyfit(sub["order_strength"], sub["relative_gap"] * 100, 1)
            p = np.poly1d(z)
            x_vals = np.linspace(sub["order_strength"].min(), sub["order_strength"].max(), 50)
            ax.plot(x_vals, p(x_vals), color=colors.get(m, "gray"), linestyle="--", linewidth=2)

    ax.set_xlabel("Order Strength (OS)")
    ax.set_ylabel("Optimality Gap (%)")
    ax.set_title("E2: Influence of Order Strength on Heuristic Optimality Gap")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def run_e2_structure_pipeline(
    evaluated_df: pd.DataFrame,
    instances_yaml: str = "configs/instances.yaml",
    output_dir: str = "results",
) -> dict[str, Any]:
    """Runs the full E2 structural regression and correlation analysis pipeline."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Augment with graph measures
    augmented = augment_with_instance_measures(evaluated_df, instances_yaml)
    augmented.to_csv(out_p / "e2_augmented_runs.csv", index=False)

    # 2. Fit regression for key methods
    reg_rpw = fit_structural_regression(augmented, method="RPW_FORWARD")
    reg_multi = fit_structural_regression(augmented, method="MULTI_PASS_RPW_K500")

    # Export regression coefficients
    df_rpw = pd.DataFrame(reg_rpw["coefficients"])
    df_rpw["method"] = "RPW_FORWARD"
    df_multi = pd.DataFrame(reg_multi["coefficients"])
    df_multi["method"] = "MULTI_PASS_RPW_K500"

    combined_reg = pd.concat([df_rpw, df_multi], ignore_index=True)
    combined_reg.to_csv(out_p / "e2_regression_coefficients.csv", index=False)

    # 3. Figure
    fig_path = str(out_p / "figures" / "e2_gap_vs_order_strength.png")
    plot_e2_gap_vs_order_strength(augmented, fig_path)

    return {
        "status": "COMPLETED",
        "augmented_runs_path": str(out_p / "e2_augmented_runs.csv"),
        "regression_results": {
            "RPW_FORWARD": {
                "r_squared": reg_rpw["r_squared"],
                "p_value": reg_rpw["f_pvalue"],
                "multicollinearity_warning": reg_rpw["multicollinearity_warning"],
            },
            "MULTI_PASS_RPW_K500": {
                "r_squared": reg_multi["r_squared"],
                "p_value": reg_multi["f_pvalue"],
                "multicollinearity_warning": reg_multi["multicollinearity_warning"],
            },
        },
        "figures": [fig_path],
    }
