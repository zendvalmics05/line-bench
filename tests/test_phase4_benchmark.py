"""Tests for Phase 4.1 Reference Table and Phase 4.2 E1 Benchmark Evaluation."""

import pandas as pd

from linebench.evaluation.benchmark import (
    compute_e1_evaluation_metrics,
    run_e1_evaluation_pipeline,
    summarize_e1_by_method,
    summarize_e1_by_stratum,
)
from linebench.evaluation.reference import (
    build_reference_table,
    export_reference_table,
    get_size_tier,
)


def test_reference_table_generation():
    """Phase 4.1: Verify reference table has all 395 instances and correct columns."""
    ref_df = build_reference_table("configs/instances.yaml")
    assert len(ref_df) == 395
    assert "reference_stations" in ref_df.columns
    assert "is_proven" in ref_df.columns
    assert "lower_bound" in ref_df.columns
    assert "lb_gap_stations" in ref_df.columns
    assert "size_tier" in ref_df.columns

    # Reference station count must be >= theoretical LB for all instances
    assert (ref_df["reference_stations"] >= ref_df["lower_bound"]).all()
    # Ensure unproven instances are flagged
    unproven_count = (~ref_df["is_proven"]).sum()
    assert unproven_count > 0, "Expected unproven instances to be explicitly flagged"


def test_get_size_tier():
    """Verify size tier mapping."""
    assert get_size_tier(8) == "small"
    assert get_size_tier(20) == "small"
    assert get_size_tier(50) == "medium"
    assert get_size_tier(100) == "large"
    assert get_size_tier(1000) == "very_large"


def test_export_reference_table(tmp_path):
    """Verify export to CSV."""
    out_csv = str(tmp_path / "ref_table_test.csv")
    df = export_reference_table("configs/instances.yaml", out_csv)
    assert len(df) == 395
    loaded = pd.read_csv(out_csv)
    assert len(loaded) == 395


def test_e1_evaluation_pipeline(tmp_path):
    """Phase 4.2: Verify end-to-end E1 benchmark evaluation pipeline."""
    res = run_e1_evaluation_pipeline(
        instances_yaml="configs/instances.yaml",
        heuristic_results_path="results/heuristic_runs.jsonl",
        exact_results_path=None,
        output_dir=str(tmp_path / "eval_out"),
    )

    assert res["status"] == "COMPLETED"
    assert res["total_evaluated_records"] == 5135  # 395 * 13 methods
    assert len(res["methods_evaluated"]) == 13
    assert len(res["figures_generated"]) == 2

    # Check generated files
    eval_csv = tmp_path / "eval_out" / "e1_evaluated_runs.csv"
    method_csv = tmp_path / "eval_out" / "e1_method_summary.csv"
    stratum_csv = tmp_path / "eval_out" / "e1_stratum_summary.csv"

    assert eval_csv.exists()
    assert method_csv.exists()
    assert stratum_csv.exists()

    df_methods = pd.read_csv(method_csv)
    assert "hit_rate_pct" in df_methods.columns
    assert "mean_relative_gap_pct" in df_methods.columns
    assert "mean_runtime_ms" in df_methods.columns

    # Randomized multi-pass K=500 should achieve the lowest gap among heuristics
    best_row = df_methods.iloc[0]
    assert "MULTI_PASS" in best_row["method"]
    assert best_row["hit_rate_pct"] > 60.0


def test_compute_e1_evaluation_metrics_unit():
    """Verify metric calculations (station_gap, relative_gap, hit_rate)."""
    runs = pd.DataFrame([
        {
            "instance_id": "inst_1",
            "method": "M1",
            "stations": 10,
            "duration_seconds": 0.05,
            "is_feasible": True,
            "is_proven_optimal": False,
        },
        {
            "instance_id": "inst_1",
            "method": "M2",
            "stations": 8,
            "duration_seconds": 0.01,
            "is_feasible": True,
            "is_proven_optimal": True,
        },
    ])
    refs = pd.DataFrame([
        {
            "instance_id": "inst_1",
            "reference_stations": 8,
            "is_proven": True,
            "lower_bound": 8,
            "stratum": "classical",
            "size_tier": "small",
        }
    ])

    evaluated = compute_e1_evaluation_metrics(runs, refs)
    assert len(evaluated) == 2

    m1_row = evaluated[evaluated["method"] == "M1"].iloc[0]
    assert m1_row["station_gap"] == 2
    assert m1_row["relative_gap"] == 0.25
    assert bool(m1_row["hit_reference"]) is False

    m2_row = evaluated[evaluated["method"] == "M2"].iloc[0]
    assert m2_row["station_gap"] == 0
    assert m2_row["relative_gap"] == 0.0
    assert bool(m2_row["hit_reference"]) is True

    summary = summarize_e1_by_method(evaluated)
    assert len(summary) == 2

    stratum_summary = summarize_e1_by_stratum(evaluated)
    assert len(stratum_summary) == 2


def test_compare_paired_methods():
    """Phase 4.3: Verify paired comparisons, Wilcoxon test, Cohen's d, and Holm-Bonferroni."""
    from linebench.evaluation.compare import (
        compare_paired_methods,
        run_declared_paired_comparisons,
    )

    eval_df = pd.read_csv("results/e1_evaluated_runs.csv")

    # Compare MULTI_PASS vs RPW_FORWARD
    res = compare_paired_methods(eval_df, "MULTI_PASS_RPW_K500", "RPW_FORWARD")
    assert res.n_instances == 395
    assert res.wins_a > res.losses_a  # Multi-pass should outperform single-pass RPW
    assert res.mean_diff < 0  # Mean difference negative (fewer stations)
    assert res.wilcoxon_p_value < 0.001

    # Run full declared battery with Holm-Bonferroni correction
    df_comp = run_declared_paired_comparisons(eval_df)
    assert len(df_comp) >= 5
    assert "p_adjusted_holm" in df_comp.columns
    assert "significant_at_05" in df_comp.columns


def test_e2_structure_pipeline(tmp_path):
    """Phase 4.4: Verify E2 structural measures augmentation and regression."""
    from linebench.evaluation.structure import (
        augment_with_instance_measures,
        fit_structural_regression,
    )

    eval_df = pd.read_csv("results/e1_evaluated_runs.csv")

    # Test augmentation on a small sample
    sample_df = eval_df.iloc[:20].copy()
    augmented = augment_with_instance_measures(sample_df, "configs/instances.yaml")
    assert "order_strength" in augmented.columns
    assert "tightness_ratio" in augmented.columns
    assert "task_time_cv" in augmented.columns

    # Test regression on full evaluated runs
    full_augmented = augment_with_instance_measures(eval_df, "configs/instances.yaml")
    reg_res = fit_structural_regression(full_augmented, method="RPW_FORWARD")
    assert "r_squared" in reg_res
    assert "coefficients" in reg_res
    assert "vif" in reg_res
    assert isinstance(reg_res["multicollinearity_warning"], bool)


def test_e3_portfolio_metrics():
    """Phase 4.5: Verify E3 portfolio evaluation metrics."""
    from linebench.evaluation.portfolio_e3 import compute_e3_portfolio_metrics

    eval_df = pd.read_csv("results/e1_evaluated_runs.csv")
    res = compute_e3_portfolio_metrics(eval_df)

    assert res["status"] == "COMPLETED"
    assert len(res["portfolio_summary"]) == 2

    # Deterministic portfolio should achieve higher hit rate than individual rules (individual best was 57.5%)
    det_sum = res["portfolio_summary"][0]
    assert det_sum["hit_rate_pct"] >= 65.0
    assert det_sum["mean_relative_gap_pct"] < 3.0

    # Full portfolio (with multi-pass) should exceed 70%
    full_sum = res["portfolio_summary"][1]
    assert full_sum["hit_rate_pct"] >= 70.0
    assert full_sum["mean_relative_gap_pct"] < 2.5


def test_ablation_pipeline():
    """Phase 4.6: Verify K-pass ablation diminishing returns scaling."""
    from linebench.evaluation.ablations import run_multipass_k_ablation

    with open("configs/instances.yaml", "r", encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)

    # Pick 3 small instances
    bench = cfg["benchmark_instances"]
    sample = [x for x in bench if x["num_tasks"] <= 20][:3]

    df_ab = run_multipass_k_ablation(sample, k_values=[10, 50, 100], seed=42)
    assert len(df_ab) == 3
    assert "mean_stations" in df_ab.columns
    assert "mean_time_per_instance_ms" in df_ab.columns

    # Runtime should scale monotonically with K
    times = df_ab["mean_time_per_instance_ms"].values
    assert times[2] > times[0]

    # Best-of-K station count should be non-increasing with K
    stations = df_ab["mean_stations"].values
    assert stations[2] <= stations[0]
