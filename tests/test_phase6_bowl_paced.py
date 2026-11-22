"""Tests for Phase 6.5 (E7 Bowl Experiment) and Phase 6.6 (Paced Line Analysis)."""

from pathlib import Path

import pandas as pd
import pytest

from linebench.experiments.e6_paced import run_paced_analysis


def test_paced_analysis_execution(tmp_path):
    """Verify Phase 6.6 paced line analysis outputs."""
    out_detailed = tmp_path / "test_paced_detailed.csv"
    out_summary = tmp_path / "test_paced_summary.csv"

    sim_jsonl = Path("results/simulation_runs.jsonl")
    if not sim_jsonl.exists():
        pytest.skip("simulation_runs.jsonl missing")

    res = run_paced_analysis(
        simulation_jsonl=str(sim_jsonl),
        output_detailed=str(out_detailed),
        output_summary=str(out_summary),
    )

    assert res["status"] == "COMPLETED"
    assert out_detailed.exists()
    assert out_summary.exists()

    df_detail = pd.read_csv(out_detailed)
    df_summary = pd.read_csv(out_summary)

    assert len(df_detail) > 0
    assert "nominal_max_station_load" in df_detail.columns
    assert "empirical_max_stn_overload_prob" in df_detail.columns
    assert "norm_approx_max_stn_overload_prob" in df_detail.columns
    assert "abs_error_max_stn_overload" in df_detail.columns

    # Probabilities must be within [0, 1]
    assert (df_detail["empirical_max_stn_overload_prob"] >= 0.0).all()
    assert (df_detail["empirical_max_stn_overload_prob"] <= 1.0).all()

    # Normal approximation MAE should be reasonable (< 0.25)
    assert df_summary["mae_normal_approx"].mean() < 0.25


def test_bowl_experiment_artifacts():
    """Verify pre-generated Phase 6.5 bowl experiment CSV artifacts."""
    summary_path = Path("results/e7_bowl_summary.csv")
    paired_path = Path("results/e7_bowl_paired.csv")

    assert summary_path.exists(), "results/e7_bowl_summary.csv missing"
    assert paired_path.exists(), "results/e7_bowl_paired.csv missing"

    df_summary = pd.read_csv(summary_path)
    df_paired = pd.read_csv(paired_path)

    assert len(df_summary) == 216
    assert "th_bowl" in df_summary.columns
    assert "th_so1" in df_summary.columns
    assert "pct_gain_bowl_vs_so1" in df_summary.columns

    assert len(df_paired) == 9  # 3 CVs x 3 Buffers
    assert "pct_gain_bowl_vs_so1_mean" in df_paired.columns
    assert "bowl_win_rate_pct" in df_paired.columns
    assert "pct_gain_bowl_vs_so1_mean" in df_paired.columns
    assert "bowl_win_rate_pct" in df_paired.columns
