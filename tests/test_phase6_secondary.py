"""Tests for Phase 6.3: E5 Secondary Objectives Analysis."""

from pathlib import Path

import pandas as pd
import pytest

from linebench.experiments.e5_secondary import run_e5_secondary_objectives


def test_e5_secondary_objectives_generation(tmp_path):
    """Verify that E5 analysis executes and produces valid summary and paired CSVs."""
    out_summary = tmp_path / "test_e5_summary.csv"
    out_paired = tmp_path / "test_e5_paired.csv"

    sim_jsonl = Path("results/simulation_runs.jsonl")
    if not sim_jsonl.exists():
        pytest.skip("simulation_runs.jsonl does not exist")

    res = run_e5_secondary_objectives(
        simulation_jsonl=str(sim_jsonl),
        output_summary=str(out_summary),
        output_paired=str(out_paired),
    )

    assert res["status"] == "COMPLETED"
    assert out_summary.exists()
    assert out_paired.exists()

    df_summary = pd.read_csv(out_summary)
    df_paired = pd.read_csv(out_paired)

    # Check summary table
    assert len(df_summary) > 0
    assert "exact_rel_th" in df_summary.columns
    assert "so1_rel_th" in df_summary.columns
    assert "so2_rel_th" in df_summary.columns
    assert "so1_percentile_rank" in df_summary.columns
    assert "so2_percentile_rank" in df_summary.columns
    assert "diff_so1_vs_exact" in df_summary.columns

    # Check paired aggregated table
    assert len(df_paired) == 9  # 3 CVs x 3 Buffers
    assert "so1_percentile_rank_mean" in df_paired.columns
    assert "gain_pct_so1_vs_exact_mean" in df_paired.columns
    assert (df_paired["so1_percentile_rank_boot_lo"] <= df_paired["so1_percentile_rank_boot_hi"]).all()


def test_existing_e5_artifacts():
    """Verify pre-generated results/e5_*.csv files."""
    summary_path = Path("results/e5_secondary_objectives_summary.csv")
    paired_path = Path("results/e5_secondary_objectives_paired.csv")

    assert summary_path.exists(), "results/e5_secondary_objectives_summary.csv missing"
    assert paired_path.exists(), "results/e5_secondary_objectives_paired.csv missing"

    df_s = pd.read_csv(summary_path)
    df_p = pd.read_csv(paired_path)

    assert len(df_s) == 270  # 30 instances x 3 CVs x 3 buffers
    assert len(df_p) == 9
    assert set(df_p["cv"].unique()) == {0.10, 0.25, 0.50}
    assert set(df_p["buffer_capacity"].unique()) == {0, 1, 3}
