"""Tests for Phase 6.2: E4 Robustness Analysis."""

from pathlib import Path

import pandas as pd
import pytest

from linebench.experiments.e4_robustness import run_e4_robustness


def test_e4_robustness_output_generation(tmp_path):
    """Verify that E4 analysis generates valid summary and paired comparison CSVs."""
    out_summary = tmp_path / "test_e4_summary.csv"
    out_paired = tmp_path / "test_e4_paired.csv"

    # Run on existing simulation results
    sim_jsonl = Path("results/simulation_runs.jsonl")
    if not sim_jsonl.exists():
        pytest.skip("simulation_runs.jsonl does not exist")

    res = run_e4_robustness(
        simulation_jsonl=str(sim_jsonl),
        output_summary=str(out_summary),
        output_paired=str(out_paired),
    )

    assert res["status"] == "COMPLETED"
    assert out_summary.exists()
    assert out_paired.exists()

    df_summary = pd.read_csv(out_summary)
    df_paired = pd.read_csv(out_paired)

    # Check summary columns and validity
    assert len(df_summary) > 0
    assert "rand_rel_th_mean" in df_summary.columns
    assert "rand_rel_th_spread" in df_summary.columns
    assert "heur_best_rel_th" in df_summary.columns
    assert "heur_best_percentile_rank" in df_summary.columns
    assert df_summary["rand_rel_th_mean"].isna().sum() == 0
    assert (df_summary["rand_rel_th_spread"] >= 0).all()

    # Check paired table
    assert len(df_paired) == 9  # 3 CVs x 3 Buffers
    assert "rand_spread_mean" in df_paired.columns
    assert "rand_spread_boot_lo" in df_paired.columns
    assert "rand_spread_boot_hi" in df_paired.columns
    assert "heur_best_percentile_rank_mean" in df_paired.columns
    # Check that bootstrap CI bounds are consistent
    assert (df_paired["rand_spread_boot_lo"] <= df_paired["rand_spread_boot_hi"]).all()


def test_existing_e4_artifacts():
    """Verify the pre-generated results/e4_*.csv files."""
    summary_path = Path("results/e4_robustness_summary.csv")
    paired_path = Path("results/e4_paired_differences.csv")

    assert summary_path.exists(), "results/e4_robustness_summary.csv missing"
    assert paired_path.exists(), "results/e4_paired_differences.csv missing"

    df_s = pd.read_csv(summary_path)
    df_p = pd.read_csv(paired_path)

    assert len(df_s) == 207
    assert len(df_p) == 9
    assert set(df_p["cv"].unique()) == {0.10, 0.25, 0.50}
    assert set(df_p["buffer_capacity"].unique()) == {0, 1, 3}
