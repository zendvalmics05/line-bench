"""Tests for Phase 6.4: E6 Extra Station Analysis."""

from pathlib import Path

import pandas as pd
import pytest

from linebench.experiments.e6_extra_station import run_e6_extra_station


def test_e6_extra_station_generation(tmp_path):
    """Verify that E6 analysis runs and generates valid summary and paired CSVs."""
    out_summary = tmp_path / "test_e6_summary.csv"
    out_paired = tmp_path / "test_e6_paired.csv"

    sim_jsonl = Path("results/simulation_runs.jsonl")
    if not sim_jsonl.exists():
        pytest.skip("simulation_runs.jsonl does not exist")

    res = run_e6_extra_station(
        simulation_jsonl=str(sim_jsonl),
        output_summary=str(out_summary),
        output_paired=str(out_paired),
    )

    assert res["status"] == "COMPLETED"
    assert out_summary.exists()
    assert out_paired.exists()

    df_summary = pd.read_csv(out_summary)
    df_paired = pd.read_csv(out_paired)

    # Check summary metrics
    assert len(df_summary) > 0
    assert "th_so1" in df_summary.columns
    assert "th_m_plus_1" in df_summary.columns
    assert "pct_gain_th_total" in df_summary.columns
    assert "pct_diff_th_per_stn" in df_summary.columns
    assert "benefit_cost_ratio" in df_summary.columns

    # Extra station must always have m_plus_1 = m + 1
    assert (df_summary["m_plus_1_stations"] == df_summary["m_stations"] + 1).all()

    # Check aggregated paired table
    assert len(df_paired) == 9  # 3 CVs x 3 Buffers
    assert "pct_gain_total_th_mean" in df_paired.columns
    assert "pct_change_th_per_stn_mean" in df_paired.columns
    assert "benefit_cost_ratio_mean" in df_paired.columns
    assert (df_paired["pct_gain_total_th_boot_lo"] <= df_paired["pct_gain_total_th_boot_hi"]).all()


def test_existing_e6_artifacts():
    """Verify pre-generated results/e6_*.csv files."""
    summary_path = Path("results/e6_extra_station_summary.csv")
    paired_path = Path("results/e6_extra_station_paired.csv")

    assert summary_path.exists(), "results/e6_extra_station_summary.csv missing"
    assert paired_path.exists(), "results/e6_extra_station_paired.csv missing"

    df_s = pd.read_csv(summary_path)
    df_p = pd.read_csv(paired_path)

    assert len(df_s) == 252  # 28 instances x 3 CVs x 3 buffers
    assert len(df_p) == 9
    assert set(df_p["cv"].unique()) == {0.10, 0.25, 0.50}
    assert set(df_p["buffer_capacity"].unique()) == {0, 1, 3}
