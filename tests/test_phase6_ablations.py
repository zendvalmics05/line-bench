"""Tests for Phase 6.7: Simulation Ablations."""

from pathlib import Path

import pandas as pd


def test_existing_ablation_artifacts():
    """Verify that Phase 6.7 ablation files exist and contain valid results."""
    noise_path = Path("results/ablation_noise_distribution.csv")
    buffer_path = Path("results/ablation_buffer_sensitivity.csv")

    assert noise_path.exists(), "results/ablation_noise_distribution.csv missing"
    assert buffer_path.exists(), "results/ablation_buffer_sensitivity.csv missing"

    df_noise = pd.read_csv(noise_path)
    df_buf = pd.read_csv(buffer_path)

    # Check noise ablation structure
    assert len(df_noise) > 0
    assert "pct_diff_gamma_vs_ln" in df_noise.columns
    assert "th_lognormal" in df_noise.columns
    assert "th_gamma" in df_noise.columns
    # Check that difference between distributions is minor (< 2% on average)
    assert abs(df_noise["pct_diff_gamma_vs_ln"].mean()) < 2.0

    # Check buffer sensitivity structure
    assert len(df_buf) > 0
    assert "gain_b0_to_b1_pct" in df_buf.columns
    assert "gain_b1_to_b3_pct" in df_buf.columns
    assert "marginal_return_ratio" in df_buf.columns

    # Buffer monotonicity: adding buffer should never decrease throughput on average
    assert df_buf["gain_b0_to_b1_pct"].mean() > 0.0
    assert df_buf["gain_b1_to_b3_pct"].mean() > 0.0
    # Diminishing marginal returns of buffering
    assert df_buf["marginal_return_ratio"].mean() < 1.0
