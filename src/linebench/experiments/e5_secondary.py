"""E5 Secondary Objectives Analysis (Master Plan §6.3, Phase 6.3).

Examines where the exact, SO1 (MinMaxLoad), and SO2 (Smoothness) balances fall
within the equal-station-count random-balance population distribution:
1. Percentile rank of EXACT, SO1, and SO2 relative to RANDOM_DISTINCT balances.
2. Paired differences against the arbitrary EXACT baseline and against the random population mean.
3. Summary across instances by CV level and buffer size with bootstrap confidence intervals.

Outputs
-------
- ``results/e5_secondary_objectives_summary.csv`` — per-(instance, CV, buffer) detailed metrics
  for EXACT, SO1, SO2, including percentile ranks and paired differences.
- ``results/e5_secondary_objectives_paired.csv`` — aggregated across instances with 95% bootstrap CIs
  by CV and buffer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["run_e5_secondary_objectives"]

_BOOTSTRAP_REPS = 5000
_RNG_SEED = 2026


def _bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    n_boot: int = _BOOTSTRAP_REPS,
    rng: np.random.Generator | None = None,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap percentile CI for a statistic."""
    if rng is None:
        rng = np.random.default_rng(_RNG_SEED)
    valid_vals = values[~np.isnan(values)]
    n = len(valid_vals)
    if n < 2:
        v = float(valid_vals[0]) if n == 1 else float("nan")
        return v, v
    boot = rng.choice(valid_vals, size=(n_boot, n), replace=True)
    if statistic == "mean":
        boot_stats = boot.mean(axis=1)
    elif statistic == "median":
        boot_stats = np.median(boot, axis=1)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi


def _bootstrap_col(df_grp: pd.DataFrame, col: str, rng: np.random.Generator) -> tuple[float, float, float]:
    """Extract column values, compute mean and bootstrap 95% CI."""
    vals = df_grp[col].dropna().values
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(vals))
    lo, hi = _bootstrap_ci(vals, statistic="mean", rng=rng)
    return m, lo, hi


def run_e5_secondary_objectives(
    simulation_jsonl: str = "results/simulation_runs.jsonl",
    output_summary: str = "results/e5_secondary_objectives_summary.csv",
    output_paired: str = "results/e5_secondary_objectives_paired.csv",
) -> dict[str, str]:
    """Run E5 analysis and produce summary and paired difference CSVs."""
    jsonl_path = Path(simulation_jsonl)
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    df = pd.DataFrame(records)
    df_unpaced = df[df["line_type"] == "unpaced"].copy()

    summary_rows: list[dict] = []
    group_cols = ["instance_id", "cv", "buffer_capacity"]

    for (inst_id, cv, buf), grp in df_unpaced.groupby(group_cols):
        cycle_time = float(grp["cycle_time"].iloc[0])

        rand_grp = grp[grp["balance_type"] == "RANDOM_DISTINCT"]
        n_random = len(rand_grp)
        rand_rel_th = rand_grp["rel_throughput_design"].values if n_random > 0 else np.array([])
        rand_abs_th = rand_grp["mean_throughput"].values if n_random > 0 else np.array([])

        mean_rand_rel_th = float(np.mean(rand_rel_th)) if n_random > 0 else float("nan")
        mean_rand_abs_th = float(np.mean(rand_abs_th)) if n_random > 0 else float("nan")

        exact_sub = grp[grp["balance_type"] == "EXACT"]
        exact_abs = float(exact_sub["mean_throughput"].iloc[0]) if len(exact_sub) > 0 else float("nan")
        exact_rel = float(exact_sub["rel_throughput_design"].iloc[0]) if len(exact_sub) > 0 else float("nan")
        exact_pct = (
            float(np.mean(rand_rel_th < exact_rel)) * 100.0
            if n_random > 0 and not np.isnan(exact_rel)
            else float("nan")
        )

        so1_sub = grp[grp["balance_type"] == "SO1"]
        so1_abs = float(so1_sub["mean_throughput"].iloc[0]) if len(so1_sub) > 0 else float("nan")
        so1_rel = float(so1_sub["rel_throughput_design"].iloc[0]) if len(so1_sub) > 0 else float("nan")
        so1_pct = (
            float(np.mean(rand_rel_th < so1_rel)) * 100.0
            if n_random > 0 and not np.isnan(so1_rel)
            else float("nan")
        )

        so2_sub = grp[grp["balance_type"] == "SO2"]
        so2_abs = float(so2_sub["mean_throughput"].iloc[0]) if len(so2_sub) > 0 else float("nan")
        so2_rel = float(so2_sub["rel_throughput_design"].iloc[0]) if len(so2_sub) > 0 else float("nan")
        so2_pct = (
            float(np.mean(rand_rel_th < so2_rel)) * 100.0
            if n_random > 0 and not np.isnan(so2_rel)
            else float("nan")
        )

        hb_sub = grp[grp["balance_type"] == "HEUR_BEST"]
        hb_abs = float(hb_sub["mean_throughput"].iloc[0]) if len(hb_sub) > 0 else float("nan")
        hb_rel = float(hb_sub["rel_throughput_design"].iloc[0]) if len(hb_sub) > 0 else float("nan")
        hb_pct = (
            float(np.mean(rand_rel_th < hb_rel)) * 100.0
            if n_random > 0 and not np.isnan(hb_rel)
            else float("nan")
        )

        # Paired differences relative to EXACT (arbitrary optimal balance)
        diff_so1_vs_exact = (so1_abs - exact_abs) if (not np.isnan(so1_abs) and not np.isnan(exact_abs)) else float("nan")
        diff_so2_vs_exact = (so2_abs - exact_abs) if (not np.isnan(so2_abs) and not np.isnan(exact_abs)) else float("nan")
        diff_hb_vs_exact = (hb_abs - exact_abs) if (not np.isnan(hb_abs) and not np.isnan(exact_abs)) else float("nan")

        # Relative throughput difference as % of exact rate
        rel_diff_pct_so1_vs_exact = (
            (diff_so1_vs_exact / exact_abs) * 100.0 if not np.isnan(diff_so1_vs_exact) and exact_abs > 0 else float("nan")
        )
        rel_diff_pct_so2_vs_exact = (
            (diff_so2_vs_exact / exact_abs) * 100.0 if not np.isnan(diff_so2_vs_exact) and exact_abs > 0 else float("nan")
        )

        # Paired differences relative to mean random balance
        diff_so1_vs_rand = (so1_abs - mean_rand_abs_th) if (not np.isnan(so1_abs) and not np.isnan(mean_rand_abs_th)) else float("nan")
        diff_so2_vs_rand = (so2_abs - mean_rand_abs_th) if (not np.isnan(so2_abs) and not np.isnan(mean_rand_abs_th)) else float("nan")
        diff_exact_vs_rand = (exact_abs - mean_rand_abs_th) if (not np.isnan(exact_abs) and not np.isnan(mean_rand_abs_th)) else float("nan")

        summary_rows.append({
            "instance_id": inst_id,
            "cv": cv,
            "buffer_capacity": buf,
            "cycle_time": cycle_time,
            "n_random_balances": n_random,
            # Relative throughputs
            "exact_rel_th": exact_rel,
            "so1_rel_th": so1_rel,
            "so2_rel_th": so2_rel,
            "heur_best_rel_th": hb_rel,
            "rand_mean_rel_th": mean_rand_rel_th,
            # Percentile ranks within random balance distribution
            "exact_percentile_rank": exact_pct,
            "so1_percentile_rank": so1_pct,
            "so2_percentile_rank": so2_pct,
            "heur_best_percentile_rank": hb_pct,
            # Absolute paired differences vs EXACT
            "diff_so1_vs_exact": diff_so1_vs_exact,
            "diff_so2_vs_exact": diff_so2_vs_exact,
            "diff_hb_vs_exact": diff_hb_vs_exact,
            "rel_diff_pct_so1_vs_exact": rel_diff_pct_so1_vs_exact,
            "rel_diff_pct_so2_vs_exact": rel_diff_pct_so2_vs_exact,
            # Absolute paired differences vs Random Mean
            "diff_so1_vs_rand": diff_so1_vs_rand,
            "diff_so2_vs_rand": diff_so2_vs_rand,
            "diff_exact_vs_rand": diff_exact_vs_rand,
        })

    df_summary = pd.DataFrame(summary_rows)

    # Aggregated Bootstrap Table across instances (per CV x buffer)
    paired_rows: list[dict] = []
    rng = np.random.default_rng(_RNG_SEED)

    for (cv, buf), grp in df_summary.groupby(["cv", "buffer_capacity"]):
        n_instances = len(grp)

        exact_rel_m, exact_rel_lo, exact_rel_hi = _bootstrap_col(grp, "exact_rel_th", rng)
        so1_rel_m, so1_rel_lo, so1_rel_hi = _bootstrap_col(grp, "so1_rel_th", rng)
        so2_rel_m, so2_rel_lo, so2_rel_hi = _bootstrap_col(grp, "so2_rel_th", rng)

        so1_pct_m, so1_pct_lo, so1_pct_hi = _bootstrap_col(grp, "so1_percentile_rank", rng)
        so2_pct_m, so2_pct_lo, so2_pct_hi = _bootstrap_col(grp, "so2_percentile_rank", rng)
        exact_pct_m, exact_pct_lo, exact_pct_hi = _bootstrap_col(grp, "exact_percentile_rank", rng)

        diff_so1_ex_m, diff_so1_ex_lo, diff_so1_ex_hi = _bootstrap_col(grp, "diff_so1_vs_exact", rng)
        diff_so2_ex_m, diff_so2_ex_lo, diff_so2_ex_hi = _bootstrap_col(grp, "diff_so2_vs_exact", rng)

        pct_diff_so1_ex_m, pct_diff_so1_ex_lo, pct_diff_so1_ex_hi = _bootstrap_col(grp, "rel_diff_pct_so1_vs_exact", rng)
        pct_diff_so2_ex_m, pct_diff_so2_ex_lo, pct_diff_so2_ex_hi = _bootstrap_col(grp, "rel_diff_pct_so2_vs_exact", rng)

        paired_rows.append({
            "cv": cv,
            "buffer_capacity": buf,
            "n_instances": n_instances,
            # Relative throughput means & 95% CIs
            "exact_rel_th_mean": exact_rel_m,
            "exact_rel_th_boot_lo": exact_rel_lo,
            "exact_rel_th_boot_hi": exact_rel_hi,
            "so1_rel_th_mean": so1_rel_m,
            "so1_rel_th_boot_lo": so1_rel_lo,
            "so1_rel_th_boot_hi": so1_rel_hi,
            "so2_rel_th_mean": so2_rel_m,
            "so2_rel_th_boot_lo": so2_rel_lo,
            "so2_rel_th_boot_hi": so2_rel_hi,
            # Percentile ranks within random distribution
            "so1_percentile_rank_mean": so1_pct_m,
            "so1_percentile_rank_boot_lo": so1_pct_lo,
            "so1_percentile_rank_boot_hi": so1_pct_hi,
            "so2_percentile_rank_mean": so2_pct_m,
            "so2_percentile_rank_boot_lo": so2_pct_lo,
            "so2_percentile_rank_boot_hi": so2_pct_hi,
            "exact_percentile_rank_mean": exact_pct_m,
            "exact_percentile_rank_boot_lo": exact_pct_lo,
            "exact_percentile_rank_boot_hi": exact_pct_hi,
            # Paired differences vs EXACT (absolute)
            "diff_so1_vs_exact_mean": diff_so1_ex_m,
            "diff_so1_vs_exact_boot_lo": diff_so1_ex_lo,
            "diff_so1_vs_exact_boot_hi": diff_so1_ex_hi,
            "diff_so2_vs_exact_mean": diff_so2_ex_m,
            "diff_so2_vs_exact_boot_lo": diff_so2_ex_lo,
            "diff_so2_vs_exact_boot_hi": diff_so2_ex_hi,
            # Percentage gain vs EXACT
            "gain_pct_so1_vs_exact_mean": pct_diff_so1_ex_m,
            "gain_pct_so1_vs_exact_boot_lo": pct_diff_so1_ex_lo,
            "gain_pct_so1_vs_exact_boot_hi": pct_diff_so1_ex_hi,
            "gain_pct_so2_vs_exact_mean": pct_diff_so2_ex_m,
            "gain_pct_so2_vs_exact_boot_lo": pct_diff_so2_ex_lo,
            "gain_pct_so2_vs_exact_boot_hi": pct_diff_so2_ex_hi,
        })

    df_paired = pd.DataFrame(paired_rows).sort_values(["cv", "buffer_capacity"])

    Path(output_summary).parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(output_summary, index=False)
    df_paired.to_csv(output_paired, index=False)

    print(f"E5 summary rows written : {len(df_summary):>5}  → {output_summary}")
    print(f"E5 paired rows written  : {len(df_paired):>5}  → {output_paired}")

    return {
        "status": "COMPLETED",
        "output_summary": output_summary,
        "output_paired": output_paired,
        "n_summary_rows": len(df_summary),
        "n_paired_rows": len(df_paired),
    }
