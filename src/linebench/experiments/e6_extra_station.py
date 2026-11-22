"""E6 Extra Station Analysis (Master Plan §6.3, Phase 6.4).

Compares m versus m + 1 stations (both under SO1 MinMaxLoad balance) across CV levels and buffer sizes:
1. Total line throughput comparison: TH(m+1) vs TH(m).
2. Throughput per station (efficiency/productivity per worker): TH(m+1)/(m+1) vs TH(m)/m.
3. Cost-benefit ratio: Gain in total throughput relative to the fractional cost of one extra station (1/m).
4. Bootstrap 95% confidence intervals across instances.

Outputs
-------
- ``results/e6_extra_station_summary.csv`` — per-(instance, CV, buffer) detailed metrics.
- ``results/e6_extra_station_paired.csv`` — aggregated across instances by CV and buffer with bootstrap 95% CIs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["run_e6_extra_station"]

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


def run_e6_extra_station(
    simulation_jsonl: str = "results/simulation_runs.jsonl",
    output_summary: str = "results/e6_extra_station_summary.csv",
    output_paired: str = "results/e6_extra_station_paired.csv",
) -> dict[str, str]:
    """Run E6 extra station analysis and produce summary and paired difference CSVs."""
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
        so1_sub = grp[grp["balance_type"] == "SO1"]
        m1_sub = grp[grp["balance_type"] == "M_PLUS_1"]

        if len(so1_sub) == 0 or len(m1_sub) == 0:
            continue

        cycle_time = float(grp["cycle_time"].iloc[0])

        m_stn = int(so1_sub["num_stations"].iloc[0])
        m1_stn = int(m1_sub["num_stations"].iloc[0])

        th_so1 = float(so1_sub["mean_throughput"].iloc[0])
        th_m1 = float(m1_sub["mean_throughput"].iloc[0])

        rel_th_so1 = float(so1_sub["rel_throughput_design"].iloc[0])
        rel_th_m1 = float(m1_sub["rel_throughput_design"].iloc[0])

        th_per_stn_so1 = th_so1 / m_stn if m_stn > 0 else float("nan")
        th_per_stn_m1 = th_m1 / m1_stn if m1_stn > 0 else float("nan")

        # Paired differences
        diff_th_total = th_m1 - th_so1
        pct_gain_th_total = (diff_th_total / th_so1) * 100.0 if th_so1 > 0 else float("nan")

        diff_th_per_stn = th_per_stn_m1 - th_per_stn_so1
        pct_diff_th_per_stn = (diff_th_per_stn / th_per_stn_so1) * 100.0 if th_per_stn_so1 > 0 else float("nan")

        # Station cost fraction: 1 / m
        cost_fraction = 1.0 / m_stn if m_stn > 0 else float("nan")
        # Relative gain vs station cost fraction: (gain / th_so1) / (1 / m) = (gain * m) / th_so1
        benefit_cost_ratio = (
            (diff_th_total / th_so1) / cost_fraction
            if th_so1 > 0 and cost_fraction > 0
            else float("nan")
        )

        summary_rows.append({
            "instance_id": inst_id,
            "cv": cv,
            "buffer_capacity": buf,
            "cycle_time": cycle_time,
            "m_stations": m_stn,
            "m_plus_1_stations": m1_stn,
            "th_so1": th_so1,
            "th_m_plus_1": th_m1,
            "rel_th_so1": rel_th_so1,
            "rel_th_m_plus_1": rel_th_m1,
            "th_per_stn_so1": th_per_stn_so1,
            "th_per_stn_m_plus_1": th_per_stn_m1,
            "diff_th_total": diff_th_total,
            "pct_gain_th_total": pct_gain_th_total,
            "diff_th_per_stn": diff_th_per_stn,
            "pct_diff_th_per_stn": pct_diff_th_per_stn,
            "cost_fraction_1_over_m": cost_fraction,
            "benefit_cost_ratio": benefit_cost_ratio,
        })

    df_summary = pd.DataFrame(summary_rows)

    # Aggregated Bootstrap Table across instances (per CV x buffer)
    paired_rows: list[dict] = []
    rng = np.random.default_rng(_RNG_SEED)

    for (cv, buf), grp in df_summary.groupby(["cv", "buffer_capacity"]):
        n_instances = len(grp)

        rel_so1_m, rel_so1_lo, rel_so1_hi = _bootstrap_col(grp, "rel_th_so1", rng)
        rel_m1_m, rel_m1_lo, rel_m1_hi = _bootstrap_col(grp, "rel_th_m_plus_1", rng)

        gain_pct_m, gain_pct_lo, gain_pct_hi = _bootstrap_col(grp, "pct_gain_th_total", rng)
        diff_per_stn_pct_m, diff_per_stn_pct_lo, diff_per_stn_pct_hi = _bootstrap_col(
            grp, "pct_diff_th_per_stn", rng
        )
        bcr_m, bcr_lo, bcr_hi = _bootstrap_col(grp, "benefit_cost_ratio", rng)
        cost_m, _, _ = _bootstrap_col(grp, "cost_fraction_1_over_m", rng)

        paired_rows.append({
            "cv": cv,
            "buffer_capacity": buf,
            "n_instances": n_instances,
            "rel_th_so1_mean": rel_so1_m,
            "rel_th_so1_boot_lo": rel_so1_lo,
            "rel_th_so1_boot_hi": rel_so1_hi,
            "rel_th_m_plus_1_mean": rel_m1_m,
            "rel_th_m_plus_1_boot_lo": rel_m1_lo,
            "rel_th_m_plus_1_boot_hi": rel_m1_hi,
            # Total throughput gain (%)
            "pct_gain_total_th_mean": gain_pct_m,
            "pct_gain_total_th_boot_lo": gain_pct_lo,
            "pct_gain_total_th_boot_hi": gain_pct_hi,
            # Productivity per station change (%)
            "pct_change_th_per_stn_mean": diff_per_stn_pct_m,
            "pct_change_th_per_stn_boot_lo": diff_per_stn_pct_lo,
            "pct_change_th_per_stn_boot_hi": diff_per_stn_pct_hi,
            # Cost of extra station (1/m) and Benefit-to-Cost Ratio
            "cost_fraction_1_over_m_mean": cost_m,
            "benefit_cost_ratio_mean": bcr_m,
            "benefit_cost_ratio_boot_lo": bcr_lo,
            "benefit_cost_ratio_boot_hi": bcr_hi,
        })

    df_paired = pd.DataFrame(paired_rows).sort_values(["cv", "buffer_capacity"])

    Path(output_summary).parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(output_summary, index=False)
    df_paired.to_csv(output_paired, index=False)

    print(f"E6 summary rows written : {len(df_summary):>5}  → {output_summary}")
    print(f"E6 paired rows written  : {len(df_paired):>5}  → {output_paired}")

    return {
        "status": "COMPLETED",
        "output_summary": output_summary,
        "output_paired": output_paired,
        "n_summary_rows": len(df_summary),
        "n_paired_rows": len(df_paired),
    }
