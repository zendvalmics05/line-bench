"""E4 Robustness Analysis (Master Plan §6.4, Phase 6.2).

For the equal-station-count balances (HEUR_BEST, EXACT, SO1, SO2, RANDOM_DISTINCT),
this module analyses:

1. **Random-balance spread** — the distribution of relative throughput across the
   RANDOM_DISTINCT population for each (instance, CV, buffer) stratum.
2. **Bootstrap summary across instances** — pooling instance-level spread statistics
   with 95% bootstrap confidence intervals.
3. **Paired differences against reference** — HEUR_BEST minus each RANDOM_DISTINCT
   balance; CRN is preserved because all balances of the same instance share the
   same noise tensor (enforced by the sweep runner).

Outputs
-------
- ``results/e4_robustness_summary.csv`` — per-(instance, CV, buffer) random-balance
  spread statistics plus the HEUR_BEST and EXACT relative throughputs.
- ``results/e4_paired_differences.csv`` — per-(instance, CV, buffer) mean paired
  difference (HEUR_BEST - RANDOM mean) with bootstrap 95% CI across instances.

Relative throughput (§A14)
--------------------------
- ``rel_th_design``     = mean_throughput × cycle_time      (fraction of design rate)
- ``rel_th_bottleneck`` = mean_throughput × det_bottleneck  (fraction of bottleneck rate)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["run_e4_robustness"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REFERENCE_BALANCE = "HEUR_BEST"
_RANDOM_TYPE = "RANDOM_DISTINCT"
_BOOTSTRAP_REPS = 5000
_RNG_SEED = 2026


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    n = len(values)
    if n < 2:
        v = float(values[0]) if n == 1 else float("nan")
        return v, v
    boot = rng.choice(values, size=(n_boot, n), replace=True)
    if statistic == "mean":
        stats = boot.mean(axis=1)
    elif statistic == "std":
        stats = boot.std(axis=1, ddof=1)
    elif statistic == "median":
        stats = np.median(boot, axis=1)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_e4_robustness(
    simulation_jsonl: str = "results/simulation_runs.jsonl",
    output_summary: str = "results/e4_robustness_summary.csv",
    output_paired: str = "results/e4_paired_differences.csv",
) -> dict[str, str]:
    """Run the E4 robustness analysis and write CSV outputs.

    Parameters
    ----------
    simulation_jsonl:
        Path to the completed simulation sweep JSONL.
    output_summary:
        Output path for the per-stratum spread summary CSV.
    output_paired:
        Output path for the paired-difference summary CSV.

    Returns
    -------
    dict with output paths and record counts.
    """
    # ------------------------------------------------------------------
    # 1. Load simulation data
    # ------------------------------------------------------------------
    jsonl_path = Path(simulation_jsonl)
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    df = pd.DataFrame(records)

    # Focus on unpaced lines for E4 (paced is covered in 6.6)
    df_unpaced = df[df["line_type"] == "unpaced"].copy()

    # ------------------------------------------------------------------
    # 2. Build per-stratum spread table
    # ------------------------------------------------------------------
    # For each (instance_id, cv, buffer_capacity):
    #   - random_balance_population stats from RANDOM_DISTINCT
    #   - reference rel_th from HEUR_BEST
    #   - secondary rel_th from EXACT (where available)

    summary_rows: list[dict] = []
    rng = np.random.default_rng(_RNG_SEED)

    group_cols = ["instance_id", "cv", "buffer_capacity"]

    for (inst_id, cv, buf), grp in df_unpaced.groupby(group_cols):
        cycle_time = float(grp["cycle_time"].iloc[0])

        # Random population
        rand_grp = grp[grp["balance_type"] == _RANDOM_TYPE]
        n_random = len(rand_grp)

        if n_random == 0:
            # No random balances for this stratum — skip
            continue

        rand_rel_th = rand_grp["rel_throughput_design"].values
        rand_th_abs = rand_grp["mean_throughput"].values

        rand_mean = float(np.mean(rand_rel_th))
        rand_std = float(np.std(rand_rel_th, ddof=1)) if n_random > 1 else 0.0
        rand_min = float(np.min(rand_rel_th))
        rand_max = float(np.max(rand_rel_th))
        rand_spread = rand_max - rand_min  # max minus min spread

        # Reference: HEUR_BEST
        hb_grp = grp[grp["balance_type"] == _REFERENCE_BALANCE]
        ref_rel_th = float(hb_grp["rel_throughput_design"].iloc[0]) if len(hb_grp) > 0 else float("nan")
        ref_th_abs = float(hb_grp["mean_throughput"].iloc[0]) if len(hb_grp) > 0 else float("nan")

        # Exact balance
        ex_grp = grp[grp["balance_type"] == "EXACT"]
        exact_rel_th = float(ex_grp["rel_throughput_design"].iloc[0]) if len(ex_grp) > 0 else float("nan")

        # SO1, SO2
        so1_grp = grp[grp["balance_type"] == "SO1"]
        so1_rel_th = float(so1_grp["rel_throughput_design"].iloc[0]) if len(so1_grp) > 0 else float("nan")
        so2_grp = grp[grp["balance_type"] == "SO2"]
        so2_rel_th = float(so2_grp["rel_throughput_design"].iloc[0]) if len(so2_grp) > 0 else float("nan")

        # Percentile rank of reference within random distribution
        ref_percentile = float(np.mean(rand_rel_th < ref_rel_th)) * 100 if not np.isnan(ref_rel_th) else float("nan")

        # Paired difference: ref minus each random (CRN; both drawn from same noise)
        paired_diffs = ref_th_abs - rand_th_abs  # shape (n_random,)
        mean_paired_diff = float(np.mean(paired_diffs))

        summary_rows.append({
            "instance_id": inst_id,
            "cv": cv,
            "buffer_capacity": buf,
            "cycle_time": cycle_time,
            "n_random_balances": n_random,
            # Random population stats (relative throughput = TH * c)
            "rand_rel_th_mean": rand_mean,
            "rand_rel_th_std": rand_std,
            "rand_rel_th_min": rand_min,
            "rand_rel_th_max": rand_max,
            "rand_rel_th_spread": rand_spread,
            # Reference and special balances
            "heur_best_rel_th": ref_rel_th,
            "exact_rel_th": exact_rel_th,
            "so1_rel_th": so1_rel_th,
            "so2_rel_th": so2_rel_th,
            # Position of reference in random distribution
            "heur_best_percentile_rank": ref_percentile,
            # Paired difference summary (absolute throughput)
            "mean_paired_diff_heur_vs_rand": mean_paired_diff,
        })

    df_summary = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # 3. Bootstrap summary across instances (per CV × buffer stratum)
    # ------------------------------------------------------------------
    paired_rows: list[dict] = []

    for (cv, buf), grp in df_summary.groupby(["cv", "buffer_capacity"]):
        n_instances = len(grp)

        # Spread of random population (instance-level values to bootstrap over)
        spread_vals = grp["rand_rel_th_spread"].values
        spread_mean = float(np.mean(spread_vals))
        spread_lo, spread_hi = _bootstrap_ci(spread_vals, statistic="mean", rng=rng)

        # Mean random rel throughput across instances
        rand_mean_vals = grp["rand_rel_th_mean"].values
        overall_rand_mean = float(np.mean(rand_mean_vals))
        rand_mean_lo, rand_mean_hi = _bootstrap_ci(rand_mean_vals, statistic="mean", rng=rng)

        # HEUR_BEST rel throughput across instances
        hb_vals = grp["heur_best_rel_th"].dropna().values
        hb_mean = float(np.mean(hb_vals)) if len(hb_vals) > 0 else float("nan")
        hb_lo, hb_hi = _bootstrap_ci(hb_vals, statistic="mean", rng=rng) if len(hb_vals) >= 2 else (float("nan"), float("nan"))

        # Paired diff across instances
        pd_vals = grp["mean_paired_diff_heur_vs_rand"].dropna().values
        pd_mean = float(np.mean(pd_vals)) if len(pd_vals) > 0 else float("nan")
        pd_lo, pd_hi = _bootstrap_ci(pd_vals, statistic="mean", rng=rng) if len(pd_vals) >= 2 else (float("nan"), float("nan"))

        # EXACT rel throughput
        ex_vals = grp["exact_rel_th"].dropna().values
        ex_mean = float(np.mean(ex_vals)) if len(ex_vals) > 0 else float("nan")
        ex_lo, ex_hi = _bootstrap_ci(ex_vals, statistic="mean", rng=rng) if len(ex_vals) >= 2 else (float("nan"), float("nan"))

        # SO1 rel throughput
        so1_vals = grp["so1_rel_th"].dropna().values
        so1_mean = float(np.mean(so1_vals)) if len(so1_vals) > 0 else float("nan")
        so1_lo, so1_hi = _bootstrap_ci(so1_vals, statistic="mean", rng=rng) if len(so1_vals) >= 2 else (float("nan"), float("nan"))

        # Percentile rank of HEUR_BEST within random distribution
        pct_vals = grp["heur_best_percentile_rank"].dropna().values
        pct_mean = float(np.mean(pct_vals)) if len(pct_vals) > 0 else float("nan")
        pct_lo, pct_hi = _bootstrap_ci(pct_vals, statistic="mean", rng=rng) if len(pct_vals) >= 2 else (float("nan"), float("nan"))

        paired_rows.append({
            "cv": cv,
            "buffer_capacity": buf,
            "n_instances": n_instances,
            # Random population spread (key E4 metric)
            "rand_spread_mean": spread_mean,
            "rand_spread_boot_lo": spread_lo,
            "rand_spread_boot_hi": spread_hi,
            # Random mean rel throughput
            "rand_rel_th_mean": overall_rand_mean,
            "rand_rel_th_boot_lo": rand_mean_lo,
            "rand_rel_th_boot_hi": rand_mean_hi,
            # HEUR_BEST
            "heur_best_rel_th_mean": hb_mean,
            "heur_best_rel_th_boot_lo": hb_lo,
            "heur_best_rel_th_boot_hi": hb_hi,
            # EXACT
            "exact_rel_th_mean": ex_mean,
            "exact_rel_th_boot_lo": ex_lo,
            "exact_rel_th_boot_hi": ex_hi,
            # SO1
            "so1_rel_th_mean": so1_mean,
            "so1_rel_th_boot_lo": so1_lo,
            "so1_rel_th_boot_hi": so1_hi,
            # Paired difference: HEUR_BEST vs RANDOM mean (absolute throughput)
            "paired_diff_heur_vs_rand_mean": pd_mean,
            "paired_diff_heur_vs_rand_boot_lo": pd_lo,
            "paired_diff_heur_vs_rand_boot_hi": pd_hi,
            # Percentile rank of HEUR_BEST within random distribution
            "heur_best_percentile_rank_mean": pct_mean,
            "heur_best_percentile_rank_boot_lo": pct_lo,
            "heur_best_percentile_rank_boot_hi": pct_hi,
        })

    df_paired = pd.DataFrame(paired_rows).sort_values(["cv", "buffer_capacity"])

    # ------------------------------------------------------------------
    # 4. Write outputs
    # ------------------------------------------------------------------
    Path(output_summary).parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(output_summary, index=False)
    df_paired.to_csv(output_paired, index=False)

    print(f"E4 summary rows written : {len(df_summary):>5}  → {output_summary}")
    print(f"E4 paired rows written  : {len(df_paired):>5}  → {output_paired}")

    return {
        "status": "COMPLETED",
        "output_summary": output_summary,
        "output_paired": output_paired,
        "n_summary_rows": len(df_summary),
        "n_paired_rows": len(df_paired),
    }
