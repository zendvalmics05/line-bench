"""Paired statistical comparisons for LINEBENCH benchmark methods.

Computes pairwise comparisons between methods across benchmark instances:
  - Win, tie, loss counts (method A vs method B)
  - Paired difference in station counts: diff = stations_A - stations_B
  - Wilcoxon signed-rank test (non-parametric)
  - Paired Student's t-test
  - Effect sizes:
      * Matched-pairs rank biserial correlation: r = (W_pos - W_neg) / (W_pos + W_neg)
      * Cohen's d_z for paired samples: mean(diff) / std(diff)
  - Multiple testing adjustments (Holm-Bonferroni / Benjamini-Hochberg FDR)
  - Pre-declared focal pairs:
      1. RPW_FORWARD vs UBT_FORWARD (Ranked Positional Weight vs Longest Task Time)
      2. RPW_FORWARD vs RPW_BACKWARD (Forward vs Backward direction)
      3. MULTI_PASS_RPW_K500 vs RPW_FORWARD (Multi-pass vs Deterministic baseline)
      4. MULTI_PASS_RPW_K500 vs UBT_BACKWARD (Top multi-pass vs Top deterministic heuristic)

Ref: Master Plan §5.3 Module D, Implementation Plan Phase 4.3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class PairedComparisonResult:
    """Outcome of a paired statistical comparison between two methods."""

    method_a: str
    method_b: str
    n_instances: int
    wins_a: int  # method A used fewer stations than method B
    ties: int  # method A used equal stations to method B
    losses_a: int  # method A used more stations than method B
    mean_diff: float  # mean(stations_A - stations_B) -> negative means A is better
    std_diff: float
    median_diff: float
    cohens_d: float  # paired Cohen's d_z
    wilcoxon_stat: float | None
    wilcoxon_p_value: float | None
    rank_biserial_r: float | None
    t_stat: float
    t_p_value: float
    p_adjusted_holm: float | None = None
    significant_at_05: bool | None = None


def compare_paired_methods(
    evaluated_df: pd.DataFrame,
    method_a: str,
    method_b: str,
) -> PairedComparisonResult:
    """Performs paired statistical test comparing method A against method B."""
    # Pivot to get instance-level paired rows
    sub = evaluated_df[evaluated_df["method"].isin([method_a, method_b])]
    pivoted = sub.pivot(index="instance_id", columns="method", values="stations").dropna()

    if method_a not in pivoted.columns or method_b not in pivoted.columns:
        raise ValueError(f"Both {method_a} and {method_b} must be present in evaluated_df")

    vals_a = pivoted[method_a].values.astype(float)
    vals_b = pivoted[method_b].values.astype(float)

    diff = vals_a - vals_b
    n = len(diff)

    wins_a = int(np.sum(diff < 0))  # Fewer stations is better
    ties = int(np.sum(diff == 0))
    losses_a = int(np.sum(diff > 0))

    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    median_d = float(np.median(diff))

    cohens_d = (mean_d / std_d) if std_d > 1e-9 else 0.0

    # Paired Student's t-test
    if std_d > 1e-9:
        t_res = stats.ttest_rel(vals_a, vals_b)
        t_stat = float(t_res.statistic)
        t_pval = float(t_res.pvalue)
    else:
        t_stat = 0.0
        t_pval = 1.0

    # Wilcoxon signed-rank test (on non-zero differences)
    non_zero_diff = diff[diff != 0]
    if len(non_zero_diff) > 0:
        w_res = stats.wilcoxon(vals_a, vals_b, zero_method="pratt", alternative="two-sided")
        w_stat = float(w_res.statistic)
        w_pval = float(w_res.pvalue)

        # Matched-pairs rank biserial correlation: r = (W_pos - W_neg) / (W_pos + W_neg)
        # We can calculate positive ranks and negative ranks
        ranks = stats.rankdata(np.abs(diff))
        w_pos = np.sum(ranks[diff > 0])
        w_neg = np.sum(ranks[diff < 0])
        w_tot = w_pos + w_neg
        rank_biserial = float((w_pos - w_neg) / w_tot) if w_tot > 0 else 0.0
    else:
        w_stat = None
        w_pval = 1.0
        rank_biserial = 0.0

    return PairedComparisonResult(
        method_a=method_a,
        method_b=method_b,
        n_instances=n,
        wins_a=wins_a,
        ties=ties,
        losses_a=losses_a,
        mean_diff=mean_d,
        std_diff=std_d,
        median_diff=median_d,
        cohens_d=cohens_d,
        wilcoxon_stat=w_stat,
        wilcoxon_p_value=w_pval,
        rank_biserial_r=rank_biserial,
        t_stat=t_stat,
        t_p_value=t_pval,
    )


def run_declared_paired_comparisons(
    evaluated_df: pd.DataFrame,
    declared_pairs: list[tuple[str, str]] | None = None,
    output_csv: str = "results/paired_comparisons.csv",
) -> pd.DataFrame:
    """Evaluates all pre-declared method pairs and applies Holm-Bonferroni correction."""
    if declared_pairs is None:
        declared_pairs = [
            ("RPW_FORWARD", "UBT_FORWARD"),
            ("RPW_FORWARD", "RPW_BACKWARD"),
            ("UBT_FORWARD", "UBT_BACKWARD"),
            ("MULTI_PASS_RPW_K500", "RPW_FORWARD"),
            ("MULTI_PASS_RPW_K500", "UBT_BACKWARD"),
            ("MULTI_PASS_RPW_K500", "MTS_FORWARD"),
            ("RPW_FORWARD", "EST_FORWARD"),
        ]

    results: list[dict[str, Any]] = []
    p_vals: list[float] = []

    for m_a, m_b in declared_pairs:
        res = compare_paired_methods(evaluated_df, m_a, m_b)
        rec = asdict(res)
        results.append(rec)
        p_vals.append(res.wilcoxon_p_value if res.wilcoxon_p_value is not None else 1.0)

    # Apply Holm-Bonferroni correction across the family of tests
    # to rigorously address the multiple-comparisons problem
    m_tests = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    adj_p = np.zeros(m_tests)

    for rank, idx in enumerate(sorted_indices):
        multiplier = m_tests - rank
        adj = min(1.0, p_vals[idx] * multiplier)
        adj_p[idx] = adj

    # Enforce monotonicity
    for i in range(1, len(sorted_indices)):
        prev_idx = sorted_indices[i - 1]
        cur_idx = sorted_indices[i]
        adj_p[cur_idx] = max(adj_p[cur_idx], adj_p[prev_idx])

    for i in range(m_tests):
        results[i]["p_adjusted_holm"] = float(adj_p[i])
        results[i]["significant_at_05"] = bool(adj_p[i] < 0.05)

    df = pd.DataFrame(results)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df
