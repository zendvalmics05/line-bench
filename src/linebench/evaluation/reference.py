"""Reference table builder for LINEBENCH benchmark instances.

Constructs the standardized reference table containing:
  - Instance ID, stratum (classical vs generated), size tier, num_tasks, cycle_time
  - Reference station count: proven optimum m* or best-known value
  - Provenance status: is_proven_optimal (True/False)
  - Lower bounds: LB1, LB2, and combined LB = max(LB1, LB2)
  - Lower bound gap: (m_ref - LB) / m_ref

Ref: Master Plan §5.1, Implementation Plan Phase 4.1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from linebench.instances.bounds import compute_lb1, compute_lb2, compute_lower_bound
from linebench.instances.loader import Instance
from linebench.instances.subset import load_subset_instance


@dataclass(frozen=True)
class ReferenceEntry:
    """Standardized reference metadata for a single benchmark instance."""

    instance_id: str
    stratum: str  # "classical" or "generated"
    num_tasks: int
    cycle_time: int
    size_tier: str  # "small", "medium", "large", "very_large"
    reference_stations: int  # optimal or best-known
    is_proven: bool  # True if mathematically proven optimal in literature/by bound
    lower_bound_1: int  # LB1 = ceil(sum(t) / c)
    lower_bound_2: int  # LB2 = half-task count bound
    lower_bound: int  # max(LB1, LB2)
    lb_gap_stations: int  # reference_stations - lower_bound
    lb_relative_gap: float  # (reference_stations - lower_bound) / reference_stations
    source_file: str
    cell: str | None = None


def get_size_tier(num_tasks: int) -> str:
    """Categorizes task count into size tiers."""
    if num_tasks <= 25:
        return "small"
    elif num_tasks <= 60:
        return "medium"
    elif num_tasks <= 200:
        return "large"
    else:
        return "very_large"


def build_reference_table(
    instances_yaml: str = "configs/instances.yaml",
) -> pd.DataFrame:
    """Builds a standardized DataFrame containing reference optima, bounds, and status flags.

    Args:
        instances_yaml: Path to the frozen instances configuration YAML.

    Returns:
        pandas DataFrame indexed by instance_id with reference metrics.
    """
    with open(instances_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    entries: list[dict[str, Any]] = []

    for item in cfg["benchmark_instances"]:
        inst_id = item["id"]
        stratum = item["stratum"]
        num_tasks = int(item["num_tasks"])
        cycle_time = int(item["cycle_time"])
        source_file = item["source_file"]
        cell = item.get("cell")

        # Load instance to compute lower bounds
        inst: Instance = load_subset_instance(item)
        lb1 = compute_lb1(inst)
        lb2 = compute_lb2(inst)
        lb = compute_lower_bound(inst)

        opt = item.get("optimal_stations")
        best = item.get("best_known_stations")

        if opt is not None:
            ref_val = int(opt)
            is_proven = bool(item.get("is_optimum_proven", True))
        elif best is not None:
            ref_val = int(best)
            is_proven = False
        else:
            # If no published value, reference is the theoretical LB
            ref_val = lb
            is_proven = False

        lb_gap = max(0, ref_val - lb)
        rel_gap = float(lb_gap / ref_val) if ref_val > 0 else 0.0

        size_tier = get_size_tier(num_tasks)

        entry = ReferenceEntry(
            instance_id=inst_id,
            stratum=stratum,
            num_tasks=num_tasks,
            cycle_time=cycle_time,
            size_tier=size_tier,
            reference_stations=ref_val,
            is_proven=is_proven,
            lower_bound_1=lb1,
            lower_bound_2=lb2,
            lower_bound=lb,
            lb_gap_stations=lb_gap,
            lb_relative_gap=rel_gap,
            source_file=source_file,
            cell=cell,
        )
        entries.append(asdict(entry))

    df = pd.DataFrame(entries)
    return df


def export_reference_table(
    instances_yaml: str = "configs/instances.yaml",
    output_csv: str = "results/reference_table.csv",
) -> pd.DataFrame:
    """Builds and saves the reference table to CSV."""
    df = build_reference_table(instances_yaml)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
