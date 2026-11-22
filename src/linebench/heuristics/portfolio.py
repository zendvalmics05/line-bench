"""
Portfolio helper for heuristic methods.
Extracts the best station count found across all heuristic variants for an instance,
calculates aggregated runtimes, identifies winning heuristics, and evaluates portfolio hit rates.
Ref: Master Plan §4 (RQ3), Implementation Plan Phase 2.6.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from linebench.experiments.run_heuristics import run_heuristics_on_instance
from linebench.instances.loader import Instance


@dataclass(frozen=True)
class PortfolioResult:
    """Summary of a portfolio run across heuristic methods for an instance."""

    instance_name: str
    best_stations: int
    winning_methods: list[str]
    total_runtime_seconds: float
    method_stations: dict[str, int]
    method_runtimes: dict[str, float]
    all_feasible: bool


def evaluate_heuristic_portfolio(
    instance: Instance,
    heuristic_results: Sequence[dict[str, Any]] | None = None,
    rules: Sequence[str] = ("RPW", "UBT", "MTS", "MDS", "EST", "LST"),
    directions: Sequence[str] = ("forward", "backward"),
    include_multi_pass: bool = True,
    multi_pass_passes: int = 500,
    multi_pass_seed: int = 1001,
) -> PortfolioResult:
    """Evaluates the heuristic portfolio for a given instance.

    If heuristic_results is provided (e.g. from runner logs), aggregates them.
    Otherwise, executes the heuristics on the instance.

    Args:
        instance: The SALBP-1 instance.
        heuristic_results: Optional precomputed list of result dicts from run_heuristics_on_instance.
        rules: Priority rules to include.
        directions: Directions ('forward', 'backward') to include.
        include_multi_pass: Whether to include multi-pass in the portfolio.
        multi_pass_passes: Passes for randomized multi-pass.
        multi_pass_seed: Seed for randomized multi-pass.

    Returns:
        PortfolioResult dataclass.
    """
    if heuristic_results is None:
        raw_results = run_heuristics_on_instance(
            instance,
            rules=rules,
            directions=directions,
            multi_pass_passes=multi_pass_passes,
            multi_pass_seed=multi_pass_seed,
        )
    else:
        raw_results = list(heuristic_results)

    if not include_multi_pass:
        raw_results = [r for r in raw_results if "MULTI_PASS" not in r["method"]]

    if not raw_results:
        raise ValueError("No heuristic results available to evaluate portfolio")

    method_stations: dict[str, int] = {}
    method_runtimes: dict[str, float] = {}
    all_feasible = True

    for r in raw_results:
        m_name = r["method"]
        m_stations = int(r["num_stations"])
        m_time = float(r["duration_seconds"])

        method_stations[m_name] = m_stations
        method_runtimes[m_name] = m_time
        if not r.get("is_feasible", True):
            all_feasible = False

    best_m = min(method_stations.values())
    winning_methods = sorted([m for m, stations in method_stations.items() if stations == best_m])
    total_time = sum(method_runtimes.values())

    return PortfolioResult(
        instance_name=instance.name,
        best_stations=best_m,
        winning_methods=winning_methods,
        total_runtime_seconds=total_time,
        method_stations=method_stations,
        method_runtimes=method_runtimes,
        all_feasible=all_feasible,
    )
