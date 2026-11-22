"""
Structural measures and characteristics of SALBP-1 instances.
Computes order strength, task-time distributions, graph shape,
cycle-time tightness, and theoretical lower bounds.
"""

from dataclasses import asdict, dataclass

import numpy as np

from linebench.instances.bounds import compute_lb1, compute_lb2, compute_lower_bound
from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance


@dataclass(frozen=True)
class InstanceMeasures:
    """Structural properties of an instance."""

    name: str
    num_tasks: int
    cycle_time: int
    total_work: int

    # Order strength and graph topology
    order_strength: float
    num_precedence_pairs_direct: int
    num_precedence_pairs_transitive: int
    num_sources: int
    num_sinks: int
    avg_in_degree: float
    avg_out_degree: float
    max_in_degree: int
    max_out_degree: int

    # Task time statistics
    t_min: int
    t_max: int
    t_mean: float
    t_std: float
    tightness_ratio: float  # c / t_max

    # Lower bounds
    lb1: int
    lb2: int
    lb: int

    def to_dict(self) -> dict[str, object]:
        """Convert measures to dictionary format."""
        return asdict(self)


def compute_instance_measures(
    instance: Instance, graph: PrecedenceGraph | None = None
) -> InstanceMeasures:
    """Computes all structural measures for a given instance."""
    if graph is None:
        graph = PrecedenceGraph(instance)

    n = instance.num_tasks
    c = instance.cycle_time
    times = list(instance.task_times.values())
    total_work = sum(times)

    # Order strength: transitive pairs / (n * (n - 1) / 2)
    transitive_pairs = graph.total_precedence_pairs()
    max_possible_pairs = n * (n - 1) / 2.0 if n > 1 else 1.0
    order_strength = transitive_pairs / max_possible_pairs if n > 1 else 0.0

    # In and out degree stats
    in_degrees = [len(graph.immediate_predecessors(i)) for i in range(1, n + 1)]
    out_degrees = [len(graph.immediate_successors(i)) for i in range(1, n + 1)]

    t_min = min(times)
    t_max = max(times)
    t_mean = float(np.mean(times))
    t_std = float(np.std(times))
    tightness_ratio = c / t_max if t_max > 0 else float("inf")

    lb1 = compute_lb1(instance)
    lb2 = compute_lb2(instance)
    lb = compute_lower_bound(instance)

    return InstanceMeasures(
        name=instance.name,
        num_tasks=n,
        cycle_time=c,
        total_work=total_work,
        order_strength=round(order_strength, 6),
        num_precedence_pairs_direct=len(instance.precedences),
        num_precedence_pairs_transitive=transitive_pairs,
        num_sources=len(graph.sources()),
        num_sinks=len(graph.sinks()),
        avg_in_degree=float(np.mean(in_degrees)),
        avg_out_degree=float(np.mean(out_degrees)),
        max_in_degree=max(in_degrees),
        max_out_degree=max(out_degrees),
        t_min=t_min,
        t_max=t_max,
        t_mean=round(t_mean, 4),
        t_std=round(t_std, 4),
        tightness_ratio=round(tightness_ratio, 4),
        lb1=lb1,
        lb2=lb2,
        lb=lb,
    )
