"""
Station bounds calculation for SALBP-1.
Computes earliest station E(j) and latest station L(j) for each task j
given cycle time c, precedence relations, and an upper bound m_ub on stations.
Ref: Scholl (1999), Master Plan Appendix A5.
"""

import math
from dataclasses import dataclass

from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance


@dataclass(frozen=True)
class StationBounds:
    """Interval [earliest, latest] of stations to which task j can be assigned."""

    earliest: dict[int, int]
    latest: dict[int, int]
    upper_bound_stations: int


def compute_station_bounds(
    instance: Instance,
    graph: PrecedenceGraph | None = None,
    upper_bound_stations: int | None = None,
) -> StationBounds:
    """Computes earliest station E(j) and latest station L(j) for all tasks j.

    Formulas:
      E(j) = ceil((t_j + sum_{i in AllPreds(j)} t_i) / c)
      L(j) = m_ub + 1 - ceil((t_j + sum_{k in AllSuccs(j)} t_k) / c)

    Args:
        instance: The SALBP-1 instance.
        graph: Optional PrecedenceGraph (constructed if None).
        upper_bound_stations: Optional upper bound m_ub on stations.
                              Defaults to instance.num_tasks if not provided.

    Returns:
        StationBounds containing earliest and latest mappings.
    """
    if graph is None:
        graph = PrecedenceGraph(instance)

    n = instance.num_tasks
    c = instance.cycle_time
    m_ub = upper_bound_stations if upper_bound_stations is not None else n

    if m_ub <= 0:
        raise ValueError(f"upper_bound_stations must be positive, got {m_ub}")

    earliest: dict[int, int] = {}
    latest: dict[int, int] = {}

    for j in range(1, n + 1):
        t_j = instance.task_times[j]

        # Predecessor work content including task j
        pred_work = t_j + sum(instance.task_times[i] for i in graph.all_predecessors(j))
        e_j = math.ceil(pred_work / c)

        # Successor work content including task j
        succ_work = t_j + sum(instance.task_times[k] for k in graph.all_successors(j))
        l_j = m_ub + 1 - math.ceil(succ_work / c)

        earliest[j] = e_j
        latest[j] = l_j

    return StationBounds(
        earliest=earliest,
        latest=latest,
        upper_bound_stations=m_ub,
    )
