"""
Priority functions for SALBP-1 constructive heuristics.
Computes static and dynamic priority values for tasks based on graph topology,
durations, and station bounds.
"""

from collections.abc import Callable

from linebench.instances.graph import PrecedenceGraph
from linebench.instances.loader import Instance
from linebench.instances.station_bounds import compute_station_bounds


def compute_rpw_priorities(instance: Instance, graph: PrecedenceGraph) -> dict[int, float]:
    """Ranked Positional Weight (Helgeson & Birnie, 1961).

    RPW(j) = t_j + sum_{k in AllSuccessors(j)} t_k
    Higher value -> higher priority.
    """
    priorities: dict[int, float] = {}
    for j in range(1, instance.num_tasks + 1):
        t_j = instance.task_times[j]
        succ_work = sum(instance.task_times[k] for k in graph.all_successors(j))
        priorities[j] = float(t_j + succ_work)
    return priorities


def compute_ubt_priorities(instance: Instance, graph: PrecedenceGraph) -> dict[int, float]:
    """Longest Task Time / Maximum Duration (Jackson, 1956).

    UBT(j) = t_j
    Higher value -> higher priority.
    """
    return {j: float(instance.task_times[j]) for j in range(1, instance.num_tasks + 1)}


def compute_mts_priorities(instance: Instance, graph: PrecedenceGraph) -> dict[int, float]:
    """Maximum Total Successors / Most Followers.

    MTS(j) = |AllSuccessors(j)|
    Higher value -> higher priority.
    """
    return {j: float(len(graph.all_successors(j))) for j in range(1, instance.num_tasks + 1)}


def compute_mds_priorities(instance: Instance, graph: PrecedenceGraph) -> dict[int, float]:
    """Maximum Direct Successors / Immediate Followers.

    MDS(j) = |ImmediateSuccessors(j)|
    Higher value -> higher priority.
    """
    return {j: float(len(graph.immediate_successors(j))) for j in range(1, instance.num_tasks + 1)}


def compute_est_priorities(instance: Instance, graph: PrecedenceGraph) -> dict[int, float]:
    """Earliest Station Bound (Scholl, 1999).

    E(j) = ceil((t_j + sum_{i in AllPredecessors(j)} t_i) / c)
    Smaller E(j) means earlier urgency, so priority score = -E(j).
    Higher value -> higher priority.
    """
    sb = compute_station_bounds(instance, graph=graph)
    return {j: -float(sb.earliest[j]) for j in range(1, instance.num_tasks + 1)}


def compute_lst_priorities(instance: Instance, graph: PrecedenceGraph) -> dict[int, float]:
    """Latest Station Bound (Scholl, 1999).

    L(j) = m_ub + 1 - ceil((t_j + sum_{k in AllSuccessors(j)} t_k) / c)
    Smaller L(j) means tighter deadline, so priority score = -L(j).
    Higher value -> higher priority.
    """
    sb = compute_station_bounds(instance, graph=graph)
    return {j: -float(sb.latest[j]) for j in range(1, instance.num_tasks + 1)}


# Registry of supported priority rule functions
PRIORITY_RULE_REGISTRY: dict[str, Callable[[Instance, PrecedenceGraph], dict[int, float]]] = {
    "RPW": compute_rpw_priorities,
    "UBT": compute_ubt_priorities,
    "MTS": compute_mts_priorities,
    "MDS": compute_mds_priorities,
    "EST": compute_est_priorities,
    "LST": compute_lst_priorities,
}
