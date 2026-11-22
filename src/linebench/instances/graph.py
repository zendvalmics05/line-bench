"""
Graph algorithms and transitive closure for SALBP precedence graphs.
Provides topological sorting, all predecessors/successors (transitive closure),
cycle detection, and graph reversal.
"""

from collections import deque

import networkx as nx

from linebench.instances.loader import Instance


class PrecedenceGraph:
    """Precedence DAG wrapper providing structural queries and transitive closures."""

    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.num_tasks = instance.num_tasks

        # Build directed graph
        self.graph = nx.DiGraph()
        for i in range(1, self.num_tasks + 1):
            self.graph.add_node(i, duration=instance.task_times[i])

        for u, v in instance.precedences:
            self.graph.add_edge(u, v)

        # Check for acyclicity
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise ValueError(f"Precedence graph contains cycles: {cycles}")

        # Compute transitive closure graph (reachability)
        self.transitive_closure_graph = nx.transitive_closure(self.graph)

        # Precompute immediate and all predecessors / successors for each task
        self._immediate_preds: dict[int, set[int]] = {
            i: set(self.graph.predecessors(i)) for i in range(1, self.num_tasks + 1)
        }
        self._immediate_succs: dict[int, set[int]] = {
            i: set(self.graph.successors(i)) for i in range(1, self.num_tasks + 1)
        }
        self._all_preds: dict[int, set[int]] = {
            i: set(self.transitive_closure_graph.predecessors(i))
            for i in range(1, self.num_tasks + 1)
        }
        self._all_succs: dict[int, set[int]] = {
            i: set(self.transitive_closure_graph.successors(i))
            for i in range(1, self.num_tasks + 1)
        }

    def immediate_predecessors(self, task: int) -> set[int]:
        """Returns direct predecessors of task."""
        return self._immediate_preds[task]

    def immediate_successors(self, task: int) -> set[int]:
        """Returns direct successors of task."""
        return self._immediate_succs[task]

    def all_predecessors(self, task: int) -> set[int]:
        """Returns all transitive predecessors of task."""
        return self._all_preds[task]

    def all_successors(self, task: int) -> set[int]:
        """Returns all transitive successors of task."""
        return self._all_succs[task]

    def topological_order(self) -> list[int]:
        """Returns a valid topological sort of tasks (Kahn's algorithm, stable tie-break)."""
        in_degree = {i: len(self._immediate_preds[i]) for i in range(1, self.num_tasks + 1)}
        # Queue with smallest task index first for stable ordering
        queue: deque[int] = deque(sorted([i for i, deg in in_degree.items() if deg == 0]))
        order: list[int] = []

        while queue:
            # Pop left (sorted smallest index first)
            curr = queue.popleft()
            order.append(curr)

            # Reduce in-degree for immediate successors
            for succ in sorted(self._immediate_succs[curr]):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(order) != self.num_tasks:
            raise RuntimeError("Topological sort failed: graph has a cycle")

        return order

    def reversed_instance(self) -> Instance:
        """Returns a new Instance with all precedence directions reversed.

        Task times and cycle time remain identical; an edge u -> v becomes v -> u.
        """
        reversed_precedences = {(v, u) for (u, v) in self.instance.precedences}
        return Instance(
            name=f"{self.instance.name}_reversed",
            num_tasks=self.instance.num_tasks,
            cycle_time=self.instance.cycle_time,
            task_times=dict(self.instance.task_times),
            precedences=reversed_precedences,
            order_strength=self.instance.order_strength,
        )

    def total_precedence_pairs(self) -> int:
        """Total number of ordered task pairs related by precedence in transitive closure."""
        return sum(len(succs) for succs in self._all_succs.values())

    def sources(self) -> list[int]:
        """Tasks with no predecessors."""
        return sorted([i for i in range(1, self.num_tasks + 1) if not self._immediate_preds[i]])

    def sinks(self) -> list[int]:
        """Tasks with no successors."""
        return sorted([i for i in range(1, self.num_tasks + 1) if not self._immediate_succs[i]])
