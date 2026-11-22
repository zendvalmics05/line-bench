"""Station-time builder (Master Plan §A9 preamble, Phase 5.2).

Converts a task assignment (station -> list of tasks) and noisy task times
into station times.

Features:
  - assignment_to_matrix: converts a station-to-task mapping to an (m, n) binary matrix.
  - build_station_times: computes station times via tensor dot / matrix multiplication:
      station_times = noisy_task_times @ A.T
    where A is (m, n), yielding shape (..., m).
  - Common Random Numbers (CRN):
    The exact same noise factors array is multiplied by different candidate balance
    assignment matrices, isolating the balance structural design from stochastic noise.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "assignment_to_matrix",
    "build_station_times",
]


def assignment_to_matrix(
    station_assignments: dict[int, list[int]],
    n_tasks: int,
    zero_indexed_stations: bool = False,
    zero_indexed_tasks: bool = False,
) -> np.ndarray:
    """Convert station assignment dictionary to an (m, n) binary indicator matrix.

    Parameters
    ----------
    station_assignments:
        Mapping from station identifier to list of task identifiers.
    n_tasks:
        Total number of tasks in the instance.
    zero_indexed_stations:
        If True, station identifiers in keys are already 0-indexed (0..m-1).
        If False, keys are 1-indexed (1..m). Default is False (standard SALBP 1-based).
    zero_indexed_tasks:
        If True, task IDs in lists are already 0-indexed (0..n-1).
        If False, task IDs are 1-indexed (1..n). Default is False (standard SALBP 1-based).

    Returns
    -------
    np.ndarray:
        Binary matrix of shape (m, n) where A[s, j] = 1 if task j is assigned to station s.
    """
    m = len(station_assignments)
    A = np.zeros((m, n_tasks), dtype=np.float64)

    for stn_id, tasks in station_assignments.items():
        s_idx = int(stn_id) if zero_indexed_stations else int(stn_id) - 1
        if not (0 <= s_idx < m):
            raise IndexError(f"Station index {s_idx} out of bounds for m={m}")
        for t_id in tasks:
            t_idx = int(t_id) if zero_indexed_tasks else int(t_id) - 1
            if not (0 <= t_idx < n_tasks):
                raise IndexError(f"Task index {t_idx} out of bounds for n_tasks={n_tasks}")
            A[s_idx, t_idx] = 1.0

    return A


def build_station_times(
    station_assignments: dict[int, list[int]],
    base_task_times: np.ndarray | list[float | int],
    noise_factors: np.ndarray,
    zero_indexed_stations: bool = False,
    zero_indexed_tasks: bool = False,
) -> np.ndarray:
    """Build station-time array for all replications/units via matrix multiplication.

    Parameters
    ----------
    station_assignments:
        Dict mapping station identifier to list of task identifiers.
    base_task_times:
        1D array or list of nominal deterministic task times of length n_tasks.
    noise_factors:
        Multiplicative noise factors of shape (..., n_tasks), e.g.:
          - 3D: (n_replications, n_units, n_tasks)
          - 2D: (n_replications, n_tasks)
          - 1D: (n_tasks,)
    zero_indexed_stations:
        Whether station IDs in station_assignments are 0-indexed.
    zero_indexed_tasks:
        Whether task IDs in station_assignments are 0-indexed.

    Returns
    -------
    np.ndarray:
        Station times of shape (..., m) matching the leading dimensions of noise_factors.
    """
    base_times = np.asarray(base_task_times, dtype=np.float64)
    n_tasks = len(base_times)

    if noise_factors.shape[-1] != n_tasks:
        raise ValueError(
            f"Last dimension of noise_factors ({noise_factors.shape[-1]}) "
            f"does not match n_tasks ({n_tasks})"
        )

    # Compute noisy task times: (..., n_tasks)
    noisy_task_times = base_times * noise_factors

    # Build (m, n) assignment matrix
    A = assignment_to_matrix(
        station_assignments,
        n_tasks=n_tasks,
        zero_indexed_stations=zero_indexed_stations,
        zero_indexed_tasks=zero_indexed_tasks,
    )

    # Station times: (..., n_tasks) @ (n_tasks, m) -> (..., m)
    station_times = np.matmul(noisy_task_times, A.T)
    return station_times
