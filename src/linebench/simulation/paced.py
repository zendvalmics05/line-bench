"""Paced (synchronous) assembly line model (Master Plan §A10, Phase 5.4).

In a paced line with fixed cycle time c:
  - All stations operate synchronously on a conveyor or transfer bar.
  - The line waits for the slowest station to finish before advancing (wait-for-slowest):
      cycle_length(i) = max_{k=0..K-1} S(k, i)
  - Throughput (units per unit time) is the reciprocal of the mean cycle length:
      TH = 1 / mean_{i} [ max_{k} S(k, i) ]
  - Overload probability per station:
      P(S(k, i) > c) = fraction of cycles where station k exceeds cycle time c.
  - Line overload probability:
      P(max_k S(k, i) > c) = fraction of cycles where at least one station exceeds c.
  - Normal approximation of overload probability (Master Plan §A11):
      For station k with mean load L_k and task times t_j in k:
      P(S_k > c) ≈ 1 - Φ( (c - L_k) / sqrt( CV^2 * sum_{j in k} t_j^2 ) )
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = [
    "PacedResult",
    "normal_approx_overload_probability",
    "simulate_paced_line",
    "simulate_paced_replications",
]


@dataclass
class PacedResult:
    """Output metrics from simulating a paced line."""

    throughput: float
    mean_cycle_length: float
    station_overload_prob: np.ndarray  # (K,) empirical overload prob per station
    line_overload_prob: float  # empirical prob that at least one station exceeds c


def simulate_paced_line(
    station_times: np.ndarray,
    cycle_time: float,
    warmup_units: int = 0,
) -> PacedResult:
    """Simulate a single replication of a paced assembly line.

    Parameters
    ----------
    station_times:
        Array of shape (N, K) representing service times of N units at K stations.
    cycle_time:
        Nominal cycle time c.
    warmup_units:
        Number of initial units to discard (default 0 for paced lines).

    Returns
    -------
    PacedResult
    """
    S = np.asarray(station_times, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"station_times must be 2D array of shape (N, K), got shape {S.shape}")

    N, _k = S.shape
    if N <= warmup_units:
        raise ValueError(f"Total units N={N} must be strictly greater than warmup_units={warmup_units}")

    S_meas = S[warmup_units:N, :]  # (N_meas, K)

    # Wait-for-slowest cycle length for each unit i: max across all stations
    cycle_lengths = np.max(S_meas, axis=1)  # (N_meas,)
    mean_cycle_length = float(np.mean(cycle_lengths))
    throughput = 1.0 / mean_cycle_length if mean_cycle_length > 0.0 else 0.0

    # Overload probability per station: S(k, i) > c
    station_overload_prob = np.mean(S_meas > cycle_time, axis=0)  # (K,)

    # Line overload probability: max_k S(k, i) > c
    line_overload_prob = float(np.mean(cycle_lengths > cycle_time))

    return PacedResult(
        throughput=throughput,
        mean_cycle_length=mean_cycle_length,
        station_overload_prob=station_overload_prob,
        line_overload_prob=line_overload_prob,
    )


def simulate_paced_replications(
    station_times: np.ndarray,
    cycle_time: float,
    warmup_units: int = 0,
) -> dict[str, np.ndarray]:
    """Run paced line simulation across multiple replications.

    Parameters
    ----------
    station_times:
        3D array of shape (n_replications, n_units, n_stations).
    cycle_time:
        Nominal cycle time c.
    warmup_units:
        Warmup units to discard.

    Returns
    -------
    dict:
        - "throughput": (n_replications,)
        - "mean_cycle_length": (n_replications,)
        - "station_overload_prob": (n_replications, n_stations)
        - "line_overload_prob": (n_replications,)
    """
    S = np.asarray(station_times, dtype=np.float64)
    if S.ndim != 3:
        raise ValueError(
            f"station_times must be 3D array of shape (n_replications, n_units, n_stations), got {S.shape}"
        )

    n_reps, _n, K = S.shape
    throughputs = np.zeros(n_reps, dtype=np.float64)
    mean_cycles = np.zeros(n_reps, dtype=np.float64)
    station_overloads = np.zeros((n_reps, K), dtype=np.float64)
    line_overloads = np.zeros(n_reps, dtype=np.float64)

    for r in range(n_reps):
        res = simulate_paced_line(
            station_times=S[r],
            cycle_time=cycle_time,
            warmup_units=warmup_units,
        )
        throughputs[r] = res.throughput
        mean_cycles[r] = res.mean_cycle_length
        station_overloads[r] = res.station_overload_prob
        line_overloads[r] = res.line_overload_prob

    return {
        "throughput": throughputs,
        "mean_cycle_length": mean_cycles,
        "station_overload_prob": station_overloads,
        "line_overload_prob": line_overloads,
    }


def normal_approx_overload_probability(
    station_task_times: Sequence[float | int],
    cycle_time: float,
    cv: float,
) -> float:
    """Compute normal approximation of overload probability for a station (Master Plan §A11).

    Formula:
        P(S_k > c) = 1 - Φ( (c - L_k) / sqrt( CV^2 * sum_{j in k} t_j^2 ) )
    where L_k = sum_{j in k} t_j.

    Parameters
    ----------
    station_task_times:
        Nominal deterministic task times assigned to this station.
    cycle_time:
        Nominal cycle time c.
    cv:
        Task-time coefficient of variation.

    Returns
    -------
    float:
        Approximated probability of overload in [0.0, 1.0].
    """
    times = np.asarray(station_task_times, dtype=np.float64)
    if len(times) == 0:
        return 0.0

    load_k = float(np.sum(times))
    if cv <= 0.0:
        return 1.0 if load_k > cycle_time else 0.0

    variance_k = (cv * cv) * float(np.sum(times * times))
    std_k = np.sqrt(variance_k)

    if std_k <= 1e-12:
        return 1.0 if load_k > cycle_time else 0.0

    z = (cycle_time - load_k) / std_k
    # P(S_k > c) = 1 - Phi(z) = sf(z)
    return float(norm.sf(z))
