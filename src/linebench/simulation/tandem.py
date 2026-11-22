"""Unpaced tandem line simulator (Master Plan §A9, Phase 5.3).

Implements the departure-time recursion for a finite-buffer tandem line
with Blocking-After-Service (BAS), infinite supply before station 1, and
infinite demand / room after station K.

Recursion Equations (Master Plan §A9):
--------------------------------------
For unit i = 0 .. N-1 (0-based indexing) and station k = 0 .. K-1:
  - Start time:
      s(k, i) = max( d(k-1, i), d(k, i-1) )
      with d(-1, i) = 0.0 (infinite supply) and d(k, -1) = 0.0 (station initially empty).
  - Service finish time:
      f(k, i) = s(k, i) + S(k, i)
      where S(k, i) is the processing time of unit i at station k.
  - Departure time:
      For k < K-1:
        d(k, i) = max( f(k, i), d(k+1, i - b_k - 1) )
        where b_k is intermediate buffer capacity between station k and k+1.
        (For i - b_k - 1 < 0, buffer has space, term evaluates to 0.0).
      For k = K-1 (last station, infinite demand downstream):
        d(K-1, i) = f(K-1, i)

Performance Metrics (measured over units W .. N-1, where W = warm_up):
----------------------------------------------------------------------
  - Throughput:
      TH = (N - W) / ( d(K-1, N-1) - d(K-1, W-1) )
  - Utilization (busy fraction) for station k:
      sum_{i=W}^{N-1} S(k, i) / WindowDuration
  - Starvation fraction for station k:
      fraction of measurement window spent idle waiting for unit from upstream.
  - Blocked fraction for station k:
      fraction of measurement window spent blocked after service waiting for downstream space.
  - WIP (Work In Progress):
      Average number of units in the line (at stations and in buffers) during the window.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SimulationResult",
    "simulate_tandem_line",
    "simulate_tandem_replications",
]


@dataclass
class SimulationResult:
    """Output metrics from simulating an unpaced tandem line."""

    throughput: float
    utilization: np.ndarray  # (K,) busy fraction per station
    starved_fraction: np.ndarray  # (K,) starved fraction per station
    blocked_fraction: np.ndarray  # (K,) blocked fraction per station
    mean_wip: float  # average work in progress in system
    departure_times: np.ndarray | None = None  # (N, K) departure times if requested
    start_times: np.ndarray | None = None  # (N, K) start times if requested
    finish_times: np.ndarray | None = None  # (N, K) finish times if requested


def simulate_tandem_line(
    station_times: np.ndarray,
    buffers: int | Sequence[int] | np.ndarray,
    warmup_units: int = 500,
    return_details: bool = False,
) -> SimulationResult:
    """Simulate a single replication of an unpaced tandem line.

    Parameters
    ----------
    station_times:
        Array of shape (N, K) where N is number of units and K is number of stations.
        station_times[i, k] is the service time S(k, i) of unit i at station k.
    buffers:
        Buffer capacity. If an integer b, all K-1 intermediate buffers have capacity b.
        If a sequence/array of length K-1, buffers[k] is capacity between station k and k+1.
    warmup_units:
        Number of initial units W to discard from steady-state statistics.
    return_details:
        If True, includes full (N, K) departure, start, and finish time matrices in result.

    Returns
    -------
    SimulationResult
    """
    S = np.asarray(station_times, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"station_times must be 2D array of shape (N, K), got shape {S.shape}")

    N, K = S.shape
    if N <= warmup_units:
        raise ValueError(f"Total units N={N} must be strictly greater than warmup_units={warmup_units}")

    # Prepare buffer capacities: length K - 1
    if isinstance(buffers, (int, np.integer)):
        buf = np.full(K - 1, int(buffers), dtype=np.int64)
    else:
        buf = np.asarray(buffers, dtype=np.int64)
        if len(buf) != K - 1:
            raise ValueError(f"Buffer array length must be K-1={K-1}, got {len(buf)}")

    # State matrices: s(k, i), f(k, i), d(k, i)
    s_arr = np.zeros((N, K), dtype=np.float64)
    f_arr = np.zeros((N, K), dtype=np.float64)
    d_arr = np.zeros((N, K), dtype=np.float64)

    # Unit-by-unit forward recursion
    for i in range(N):
        for k in range(K):
            # Arrival from upstream: d(k-1, i)
            arr_up = d_arr[i, k - 1] if k > 0 else 0.0
            # Previous unit departure from this station: d(k, i-1)
            prev_dep = d_arr[i - 1, k] if i > 0 else 0.0

            # Start time: s(k, i) = max( d(k-1, i), d(k, i-1) )
            st = max(prev_dep, arr_up)
            s_arr[i, k] = st

            # Finish service time: f(k, i) = s(k, i) + S(k, i)
            ft = st + S[i, k]
            f_arr[i, k] = ft

            # Departure time: d(k, i)
            if k == K - 1:
                # Last station: infinite downstream capacity
                d_arr[i, k] = ft
            else:
                # Upstream station: must wait if downstream buffer is full
                # Downstream space opens when unit (i - b_k - 1) departs station k+1
                b_k = int(buf[k])
                unblock_idx = i - b_k - 1
                unblock_t = d_arr[unblock_idx, k + 1] if unblock_idx >= 0 else 0.0
                d_arr[i, k] = max(unblock_t, ft)

    # Compute statistics over measurement window [W, N-1]
    W = warmup_units
    n_meas = N - W

    # Window elapsed time measured by output of line (Station K-1)
    # Start of measurement window: departure of unit W-1 (or 0.0 if W=0)
    t_start = d_arr[W - 1, K - 1] if W > 0 else 0.0
    t_end = d_arr[N - 1, K - 1]
    window_duration = t_end - t_start

    if window_duration <= 0.0:
        throughput = 0.0
    else:
        throughput = n_meas / window_duration

    # Busy, blocked, and starved times per station during window
    # For station k, total busy time in measurement window = sum_{i=W}^{N-1} S(k, i)
    busy_times = np.sum(S[W:N, :], axis=0)
    utilization = np.zeros(K, dtype=np.float64)
    if window_duration > 0:
        utilization = busy_times / window_duration

    # Blocked time: time spent waiting between service finish and departure
    # f(k, i) -> d(k, i)
    blocked_times = np.sum(d_arr[W:N, :] - f_arr[W:N, :], axis=0)
    blocked_fraction = np.zeros(K, dtype=np.float64)
    if window_duration > 0:
        blocked_fraction = blocked_times / window_duration

    # Starved time: time station spent idle waiting for unit arrival
    # Between d(k, i-1) and s(k, i)
    starved_times = np.zeros(K, dtype=np.float64)
    for k in range(1, K):
        for i in range(W, N):
            prev_d = d_arr[i - 1, k] if i > 0 else 0.0
            st = s_arr[i, k]
            if st > prev_d:
                starved_times[k] += st - prev_d

    starved_fraction = np.zeros(K, dtype=np.float64)
    if window_duration > 0:
        starved_fraction = starved_times / window_duration

    # Mean WIP via Little's Law over the measurement window:
    # WIP = Throughput * Mean Flow Time through line during measurement window
    # Flow time of unit i in system = d(K-1, i) - s(0, i)
    flow_times = d_arr[W:N, K - 1] - s_arr[W:N, 0]
    mean_flow_time = float(np.mean(flow_times)) if n_meas > 0 else 0.0
    mean_wip = throughput * mean_flow_time

    return SimulationResult(
        throughput=float(throughput),
        utilization=utilization,
        starved_fraction=starved_fraction,
        blocked_fraction=blocked_fraction,
        mean_wip=float(mean_wip),
        departure_times=d_arr if return_details else None,
        start_times=s_arr if return_details else None,
        finish_times=f_arr if return_details else None,
    )


def simulate_tandem_replications(
    station_times: np.ndarray,
    buffers: int | Sequence[int] | np.ndarray,
    warmup_units: int = 500,
) -> dict[str, np.ndarray]:
    """Vectorized / batch runner across multiple replications.

    Parameters
    ----------
    station_times:
        3D array of shape (n_replications, n_units, n_stations).
    buffers:
        Buffer capacity (scalar or sequence of length K-1).
    warmup_units:
        Warmup units to discard.

    Returns
    -------
    dict:
        - "throughput": 1D array of shape (n_replications,)
        - "utilization": 2D array of shape (n_replications, n_stations)
        - "blocked_fraction": 2D array of shape (n_replications, n_stations)
        - "starved_fraction": 2D array of shape (n_replications, n_stations)
        - "mean_wip": 1D array of shape (n_replications,)
    """
    S = np.asarray(station_times, dtype=np.float64)
    if S.ndim != 3:
        raise ValueError(
            f"station_times must be 3D array of shape (n_replications, n_units, n_stations), got {S.shape}"
        )

    n_reps, _n, K = S.shape
    throughputs = np.zeros(n_reps, dtype=np.float64)
    utils = np.zeros((n_reps, K), dtype=np.float64)
    blocked = np.zeros((n_reps, K), dtype=np.float64)
    starved = np.zeros((n_reps, K), dtype=np.float64)
    wips = np.zeros(n_reps, dtype=np.float64)

    for r in range(n_reps):
        res = simulate_tandem_line(
            station_times=S[r],
            buffers=buffers,
            warmup_units=warmup_units,
            return_details=False,
        )
        throughputs[r] = res.throughput
        utils[r] = res.utilization
        blocked[r] = res.blocked_fraction
        starved[r] = res.starved_fraction
        wips[r] = res.mean_wip

    return {
        "throughput": throughputs,
        "utilization": utils,
        "blocked_fraction": blocked,
        "starved_fraction": starved,
        "mean_wip": wips,
    }
