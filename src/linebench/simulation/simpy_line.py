"""SimPy discrete-event validator for unpaced tandem lines (Master Plan §5.6, Phase 5.5).

Provides an independent, event-driven discrete-event simulation model of the exact
same tandem line with Blocking-After-Service (BAS), finite buffer capacities,
infinite supply upstream, and infinite demand downstream.

Validation Principle:
---------------------
When fed the *identical* station-time matrix S of shape (N, K) and buffer configuration,
the discrete-event simulation in SimPy and the dynamic programming recursion in `tandem.py`
must produce bit-for-bit identical departure times d(k, i) within numerical floating-point
tolerance (|D_simpy - D_recursion| < 1e-9).

This is a structural mathematical verification, far stronger than mere statistical agreement.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import simpy

__all__ = ["simulate_simpy_line"]


def simulate_simpy_line(
    station_times: np.ndarray,
    buffers: int | Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run an event-driven SimPy simulation of an unpaced tandem line with BAS.

    Parameters
    ----------
    station_times:
        2D array of shape (N, K) with processing time S(k, i) of unit i at station k.
    buffers:
        Buffer capacities between adjacent stations. Either a scalar b or sequence of length K-1.

    Returns
    -------
    tuple of (start_times, finish_times, departure_times):
        Each is an ndarray of shape (N, K) containing the event timestamps for every unit
        at every station.
    """
    S = np.asarray(station_times, dtype=np.float64)
    N, K = S.shape

    if isinstance(buffers, (int, np.integer)):
        buf_caps = [int(buffers)] * (K - 1)
    else:
        buf_caps = [int(b) for b in buffers]
        if len(buf_caps) != K - 1:
            raise ValueError(f"buffers length must be K-1={K-1}, got {len(buf_caps)}")

    env = simpy.Environment()

    # Station resources: capacity 1 (one unit in service at a time)
    stations = [simpy.Resource(env, capacity=1) for _ in range(K)]

    # Downstream spaces:
    # In Blocking-After-Service (BAS):
    # Total capacity between station k and station k+1 is:
    #   buffer capacity (b_k) + downstream server (1) = b_k + 1.
    spaces = [simpy.Resource(env, capacity=buf_caps[k] + 1) for k in range(K - 1)]

    # Result arrays
    start_times = np.zeros((N, K), dtype=np.float64)
    finish_times = np.zeros((N, K), dtype=np.float64)
    departure_times = np.zeros((N, K), dtype=np.float64)

    def unit_process(unit_idx: int):
        prev_space_req = None
        # Unit moves through station 0 .. K-1 in sequence
        for k in range(K):
            # Request station k resource
            req_stn = stations[k].request()
            yield req_stn

            # Station service starts
            start_times[unit_idx, k] = env.now
            st_time = float(S[unit_idx, k])
            yield env.timeout(st_time)
            finish_times[unit_idx, k] = env.now

            # Service is finished. Can we depart station k?
            if k < K - 1:
                # Need space in downstream buffer / station k+1
                req_space = spaces[k].request()
                yield req_space

                # Departure from station k
                departure_times[unit_idx, k] = env.now
                stations[k].release(req_stn)

                # Now that we departed station k, we vacate a space in upstream buffer
                if k > 0 and prev_space_req is not None:
                    spaces[k - 1].release(prev_space_req)

                prev_space_req = req_space
            else:
                # Last station: infinite room downstream
                departure_times[unit_idx, k] = env.now
                stations[k].release(req_stn)
                if k > 0 and prev_space_req is not None:
                    spaces[k - 1].release(prev_space_req)

    # Spawn processes in arrival order (FIFO: unit 0, 1, 2...)
    for i in range(N):
        env.process(unit_process(i))

    env.run()

    return start_times, finish_times, departure_times
