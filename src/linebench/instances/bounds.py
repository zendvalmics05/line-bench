"""
Theoretical lower bounds for SALBP-1 instances.
Computes:
  LB1: Total work content lower bound ceil(sum(t_j) / c)
  LB2: Half-cycle task counting lower bound
  LB: max(LB1, LB2)
"""

import math

from linebench.instances.loader import Instance


def compute_lb1(instance: Instance) -> int:
    """Computes LB1 = ceil(sum(t_j) / c).

    Based on the total work content divided by the station capacity.
    """
    total_time = sum(instance.task_times.values())
    return math.ceil(total_time / instance.cycle_time)


def compute_lb2(instance: Instance) -> int:
    """Computes LB2 based on tasks requiring more than half the cycle time.

    Tasks with t_j > c / 2 cannot share a station with any other task with t_k > c / 2.
    Tasks with t_j = c / 2 can share a station at most in pairs.
    LB2 = count(t_j > c/2) + ceil(count(t_j == c/2) / 2)
    """
    half_c = instance.cycle_time / 2.0
    strictly_greater = sum(1 for t in instance.task_times.values() if t > half_c)
    equal_half = sum(1 for t in instance.task_times.values() if t == half_c)
    return strictly_greater + math.ceil(equal_half / 2.0)


def compute_lower_bound(instance: Instance) -> int:
    """Computes the combined theoretical lower bound LB = max(LB1, LB2)."""
    return max(compute_lb1(instance), compute_lb2(instance))
