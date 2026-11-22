"""
Instance data structure and loaders for SALBP-1 benchmark files.
Supports both the modern .ALB tagged format (Otto et al. 2013)
and the classical .IN2 precedence graph format (Scholl 1993).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Instance:
    """Represents a Simple Assembly Line Balancing Problem (SALBP-1) instance.

    Attributes:
        name: Name or identifier of the instance.
        num_tasks: Total number of tasks n.
        cycle_time: Cycle time c (available time per station).
        task_times: Dictionary mapping task ID (1-indexed) to its integer processing time.
        precedences: Set of direct precedence pairs (i, j) where task i must precede task j.
        order_strength: Optional order strength float if declared in file metadata.
    """

    name: str
    num_tasks: int
    cycle_time: int
    task_times: dict[int, int]
    precedences: set[tuple[int, int]] = field(default_factory=set)
    order_strength: float | None = None

    def __post_init__(self) -> None:
        if self.num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {self.num_tasks}")
        if self.cycle_time <= 0:
            raise ValueError(f"cycle_time must be positive, got {self.cycle_time}")
        if len(self.task_times) != self.num_tasks:
            raise ValueError(
                f"Mismatch: num_tasks={self.num_tasks} but task_times has {len(self.task_times)} entries"
            )
        # Ensure tasks are 1-indexed up to n
        expected_tasks = set(range(1, self.num_tasks + 1))
        actual_tasks = set(self.task_times.keys())
        if expected_tasks != actual_tasks:
            missing = expected_tasks - actual_tasks
            extra = actual_tasks - expected_tasks
            raise ValueError(f"Task index mismatch: missing={missing}, extra={extra}")

        # Check task times
        for t_id, duration in self.task_times.items():
            if duration <= 0:
                raise ValueError(f"Task {t_id} duration must be positive, got {duration}")
            if duration > self.cycle_time:
                raise ValueError(
                    f"Task {t_id} duration ({duration}) exceeds cycle time ({self.cycle_time})"
                )

        # Check precedence pairs
        for u, v in self.precedences:
            if u not in expected_tasks or v not in expected_tasks:
                raise ValueError(f"Precedence pair ({u}, {v}) refers to unknown task ID")
            if u == v:
                raise ValueError(f"Self-loop precedence detected for task {u}")


def load_alb(file_path: str | Path) -> Instance:
    """Parses an instance file in .ALB format (Otto et al. 2013).

    Block format specification:
      <number of tasks>
      n
      <cycle time>
      c
      <order strength> (optional)
      os
      <task times>
      task_id duration
      <precedence relations>
      pred_id,succ_id
      <end>
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Instance file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f]

    current_section = None
    num_tasks = None
    cycle_time = None
    order_strength = None
    task_times: dict[int, int] = {}
    precedences: set[tuple[int, int]] = set()

    for line in lines:
        if not line:
            continue
        if line.startswith("<") and line.endswith(">"):
            current_section = line.lower()
            if current_section == "<end>":
                break
            continue

        if current_section == "<number of tasks>":
            num_tasks = int(line)
        elif current_section == "<cycle time>":
            cycle_time = int(line)
        elif current_section == "<order strength>":
            # Some files use European comma decimals: '0,268'
            order_strength = float(line.replace(",", "."))
        elif current_section == "<task times>":
            parts = line.split()
            if len(parts) >= 2:
                task_id = int(parts[0])
                duration = int(parts[1])
                task_times[task_id] = duration
        elif current_section == "<precedence relations>":
            # Supports comma-separated '1,6' or whitespace-separated '1 6'
            parts = re.split(r"[,;\s]+", line)
            if len(parts) >= 2 and parts[0] and parts[1]:
                u = int(parts[0])
                v = int(parts[1])
                precedences.add((u, v))

    if num_tasks is None:
        raise ValueError(f"Missing <number of tasks> block in {path}")
    if cycle_time is None:
        raise ValueError(f"Missing <cycle time> block in {path}")

    return Instance(
        name=path.stem,
        num_tasks=num_tasks,
        cycle_time=cycle_time,
        task_times=task_times,
        precedences=precedences,
        order_strength=order_strength,
    )


def load_in2(file_path: str | Path, cycle_time: int, name: str | None = None) -> Instance:
    """Parses a precedence graph file in .IN2 format (Scholl 1993).

    Format specification:
      Line 1: number of tasks n
      Lines 2 .. n+1: integer task times (for tasks 1 .. n)
      Lines n+2 ..: direct precedence relations 'i,j'
      Last line: optional end mark '-1,-1' or citation notes
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"IN2 file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    if not raw_lines:
        raise ValueError(f"Empty IN2 file: {path}")

    num_tasks = int(raw_lines[0])
    task_times: dict[int, int] = {}
    for i in range(1, num_tasks + 1):
        if i >= len(raw_lines):
            raise ValueError(f"Unexpected end of file while reading task times in {path}")
        task_times[i] = int(raw_lines[i])

    precedences: set[tuple[int, int]] = set()
    for line in raw_lines[num_tasks + 1 :]:
        if line.startswith("-1") or line == "-1,-1":
            break
        # Ignore comments or trailing attribution lines that don't match precedence pattern
        parts = re.split(r"[,;\s]+", line)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            u = int(parts[0])
            v = int(parts[1])
            precedences.add((u, v))

    instance_name = name or path.stem
    return Instance(
        name=instance_name,
        num_tasks=num_tasks,
        cycle_time=cycle_time,
        task_times=task_times,
        precedences=precedences,
    )
