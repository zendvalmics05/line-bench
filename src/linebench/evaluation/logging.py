"""
Seed and Reproducibility Utilities for LINEBENCH.
Provides NumPy SeedSequence hierarchy and result logging helpers.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_yaml_config(path: str) -> dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_hash(config_dict: dict[str, Any]) -> str:
    """Computes a SHA-256 hash of a configuration dictionary."""
    serialized = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def create_seed_sequence(master_seed: int, num_subseeds: int = 1) -> list[np.random.Generator]:
    """Generates independent NumPy random generators using SeedSequence."""
    ss = np.random.SeedSequence(master_seed)
    child_seeds = ss.spawn(num_subseeds)
    return [np.random.default_rng(s) for s in child_seeds]


class ResultsLogger:
    """Appends benchmark or simulation runs to structured JSONL logs."""

    def __init__(self, log_path: str = "results/run_log.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        run_id: str,
        config_hash: str,
        seed: int,
        instance_id: str,
        method: str,
        balance_id: str | None,
        factor_settings: dict[str, Any],
        metric: str,
        value: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Appends a single structured record to the log."""
        record = {
            "timestamp": time.time(),
            "run_id": run_id,
            "config_hash": config_hash,
            "seed": seed,
            "instance_id": instance_id,
            "method": method,
            "balance_id": balance_id,
            "factor_settings": factor_settings,
            "metric": metric,
            "value": value,
            "extra": extra or {},
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
