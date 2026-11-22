"""Smoke tests for LINEBENCH Phase 0 setup."""

from linebench.evaluation.logging import (
    ResultsLogger,
    create_seed_sequence,
    get_config_hash,
    load_yaml_config,
)
from linebench.exact.prototype import solve_salbp1_prototype


def test_prototype_salbp1():
    """Verify that CP-SAT prototype correctly solves Bowman-8 (c=20) to optimum 5."""
    times = [11, 17, 9, 5, 8, 12, 10, 3]
    prec = [
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
        (2, 5),
        (3, 5),
        (4, 6),
        (5, 7),
    ]
    res = solve_salbp1_prototype(times, prec, cycle_time=20, time_limit_seconds=5.0)
    assert res["status"] == "OPTIMAL"
    assert res["stations_used"] == 5


def test_config_loading():
    """Verify loading of configuration files."""
    cfg = load_yaml_config("configs/default.yaml")
    assert cfg["project"]["name"] == "linebench"
    cfg_hash = get_config_hash(cfg)
    assert len(cfg_hash) == 12


def test_seed_sequence():
    """Verify independent random generator generation."""
    rngs = create_seed_sequence(42, num_subseeds=3)
    assert len(rngs) == 3
    val1 = rngs[0].random()
    val2 = rngs[1].random()
    assert val1 != val2


def test_results_logger(tmp_path):
    """Verify logging helper."""
    log_file = tmp_path / "test_log.jsonl"
    logger = ResultsLogger(str(log_file))
    logger.log_run(
        run_id="run_001",
        config_hash="abc12345",
        seed=42,
        instance_id="bowman8_c20",
        method="cp_sat",
        balance_id="b1",
        factor_settings={"time_limit": 5.0},
        metric="stations",
        value=5.0,
    )
    assert log_file.exists()
    with open(log_file) as f:
        lines = f.readlines()
        assert len(lines) == 1
        assert "bowman8_c20" in lines[0]
