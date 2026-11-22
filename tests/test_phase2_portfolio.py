"""
Tests for Phase 2.6 (Portfolio Helper).
Verifies portfolio aggregation, identification of winning heuristics,
runtime summation, and handling of optional multi-pass.
"""

import pytest

from linebench.heuristics.portfolio import evaluate_heuristic_portfolio
from linebench.instances.loader import load_in2


@pytest.fixture
def bowman_instance():
    """Bowman-8 instance fixture."""
    return load_in2("data/raw/scholl1993/precedence graphs/BOWMAN8.IN2", cycle_time=20)


def test_portfolio_evaluation_bowman(bowman_instance):
    """Verify portfolio evaluation on Bowman-8."""
    portfolio = evaluate_heuristic_portfolio(
        bowman_instance,
        rules=["RPW", "UBT", "MTS", "MDS", "EST", "LST"],
        directions=["forward", "backward"],
        include_multi_pass=True,
        multi_pass_passes=50,
        multi_pass_seed=42,
    )

    # Optimum for Bowman-8 is 5 stations
    assert portfolio.best_stations == 5
    assert len(portfolio.winning_methods) >= 1
    assert portfolio.all_feasible
    assert portfolio.total_runtime_seconds > 0.0

    # 12 deterministic methods + 1 multi-pass = 13 methods
    assert len(portfolio.method_stations) == 13
    assert len(portfolio.method_runtimes) == 13

    # Check that winning methods all achieve 5 stations
    for m in portfolio.winning_methods:
        assert portfolio.method_stations[m] == 5


def test_portfolio_without_multi_pass(bowman_instance):
    """Verify deterministic-only portfolio."""
    portfolio = evaluate_heuristic_portfolio(
        bowman_instance,
        include_multi_pass=False,
    )
    assert len(portfolio.method_stations) == 12
    assert all("MULTI_PASS" not in m for m in portfolio.method_stations)
