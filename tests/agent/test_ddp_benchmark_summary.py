"""CPU checks for the persistent launcher's performance accounting."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def launcher():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_ddp_ppo.py"
    spec = importlib.util.spec_from_file_location("ddp_benchmark_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iteration(games, wall, clock, *, collect=0.0, update=0.0):
    return {
        "games": games,
        "rows": games * 20,
        "iteration_wall_s": wall,
        "global_committed_matches": clock,
        "digests_equal": True,
        "operational_errors": 0,
        "collect_wall_s": collect,
        "update_wall_s": update,
    }


def test_uses_measured_makespan_not_rank_sum_or_sum_of_stage_maxima(launcher):
    # Different ranks are slow in different stages. Stage maxima (9 + 9)
    # would overstate the measured critical rank wall of 10 seconds.
    ranks = [
        {"iterations": [iteration(100, 10, 200, collect=9, update=1)]},
        {"iterations": [iteration(100, 10, 200, collect=1, update=9)]},
    ]
    summaries, steady = launcher.summarize_iterations(ranks)
    assert summaries[0]["iteration_wall_s"] == 10
    assert summaries[0]["committed_games_per_s"] == 20
    assert summaries[0]["max_rank_stage_wall_s"]["collect_wall_s"] == 9
    assert summaries[0]["max_rank_stage_wall_s"]["update_wall_s"] == 9
    assert steady is None


def test_steady_state_excludes_cold_iteration_and_weights_by_wall(launcher):
    ranks = [
        {"iterations": [
            iteration(100, 100, 200),
            iteration(100, 5, 400),
            iteration(200, 20, 800),
        ]},
        {"iterations": [
            iteration(100, 90, 200),
            iteration(100, 4, 400),
            iteration(200, 19, 800),
        ]},
    ]
    summaries, steady = launcher.summarize_iterations(ranks)
    assert [entry["global_committed_matches"] for entry in summaries] == [200, 400, 800]
    assert steady["first_iteration"] == 2
    assert steady["iterations"] == 2
    assert steady["total_games"] == 600
    assert steady["wall_s"] == 25
    assert steady["committed_games_per_s"] == 24  # not mean(40, 20)


def test_summary_rejects_incomplete_rank_series(launcher):
    with pytest.raises(ValueError, match="same nonzero iteration count"):
        launcher.summarize_iterations([
            {"iterations": [iteration(1, 1, 2)]}, {"iterations": []},
        ])


def test_summary_rejects_divergent_lr_clock(launcher):
    with pytest.raises(ValueError, match="committed-match clock"):
        launcher.summarize_iterations([
            {"iterations": [iteration(1, 1, 2)]},
            {"iterations": [iteration(1, 1, 3)]},
        ])


def test_persistent_cli_preserves_one_iteration_default(launcher):
    defaults = launcher.parse_args([])
    assert defaults.iterations == 1
    assert defaults.batch_cache == "none"
    options = launcher.parse_args(["--iterations", "3", "--batch-cache", "cuda"])
    assert options.iterations == 3
    assert options.batch_cache == "cuda"


def test_cli_rejects_zero_iterations(launcher):
    with pytest.raises(SystemExit):
        launcher.parse_args(["--iterations", "0"])
