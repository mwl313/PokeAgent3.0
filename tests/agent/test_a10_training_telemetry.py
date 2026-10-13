"""Tests for the additive, rank-local PPO training metrics writer."""
import json
from types import SimpleNamespace

import pytest

from agent.ppo.telemetry import append_training_update


def fake_report():
    return {
        "policy_loss": -0.1, "value_loss": 0.2, "entropy": 0.7,
        "uniform_kl": 0.01, "approx_kl": 0.004,
        "epoch_approx_kl": [0.001, 0.004],
        "ratio_mean": 1.02, "clip_fraction": 0.03,
        "grad_norm": 0.4, "grad_norm_max": 0.6,
        "learning_rate": 0.00015, "optimizer_steps": 100,
        "optimizer_steps_skipped": 0, "epochs_run": 2,
        "stopped_early": False, "scaler_scale": 65536.0,
        "rows": 9000, "actor_rows": 9000,
    }


def test_write_two_iterations(tmp_path):
    path = tmp_path / "metrics.rank0.jsonl"
    for i in (1, 2):
        append_training_update(
            path, iteration=i, committed_matches=i*4096, rank=0,
            update=fake_report(), collector={"games": 2048, "operational_errors": 0},
            iteration_matches=4096, iteration_wall_s=40.0,
            timing={"update_wall_s": 20.0},
            recompute={"within_gate": True},
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert [r["identity"]["absolute_iteration"] for r in rows] == [1, 2]
    assert rows[0]["ppo"]["approx_kl"] == 0.004
    assert rows[1]["performance"]["training_games_per_second"] == 102.4


def test_reject_nonfinite_metric(tmp_path):
    update = fake_report()
    update["approx_kl"] = float("nan")
    with pytest.raises(ValueError):
        append_training_update(
            tmp_path / "bad.jsonl", iteration=1, committed_matches=4096,
            rank=0, update=update, collector={}, iteration_matches=4096,
            iteration_wall_s=40.0,
        )
    assert not (tmp_path / "bad.jsonl").exists()


def test_reject_incomplete_report(tmp_path):
    with pytest.raises(ValueError, match="missing telemetry"):
        append_training_update(
            tmp_path / "missing.jsonl", iteration=1, committed_matches=4096,
            rank=0, update={}, collector={}, iteration_matches=4096,
            iteration_wall_s=40.0,
        )
