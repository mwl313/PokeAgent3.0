import copy

import pytest
import torch

from agent.model import build_model
from agent.ppo import PPOConfig, PPOLearner
from agent.ppo.experimental_checkpoint import (
    build_envelope,
    fork_for_epochs,
    restore_envelope,
    load_envelope,
    save_envelope,
    validate_epoch_override,
    validate_envelope,
)


def _make(model_factory, epochs=4):
    model = model_factory(seed=20261011)
    learner = PPOLearner(
        model,
        PPOConfig(ppo_epochs=epochs, global_minibatch_size=8,
                  per_rank_minibatch_size=8, microbatch_size=4),
        device="cpu", amp=False,
    )
    parameter = next(learner.model.parameters())
    learner.optimizer.state[parameter] = {
        "step": torch.tensor(7.0),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.ones_like(parameter),
    }
    learner.scheduler.step_to(321)
    learner.optimizer_steps = 19
    return learner


def _envelope(learner, generator):
    return build_envelope(
        learner,
        source_sha="source",
        dataset_manifest_hash="dataset",
        action_schema_hash="actions",
        opponent_manifest={"current": "current", "historical": []},
        experiment={"name": "paired_epochs", "ppo_epochs": 4},
        rank=0, world_size=2, executor="ddp", iteration=3,
        committed_matches=321, natural_games=128, learner_rows=256,
        collector_restart={"seed": 20261011, "cohorts": 4,
                           "match_counter": 128, "slot_counter": 128,
                           "config_hash": "collector"},
        rng_generators=[generator], rank_digests={"0": "digest", "1": "digest"},
    )


def test_envelope_round_trip_restores_optimizer_scheduler_scaler_and_rng(model_factory, tmp_path):
    learner = _make(model_factory)
    generator = torch.Generator(device="cpu").manual_seed(17)
    envelope = _envelope(learner, generator)
    path = tmp_path / "iteration-boundary.pt"
    save_envelope(path, envelope)
    envelope = load_envelope(path)
    expected_model = {k: v.detach().clone() for k, v in learner.model.state_dict().items()}
    expected_optimizer = copy.deepcopy(learner.optimizer.state_dict())
    expected_draw = torch.randperm(8, generator=generator)

    restored = _make(model_factory)
    other = torch.Generator(device="cpu").manual_seed(999)
    restore_envelope(
        restored, envelope, dataset_manifest_hash="dataset",
        action_schema_hash="actions", generators=[other],
    )
    assert all(torch.equal(value, restored.model.state_dict()[key])
               for key, value in expected_model.items())
    assert restored.optimizer.state_dict()["state"].keys() == expected_optimizer["state"].keys()
    assert restored.scheduler.matches == 321
    assert restored.optimizer_steps == 19
    assert torch.equal(torch.randperm(8, generator=other), expected_draw)


def test_envelope_rejects_hash_and_boundary_or_rank_mismatches(model_factory):
    learner = _make(model_factory)
    generator = torch.Generator(device="cpu").manual_seed(17)
    kwargs = dict(
        source_sha="source", dataset_manifest_hash="dataset", action_schema_hash="actions",
        opponent_manifest={"current": "current", "historical": []},
        experiment={"name": "paired_epochs", "ppo_epochs": 4}, rank=0, world_size=2,
        executor="ddp", iteration=3, committed_matches=321, natural_games=128,
        learner_rows=256, collector_restart={"seed": 1, "cohorts": 1,
        "match_counter": 1, "slot_counter": 1, "config_hash": "collector"},
        rng_generators=[generator],
    )
    with pytest.raises(ValueError, match="in-flight"):
        build_envelope(learner, **kwargs, in_flight_battles=1)
    with pytest.raises(ValueError, match="divergent"):
        build_envelope(learner, **kwargs, rank_digests={"0": "a", "1": "b"})
    envelope = build_envelope(learner, **kwargs, rank_digests={"0": "a", "1": "a"})
    with pytest.raises(ValueError, match="dataset manifest"):
        validate_envelope(envelope, dataset_manifest_hash="wrong", action_schema_hash="actions")


def test_epoch_fork_changes_only_explicit_epoch_override(model_factory):
    envelope = _envelope(_make(model_factory), torch.Generator(device="cpu").manual_seed(17))
    fork = fork_for_epochs(envelope, 2)
    assert fork["experiment"]["ppo_epochs"] == 2
    assert fork["learner"]["config"]["ppo"]["ppo_epochs"] == 4
    assert fork["learner"]["config"]["ppo"] == envelope["learner"]["config"]["ppo"]
    for key, value in envelope["learner"]["model"].items():
        assert torch.equal(value, fork["learner"]["model"][key])
    validate_epoch_override(envelope, PPOConfig(ppo_epochs=2, global_minibatch_size=8,
                                                per_rank_minibatch_size=8, microbatch_size=4), 2)
    with pytest.raises(ValueError, match="frozen config"):
        validate_epoch_override(
            envelope, PPOConfig(ppo_epochs=2, global_minibatch_size=16,
                                per_rank_minibatch_size=16, microbatch_size=4), 2
        )

    left = torch.Generator(device="cpu").manual_seed(1)
    right = torch.Generator(device="cpu").manual_seed(2)
    # The envelope owns one copied RNG stream; restoring it into two distinct
    # generators must produce the same first shuffle draw.
    from agent.ppo.experimental_checkpoint import restore_rng_state
    restore_rng_state(fork["rng"], [left])
    restore_rng_state(fork["rng"], [right])
    assert torch.equal(torch.randperm(16, generator=left),
                       torch.randperm(16, generator=right))
