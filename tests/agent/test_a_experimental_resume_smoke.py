import copy
import hashlib

import torch

from agent.mock_engine import MockNativeEngine, collect_mock_rollout
from agent.model import build_model
from agent.ppo import PPOConfig, PPOLearner
from agent.ppo.experimental_checkpoint import (
    build_envelope,
    load_envelope,
    restore_envelope,
    save_envelope,
)


def _digest(learner):
    return hashlib.sha256(b"".join(
        value.detach().cpu().contiguous().numpy().tobytes()
        for value in learner.model.state_dict().values()
    )).hexdigest()


def test_boundary_checkpoint_restore_matches_uninterrupted_second_iteration(model_factory, tmp_path):
    config = PPOConfig(global_minibatch_size=8, per_rank_minibatch_size=8,
                       microbatch_size=4, ppo_epochs=2, target_approx_kl=100.0)
    initial = model_factory(seed=20261011)
    initial_state = copy.deepcopy(initial.state_dict())
    fixture = collect_mock_rollout(
        MockNativeEngine(num_teams=4, requests_per_match=3), initial,
        envs=3, target_matches=3,
    )

    uninterrupted = PPOLearner(build_model(initial.config), config, device="cpu", amp=False)
    uninterrupted.model.load_state_dict(initial_state)
    stream_a = torch.Generator(device="cpu").manual_seed(55)
    for committed in (8, 16):
        uninterrupted.update_streaming(
            uninterrupted.prepare_streaming(fixture, cache_device="cpu"),
            committed_matches=committed, generator=stream_a,
        )

    first = PPOLearner(build_model(initial.config), config, device="cpu", amp=False)
    first.model.load_state_dict(initial_state)
    stream_b = torch.Generator(device="cpu").manual_seed(55)
    first.update_streaming(
        first.prepare_streaming(fixture, cache_device="cpu"),
        committed_matches=8, generator=stream_b,
    )
    envelope = build_envelope(
        first, source_sha="smoke", dataset_manifest_hash="dataset",
        action_schema_hash="actions", opponent_manifest={"current": "current", "historical": []},
        experiment={"name": "smoke", "ppo_epochs": 2}, rank=0, world_size=1,
        executor="manual", iteration=1, committed_matches=8,
        natural_games=8, learner_rows=len(fixture.rows),
        collector_restart={"seed": 1, "cohorts": 1, "match_counter": 8,
                           "slot_counter": 8, "config_hash": "collector"},
        rng_generators=[stream_b], rank_digests={"0": {"model": _digest(first), "optimizer": "opt"}},
    )
    path = tmp_path / "boundary.pt"
    save_envelope(path, envelope)
    resumed = PPOLearner(build_model(initial.config), config, device="cpu", amp=False)
    resumed.model.load_state_dict(initial_state)
    resumed_stream = torch.Generator(device="cpu").manual_seed(999)
    restore_envelope(resumed, load_envelope(path), dataset_manifest_hash="dataset",
                     action_schema_hash="actions", generators=[resumed_stream])
    resumed.update_streaming(
        resumed.prepare_streaming(fixture, cache_device="cpu"),
        committed_matches=16, generator=resumed_stream,
    )

    assert _digest(resumed) == _digest(uninterrupted)
    assert resumed.optimizer_steps == uninterrupted.optimizer_steps
    assert resumed.scheduler.matches == uninterrupted.scheduler.matches == 16
    assert resumed.scaler.state_dict() == uninterrupted.scaler.state_dict()
