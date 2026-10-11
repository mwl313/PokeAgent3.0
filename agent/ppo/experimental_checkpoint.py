"""Opt-in runtime state helpers for bounded PPO continuation experiments.

The production launcher does not call this module. It supplies only the
process RNG portion that can be restored independently of the learner state;
engine handles and in-flight collector state still require an iteration
boundary checkpoint.
"""

from __future__ import annotations

import random
import copy
import hashlib
import json
from typing import Iterable, Optional

import torch


SCHEMA_VERSION = 1


def hash_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capture_rng_state(generators: Optional[Iterable[torch.Generator]] = None) -> dict:
    """Capture Python, CPU/CUDA torch, and explicitly owned generators."""
    state = {
        "schema_version": SCHEMA_VERSION,
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "generators": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    else:
        state["torch_cuda"] = None
    if generators is not None:
        state["generators"] = [generator.get_state().clone() for generator in generators]
    return state


def restore_rng_state(state: dict, generators: Optional[Iterable[torch.Generator]] = None) -> None:
    """Restore a state captured by :func:`capture_rng_state`."""
    if int(state.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("unsupported experimental RNG checkpoint schema")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)
    if generators is not None:
        owned = list(generators)
        saved = state.get("generators", [])
        if len(owned) != len(saved):
            raise ValueError("owned generator count does not match checkpoint")
        for generator, value in zip(owned, saved):
            generator.set_state(value)


def build_envelope(
    learner,
    *,
    source_sha: str,
    dataset_manifest_hash: str,
    action_schema_hash: str,
    opponent_manifest: dict,
    experiment: dict,
    rank: int,
    world_size: int,
    executor: str,
    iteration: int,
    committed_matches: int,
    natural_games: int,
    learner_rows: int,
    collector_restart: dict,
    rng_generators: Optional[Iterable[torch.Generator]] = None,
    rank_digests: Optional[dict] = None,
    in_flight_battles: int = 0,
    iteration_complete: bool = True,
) -> dict:
    """Build a validated, iteration-boundary-only experimental envelope.

    DDP communicators are intentionally represented as recreation metadata;
    no process group or reducer object is serialized.
    """
    if not iteration_complete or in_flight_battles != 0:
        raise ValueError("checkpoint requires a completed iteration with no in-flight battles")
    if rank_digests is not None and len(set(rank_digests.values())) != 1:
        raise ValueError("cannot checkpoint divergent rank digests")
    if committed_matches != learner.scheduler.matches:
        raise ValueError("committed-match clock disagrees with learner scheduler")
    restart_required = {"seed", "cohorts", "match_counter", "slot_counter", "config_hash"}
    if not restart_required.issubset(collector_restart):
        missing = sorted(restart_required - set(collector_restart))
        raise ValueError(f"collector restart contract missing fields: {missing}")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "ppo_experimental_iteration_boundary",
        "source_sha": source_sha,
        "dataset_manifest_hash": dataset_manifest_hash,
        "action_schema_hash": action_schema_hash,
        "opponent_manifest": copy.deepcopy(opponent_manifest),
        "experiment": copy.deepcopy(experiment),
        "learner": learner.state_dict(),
        "rng": capture_rng_state(rng_generators),
        "ddp": {"rank": int(rank), "world_size": int(world_size), "executor": executor,
                "communicator": "recreate"},
        "accounting": {
            "iteration": int(iteration), "committed_matches": int(committed_matches),
            "natural_games": int(natural_games), "learner_rows": int(learner_rows),
            "optimizer_steps": int(learner.optimizer_steps),
        },
        "collector_restart": copy.deepcopy(collector_restart),
        "rank_digests": copy.deepcopy(rank_digests or {}),
    }


def validate_envelope(envelope: dict, *, dataset_manifest_hash: str,
                      action_schema_hash: str, source_sha: Optional[str] = None,
                      opponent_manifest: Optional[dict] = None) -> None:
    if envelope.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported experimental checkpoint schema")
    if envelope.get("kind") != "ppo_experimental_iteration_boundary":
        raise ValueError("not an experimental iteration-boundary checkpoint")
    if envelope.get("dataset_manifest_hash") != dataset_manifest_hash:
        raise ValueError("dataset manifest hash mismatch")
    if envelope.get("action_schema_hash") != action_schema_hash:
        raise ValueError("action schema hash mismatch")
    if source_sha is not None and envelope.get("source_sha") != source_sha:
        raise ValueError("source SHA mismatch")
    if opponent_manifest is not None and envelope.get("opponent_manifest") != opponent_manifest:
        raise ValueError("opponent manifest mismatch")
    digests = envelope.get("rank_digests", {})
    if digests and len(set(digests.values())) != 1:
        raise ValueError("checkpoint contains divergent rank digests")
    accounting = envelope.get("accounting", {})
    if accounting.get("committed_matches") != envelope["learner"]["scheduler"].get("matches"):
        raise ValueError("checkpoint accounting does not match scheduler clock")


def restore_envelope(learner, envelope: dict, *, dataset_manifest_hash: str,
                     action_schema_hash: str, generators=None,
                     source_sha: Optional[str] = None,
                     opponent_manifest: Optional[dict] = None) -> None:
    validate_envelope(
        envelope, dataset_manifest_hash=dataset_manifest_hash,
        action_schema_hash=action_schema_hash, source_sha=source_sha,
        opponent_manifest=opponent_manifest,
    )
    learner.load_state_dict(envelope["learner"])
    restore_rng_state(envelope["rng"], generators)


def validate_epoch_override(envelope: dict, config, epochs: int) -> None:
    """Allow only the explicit 2/4 epoch A/B change in a fork config."""
    if epochs not in (2, 4):
        raise ValueError("experimental PPO fork must use 2 or 4 epochs")
    saved = envelope["learner"]["config"]["ppo"]
    current = dict(config.__dict__)
    for key, value in saved.items():
        if key == "ppo_epochs":
            continue
        if current.get(key) != value:
            raise ValueError(f"PPO fork changed frozen config field: {key}")


def fork_for_epochs(envelope: dict, epochs: int) -> dict:
    """Clone a verified envelope with only the explicit PPO epoch override."""
    if epochs not in (2, 4):
        raise ValueError("experimental PPO fork must use 2 or 4 epochs")
    fork = copy.deepcopy(envelope)
    fork["experiment"]["ppo_epochs"] = int(epochs)
    fork["experiment"]["fork_parent_schema_version"] = SCHEMA_VERSION
    return fork


def save_envelope(path, envelope: dict) -> None:
    """Persist an already validated envelope at an experiment boundary."""
    torch.save(envelope, path)


def load_envelope(path) -> dict:
    """Load an envelope without implicitly restoring any process state."""
    envelope = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(envelope, dict):
        raise ValueError("experimental checkpoint is not an envelope mapping")
    return envelope
