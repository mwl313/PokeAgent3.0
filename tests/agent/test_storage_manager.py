import json
from pathlib import Path

import pytest
import torch

from agent.ppo.storage_manager import CheckpointGenerationStore, StorageBudgetError


META = {"schema": "ppo-full-v1", "iteration_complete": True, "world_size": 2, "iteration": 4, "committed_matches": 4096}


def test_generation_publish_rotation_and_latest(tmp_path):
    store = CheckpointGenerationStore(tmp_path / "rolling", reserve_bytes=0)
    for i in range(4):
        store.publish(f"g{i}", {0: f"zero-{i}".encode(), 1: f"one-{i}".encode()}, META)
    ready = store.generations_ready()
    assert len(ready) == 3
    assert store.latest() == ready[0]
    assert json.loads((store.latest() / "manifest.json").read_text())["ranks"]["1"]["sha256"]


def test_missing_rank_and_incomplete_metadata_are_rejected(tmp_path):
    store = CheckpointGenerationStore(tmp_path / "rolling", reserve_bytes=0)
    with pytest.raises(ValueError): store.publish("bad", {0: b"x"}, META)
    with pytest.raises(ValueError): store.publish("bad", {0: b"x", 1: b"y"}, {**META, "iteration_complete": False})


def test_corrupt_latest_falls_back_and_cleanup_is_safe(tmp_path):
    store = CheckpointGenerationStore(tmp_path / "rolling", reserve_bytes=0)
    store.publish("good", {0: b"a", 1: b"b"}, META)
    previous = store.latest()
    store.publish("new", {0: b"c", 1: b"d"}, META)
    latest = store.latest(); (latest / "rank1.pt").write_bytes(b"corrupt")
    assert store.latest() == previous
    assert store.generations_ready() == [previous]
    incomplete = store.generations / ".incomplete-crash"; incomplete.mkdir()
    assert str(incomplete) in store.cleanup_incomplete()


def test_insufficient_space_refuses_write(tmp_path):
    store = CheckpointGenerationStore(tmp_path / "rolling", reserve_bytes=10**18)
    with pytest.raises(StorageBudgetError): store.publish("no", {0: b"a", 1: b"b"}, META)


def test_protected_generation_is_not_rotated(tmp_path):
    store = CheckpointGenerationStore(tmp_path / "rolling", reserve_bytes=0)
    first = store.publish("g0", {0: b"0", 1: b"0"}, META)
    first_dir = store.latest()
    store.protect(first_dir, reason="A5 control anchor")
    for i in range(1, 5): store.publish(f"g{i}", {0: str(i).encode(), 1: str(i).encode()}, META)
    assert first_dir.exists()


def test_real_torch_payload_round_trip_and_metadata_parity(tmp_path):
    store = CheckpointGenerationStore(tmp_path / "rolling", reserve_bytes=0)
    payloads = {}
    for rank in (0, 1):
        payload = {"model": {"weight": torch.arange(8) + rank},
                   "optimizer": {"state": {"step": torch.tensor(3)}},
                   "scheduler": {"matches": 4096}, "scaler": {"scale": 65536.0},
                   "dataset_manifest_hash": "d" * 64, "precision_mode": "D"}
        path = tmp_path / f"rank{rank}.pt"
        torch.save(payload, path)
        payloads[rank] = path.read_bytes()
    manifest = store.publish("torch", payloads, {**META, "dataset_manifest_hash": "d" * 64, "precision_mode": "D"})
    generation = store.latest()
    for rank in (0, 1):
        restored = torch.load(generation / manifest["ranks"][str(rank)]["path"], weights_only=True)
        assert restored["optimizer"]["state"]["step"].item() == 3
        assert restored["scheduler"]["matches"] == 4096
        assert restored["dataset_manifest_hash"] == "d" * 64
