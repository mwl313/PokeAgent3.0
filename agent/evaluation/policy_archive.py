"""Small immutable JSON policy archive and non-blocking evaluation records."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from agent.evaluation.frozen_policy import FrozenPPOPolicy


ARCHIVE_SCHEMA = "pa3-policy-archive-v1"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register_checkpoint(registry_path, *, policy_id, label, checkpoint_path,
                        committed_matches=None, source_sha=None, incumbent=False):
    """Register a local checkpoint; refuse replacement with a new digest."""
    checkpoint_path = os.path.abspath(os.fspath(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"checkpoint is missing: {checkpoint_path}")
    file_hash = _sha256(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "learner" not in payload:
        raise ValueError("archive requires an experimental learner envelope")
    model = payload["learner"]["model"]
    model_hash = hashlib.sha256()
    for name, value in model.items():
        model_hash.update(name.encode())
        model_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    registry_path = os.path.abspath(os.fspath(registry_path))
    if os.path.exists(registry_path):
        registry = json.load(open(registry_path))
    else:
        registry = {"schema": ARCHIVE_SCHEMA, "policies": {}, "evaluations": [], "latest": None, "incumbent": None}
    if registry.get("schema") != ARCHIVE_SCHEMA:
        raise ValueError("unsupported policy archive schema")
    old = registry["policies"].get(policy_id)
    if old and old["file_sha256"] != file_hash:
        raise ValueError("immutable archive entry cannot be replaced by a different checkpoint")
    entry = {
        "id": policy_id, "label": label, "path": checkpoint_path,
        "file_sha256": file_hash, "model_digest": model_hash.hexdigest(),
        "committed_matches": committed_matches,
        "source_sha": source_sha or payload.get("source_sha"),
        "observation_schema": "pa3-entity-observation-v1",
        "action_schema": "atomic-action-v1",
    }
    registry["policies"][policy_id] = entry
    registry["latest"] = policy_id
    if incumbent:
        registry["incumbent"] = policy_id
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".archive-", dir=os.path.dirname(registry_path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(registry, f, indent=2, sort_keys=True)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, registry_path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return entry


def verify_archive(registry_path):
    registry = json.load(open(registry_path))
    if registry.get("schema") != ARCHIVE_SCHEMA:
        raise ValueError("unsupported policy archive schema")
    for entry in registry["policies"].values():
        if _sha256(entry["path"]) != entry["file_sha256"]:
            raise ValueError(f"archived checkpoint digest mismatch: {entry['id']}")
    return registry
