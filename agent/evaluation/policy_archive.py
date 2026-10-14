"""Verified local policy archive used by the evaluation league.

``verification-only`` entries point to protected local artifacts and are
re-verified before every evaluation.  They are deliberately not called
immutable storage.  Content-addressed copies are optional for new artifacts.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import torch
from agent.evaluation.identity import canonical_model_digest, sha256_file, MODEL_DIGEST_ALGORITHM

from agent.contracts import OBSERVATION_SCHEMA, ACTION_SCHEMA, require_checkpoint_identity
ARCHIVE_SCHEMA = "pa3-policy-archive-v2"


def _sha256(path):
    return sha256_file(path)


def _model_digest(model):
    return canonical_model_digest(model)


def _resolve(registry_path, entry, key="path"):
    path = entry[key]
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(os.path.dirname(registry_path), path))


def _load_envelope(path):
    if "://" in path or not os.path.isfile(path):
        raise ValueError("evaluation checkpoints must be trusted local files")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("learner"), dict):
        raise ValueError("archive requires an experimental learner envelope")
    if not isinstance(payload["learner"].get("model"), dict):
        raise ValueError("archive envelope has no model state")
    require_checkpoint_identity(payload['learner'])
    if not isinstance(payload['learner'].get('config',{}).get('model'),dict):
        raise ValueError('archive requires V2 model architecture identity')
    from agent.ppo.experimental_checkpoint import hash_json
    if payload.get('model_config_hash') != hash_json(payload['learner']['config']['model']):
        raise ValueError('archive model configuration hash mismatch')
    return payload


def _atomic_json(path, value):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".archive-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _validate_rank_digests(payload, policy_id):
    digests = payload.get("rank_digests")
    world_size=payload.get('ddp',{}).get('world_size')
    if world_size != 2 or not isinstance(digests, dict) or set(digests)!={'0','1'}:
        raise ValueError(f"archived rank digests missing: {policy_id}")
    values = list(digests.values())
    if any(not row.get("model") or not row.get("optimizer") for row in values):
        raise ValueError(f"archived rank digests incomplete: {policy_id}")
    if len({row["model"] for row in values}) != 1 or len({row["optimizer"] for row in values}) != 1:
        raise ValueError(f"archived rank digests inconsistent: {policy_id}")


def register_checkpoint(registry_path, *, policy_id, label, checkpoint_path,
                        committed_matches=None, source_sha=None, incumbent=False,
                        latest=False, storage_mode="verification-only", archive_root=None,
                        rank_peer_path=None):
    """Add an identity-pinned artifact without changing active policy by default."""
    checkpoint_path = os.path.abspath(os.fspath(checkpoint_path))
    payload = _load_envelope(checkpoint_path)
    file_hash = _sha256(checkpoint_path)
    if storage_mode == "content-addressed":
        if not archive_root: raise ValueError("content-addressed registration requires archive_root")
        destination = os.path.join(os.path.abspath(archive_root), file_hash[:2], file_hash)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if not os.path.exists(destination):
            if shutil.disk_usage(os.path.abspath(archive_root)).free < os.path.getsize(checkpoint_path):
                raise ValueError("insufficient archive disk space")
            shutil.copy2(checkpoint_path, destination)
            os.chmod(destination, 0o444)
        if _sha256(destination) != file_hash: raise ValueError("content-addressed copy verification failed")
        checkpoint_path, storage_mode = destination, "content-addressed"
    elif storage_mode != "verification-only":
        raise ValueError("unsupported archive storage mode")
    registry_path = os.path.abspath(os.fspath(registry_path))
    registry = json.load(open(registry_path)) if os.path.exists(registry_path) else {
        "schema": ARCHIVE_SCHEMA, "policies": {}, "evaluations": [], "latest": None, "incumbent": None,
    }
    if registry.get("schema") != ARCHIVE_SCHEMA: raise ValueError("unsupported policy archive schema")
    old = registry["policies"].get(policy_id)
    if old and old["file_sha256"] != file_hash:
        raise ValueError("immutable archive entry cannot be replaced by a different checkpoint")
    stored_path = checkpoint_path
    if old and not os.path.isabs(old.get("path", "")) and storage_mode == "verification-only":
        stored_path = os.path.relpath(checkpoint_path, os.path.dirname(registry_path))
    entry = {
        "id": policy_id, "label": label, "path": stored_path, "storage_mode": storage_mode,
        "file_sha256": file_hash, "model_digest": _model_digest(payload["learner"]["model"]),
        "model_digest_algorithm": MODEL_DIGEST_ALGORITHM,
        "committed_matches": committed_matches if committed_matches is not None else payload.get("accounting", {}).get("committed_matches"),
        "source_sha": source_sha or payload.get("source_sha"), "model_config_hash": payload.get("model_config_hash"),
        "action_schema_hash": payload.get("action_schema_hash"),
        "observation_schema": OBSERVATION_SCHEMA, "action_schema": ACTION_SCHEMA,
        "precision_mode": payload.get("experiment", {}).get("precision_mode", "A"),
        "precision_stability": payload.get("experiment", {}).get("precision_stability", "default"),
        "precision_contract_version": payload.get("experiment", {}).get("precision_contract_version", "legacy"),
    }
    if rank_peer_path:
        peer = os.path.abspath(os.fspath(rank_peer_path))
        entry["rank_peer_path"] = os.path.relpath(peer, os.path.dirname(registry_path)) if old and not os.path.isabs(old.get("rank_peer_path", "")) else peer
    registry["policies"][policy_id] = entry
    if latest: registry["latest"] = policy_id
    if incumbent: registry["incumbent"] = policy_id
    _atomic_json(registry_path, registry)
    return entry


def verify_entry(registry_path, entry):
    """Verify identity, architecture contract, provenance, and optional DDP peer."""
    path = _resolve(registry_path, entry)
    if not os.path.isfile(path): raise ValueError(f"archived checkpoint missing: {entry['id']}")
    if _sha256(path) != entry["file_sha256"]: raise ValueError(f"archived checkpoint SHA mismatch: {entry['id']}")
    payload = _load_envelope(path)
    _validate_rank_digests(payload, entry["id"])
    if entry.get("model_digest_algorithm", MODEL_DIGEST_ALGORITHM) != MODEL_DIGEST_ALGORITHM:
        raise ValueError(f"unsupported archived model digest algorithm: {entry['id']}")
    if _model_digest(payload["learner"]["model"]) != entry["model_digest"]:
        raise ValueError(f"archived model digest mismatch: {entry['id']}")
    for key in ("source_sha", "model_config_hash", "action_schema_hash"):
        if not entry.get(key) or payload.get(key) != entry[key]:
            raise ValueError(f"archived {key} mismatch: {entry['id']}")
    experiment = payload.get("experiment", {})
    if entry.get("precision_mode", "A") != experiment.get("precision_mode", "A"):
        raise ValueError(f"archived precision-mode mismatch: {entry['id']}")
    if entry.get("precision_stability", "default") != experiment.get("precision_stability", "default"):
        raise ValueError(f"archived precision-stability mismatch: {entry['id']}")
    if payload.get("accounting", {}).get("committed_matches") != entry.get("committed_matches"):
        raise ValueError(f"archived committed-match mismatch: {entry['id']}")
    if entry.get("observation_schema") != OBSERVATION_SCHEMA or entry.get("action_schema") != ACTION_SCHEMA:
        raise ValueError(f"archived schema mismatch: {entry['id']}")
    if entry.get("rank_peer_path"):
        peer_payload = _load_envelope(_resolve(registry_path, entry, "rank_peer_path"))
        if peer_payload.get("rank_digests") != payload.get("rank_digests") or peer_payload.get("source_sha") != payload.get("source_sha"):
            raise ValueError(f"archived DDP rank mismatch: {entry['id']}")
    return path


def verify_archive(registry_path):
    registry = json.load(open(registry_path))
    if registry.get("schema") != ARCHIVE_SCHEMA: raise ValueError("unsupported policy archive schema")
    for entry in registry["policies"].values(): verify_entry(registry_path, entry)
    return registry
