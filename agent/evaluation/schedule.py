"""Durable non-blocking evaluation scheduling keyed by committed matches."""
from __future__ import annotations

import json
import os
import tempfile

SCHEDULE_SCHEMA = "pa3-evaluation-schedule-v2"


def _load(path, *, anchor, light_interval, full_interval):
    if not os.path.exists(path):
        return {"schema": SCHEDULE_SCHEMA, "anchor": int(anchor), "light_interval": int(light_interval), "full_interval": int(full_interval), "jobs": {}}
    with open(path) as handle:
        state = json.load(handle)
    if state.get("schema") != SCHEDULE_SCHEMA:
        raise ValueError("unsupported evaluation schedule schema")
    for key, value in (("anchor", anchor), ("light_interval", light_interval), ("full_interval", full_interval)):
        if int(state[key]) != int(value):
            raise ValueError(f"evaluation schedule configuration mismatch: {key}")
    return state


def _save(path, state):
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".schedule-", dir=os.path.dirname(target))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def boundary_jobs(state_path, committed_matches, *, anchor=352256, light_interval=50000, full_interval=200000, checkpoint=None):
    """Persist and return every boundary due through a safe checkpoint."""
    state = _load(state_path, anchor=anchor, light_interval=light_interval, full_interval=full_interval)
    now, created = int(committed_matches), []
    checkpoint_claimed = any(job.get("captured_checkpoint") == checkpoint for job in state["jobs"].values()) if checkpoint else False
    if now >= anchor + light_interval:
        for index in range(1, (now - anchor) // light_interval + 1):
            boundary = anchor + index * light_interval
            kind = "full" if (index * light_interval) % full_interval == 0 else "light"
            identity = f"{kind}:{boundary}"
            if identity not in state["jobs"]:
                exact_snapshot = checkpoint is not None and not checkpoint_claimed
                checkpoint_claimed = checkpoint_claimed or exact_snapshot
                state["jobs"][identity] = {"id": identity, "kind": kind, "scheduled_boundary": boundary,
                    "status": "pending" if exact_snapshot else "deferred", "captured_checkpoint": checkpoint if exact_snapshot else None,
                    "captured_matches": now if exact_snapshot else None,
                    "missing_snapshot": not exact_snapshot,
                    "reason": None if exact_snapshot else (
                        "catch-up policy assigns one safe checkpoint to the earliest crossed boundary" if checkpoint else
                        "no safe boundary checkpoint was supplied")}
                created.append(dict(state["jobs"][identity]))
            elif checkpoint and not checkpoint_claimed:
                job = state["jobs"][identity]
                if job.get("missing_snapshot") and job.get("status") in {"deferred", "failed"}:
                    job.update({"status": "pending", "captured_checkpoint": checkpoint, "captured_matches": now,
                                "missing_snapshot": False, "reason": "delayed evaluation uses first safe checkpoint after scheduled boundary"})
                    checkpoint_claimed = True
                    created.append(dict(job))
    _save(state_path, state)
    return created


def pending_jobs(state_path, *, anchor=352256, light_interval=50000, full_interval=200000):
    state = _load(state_path, anchor=anchor, light_interval=light_interval, full_interval=full_interval)
    return [dict(job) for job in state["jobs"].values() if job["status"] in {"pending", "deferred", "failed"}]


def bind_job(state_path, job_id, expected, *, anchor=352256, light_interval=50000, full_interval=200000):
    """Pin candidate/panel metadata before any evaluator is launched."""
    state = _load(state_path, anchor=anchor, light_interval=light_interval, full_interval=full_interval)
    job = state["jobs"].get(job_id)
    if job is None: raise ValueError("unknown evaluation job")
    if job.get("missing_snapshot"): raise ValueError("cannot bind unavailable historical snapshot")
    if job.get("expected") and job["expected"] != expected:
        raise ValueError("job already has a conflicting evaluation plan")
    job["expected"] = expected
    _save(state_path, state)
    return dict(job)


def fail_job(state_path, job_id, reason, *, anchor=352256, light_interval=50000, full_interval=200000):
    state = _load(state_path, anchor=anchor, light_interval=light_interval, full_interval=full_interval)
    job = state["jobs"].get(job_id)
    if job is None: raise ValueError("unknown evaluation job")
    if job.get("status") != "completed":
        job["status"] = "failed"; job["reason"] = str(reason); _save(state_path, state)
    return dict(job)


def complete_job(state_path, job_id, receipt, *, anchor=352256, light_interval=50000, full_interval=200000):
    state = _load(state_path, anchor=anchor, light_interval=light_interval, full_interval=full_interval)
    job = state["jobs"].get(job_id)
    if job is None: raise ValueError("unknown evaluation job")
    if not isinstance(receipt, dict) or receipt.get("status") != "success":
        raise ValueError("completion requires a successful evaluation receipt")
    expected = job.get("expected")
    if job.get("missing_snapshot") or not job.get("captured_checkpoint"):
        raise ValueError("cannot complete an evaluation without its exact checkpoint snapshot")
    if receipt.get("job_id") != job_id or receipt.get("scheduled_boundary") != job["scheduled_boundary"]:
        raise ValueError("receipt does not belong to this scheduled job")
    checkpoint = job["captured_checkpoint"]
    if receipt.get("checkpoint_sha256") != checkpoint.get("file_sha256"):
        raise ValueError("receipt checkpoint identity mismatch")
    if expected and receipt.get("candidate_id") != expected.get("candidate_id"):
        raise ValueError("receipt candidate identity mismatch")
    if expected and receipt.get("candidate_model_digest") != expected.get("candidate_model_digest"):
        raise ValueError("receipt candidate model digest mismatch")
    expected_panels = (expected or {}).get("panels", [])
    observed = receipt.get("matchups", [])
    if len(observed) != len(expected_panels):
        raise ValueError("receipt does not contain every required matchup")
    for panel, result in zip(expected_panels, observed):
        if result.get("opponent_id") != panel["opponent_id"] or result.get("matches") != panel["games"]:
            raise ValueError("receipt matchup panel mismatch")
        if result.get("status") != "success" or not result.get("identity") or not result.get("fixture_manifest_hash"):
            raise ValueError("receipt contains an invalid matchup")
    if job["status"] == "completed":
        if job.get("receipt") != receipt: raise ValueError("conflicting duplicate evaluation completion")
        return dict(job)
    job["status"] = "completed"; job["receipt"] = receipt
    _save(state_path, state)
    return dict(job)


def defer_job(state_path, job_id, reason, *, anchor=352256, light_interval=50000, full_interval=200000):
    state = _load(state_path, anchor=anchor, light_interval=light_interval, full_interval=full_interval)
    job = state["jobs"].get(job_id)
    if job is None: raise ValueError("unknown evaluation job")
    if job["status"] != "completed":
        job["status"] = "deferred"; job["reason"] = str(reason); _save(state_path, state)
    return dict(job)


def due_evaluation(state_path, committed_matches, *, anchor=352256):
    created = boundary_jobs(state_path, committed_matches, anchor=anchor)
    jobs = created or pending_jobs(state_path, anchor=anchor)
    if not jobs: return {"kind": None, "key": None, "committed_matches": int(committed_matches), "pending": False}
    first = sorted(jobs, key=lambda row: row["scheduled_boundary"])[0]
    return {"kind": first["kind"], "key": first["id"], "committed_matches": int(committed_matches), "pending": True}


def mark_completed(state_path, key, receipt):
    return complete_job(state_path, key, receipt)
