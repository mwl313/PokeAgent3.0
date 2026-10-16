#!/usr/bin/env python3
"""V2 evaluation-side checkpoint watcher and registrar.

Reads the production rolling checkpoint store read-only, verifies fully
published two-rank generations, derives evaluation-owned model-only snapshots
and refreshes the milestone schedule.  Never writes into production and never
inspects partially written rank files (discovery relies on
``CheckpointGenerationStore.generations_ready``).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "engine" / "python")]

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from agent.evaluation.v2_league import (  # noqa: E402
    DEFAULT_CAMPAIGN_START,
    FULL_INTERVAL,
    LIGHT_INTERVAL,
    atomic_json,
    capture_frozen_snapshot,
    disk_headroom_bytes,
    initial_incumbent,
    read_json,
    register_snapshot,
    snapshot_entries,
    update_schedule,
)
from agent.ppo.storage_manager import CheckpointGenerationStore  # noqa: E402

from agent.evaluation.identity import sha256_file  # noqa: E402


def current_committed_matches(reports_dir, fallback):
    rank0 = Path(reports_dir) / "continuation.json.rank0.ppo.jsonl"
    latest = fallback
    if rank0.is_file():
        try:
            with rank0.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 65536))
                tail = handle.read().decode("utf-8", errors="ignore").strip().splitlines()
            for line in reversed(tail):
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                value = payload.get("identity", {}).get("committed_matches")
                if isinstance(value, int):
                    latest = max(latest, value)
                    break
        except OSError:
            pass
    return int(latest)


def ingest_generation(*, generation, registry_path, snapshots_root, label_prefix, incumbent=False,
                      latest=False, extra=None):
    snapshot, snapshot_path, clock = capture_frozen_snapshot(generation, snapshots_root)
    source = snapshot["source_sha"]
    policy_id = f"v2-{clock:012d}-{source[:8]}"
    if extra and extra.get("bootstrap"):
        policy_id = f"v2-{clock:012d}-bootstrap"
    entry = register_snapshot(
        registry_path,
        policy_id=policy_id,
        label=f"{label_prefix} {clock}",
        snapshot_path=snapshot_path,
        snapshot=snapshot,
        incumbent=incumbent,
        latest=latest,
        extra=extra,
    )
    return entry


def discover(args):
    eval_root = Path(args.eval_root)
    snapshots_root = eval_root / "snapshots"
    registry_path = eval_root / "registry.json"
    receipts_dir = eval_root / "receipts"
    schedule_path = eval_root / "schedule.json"
    eval_root.mkdir(parents=True, exist_ok=True)
    (eval_root / "locks").mkdir(exist_ok=True)

    lock_path = eval_root / "locks" / "discovery.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "already_running"}
        if disk_headroom_bytes(eval_root) < int(args.reserve_gib * (1 << 30)):
            return {"status": "resource_deferred", "reason": f"{args.reserve_gib:g}GiB evaluation reserve"}

        registry = read_json(registry_path) or {"schema": "pa3-policy-archive-v2", "policies": {},
                                                "evaluations": [], "latest": None, "incumbent": None}
        processed = {entry.get("checkpoint_generation") for entry in registry["policies"].values()
                     if entry.get("checkpoint_generation")}
        registered, rejected = [], {}

        if args.bootstrap_generation:
            bootstrap = Path(args.bootstrap_generation)
            if bootstrap.is_dir() and bootstrap.name not in processed:
                try:
                    entry = ingest_generation(
                        generation=bootstrap, registry_path=registry_path,
                        snapshots_root=snapshots_root, label_prefix="V2 bootstrap",
                        incumbent=registry.get("incumbent") is None,
                        latest=registry.get("latest") is None,
                        extra={"bootstrap": True, "bootstrap_source": str(bootstrap)},
                    )
                    registered.append(entry["id"])
                except (ValueError, OSError, KeyError, RuntimeError) as error:
                    rejected[bootstrap.name] = str(error)
            registry = read_json(registry_path)

        if args.rolling:
            store = CheckpointGenerationStore(args.rolling, keep_generations=3)
            for generation in store.generations_ready():
                if generation.name in processed:
                    continue
                try:
                    entry = ingest_generation(
                        generation=generation, registry_path=registry_path,
                        snapshots_root=snapshots_root, label_prefix="V2",
                        latest=True,
                    )
                    registered.append(entry["id"])
                except (ValueError, OSError, KeyError, RuntimeError) as error:
                    rejected[generation.name] = str(error)

        registry, entries = snapshot_entries(registry_path) if registry_path.exists() else (registry, [])
        fallback = max([entry["committed_matches"] for entry in entries], default=DEFAULT_CAMPAIGN_START)
        committed = current_committed_matches(args.reports_dir, fallback) if args.reports_dir else fallback
        schedule = update_schedule(
            schedule_path,
            [{"id": entry["id"], "committed_matches": entry["committed_matches"]} for entry in entries],
            committed,
            campaign_start=args.campaign_start,
        )
        if registry.get("incumbent") is None and entries:
            bootstrap_entry = min(entries, key=lambda entry: entry["committed_matches"])
            atomic_json(eval_root / "best_policy.json",
                        initial_incumbent(bootstrap_entry["id"], bootstrap_entry))
        state = {
            "status": "ok",
            "committed_matches": committed,
            "policies": len(entries),
            "registered": registered,
            "rejected": rejected,
            "latest_generation": max((entry.get("checkpoint_generation") or "" for entry in entries), default=None),
            "pending_jobs": sum(1 for job in schedule["jobs"].values() if job["status"] == "pending"),
            "updated_at": time.time(),
        }
        atomic_json(eval_root / "state" / "discovery.json", state)
        return state


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling", help="production rolling checkpoint root (read-only)")
    parser.add_argument("--reports-dir", help="production reports dir for the live committed-match counter")
    parser.add_argument("--eval-root", default="/home/minidc/eval-league/v2")
    parser.add_argument("--bootstrap-generation", help="explicit recovered baseline generation directory")
    parser.add_argument("--campaign-start", type=int, default=DEFAULT_CAMPAIGN_START)
    parser.add_argument("--reserve-gib", type=float, default=12.0)
    args = parser.parse_args(argv)
    print(json.dumps(discover(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
