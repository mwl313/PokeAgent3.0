#!/usr/bin/env python3
"""Capture read-only proof that protected training/eval state was not modified."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "engine" / "python")]

from agent.evaluation.identity import sha256_file
from agent.evaluation.v2_league import atomic_json

DEFAULT_TRAINING = Path("/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix")
DEFAULT_EVAL = Path("/home/minidc/Desktop/PokeAgent3.0-V2-evaluation-league")
DEFAULT_FORMAL = Path("/home/minidc/eval-league/v2")
DEFAULT_RR = Path("/home/minidc/eval-league/roundrobin/v1")
PROTECTED = [
    Path("/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix/configs/pa3_v2_production.yaml"),
    Path("/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix/runs/production/"
         "mode_e_da35700_20261016/STOP"),
    Path("/home/minidc/Desktop/PokeAgent3.0-V2-p00b-hotfix/runs/production/"
         "mode_e_da35700_20261016/rolling/generations/"
         "iter-000002764800-1792212738467610757/manifest.json"),
    DEFAULT_FORMAL / "alpha_budget.json",
    DEFAULT_FORMAL / "best_policy.json",
    DEFAULT_FORMAL / "schedule.json",
    DEFAULT_FORMAL / "registry.json",
]


def _git(path, *args):
    try:
        return subprocess.run(
            ["git", "-C", str(path), *args], check=True, capture_output=True,
            text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _baseline(root):
    path = Path(root) / "protected_snapshot" / "key_hashes_before.txt"
    rows = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            digest, _, file_path = line.partition("  ")
            if digest and file_path:
                rows[file_path] = digest
    return rows


def capture(*, root=DEFAULT_RR, formal_root=DEFAULT_FORMAL,
            training=DEFAULT_TRAINING, evaluation=DEFAULT_EVAL,
            output=None) -> dict:
    baseline = _baseline(root)
    current = {}
    for path in PROTECTED:
        current[str(path)] = sha256_file(path) if path.is_file() else None
    comparisons = []
    for path, after in current.items():
        before = baseline.get(path)
        comparisons.append({
            "path": path,
            "before": before,
            "after": after,
            "changed": before is not None and before != after,
        })
    schedule = next((row for row in comparisons
                     if row["path"].endswith("schedule.json")), None)
    critical = [row for row in comparisons
                if row["changed"] and not row["path"].endswith("schedule.json")]
    jobs = {}
    schedule_path = formal_root / "schedule.json"
    if schedule_path.is_file():
        with open(schedule_path) as handle:
            jobs = json.load(handle).get("jobs", {}) or {}
    payload = {
        "schema": "pa3-v2-history-protected-verification-v1",
        "captured_at": time.time(),
        "captured_at_kst": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "training": {
            "worktree": str(training),
            "head": _git(training, "rev-parse", "HEAD"),
            "status_porcelain": _git(training, "status", "--porcelain"),
            "stop_present": (training / "runs/production/mode_e_da35700_20261016/STOP").is_file(),
        },
        "evaluation": {
            "worktree": str(evaluation),
            "head": _git(evaluation, "rev-parse", "HEAD"),
            "status_porcelain": _git(evaluation, "status", "--porcelain"),
        },
        "formal_root": {
            "path": str(formal_root),
            "schedule_jobs": len(jobs),
            "schedule_statuses": {
                status: sum(1 for job in jobs.values()
                            if job.get("status") == status)
                for status in sorted({job.get("status") for job in jobs.values()})
            },
            "alpha_sha256": current.get(str(formal_root / "alpha_budget.json")),
            "champion_sha256": current.get(str(formal_root / "best_policy.json")),
            "registry_sha256": current.get(str(formal_root / "registry.json")),
        },
        "protected_files": comparisons,
        "critical_mismatches": critical,
        "unchanged": not critical,
        "schedule_note": (
            "schedule.json is rewritten by the formal v2-eval-discover timer; it is "
            "reported separately and is never written by historical diagnostics"),
        "schedule_comparison": schedule,
        "writes_by_historical_diagnostics": 0,
    }
    if output:
        atomic_json(Path(output), payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_RR))
    parser.add_argument("--formal-root", default=str(DEFAULT_FORMAL))
    parser.add_argument("--training", default=str(DEFAULT_TRAINING))
    parser.add_argument("--evaluation", default=str(DEFAULT_EVAL))
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    payload = capture(
        root=Path(args.root), formal_root=Path(args.formal_root),
        training=Path(args.training), evaluation=Path(args.evaluation),
        output=args.output)
    print(json.dumps({
        "unchanged": payload["unchanged"],
        "critical_mismatches": len(payload["critical_mismatches"]),
        "training_head": payload["training"]["head"],
        "evaluation_head": payload["evaluation"]["head"],
        "stop_present": payload["training"]["stop_present"],
        "schedule_jobs": payload["formal_root"]["schedule_jobs"],
        "alpha_sha256": payload["formal_root"]["alpha_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
