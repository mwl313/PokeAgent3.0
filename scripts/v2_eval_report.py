#!/usr/bin/env python3
"""Read-only V2 evaluation league reporting and control."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from agent.evaluation.v2_league import (  # noqa: E402
    internal_ratings,
    read_json,
    snapshot_entries,
    heldout_status,
)

EVAL_ROOT = Path("/home/minidc/eval-league/v2")


def _counts(schedule):
    counts = {}
    for job in schedule.get("jobs", {}).values():
        counts[job["status"]] = counts.get(job["status"], 0) + 1
    return counts


def _job_receipts(eval_root):
    rows = []
    for path in sorted((eval_root / "receipts" / "jobs").glob("*.json")):
        payload = read_json(path)
        if payload:
            rows.append(payload)
    return sorted(rows, key=lambda row: row.get("candidate_clock", 0))


def _raw_receipts(eval_root):
    rows = []
    receipts_dir = eval_root / "receipts"
    for path in sorted(receipts_dir.glob("*.json")):
        payload = read_json(path)
        if payload and "left_id" in payload:
            rows.append(payload)
    return rows


def status(args):
    eval_root = Path(args.eval_root)
    schedule = read_json(eval_root / "schedule.json", {}) or {"jobs": {}}
    registry_path = eval_root / "registry.json"
    entries = []
    if registry_path.exists():
        _registry, entries = snapshot_entries(registry_path)
    boundaries = list((schedule.get("boundaries") or {}).values())
    return {
        "eval_root": str(eval_root),
        "discovery": read_json(eval_root / "state" / "discovery.json"),
        "worker": read_json(eval_root / "state" / "worker_state.json"),
        "active": read_json(eval_root / "state" / "active.json"),
        "resources": read_json(eval_root / "state" / "resource_state.json"),
        "paused": (eval_root / "PAUSE").exists(),
        "best_policy": read_json(eval_root / "best_policy.json"),
        "latest_policy": entries[-1]["id"] if entries else None,
        "training_policy_note": "training policy is owned by PPO and never changes from evaluation",
        "policies": [{"id": entry["id"], "clock": entry["committed_matches"], "precision_mode": entry["precision_mode"]}
                     for entry in entries],
        "schedule": {
            "campaign_start": schedule.get("campaign_start"),
            "light_interval": schedule.get("light_interval"),
            "full_interval": schedule.get("full_interval"),
            "boundary_count": len(boundaries),
            "boundaries": sorted(boundaries, key=lambda row: row["boundary"]),
        },
        "job_counts": _counts(schedule),
        "heldout": read_json(eval_root / "heldout" / "heldout_status.json"),
    }


def jobs(args):
    schedule = read_json(Path(args.eval_root) / "schedule.json", {}) or {"jobs": {}}
    rows = sorted(schedule.get("jobs", {}).values(), key=lambda job: job.get("candidate_clock", 0))
    return {"jobs": rows}


def receipts(args):
    rows = []
    for row in _job_receipts(Path(args.eval_root)):
        rows.append({
            "job_id": row.get("job_id"),
            "tier": row.get("tier"),
            "candidate_id": row.get("candidate_id"),
            "candidate_clock": row.get("candidate_clock"),
            "incumbent_id": row.get("incumbent_id"),
            "games": row.get("games"),
            "status": row.get("status"),
            "decision": row.get("decision", {}).get("decision"),
            "promoted": row.get("promoted"),
            "matchups": [{k: m[k] for k in ("opponent_id", "games", "wins", "losses", "draws",
                                            "score_mean", "score_ci95", "status")} for m in row.get("matchups", [])],
        })
    return {"completed": rows}


def best(args):
    return read_json(Path(args.eval_root) / "best_policy.json", {})


def trend(args):
    eval_root = Path(args.eval_root)
    rows = []
    for row in _job_receipts(eval_root):
        rows.append({
            "candidate_id": row.get("candidate_id"),
            "committed_matches": row.get("candidate_clock"),
            "tier": row.get("tier"),
            "decision": row.get("decision", {}).get("decision"),
            "promoted": row.get("promoted"),
            "vs_baseline": next(({"score": m["score_mean"], "ci95": m["score_ci95"], "games": m["matches"]}
                                 for m in row.get("matchups", []) if m["opponent_id"] == row.get("baseline_id")), None),
            "vs_incumbent": next(({"score": m["score_mean"], "ci95": m["score_ci95"], "games": m["matches"],
                                   "opponent": m["opponent_id"]}
                                  for m in row.get("matchups", []) if m["opponent_id"] == row.get("incumbent_id")), None),
        })
    return {"trend": rows, "internal_ratings": internal_ratings(_raw_receipts(eval_root))}


def heldout(args):
    eval_root = Path(args.eval_root)
    return heldout_status(eval_root / "heldout", training_teams_path=str(ROOT / "engine" / "data" / "a10-training-teams.json"),
                          heldout_path=args.heldout_path)


def pause(args):
    (Path(args.eval_root) / "PAUSE").write_text("paused\n")
    return {"status": "paused", "path": str(Path(args.eval_root) / "PAUSE")}


def resume(args):
    path = Path(args.eval_root) / "PAUSE"
    if path.exists():
        path.unlink()
    return {"status": "resumed"}


def write_report(args, view):
    eval_root = Path(args.eval_root)
    reports = eval_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    from agent.evaluation.v2_league import atomic_json
    atomic_json(reports / "league_report.json", view)
    return {"written": str(reports / "league_report.json")}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "jobs", "receipts", "best", "trend", "heldout", "pause", "resume", "report"))
    parser.add_argument("--eval-root", default=str(EVAL_ROOT))
    parser.add_argument("--heldout-path")
    args = parser.parse_args(argv)
    if args.command == "status":
        view = status(args)
    elif args.command == "jobs":
        view = jobs(args)
    elif args.command == "receipts":
        view = receipts(args)
    elif args.command == "best":
        view = best(args)
    elif args.command == "trend":
        view = trend(args)
    elif args.command == "heldout":
        view = heldout(args)
    elif args.command == "pause":
        view = pause(args)
    elif args.command == "resume":
        view = resume(args)
    else:
        combined = {"status": status(args), "trend": trend(args), "receipts": receipts(args)}
        view = {**combined, "written": write_report(args, combined)}
    print(json.dumps(view, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
