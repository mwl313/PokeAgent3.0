#!/usr/bin/env python3
"""External, non-blocking checkpoint-to-evaluation-league entrypoint."""
from __future__ import annotations
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "engine", "python")]

import pa3_engine
import torch
from agent.evaluation.fixed_policy import build_fixture_manifest
from agent.evaluation.league import environment_identity, matchup_matrix, regression_report, run_pair
from agent.evaluation.identity import sha256_file
from agent.evaluation.policy_archive import verify_archive, verify_entry
from agent.evaluation.schedule import boundary_jobs, bind_job, complete_job, fail_job, pending_jobs


def _safe_checkpoint(path):
    """Reject anything other than an observed drained rank-0 boundary artifact."""
    path = os.path.abspath(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != "ppo_experimental_iteration_boundary":
        raise ValueError("not an experimental iteration-boundary checkpoint")
    drain, ddp, accounting = payload.get("collector_drain"), payload.get("ddp"), payload.get("accounting")
    if not isinstance(drain, dict) or drain.get("complete") is not True or int(drain.get("in_flight_battles", -1)) != 0:
        raise ValueError("checkpoint was not observed fully drained")
    if not isinstance(ddp, dict) or ddp.get("rank") != 0 or int(ddp.get("world_size", 0)) < 1 or ddp.get("communicator") != "recreate":
        raise ValueError("checkpoint is not a rank-0 recreatable DDP boundary")
    if not isinstance(accounting, dict) or int(accounting.get("committed_matches", -1)) < 0:
        raise ValueError("checkpoint has invalid global committed-match accounting")
    digests = payload.get("rank_digests", {})
    if set(digests) != {str(i) for i in range(ddp["world_size"])} or len({(x.get("model"), x.get("optimizer")) for x in digests.values()}) != 1:
        raise ValueError("checkpoint has inconsistent DDP digests")
    peer = path.rsplit(".rank0", 1)[0] + ".rank1" if path.endswith(".rank0") else None
    if int(ddp["world_size"]) > 1:
        if not peer or not os.path.isfile(peer): raise ValueError("checkpoint companion rank is missing")
        other = torch.load(peer, map_location="cpu", weights_only=False)
        if other.get("rank_digests") != digests or other.get("source_sha") != payload.get("source_sha"):
            raise ValueError("checkpoint companion rank differs")
    return {"path": path, "file_sha256": sha256_file(path), "source_sha": payload.get("source_sha"),
            "committed_matches": int(accounting["committed_matches"]), "rank_digests": digests,
            "model_config_hash": payload.get("model_config_hash"), "action_schema_hash": payload.get("action_schema_hash")}


def _candidate_for(registry_path, checkpoint):
    registry = verify_archive(registry_path)
    matches = [entry for entry in registry["policies"].values() if entry["file_sha256"] == checkpoint["file_sha256"]]
    if len(matches) != 1: raise ValueError("safe checkpoint is not registered as exactly one policy")
    entry = matches[0]; verify_entry(registry_path, entry)
    return registry, entry


def _plan(job, registry, candidate, *, smoke_games=None):
    games = int(smoke_games) if smoke_games else None
    if games is not None and games <= 0 or games is not None and games % 2:
        raise ValueError("smoke game override must be a positive even count")
    anchors = [x for x in ("ppo2_249856", "a4_ppo2_lr05_301056", "a5_historical_352256") if x in registry["policies"] and x != candidate["id"]]
    incumbent = registry["incumbent"]
    if job["kind"] == "light":
        panels = [{"opponent_id": anchors[0], "games": games or 256}]
    else:
        other = [x for x in anchors if x != incumbent][:2]
        if len(other) < 2: raise ValueError("archive lacks two full-league historical anchors")
        panels = [{"opponent_id": incumbent, "games": games or 1024},
                  {"opponent_id": other[0], "games": games or 512}, {"opponent_id": other[1], "games": games or 512}]
    return {"candidate_id": candidate["id"], "candidate_model_digest": candidate["model_digest"], "panels": panels}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "schedule", "run-pair", "process-pending", "report"))
    parser.add_argument("--registry", default=os.path.join(ROOT, "docs/experiments/policy_archive.json"))
    parser.add_argument("--receipts", default=os.path.join(ROOT, "runs/experiments/a/evaluation_league/receipts.json"))
    parser.add_argument("--schedule-state", default=os.path.join(ROOT, "runs/experiments/a/evaluation_league/schedule.json"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--anchor", type=int, default=352256)
    parser.add_argument("--candidate"); parser.add_argument("--opponent")
    parser.add_argument("--pairs", type=int, default=128); parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke-games", type=int)
    parser.add_argument("--backend", choices=("sequential", "batched"), default="batched")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data", default=os.path.join(ROOT, "engine", "data"))
    parser.add_argument("--teams", default=os.path.join(ROOT, "engine", "data", "training-teams.json"))
    args = parser.parse_args(argv)
    if args.command == "verify":
        registry = verify_archive(args.registry); print(json.dumps({"verified": sorted(registry["policies"])})); return
    if args.command == "schedule":
        if not args.checkpoint: parser.error("schedule requires --checkpoint")
        checkpoint = _safe_checkpoint(args.checkpoint)
        _registry, _candidate = _candidate_for(args.registry, checkpoint)
        created = boundary_jobs(args.schedule_state, checkpoint["committed_matches"], anchor=args.anchor, checkpoint=checkpoint)
        print(json.dumps({"created": created,
                          "pending": pending_jobs(args.schedule_state, anchor=args.anchor)}, indent=2)); return
    if args.command == "run-pair":
        if not args.candidate or not args.opponent: parser.error("run-pair needs --candidate and --opponent")
        engine = pa3_engine.NativeEngine(args.data, args.teams, workers=args.workers)
        environment = environment_identity(engine_path=pa3_engine.__file__, data_path=args.data, teams_path=args.teams)
        fixtures = build_fixture_manifest(pairs=args.pairs, team_count=engine.team_count(), seed=args.seed)
        receipt, cached = run_pair(args.registry, args.receipts, engine, fixtures, args.candidate, args.opponent,
                                   device=args.device, seed=args.seed, environment=environment, backend=args.backend, batch_size=args.batch_size)
        print(json.dumps({"cached": cached, **{k: receipt[k] for k in ("identity", "matches", "wins", "losses", "draws", "operational_errors", "illegal_actions", "status")}}, indent=2)); return
    if args.command == "process-pending":
        registry = verify_archive(args.registry); results = []
        engine = pa3_engine.NativeEngine(args.data, args.teams, workers=args.workers)
        environment = environment_identity(engine_path=pa3_engine.__file__, data_path=args.data, teams_path=args.teams)
        for job in pending_jobs(args.schedule_state, anchor=args.anchor):
            if job.get("missing_snapshot"):
                results.append({"job": job["id"], "status": "deferred", "reason": job["reason"]}); continue
            try:
                checkpoint = _safe_checkpoint(job["captured_checkpoint"]["path"])
                if checkpoint["file_sha256"] != job["captured_checkpoint"]["file_sha256"]: raise ValueError("scheduled checkpoint changed")
                _registry, candidate = _candidate_for(args.registry, checkpoint)
                plan = _plan(job, registry, candidate, smoke_games=args.smoke_games)
                bind_job(args.schedule_state, job["id"], plan, anchor=args.anchor)
                matchups = []
                for panel in plan["panels"]:
                    fixtures = build_fixture_manifest(pairs=panel["games"] // 2, team_count=engine.team_count(), seed=args.seed)
                    result, _cached = run_pair(args.registry, args.receipts, engine, fixtures, candidate["id"], panel["opponent_id"], device=args.device, seed=args.seed, environment=environment, backend=args.backend, batch_size=args.batch_size)
                    matchups.append({"opponent_id": panel["opponent_id"], "matches": result["matches"], "identity": result["identity"], "fixture_manifest_hash": result["fixture_manifest_hash"], "status": result["status"]})
                receipt = {"status": "success", "job_id": job["id"], "scheduled_boundary": job["scheduled_boundary"], "checkpoint_sha256": checkpoint["file_sha256"], "candidate_id": candidate["id"], "candidate_model_digest": candidate["model_digest"], "matchups": matchups}
                complete_job(args.schedule_state, job["id"], receipt, anchor=args.anchor); results.append({"job": job["id"], "status": "completed", "matchups": matchups})
            except Exception as error:
                fail_job(args.schedule_state, job["id"], str(error), anchor=args.anchor); results.append({"job": job["id"], "status": "failed", "error": str(error)})
        print(json.dumps({"results": results}, indent=2)); return
    registry = json.load(open(args.registry)); report = {"matrix": matchup_matrix(args.receipts)}
    if args.candidate: report["regression"] = regression_report(args.receipts, candidate_id=args.candidate, incumbent_id=registry["incumbent"])
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__": main()
