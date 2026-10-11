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
from agent.evaluation.league import matchup_matrix, regression_report, run_pair
from agent.evaluation.policy_archive import verify_archive
from agent.evaluation.schedule import boundary_jobs, complete_job, pending_jobs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "schedule", "run-pair", "report"))
    parser.add_argument("--registry", default=os.path.join(ROOT, "docs/experiments/policy_archive.json"))
    parser.add_argument("--receipts", default=os.path.join(ROOT, "runs/experiments/a/evaluation_league/receipts.json"))
    parser.add_argument("--schedule-state", default=os.path.join(ROOT, "runs/experiments/a/evaluation_league/schedule.json"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--anchor", type=int, default=352256)
    parser.add_argument("--candidate"); parser.add_argument("--opponent")
    parser.add_argument("--pairs", type=int, default=128); parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--data", default=os.path.join(ROOT, "engine", "data"))
    parser.add_argument("--teams", default=os.path.join(ROOT, "engine", "data", "training-teams.json"))
    args = parser.parse_args(argv)
    if args.command == "verify":
        registry = verify_archive(args.registry); print(json.dumps({"verified": sorted(registry["policies"])})); return
    if args.command == "schedule":
        if not args.checkpoint: parser.error("schedule requires --checkpoint")
        payload = torch.load(os.path.abspath(args.checkpoint), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not payload.get("rank_digests") or not payload.get("collector_drain"):
            raise ValueError("requires a completed experimental boundary envelope")
        if not all(v.get("model") and v.get("optimizer") for v in payload["rank_digests"].values()):
            raise ValueError("checkpoint has incomplete rank digests")
        matches = payload["accounting"]["committed_matches"]
        checkpoint = {"path": os.path.abspath(args.checkpoint), "source_sha": payload["source_sha"],
                      "committed_matches": matches, "rank_digests": payload["rank_digests"]}
        print(json.dumps({"created": boundary_jobs(args.schedule_state, matches, anchor=args.anchor, checkpoint=checkpoint),
                          "pending": pending_jobs(args.schedule_state, anchor=args.anchor)}, indent=2)); return
    if args.command == "run-pair":
        if not args.candidate or not args.opponent: parser.error("run-pair needs --candidate and --opponent")
        engine = pa3_engine.NativeEngine(args.data, args.teams, workers=args.workers)
        fixtures = build_fixture_manifest(pairs=args.pairs, team_count=engine.team_count(), seed=args.seed)
        receipt, cached = run_pair(args.registry, args.receipts, engine, fixtures, args.candidate, args.opponent,
                                   device=args.device, seed=args.seed)
        print(json.dumps({"cached": cached, **{k: receipt[k] for k in ("identity", "matches", "wins", "losses", "draws", "operational_errors", "illegal_actions", "status")}}, indent=2)); return
    registry = json.load(open(args.registry)); report = {"matrix": matchup_matrix(args.receipts)}
    if args.candidate: report["regression"] = regression_report(args.receipts, candidate_id=args.candidate, incumbent_id=registry["incumbent"])
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__": main()
