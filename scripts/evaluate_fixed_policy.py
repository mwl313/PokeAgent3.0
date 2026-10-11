#!/usr/bin/env python3
"""Evaluate one frozen policy on the pinned paired fixture panel."""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine", "python"))

import pa3_engine
from agent.evaluation.fixed_policy import (
    FixedPolicyEvaluator, build_fixture_manifest, fixture_manifest_hash,
    load_frozen_model,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--policy", choices=["random", "lexicographic_legal", "checkpoint"], default="checkpoint")
    parser.add_argument("--opponent", choices=["random", "lexicographic_legal", "checkpoint"], default="lexicographic_legal")
    parser.add_argument("--opponent-checkpoint", default="")
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--data", default=os.path.join(ROOT, "engine", "data"))
    parser.add_argument("--teams", default=os.path.join(ROOT, "engine", "data", "training-teams.json"))
    args = parser.parse_args(argv)
    engine = pa3_engine.NativeEngine(args.data, args.teams, workers=args.workers)
    fixtures = build_fixture_manifest(pairs=args.pairs, team_count=engine.team_count(), seed=args.seed)
    manifest = {"schema": "a3-fixed-panel-v1", "seed": args.seed, "pairs": args.pairs,
                "team_pool_size": engine.team_count(), "fixture_manifest_hash": fixture_manifest_hash(fixtures),
                "fixtures": [{"fixture_id": f.fixture_id, "pair_id": f.pair_id,
                              "team_a": f.team_a, "team_b": f.team_b, "seed": list(f.seed),
                              "policy_side": f.policy_side} for f in fixtures]}
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    model = load_frozen_model(args.checkpoint, device=args.device) if args.policy == "checkpoint" else None
    opponent_model = load_frozen_model(args.opponent_checkpoint, device=args.device) if args.opponent == "checkpoint" else None
    report = FixedPolicyEvaluator(engine, device=args.device, seed=args.seed).run(
        fixtures, policy=args.policy, model=model, opponent=args.opponent,
        opponent_model=opponent_model,
    )
    report.update({"source_checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else None,
                   "manifest_path": os.path.abspath(args.manifest), "team_pool_size": engine.team_count()})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps({key: report[key] for key in
                      ("matches", "wins", "losses", "draws", "operational_errors", "illegal_actions",
                       "score_mean", "score_ci95", "runtime_seconds", "fixture_manifest_hash")}, indent=2))


if __name__ == "__main__":
    main()
