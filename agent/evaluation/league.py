"""Evaluation-only league matrix; scores never gate or roll back training."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from agent.evaluation.fixed_policy import FixedPolicyEvaluator, load_frozen_model


def evaluate_pair(engine, fixtures, left_entry, right_entry, *, device="cuda:0", seed=20261011):
    left = load_frozen_model(left_entry["path"], device=device)
    right = load_frozen_model(right_entry["path"], device=device)
    result = FixedPolicyEvaluator(engine, device=device, seed=seed).run(
        fixtures, policy="checkpoint", model=left, opponent="checkpoint", opponent_model=right
    )
    return {
        "left_id": left_entry["id"], "right_id": right_entry["id"],
        "left_digest": left_entry["model_digest"], "right_digest": right_entry["model_digest"],
        "fixture_manifest_hash": result["fixture_manifest_hash"],
        "matches": result["matches"], "wins": result["wins"], "losses": result["losses"],
        "draws": result["draws"], "operational_errors": result["operational_errors"],
        "illegal_actions": result["illegal_actions"], "score_mean": result["score_mean"],
        "score_ci95": result["score_ci95"], "runtime_seconds": result["runtime_seconds"],
        "status": "evaluation_invalid" if result["operational_errors"] or result["illegal_actions"] else "descriptive_only",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def append_receipt(path, receipt):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(receipt, sort_keys=True) + "\n")
