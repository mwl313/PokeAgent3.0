"""Small durable frozen-policy evaluation league.

It is deliberately outside PPO: results warn about regressions but never
change learner state, block a checkpoint, or promote/roll back a policy.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone

from agent.evaluation.fixed_policy import FixedPolicyEvaluator, fixture_manifest_hash, load_frozen_model
from agent.evaluation.policy_archive import verify_entry

EVALUATOR_VERSION = "pa3-fixed-policy-league-v1"
ENGINE_ID = "pa3-native-fixed-policy-v1"


def _atomic_json(path, value):
    path = os.path.abspath(path); os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".league-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _load_store(path):
    if not os.path.exists(path): return {"schema": "pa3-evaluation-league-v1", "receipts": {}}
    with open(path) as handle: store = json.load(handle)
    if store.get("schema") != "pa3-evaluation-league-v1": raise ValueError("unsupported league receipt schema")
    return store


def _team_digest(fixtures):
    team_ids = sorted({x.team_a for x in fixtures} | {x.team_b for x in fixtures})
    return hashlib.sha256(json.dumps(team_ids, separators=(",", ":")).encode()).hexdigest()


def evaluation_identity(left, right, fixtures, *, seed, max_rounds, engine_identity=ENGINE_ID):
    body = {"candidate": [left["id"], left["file_sha256"], left["model_digest"]],
            "opponent": [right["id"], right["file_sha256"], right["model_digest"]],
            "fixture_manifest_hash": fixture_manifest_hash(fixtures), "team_pool_digest": _team_digest(fixtures),
            "engine": engine_identity, "evaluator": EVALUATOR_VERSION,
            "seed": int(seed), "max_rounds": int(max_rounds), "sampling": "frozen-torch-generator-per-fixture-v1"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), body


def evaluate_pair(engine, fixtures, left_entry, right_entry, *, device="cuda:0", seed=20261011, max_rounds=2000):
    left = load_frozen_model(left_entry["path"], device=device)
    right = load_frozen_model(right_entry["path"], device=device)
    result = FixedPolicyEvaluator(engine, device=device, seed=seed).run(
        fixtures, policy="checkpoint", model=left, opponent="checkpoint", opponent_model=right, max_rounds=max_rounds
    )
    identity, identity_body = evaluation_identity(left_entry, right_entry, fixtures, seed=seed, max_rounds=max_rounds)
    status = "evaluation_invalid" if result["operational_errors"] or result["illegal_actions"] else "success"
    return {"identity": identity, **identity_body, "left_id": left_entry["id"], "right_id": right_entry["id"],
            "candidate_checkpoint_sha256": left_entry["file_sha256"], "opponent_checkpoint_sha256": right_entry["file_sha256"],
            "matches": result["matches"], "wins": result["wins"], "losses": result["losses"], "draws": result["draws"],
            "operational_errors": result["operational_errors"], "illegal_actions": result["illegal_actions"],
            "score_mean": result["score_mean"], "paired_score_mean": result["paired_score_mean"],
            "paired_fixtures": result["paired_fixtures"], "score_ci95": result["score_ci95"],
            "runtime_seconds": result["runtime_seconds"], "outcomes": result["outcomes"],
            "team_group_results": result["team_group_results"], "status": status,
            "evaluated_at": datetime.now(timezone.utc).isoformat()}


def run_pair(registry_path, receipt_path, engine, fixtures, candidate_id, opponent_id, *, device="cuda:0", seed=20261011, max_rounds=2000):
    """Verify, run, and atomically persist one matchup; same identity is cached."""
    registry = json.load(open(registry_path))
    candidate, opponent = registry["policies"][candidate_id], registry["policies"][opponent_id]
    candidate = dict(candidate, path=verify_entry(registry_path, candidate))
    opponent = dict(opponent, path=verify_entry(registry_path, opponent))
    identity, _ = evaluation_identity(candidate, opponent, fixtures, seed=seed, max_rounds=max_rounds)
    store = _load_store(receipt_path)
    if identity in store["receipts"]: return store["receipts"][identity], True
    receipt = evaluate_pair(engine, fixtures, candidate, opponent, device=device, seed=seed, max_rounds=max_rounds)
    if receipt["identity"] != identity: raise RuntimeError("evaluation identity changed while running")
    store["receipts"][identity] = receipt
    _atomic_json(receipt_path, store)
    return receipt, False


def matchup_matrix(receipt_path):
    rows = _load_store(receipt_path)["receipts"].values()
    matrix = {}
    for item in rows:
        if item["status"] != "success": continue
        matrix.setdefault(item["left_id"], {})[item["right_id"]] = {
            "score_mean": item["score_mean"], "score_ci95": item["score_ci95"], "matches": item["matches"], "identity": item["identity"]}
    return matrix


def regression_report(receipt_path, *, candidate_id, incumbent_id):
    """Descriptive warning only; incomparable panels remain explicitly inconclusive."""
    matrix = matchup_matrix(receipt_path)
    rows = matrix.get(candidate_id, {})
    if not rows: return {"candidate": candidate_id, "status": "insufficient_coverage", "matchups": []}
    matchups, warnings = [], []
    for opponent, row in sorted(rows.items()):
        if row["score_ci95"][1] < 0.5: status = "regression_warning"; warnings.append(opponent)
        elif row["score_ci95"][0] > 0.5: status = "improvement_evidence"
        else: status = "inconclusive"
        matchups.append({"opponent": opponent, **row, "status": status})
    incumbent = next((row for row in matchups if row["opponent"] == incumbent_id), None)
    return {"candidate": candidate_id, "incumbent": incumbent_id, "incumbent_comparison": incumbent,
            "worst_opponent": min(matchups, key=lambda x: x["score_mean"])["opponent"],
            "regression_warnings": warnings, "status": "regression_warning" if warnings else "inconclusive",
            "matchups": matchups, "non_blocking": True}
