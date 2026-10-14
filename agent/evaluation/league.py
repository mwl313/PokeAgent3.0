"""Small durable frozen-policy evaluation league.

It is deliberately outside PPO: results warn about regressions but never
change learner state, block a checkpoint, or promote/roll back a policy.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import contextlib
from datetime import datetime, timezone

from agent.evaluation.fixed_policy import FixedPolicyEvaluator, fixture_manifest_hash, load_frozen_model
from agent.evaluation.policy_archive import verify_entry
from agent.evaluation.identity import sha256_file

EVALUATOR_VERSION = "pa3-fixed-policy-league-v2"
ENGINE_ID = "pa3-native-fixed-policy-v2"


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
    if not os.path.exists(path): return {"schema": "pa3-evaluation-league-v2", "receipts": {}}
    with open(path) as handle: store = json.load(handle)
    if store.get("schema") == "pa3-evaluation-league-v1":
        # Old receipts remain inspectable but cannot satisfy the v2 cache key.
        store["schema"] = "pa3-evaluation-league-v2"; store.setdefault("legacy_receipts", store.pop("receipts", {})); store["receipts"] = {}
    if store.get("schema") != "pa3-evaluation-league-v2": raise ValueError("unsupported league receipt schema")
    return store


def environment_identity(*, engine_path=None, data_path=None, teams_path=None):
    """Content identities required for v2 cache reuse; unknown inputs fail closed."""
    if not engine_path or not data_path or not teams_path:
        raise ValueError("v2 evaluation cache requires engine, data, and team artifact paths")
    if not os.path.isfile(engine_path) or not os.path.isfile(teams_path) or not os.path.isdir(data_path):
        raise ValueError("v2 evaluation identity inputs are missing")
    digest = hashlib.sha256()
    for root, _dirs, files in os.walk(data_path):
        for name in sorted(files):
            path = os.path.join(root, name); digest.update(os.path.relpath(path, data_path).encode()); digest.update(bytes.fromhex(sha256_file(path)))
    return {"engine_sha256": sha256_file(engine_path), "team_dataset_sha256": sha256_file(teams_path), "mechanics_data_sha256": digest.hexdigest()}


def evaluation_identity(left, right, fixtures, *, seed, max_rounds, environment):
    from agent.contracts import CONTRACT_HASH
    from agent.evaluation.batched_policy import BATCHED_EVALUATOR_VERSION, SAMPLING_SEMANTICS
    def contract(entry):
        return {"precision_mode": entry.get("precision_mode", "A"),
                "precision_stability": entry.get("precision_stability", "default"),
                "precision_contract_version": entry.get("precision_contract_version", "legacy"),
                "observation_quantization": "fp16-round-continuous-v1" if entry.get("precision_mode", "A") == "D" else "none"}
    body = {"identity_version": "pa3-evaluation-identity-v4", "semantic_contract": CONTRACT_HASH,
            "candidate": [left["id"], left["file_sha256"], left["model_digest"], contract(left)],
            "opponent": [right["id"], right["file_sha256"], right["model_digest"], contract(right)],
            "fixture_manifest_hash": fixture_manifest_hash(fixtures), "environment": environment,
            "engine": ENGINE_ID, "evaluator": BATCHED_EVALUATOR_VERSION if environment.get('backend')=='batched' else EVALUATOR_VERSION,
            "seed": int(seed), "max_rounds": int(max_rounds),
            "sampling": (SAMPLING_SEMANTICS if environment.get("backend") == "batched" else "sequential-reference-v1")}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), body


def evaluate_pair(engine, fixtures, left_entry, right_entry, *, device="cuda:0", seed=20261011, max_rounds=2000, environment, backend="sequential", batch_size=32):
    left = load_frozen_model(left_entry["path"], device=device)
    right = load_frozen_model(right_entry["path"], device=device)
    if backend == "batched":
        from agent.evaluation.batched_policy import BatchedPolicyEvaluator
        result = BatchedPolicyEvaluator(engine, device=device, seed=seed, batch_size=batch_size).run(
            fixtures, policy="checkpoint", model=left, opponent="checkpoint", opponent_model=right, max_rounds=max_rounds)
    elif backend == "sequential":
        result = FixedPolicyEvaluator(engine, device=device, seed=seed).run(
            fixtures, policy="checkpoint", model=left, opponent="checkpoint", opponent_model=right, max_rounds=max_rounds)
    else: raise ValueError("unknown evaluation backend")
    identity, identity_body = evaluation_identity(left_entry, right_entry, fixtures, seed=seed, max_rounds=max_rounds, environment=environment)
    status = "evaluation_invalid" if result["operational_errors"] or result["illegal_actions"] else "success"
    return {"identity": identity, **identity_body, "left_id": left_entry["id"], "right_id": right_entry["id"],
            "candidate_checkpoint_sha256": left_entry["file_sha256"], "opponent_checkpoint_sha256": right_entry["file_sha256"],
            "matches": result["matches"], "wins": result["wins"], "losses": result["losses"], "draws": result["draws"],
            "operational_errors": result["operational_errors"], "illegal_actions": result["illegal_actions"],
            "score_mean": result["score_mean"], "paired_score_mean": result["paired_score_mean"],
            "paired_fixtures": result["paired_fixtures"], "score_ci95": result["score_ci95"],
            "runtime_seconds": result["runtime_seconds"], "outcomes": result["outcomes"],
            "team_group_results": result["team_group_results"], "status": status,
            "timing_seconds": result.get("timing_seconds"), "gpu_peak_bytes": result.get("gpu_peak_bytes"),
            "evaluator_version": result.get("evaluator_version", EVALUATOR_VERSION), "sampling_semantics": result.get("sampling", "reference-v1"),
            "evaluated_at": datetime.now(timezone.utc).isoformat()}


@contextlib.contextmanager
def _receipt_lock(path):
    import fcntl
    lock_path = os.path.abspath(path) + ".lock"; os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_pair(registry_path, receipt_path, engine, fixtures, candidate_id, opponent_id, *, device="cuda:0", seed=20261011, max_rounds=2000, environment=None, backend="sequential", batch_size=32):
    """Verify, run, and atomically persist one matchup; same identity is cached."""
    registry = json.load(open(registry_path))
    candidate, opponent = registry["policies"][candidate_id], registry["policies"][opponent_id]
    candidate = dict(candidate, path=verify_entry(registry_path, candidate))
    opponent = dict(opponent, path=verify_entry(registry_path, opponent))
    if environment is None:
        # Mock-only callers cannot safely share a cache across processes.
        environment = {"mock_engine": type(engine).__name__, "cacheable": True}
    environment = dict(environment, backend=backend, batch_size=int(batch_size))
    identity, _ = evaluation_identity(candidate, opponent, fixtures, seed=seed, max_rounds=max_rounds, environment=environment)
    with _receipt_lock(receipt_path):
        store = _load_store(receipt_path)
        if environment.get("cacheable", True) and identity in store["receipts"]: return store["receipts"][identity], True
        receipt = evaluate_pair(engine, fixtures, candidate, opponent, device=device, seed=seed, max_rounds=max_rounds, environment=environment, backend=backend, batch_size=batch_size)
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
    """Compare models on shared fixtures; absolute weakness is never forgetting."""
    matrix = matchup_matrix(receipt_path)
    rows = matrix.get(candidate_id, {})
    if not rows: return {"candidate": candidate_id, "status": "insufficient_coverage", "matchups": []}
    store = _load_store(receipt_path)
    receipts = [x for x in store["receipts"].values() if x.get("status") == "success"]
    matchups, warnings = [], []
    for opponent, row in sorted(rows.items()):
        current = next(x for x in receipts if x["identity"] == row["identity"])
        comparable = [x for x in receipts if x["left_id"] != candidate_id and x["right_id"] == opponent
                      and x.get("fixture_manifest_hash") == current.get("fixture_manifest_hash")
                      and x.get("environment") == current.get("environment")]
        comparison = {"status": "insufficient_coverage"}
        if comparable:
            previous = comparable[-1]
            score = {"win": 1.0, "draw": .5, "loss": 0.0}
            old = {x["fixture_id"]: score[x["outcome"]] for x in previous.get("outcomes", []) if x["outcome"] in score}
            new = {x["fixture_id"]: score[x["outcome"]] for x in current.get("outcomes", []) if x["outcome"] in score}
            shared = sorted(set(old) & set(new))
            if shared:
                diffs = [new[i] - old[i] for i in shared]
                import random
                rng = random.Random(20261011); samples = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(1000))
                ci = [samples[24], samples[974]]
                comparison = {"reference_candidate": previous["left_id"], "paired_games": len(shared),
                              "score_difference": sum(diffs) / len(diffs), "difference_ci95": ci,
                              "status": "regression_warning" if ci[1] < 0 else ("improvement_evidence" if ci[0] > 0 else "inconclusive")}
        if comparison["status"] == "regression_warning": warnings.append(opponent)
        matchups.append({"opponent": opponent, **row, "absolute_status": "weak_matchup" if row["score_mean"] < .5 else "competitive", "comparison": comparison})
    incumbent = next((row for row in matchups if row["opponent"] == incumbent_id), None)
    return {"candidate": candidate_id, "incumbent": incumbent_id, "incumbent_comparison": incumbent,
            "worst_opponent": min(matchups, key=lambda x: x["score_mean"])["opponent"],
            "regression_warnings": warnings, "status": "regression_warning" if warnings else "inconclusive",
            "matchups": matchups, "non_blocking": True}
