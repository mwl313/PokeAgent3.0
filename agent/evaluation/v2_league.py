"""PokeAgent 3.0 V2 evaluation league core.

This module adapts the A-series evaluation league to the running V2 Mode E
campaign.  It owns only evaluation-side state: frozen inference snapshots,
an incumbent registry, milestone scheduling, paired receipts and reporting.
It never loads optimizer state into a learner, never writes a PPO counter and
never mutates training state.

Design constraints honoured here:

* V2-only.  Every snapshot must satisfy the V2 observation/action/precision
  contract identity; V1 artefacts fail closed.
* Immutable frozen policy artefacts derived from fully verified two-rank
  checkpoint generations.  Only model state is retained, not optimizer state.
* Absolute global committed-match milestones (light every 50k, full every
  200k) recorded separately from the actual checkpoint clock.
* Conservative, predeclared promotion evidence; inconclusive evidence retains
  the incumbent.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import torch

from agent.contracts import OBSERVATION_SCHEMA, ACTION_SCHEMA, require_checkpoint_identity
from agent.evaluation.identity import (
    MODEL_DIGEST_ALGORITHM,
    canonical_model_digest,
    sha256_file,
)
from agent.evaluation.policy_archive import (
    ARCHIVE_SCHEMA,
    register_checkpoint,
    verify_archive,
    verify_entry,
)
from agent.ppo.experimental_checkpoint import hash_json

LEAGUE_VERSION = "pa3-v2-evaluation-league-v1"
LEAGUE_VERSION_V2 = "pa3-v2-evaluation-league-v2"
SNAPSHOT_KIND = "v2_frozen_evaluation_snapshot"
SCHEDULE_SCHEMA = "pa3-v2-eval-schedule-v1"
JOB_LEDGER_SCHEMA = "pa3-v2-eval-jobs-v1"
JOB_RECEIPT_SCHEMA_V2 = "pa3-v2-eval-job-receipt-v2"
INCUMBENT_SCHEMA = "pa3-v2-incumbent-registry-v1"
RULESET_VERSION = "pa3-v2-promotion-rules-v1"
RULESET_VERSION_V2 = "pa3-v2-promotion-rules-v2"
EVALUATOR_VERSION = "pa3-native-batched-evaluator-v3"

# Machine-readable promotion outcomes.  ``promotion_status`` is stored on every
# v2 job receipt and is additive: historical receipts without it remain valid.
PROMOTED = "PROMOTED"
RETAINED_CONFIRMED_INFERIOR = "RETAINED_CONFIRMED_INFERIOR"
INCONCLUSIVE_BUDGET_EXHAUSTED = "INCONCLUSIVE_BUDGET_EXHAUSTED"
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
INVALID = "INVALID"

# ---------------------------------------------------------------------------
# Evaluation League V2 statistics (see ``alpha_ledger`` + ``confirmation_rule``)
# ---------------------------------------------------------------------------
#
# The campaign-wide false-promotion budget is a family-wise error rate of
# ALPHA_TOTAL.  Because v1 promoted on two (primary + confirmatory) tests, its
# historical spend is the union bound over every formal test already performed.
# That spend is preserved verbatim and is never reset by a version change.
#
# Promotion is now decided by a *single* formal confirmation procedure on
# fixtures that are independent of the primary/screening data.  Primary
# evidence only screens candidates and plans sample sizes; because the
# confirmation sample size depends on primary (never confirmation) outcomes,
# the confirmation test remains a valid fixed-sample test by independence.
# Each promotion-eligible comparison (one full 200k milestone) spends one
# predeclared alpha from the schedule below; the union bound over comparisons
# therefore bounds the campaign-wide false-promotion rate by ALPHA_TOTAL.
ALPHA_TOTAL = 0.05
ALPHA_LEDGER_SCHEMA = "pa3-v2-alpha-budget-v1"
ALPHA_SCHEDULE = "uniform-over-promotion-eligible-comparisons-v1"
# 100M matches / 200k full milestone = 500 promotion-eligible comparisons.
ALPHA_MAX_PROMOTION_COMPARISONS = 500
# Full comparisons already resolved under pa3-v2-promotion-rules-v1 (200704
# promote, 401408 retain, 602112 promote, 802816 retain).
ALPHA_HISTORICAL_COMPARISONS = 4
# Sum of alpha over every formal v1 test (primary + confirmatory gates).
ALPHA_HISTORICAL_CONSUMED = 0.0014880952380952382

# Adaptive budgets (games).  The 4096 limits are per matchup, not summed.
LIGHT_GAMES_V2 = 256
FULL_INITIAL_GAMES = 2048
PRIMARY_MAX_GAMES = 4096
CONFIRM_MAX_GAMES = 4096
# Cumulative confirmation ladders, in multiples of complete seat-swapped pairs.
CONFIRM_STAGE_GAMES = (1024, 2048, 4096)
# Power-family (t**1.5) alpha-spending increments at information fractions
# t = 0.25, 0.5, 1.0; increments sum to 1.0 so a comparison never overspends
# its allocation.  Later (larger) looks receive the majority of the budget.
CONFIRM_STAGE_ALPHA_WEIGHTS = (0.125, 0.22855339059327373, 0.6464466094067263)
# Screening trigger: a candidate whose primary paired score reaches this is a
# potential escalation candidate.  It is a resource-allocation heuristic, not a
# significance threshold and never itself authorises promotion.
SCREEN_TRIGGER = 0.53
SCREEN_AMPLIFY_FLOOR = 0.5
# Primary fixture layout (pairs).  Baseline anchor regions are kept identical to
# the historical v1 layout so regression comparisons retain shared fixtures.
PRIMARY_PAIRS = 4096
INCUMBENT_START_PAIR = 0
INCUMBENT_INITIAL_PAIRS = FULL_INITIAL_GAMES // 2
INCUMBENT_AMP_START_PAIR = INCUMBENT_INITIAL_PAIRS
INCUMBENT_AMP_PAIRS = (PRIMARY_MAX_GAMES - FULL_INITIAL_GAMES) // 2
BASELINE_START_PAIR = 512
BASELINE_GAMES = 512
ANCHOR_START_PAIR = 768
ANCHOR_GAMES = 512
# Confirmation fixture universe.
CONFIRM_PAIRS = CONFIRM_MAX_GAMES // 2
CONFIRMATORY_SEED_BASE = 20261015

# The frozen training pool shared by the whole V2 campaign.  A snapshot from a
# different pool is refused rather than silently evaluated out of domain.
V2_DATASET_MANIFEST_HASH = "1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb"

DEFAULT_CAMPAIGN_START = 57344
LIGHT_INTERVAL = 50000
FULL_INTERVAL = 200000

# Predeclared master fixture panel.  Every candidate is scored on slices of the
# same panel so that checkpoint-to-checkpoint comparisons share fixture
# identity.  A full evaluation uses 2048 games across available opponents.
MASTER_PAIRS = 2048
MASTER_SEED = 20261014
CONFIRMATORY_SEED = 20261015
LIGHT_GAMES = 256
FULL_GAMES = 2048
CONFIRMATORY_GAMES = 1024
# Minimum paired head-to-head fixtures before promotion may be considered.
PROMOTION_MIN_PAIRS = 128


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_rank_payload(path):
    if "://" in str(path) or not os.path.isfile(path):
        raise ValueError("evaluation checkpoints must be trusted local files")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("learner"), dict):
        raise ValueError("checkpoint is not an experimental learner envelope")
    if not isinstance(payload["learner"].get("model"), dict):
        raise ValueError("checkpoint has no model state")
    return payload


def _validate_generation_manifest(manifest, *, dataset_hash):
    if manifest.get("complete") is not True or manifest.get("iteration_complete") is not True:
        raise ValueError("generation is not a completed iteration")
    if int(manifest.get("world_size", 0)) != 2:
        raise ValueError("generation is not a two-rank DDP boundary")
    if manifest.get("dataset_manifest_hash") != dataset_hash:
        raise ValueError("generation was trained on a different frozen team pool")
    source = manifest.get("source_sha")
    if not isinstance(source, str) or len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("generation has an invalid training source identity")
    ranks = manifest.get("ranks")
    if not isinstance(ranks, dict) or set(ranks) != {"0", "1"}:
        raise ValueError("generation does not contain both rank artifacts")
    return manifest


def _verify_rank_checksum(generation, manifest, rank):
    row = manifest["ranks"][str(rank)]
    path = Path(generation) / row["path"]
    if not path.is_file():
        raise ValueError(f"rank {rank} artifact is missing")
    if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
        raise ValueError(f"rank {rank} artifact checksum mismatch")
    return path


def capture_frozen_snapshot(generation_dir, snapshots_root, *, dataset_hash=V2_DATASET_MANIFEST_HASH):
    """Validate a published generation and derive an evaluation-only snapshot.

    Returns the snapshot payload plus a content-addressed path.  The source
    generation is never modified.  Only rank-0 model state is retained; the
    optimizer is intentionally dropped.
    """
    generation = Path(generation_dir).resolve()
    manifest = read_json(generation / "manifest.json")
    if not manifest:
        raise ValueError("generation manifest is missing")
    _validate_generation_manifest(manifest, dataset_hash=dataset_hash)
    payloads = []
    for rank in (0, 1):
        path = _verify_rank_checksum(generation, manifest, rank)
        payload = _load_rank_payload(path)
        require_checkpoint_identity(payload["learner"])
        if payload.get("source_sha") != manifest["source_sha"]:
            raise ValueError("checkpoint source identity disagrees with its manifest")
        if payload.get("dataset_manifest_hash") != dataset_hash:
            raise ValueError("checkpoint dataset identity disagrees with its manifest")
        saved_config = payload["learner"].get("config", {}).get("model")
        if not isinstance(saved_config, dict) or payload.get("model_config_hash") != hash_json(saved_config):
            raise ValueError("checkpoint model configuration identity is inconsistent")
        if not payload.get("action_schema_hash") or not payload.get("model_config_hash"):
            raise ValueError("checkpoint lacks required contract hashes")
        accounting = payload.get("accounting", {})
        if int(accounting.get("committed_matches", -1)) != int(manifest["committed_matches"]):
            raise ValueError("checkpoint committed-match accounting disagrees with its manifest")
        if int(accounting.get("iteration", -1)) != int(manifest["iteration"]):
            raise ValueError("checkpoint iteration accounting disagrees with its manifest")
        ddp = payload.get("ddp", {})
        if int(ddp.get("rank", -1)) != rank or int(ddp.get("world_size", 0)) != 2:
            raise ValueError("checkpoint DDP rank identity is inconsistent")
        drain = payload.get("collector_drain", {})
        if drain.get("complete") is not True or int(drain.get("in_flight_battles", -1)) != 0:
            raise ValueError("checkpoint collector was not observed fully drained")
        payloads.append(payload)
    if payloads[0].get("rank_digests") != payloads[1].get("rank_digests"):
        raise ValueError("checkpoint rank digests disagree between ranks")
    digests = payloads[0].get("rank_digests")
    if not isinstance(digests, dict) or set(digests) != {"0", "1"}:
        raise ValueError("checkpoint rank digests are incomplete")
    model = payloads[0]["learner"]["model"]
    if canonical_model_digest(model) != canonical_model_digest(payloads[1]["learner"]["model"]):
        raise ValueError("checkpoint model state disagrees between ranks")
    first = payloads[0]
    committed = int(first["accounting"]["committed_matches"])
    experiment = first.get("experiment", {})
    snapshot = {
        "schema_version": 1,
        "kind": SNAPSHOT_KIND,
        "source_sha": first["source_sha"],
        "dataset_manifest_hash": dataset_hash,
        "model_config_hash": first["model_config_hash"],
        "action_schema_hash": first["action_schema_hash"],
        "ppo_config_hash": first.get("ppo_config_hash"),
        "experiment": experiment,
        "accounting": {
            "iteration": int(first["accounting"]["iteration"]),
            "committed_matches": committed,
        },
        "ddp": first["ddp"],
        "rank_digests": digests,
        "collector_drain": first.get("collector_drain"),
        "learner": {
            "model": model,
            "config": first["learner"]["config"],
            "contracts": first["learner"]["contracts"],
        },
        "source_generation": generation.name,
        "source_generation_manifest": {
            key: value for key, value in manifest.items() if key != "team_stats_receipts"
        },
        "source_generation_manifest_sha256": sha256_file(generation / "manifest.json"),
    }
    snapshots_root = Path(snapshots_root)
    snapshots_root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".snapshot-", suffix=".pt", dir=str(snapshots_root))
    os.close(fd)
    try:
        torch.save(snapshot, tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        file_hash = sha256_file(tmp)
        destination = snapshots_root / file_hash[:2] / file_hash
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.replace(tmp, destination)
            destination.chmod(0o444)
        else:
            os.unlink(tmp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return snapshot, destination, committed


def register_snapshot(registry_path, *, policy_id, label, snapshot_path, snapshot,
                      incumbent=False, latest=False, extra=None):
    """Register an evaluation-owned snapshot with full identity pinning."""
    entry = register_checkpoint(
        registry_path,
        policy_id=policy_id,
        label=label,
        checkpoint_path=snapshot_path,
        committed_matches=int(snapshot["accounting"]["committed_matches"]),
        source_sha=snapshot.get("source_sha"),
        incumbent=incumbent,
        latest=latest,
        storage_mode="verification-only",
    )
    registry = read_json(registry_path)
    stored = registry["policies"][policy_id]
    stored["checkpoint_generation"] = snapshot.get("source_generation")
    stored["snapshot_kind"] = snapshot.get("kind")
    stored["v2_league"] = True
    stored["registered_at"] = time.time()
    if extra:
        stored.update(extra)
    atomic_json(registry_path, registry)
    return stored


def snapshot_entries(registry_path, *, include_retired=False):
    registry = verify_archive(registry_path)
    rows = [
        entry
        for entry in registry["policies"].values()
        if include_retired or not entry.get("retired")
    ]
    return registry, sorted(rows, key=lambda row: (row["committed_matches"], row["id"]))


def committed_clock(payload):
    return int(payload["accounting"]["committed_matches"])


# ---------------------------------------------------------------------------
# Milestone scheduling
# ---------------------------------------------------------------------------

def _initial_schedule(campaign_start, light_interval, full_interval):
    return {
        "schema": SCHEDULE_SCHEMA,
        "campaign_start": int(campaign_start),
        "light_interval": int(light_interval),
        "full_interval": int(full_interval),
        "boundaries": {},
        "updated_at": time.time(),
    }


def update_schedule(schedule_path, snapshots, committed_matches, *,
                    campaign_start=DEFAULT_CAMPAIGN_START,
                    light_interval=LIGHT_INTERVAL, full_interval=FULL_INTERVAL):
    """Recompute the milestone ledger and materialise one job per checkpoint.

    ``snapshots`` is an iterable of ``{"id", "committed_matches"}`` known to be
    verified.  ``committed_matches`` is the live global training counter, used
    only to decide which nominal milestones have already been crossed.

    Greedy, idempotent assignment: a checkpoint serves the earliest unserved
    milestone at or after the milestone, and every milestone crossed before that
    checkpoint is coalesced into the same job.  A job is ``full`` when it covers
    a full milestone.  Milestones with no eligible checkpoint are ``deferred``;
    milestones before the recovered campaign start are ``skipped_prestart``.
    """
    state = read_json(schedule_path) or _initial_schedule(campaign_start, light_interval, full_interval)
    if state.get("schema") != SCHEDULE_SCHEMA:
        raise ValueError("unsupported evaluation schedule schema")
    for key, value in (("campaign_start", campaign_start), ("light_interval", light_interval),
                       ("full_interval", full_interval)):
        if int(state[key]) != int(value):
            raise ValueError(f"evaluation schedule configuration mismatch: {key}")
    boundaries = state["boundaries"]
    index = 1
    while index * light_interval <= int(committed_matches):
        boundary = index * light_interval
        key = str(boundary)
        if key not in boundaries:
            boundaries[key] = {
                "boundary": boundary,
                "kind": "full" if boundary % full_interval == 0 else "light",
                "status": "unassigned",
            }
        index += 1
    for record in boundaries.values():
        if record["status"] == "unassigned" and record["boundary"] < int(campaign_start):
            record.update(status="skipped_prestart",
                          reason=f"milestone precedes recovered campaign start {campaign_start}")
        elif record["status"] == "deferred":
            # A previously unfilled milestone may be served by a checkpoint that
            # has since been published; re-attempt it without inventing a model.
            record.pop("reason", None)
            record["status"] = "unassigned"
    snaps = sorted({(row["id"], int(row["committed_matches"])) for row in snapshots}, key=lambda row: row[1])
    claimed = set()
    jobs = state.setdefault("jobs", {})
    for key in sorted(boundaries, key=lambda value: int(value)):
        record = boundaries[key]
        if record["status"] != "unassigned":
            continue
        boundary = record["boundary"]
        chosen = next(((pid, clock) for pid, clock in snaps if pid not in claimed and clock >= boundary), None)
        if chosen is None:
            record.update(status="deferred",
                          reason=f"no published checkpoint at or after milestone {boundary} yet")
            continue
        policy_id, clock = chosen
        claimed.add(policy_id)
        covered = sorted(
            int(other)
            for other, value in boundaries.items()
            if value["status"] == "unassigned" and boundary <= value["boundary"] <= clock
        )
        tier = "full" if any(boundaries[str(x)]["kind"] == "full" for x in covered) else "light"
        for other in covered:
            record_other = boundaries[str(other)]
            if other == boundary:
                status = "exact" if other == clock else "approximate"
            else:
                status = "coalesced"
            record_other.update(status=status, checkpoint_id=policy_id, checkpoint_clock=clock)
        job_id = f"{tier}:{policy_id}"
        if job_id not in jobs:
            jobs[job_id] = {
                "job_id": job_id,
                "policy_id": policy_id,
                "candidate_clock": clock,
                "tier": tier,
                "scheduled_boundaries": covered,
                "primary_boundary": min(covered),
                "status": "pending",
                "created_at": time.time(),
            }
        else:
            merged = sorted(set(jobs[job_id]["scheduled_boundaries"]) | set(covered))
            jobs[job_id]["scheduled_boundaries"] = merged
            if tier == "full":
                jobs[job_id]["tier"] = "full"
    state["updated_at"] = time.time()
    atomic_json(schedule_path, state)
    return state


def schedule_jobs(schedule_path, *, statuses=("pending", "running", "deferred", "failed")):
    state = read_json(schedule_path) or {"jobs": {}}
    return [dict(job) for job in state.get("jobs", {}).values() if job.get("status") in statuses]


def mark_job(schedule_path, job_id, status, **fields):
    state = read_json(schedule_path) or {"jobs": {}}
    job = state["jobs"].get(job_id)
    if job is None:
        raise ValueError("unknown evaluation job")
    if job.get("status") == "completed" and status != "completed":
        return dict(job)
    job["status"] = status
    job.update(fields)
    atomic_json(schedule_path, state)
    return dict(job)


def requeue_stale_jobs(schedule_path, *, status="running"):
    """Reset interrupted jobs to ``pending`` on worker startup.

    Called only while the single-writer worker lock is held, so any job still
    marked ``running`` belongs to a dead worker.  Receipts and alpha spending
    are content-addressed/idempotent, so a resumed job never double-counts.
    Jobs already ``completed`` are never touched.
    """
    state = read_json(schedule_path) or {"jobs": {}}
    requeued = []
    for job in state.get("jobs", {}).values():
        if job.get("status") == status:
            job["status"] = "pending"
            job["requeued_at"] = time.time()
            requeued.append(job["job_id"])
    if requeued:
        atomic_json(schedule_path, state)
    return requeued


# ---------------------------------------------------------------------------
# Evaluation plan and runners
# ---------------------------------------------------------------------------

def plan_panels(*, tier, candidate_id, incumbent_id, baseline_id, anchors):
    """Distribute an evaluation budget across distinct, verified opponents."""
    budget = FULL_GAMES if tier == "full" else LIGHT_GAMES
    ordered = []
    for opponent in [incumbent_id, baseline_id, *anchors]:
        if opponent and opponent != candidate_id and opponent not in ordered:
            ordered.append(opponent)
    if not ordered:
        return []
    weights = {1: [1.0], 2: [0.5, 0.5]}.get(len(ordered))
    if weights is None:
        weights = [0.5, 0.25, 0.25]
        if len(ordered) > 3:
            ordered = ordered[:3]
    panels = []
    start = 0
    for opponent, weight in zip(ordered, weights):
        games = int(round(budget * weight))
        games -= games % 2
        if games < 2:
            games = 2
        panels.append({"opponent_id": opponent, "games": games, "fixture_start_pair": start // 2})
        start += games
    return panels


def evaluation_environment(engine_path, data_path, teams_path, *, backend, batch_size, device, threads, engine_workers):
    from agent.evaluation.league import environment_identity
    from agent.evaluation.batched_policy import BATCHED_EVALUATOR_VERSION, SAMPLING_SEMANTICS
    environment = environment_identity(engine_path=engine_path, data_path=data_path, teams_path=teams_path)
    environment.update(
        backend=backend,
        batch_size=int(batch_size),
        device=device,
        threads=int(threads),
        engine_workers=int(engine_workers),
        battle_format="native-v2-frozen-team-pool",
        evaluator=BATCHED_EVALUATOR_VERSION,
        sampling=SAMPLING_SEMANTICS,
    )
    return environment


def master_fixtures(engine, *, pairs=MASTER_PAIRS, seed=MASTER_SEED):
    from agent.evaluation.fixed_policy import build_fixture_manifest
    return build_fixture_manifest(pairs=pairs, team_count=engine.team_count(), seed=seed)


def _slice_pairs(fixtures, start_pair, pairs):
    return fixtures[2 * start_pair:2 * (start_pair + pairs)]


def run_matchup_cached(registry_path, receipts_dir, engine, fixtures, candidate_id, opponent_id, *,
                       seed, max_rounds, environment, batch_size):
    """Run one verified matchup, caching the receipt under its content identity.

    Per-identity receipt files keep the league store bounded and let an
    interrupted worker resume without repeating completed matchups.
    """
    from agent.evaluation.league import evaluate_pair, evaluation_identity
    registry = read_json(registry_path)
    policies = registry["policies"]
    if candidate_id == opponent_id:
        raise ValueError("a matchup requires two distinct policies")
    candidate = dict(policies[candidate_id], path=verify_entry(registry_path, policies[candidate_id]))
    opponent = dict(policies[opponent_id], path=verify_entry(registry_path, policies[opponent_id]))
    identity, _body = evaluation_identity(candidate, opponent, fixtures, seed=seed,
                                          max_rounds=max_rounds, environment=environment)
    receipts_dir = Path(receipts_dir)
    receipts_dir.mkdir(parents=True, exist_ok=True)
    destination = receipts_dir / f"{identity}.json"
    if destination.exists():
        stored = read_json(destination)
        if stored.get("status") == "success" and stored.get("identity") == identity:
            return stored, True
    receipt = evaluate_pair(engine, fixtures, candidate, opponent, device="cpu", seed=seed,
                            max_rounds=max_rounds, environment=environment, backend="batched",
                            batch_size=batch_size)
    if receipt["identity"] != identity:
        raise RuntimeError("evaluation identity changed while running")
    atomic_json(destination, receipt)
    return receipt, False


def match_environment(environment):
    """Environment keys that must be identical for a comparison to be paired."""
    keys = ("engine_sha256", "team_dataset_sha256", "mechanics_data_sha256", "backend",
            "batch_size", "device", "battle_format", "evaluator", "sampling")
    return {key: environment.get(key) for key in keys}


# ---------------------------------------------------------------------------
# Promotion statistics
# ---------------------------------------------------------------------------

def paired_scores(outcomes):
    grouped = {}
    for row in outcomes:
        if row["outcome"] not in {"win", "loss", "draw"}:
            continue
        score = 1.0 if row["outcome"] == "win" else 0.5 if row["outcome"] == "draw" else 0.0
        grouped.setdefault(row["pair_id"], []).append(score)
    return [sum(values) / len(values) for values in grouped.values() if len(values) == 2]


def promotion_decision(outcomes, *, comparison_index, opponent_index, stage=0,
                       stage_is_final=False, already_promoted=False):
    """Pair-cluster Hoeffding bound with global alpha spending across the archive.

    Correlated seat-swapped games are collapsed into one pair before the bound
    is computed, so the interval does not treat them as independent evidence.
    """
    scores = paired_scores(outcomes)
    if not scores:
        return {"decision": "invalid", "reason": "no valid paired outcomes", "pairs": 0}
    if any(row["outcome"] == "error" for row in outcomes):
        return {"decision": "invalid", "reason": "operational evaluation failure", "pairs": len(scores)}
    alpha = 0.05 / max(1, comparison_index * (comparison_index + 1) * opponent_index * (opponent_index + 1) * 3)
    mean = sum(scores) / len(scores)
    radius = math.sqrt(math.log(1.0 / alpha) / (2.0 * len(scores)))
    result = {
        "decision": "retain",
        "score": mean,
        "lower": max(0.0, mean - radius),
        "upper": min(1.0, mean + radius),
        "alpha": alpha,
        "pairs": len(scores),
        "method": "pair-cluster-Hoeffding-global-alpha-spending-v1",
    }
    if mean - radius > 0.5:
        result["decision"] = "promote"
    elif mean + radius < 0.5:
        result["decision"] = "retain"
    elif stage_is_final or already_promoted:
        result["decision"] = "retain"
    else:
        result["decision"] = "escalate"
    return result


def paired_difference(new_outcomes, old_outcomes):
    """Paired per-fixture score difference (new minus old) on shared fixtures."""
    def scores(rows):
        return {
            row["fixture_id"]: (1.0 if row["outcome"] == "win" else 0.5 if row["outcome"] == "draw" else 0.0)
            for row in rows
            if row["outcome"] in {"win", "loss", "draw"}
        }
    new, old = scores(new_outcomes), scores(old_outcomes)
    shared = sorted(set(new) & set(old))
    if not shared:
        return None
    diffs = [new[key] - old[key] for key in shared]
    mean = sum(diffs) / len(diffs)
    if len(diffs) < 2:
        return {"games": len(diffs), "mean": mean, "ci95": [mean, mean]}
    variance = sum((value - mean) ** 2 for value in diffs) / (len(diffs) - 1)
    stderr = math.sqrt(variance / len(diffs))
    return {"games": len(diffs), "mean": mean, "ci95": [mean - 1.96 * stderr, mean + 1.96 * stderr]}


def evaluate_regression(*, candidate_baseline_outcomes, incumbent_baseline_outcomes,
                        margin=0.05):
    """Reject a statistically supported material regression against the frozen baseline."""
    result = paired_difference(candidate_baseline_outcomes, incumbent_baseline_outcomes)
    if result is None:
        return {"status": "insufficient_coverage"}
    if result["ci95"][1] < -abs(margin):
        return {"status": "regression_warning", **result}
    return {"status": "acceptable", **result}


# ---------------------------------------------------------------------------
# League V2: screening, alpha budget, sequential confirmation
# ---------------------------------------------------------------------------

def paired_variance(scores):
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    return sum((value - mean) ** 2 for value in scores) / (len(scores) - 1)


def descriptive_interval(scores, *, z=1.96):
    """Descriptive paired-confidence interval used only for screening.

    This is deliberately *not* a promotion bound: it allocates evaluation
    resources.  Formal promotion uses :func:`confirmation_rule`.
    """
    n = len(scores)
    if not scores:
        return None
    mean = sum(scores) / len(scores)
    se = math.sqrt(paired_variance(scores) / n) if n > 1 else 0.0
    return {
        "score": mean,
        "lower": max(0.0, mean - z * se),
        "upper": min(1.0, mean + z * se),
        "pairs": n,
        "method": "descriptive-paired-normal-v1",
    }


def hoeffding_interval(scores, *, alpha):
    """One-sided-safe Hoeffding confidence interval for a bounded paired mean.

    Pair scores live in [0, 1]; for ``n`` independent pairs and any
    ``alpha in (0, 1)``, Hoeffding gives
    ``P(mean > lower) >= 1 - alpha`` with ``radius = sqrt(log(1/alpha)/(2n))``.
    Seat-swapped games are collapsed into one pair before the bound is computed.
    """
    n = len(scores)
    if n == 0:
        return {"decision": "invalid", "reason": "no paired outcomes", "pairs": 0}
    mean = sum(scores) / n
    radius = math.sqrt(math.log(1.0 / alpha) / (2.0 * n))
    return {
        "score": mean,
        "lower": max(0.0, mean - radius),
        "upper": min(1.0, mean + radius),
        "alpha": alpha,
        "pairs": n,
        "method": "pair-cluster-Hoeffding-fixed-sample-v2",
    }


def primary_screening(scores, *, trigger=SCREEN_TRIGGER, floor=SCREEN_AMPLIFY_FLOOR,
                      z=1.96):
    """Classify primary evidence for resource allocation (never promotion).

    Returns one of:

    * ``confirmed_inferior`` - descriptive upper bound below the floor; the
      candidate is worse than the incumbent and is not escalated.
    * ``promising`` - point estimate at or above ``trigger``; escalate to
      independent confirmation.
    * ``amplify`` - point estimate at or above ``floor`` but below ``trigger``;
      a bounded extra primary panel may resolve the ambiguity.
    * ``weak`` - point estimate below the floor without confirmed inferiority;
      retain without spending confirmation budget.
    """
    stats = descriptive_interval(scores, z=z)
    if stats is None:
        return {"status": "invalid", "reason": "no paired outcomes", "pairs": 0}
    stats["trigger"] = trigger
    stats["floor"] = floor
    if stats["upper"] < floor:
        stats["status"] = "confirmed_inferior"
    elif stats["score"] >= trigger:
        stats["status"] = "promising"
    elif stats["score"] >= floor:
        stats["status"] = "amplify"
    else:
        stats["status"] = "weak"
    return stats


def confirmation_seed_for(candidate_id):
    """Deterministic, per-candidate confirmation seed (recorded on the job).

    Derived from the candidate identity alone so a worker restart reproduces the
    identical fresh fixture universe and never silently reuses a different set.
    """
    material = f"pa3-v2-confirmation:{candidate_id}".encode()
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def confirmation_stage_plan(*, stage_games=CONFIRM_STAGE_GAMES,
                            weights=CONFIRM_STAGE_ALPHA_WEIGHTS):
    """Predeclared cumulative confirmation ladder in complete seat-swap pairs."""
    plan, previous, cumulative_alpha = [], 0, 0.0
    for games, weight in zip(stage_games, weights):
        pairs = games // 2
        cumulative_alpha += weight
        plan.append({
            "stage": len(plan) + 1,
            "cumulative_games": int(games),
            "cumulative_pairs": int(pairs),
            "add_pairs": int(pairs - previous),
            "first_pair": int(previous),
            "cumulative_alpha_weight": cumulative_alpha,
        })
        previous = pairs
    return plan


def confirmation_rule(pooled_scores, *, alpha_total, stage):
    """Formal promotion test at one predeclared confirmation look.

    ``alpha_total`` is the whole comparison's allocation; ``stage`` is the
    cumulative stage record from :func:`confirmation_stage_plan`.  The test is a
    fixed-sample Hoeffding bound at the predeclared cumulative alpha for that
    look, so the union bound over performed looks never exceeds ``alpha_total``.
    """
    alpha = alpha_total * stage["cumulative_alpha_weight"]
    result = hoeffding_interval(pooled_scores, alpha=alpha)
    result["stage"] = stage["stage"]
    result["cumulative_alpha"] = alpha
    if result.get("decision") == "invalid":
        return result
    if result["lower"] > 0.5:
        result["decision"] = "promote"
    elif result["upper"] < 0.5:
        result["decision"] = "retain_inferior"
    else:
        result["decision"] = "continue"
    return result


def initial_alpha_ledger():
    return {
        "schema": ALPHA_LEDGER_SCHEMA,
        "total": ALPHA_TOTAL,
        "schedule": ALPHA_SCHEDULE,
        "max_promotion_comparisons": ALPHA_MAX_PROMOTION_COMPARISONS,
        "historical_comparisons": ALPHA_HISTORICAL_COMPARISONS,
        "historical_consumed": ALPHA_HISTORICAL_CONSUMED,
        "consumed": ALPHA_HISTORICAL_CONSUMED,
        "next_index": ALPHA_HISTORICAL_COMPARISONS + 1,
        "committed": [],
        "note": ("historical spend preserved from pa3-v2-promotion-rules-v1 "
                 "(union of primary+confirmatory gates); never reset"),
        "updated_at": time.time(),
    }


def load_alpha_ledger(eval_root):
    path = Path(eval_root) / "alpha_budget.json"
    ledger = read_json(path)
    if ledger is None:
        ledger = initial_alpha_ledger()
        atomic_json(path, ledger)
        return ledger
    if ledger.get("schema") != ALPHA_LEDGER_SCHEMA:
        raise ValueError("unsupported alpha ledger schema")
    return ledger


def alpha_for_comparison(ledger):
    """Return the predeclared alpha allocation for the next full comparison."""
    index = int(ledger.get("next_index", ALPHA_HISTORICAL_COMPARISONS + 1))
    maximum = int(ledger.get("max_promotion_comparisons", ALPHA_MAX_PROMOTION_COMPARISONS))
    if index > maximum:
        return {"index": index, "alpha": 0.0, "exhausted": True}
    remaining = max(0.0, float(ledger["total"]) - float(ledger["consumed"]))
    slots = maximum - (index - 1)
    return {"index": index, "alpha": remaining / slots, "exhausted": False,
            "remaining": remaining, "slots": slots}


def commit_alpha(eval_root, *, candidate_id, index, alpha, decision):
    """Record consumption once per candidate; idempotent across restarts."""
    path = Path(eval_root) / "alpha_budget.json"
    ledger = load_alpha_ledger(eval_root)
    if any(row.get("candidate_id") == candidate_id for row in ledger.get("committed", [])):
        return ledger
    ledger["consumed"] = float(ledger["consumed"]) + float(alpha)
    ledger["next_index"] = max(int(ledger.get("next_index", 1)), int(index) + 1)
    ledger.setdefault("committed", []).append({
        "candidate_id": candidate_id,
        "index": int(index),
        "alpha": float(alpha),
        "decision": decision,
        "at": time.time(),
    })
    ledger["updated_at"] = time.time()
    atomic_json(path, ledger)
    return ledger


def plan_full_panels(*, candidate_id, incumbent_id, baseline_id, anchors):
    """Primary screening layout.  Incumbent evidence is an explicit region.

    Baseline and anchor regions are byte-identical to the historical v1 layout
    so frozen regression comparisons keep shared fixtures.  All primary panels
    use the shared master universe; confirmation uses a disjoint universe.
    """
    panels = []
    if incumbent_id and incumbent_id != candidate_id:
        panels.append({"opponent_id": incumbent_id, "phase": "primary",
                       "games": FULL_INITIAL_GAMES,
                       "fixture_start_pair": INCUMBENT_START_PAIR})
    if baseline_id and baseline_id not in (candidate_id, incumbent_id):
        panels.append({"opponent_id": baseline_id, "phase": "baseline",
                       "games": BASELINE_GAMES,
                       "fixture_start_pair": BASELINE_START_PAIR})
    for anchor in anchors or []:
        if anchor in (candidate_id, incumbent_id, baseline_id):
            continue
        if any(panel["opponent_id"] == anchor for panel in panels):
            continue
        panels.append({"opponent_id": anchor, "phase": "anchor",
                       "games": ANCHOR_GAMES,
                       "fixture_start_pair": ANCHOR_START_PAIR})
        break
    return panels


def plan_light_panels(*, candidate_id, incumbent_id, baseline_id, anchors):
    """Scale the full layout down to the lightweight diagnostic budget."""
    panels = plan_full_panels(candidate_id=candidate_id, incumbent_id=incumbent_id,
                              baseline_id=baseline_id, anchors=anchors)
    if not panels:
        return []
    total = sum(panel["games"] for panel in panels)
    scaled, start = [], 0
    for panel in panels:
        games = max(2, int(round(LIGHT_GAMES_V2 * panel["games"] / total)))
        games -= games % 2
        scaled.append({"opponent_id": panel["opponent_id"], "phase": panel["phase"],
                       "games": games, "fixture_start_pair": start // 2})
        start += games
    return scaled


_PANEL_FIXTURE_KEYS = ("team_a", "team_b", "seed")


def panel_fixture_signatures(fixtures):
    """Hashable identity of a fixture slice for disjointness checks."""
    return {
        (fixture.team_a, fixture.team_b, tuple(fixture.seed), fixture.policy_side)
        for fixture in fixtures
    }


def assert_panels_disjoint(primary_fixtures, confirmation_fixtures):
    """Confirmation fixtures must not overlap the primary panel."""
    overlap = panel_fixture_signatures(primary_fixtures) & panel_fixture_signatures(confirmation_fixtures)
    if overlap:
        raise ValueError(f"confirmation fixtures overlap primary fixtures ({len(overlap)} games)")
    return True


# ---------------------------------------------------------------------------
# Incumbent registry
# ---------------------------------------------------------------------------

def initial_incumbent(policy_id, entry):
    return {
        "schema": INCUMBENT_SCHEMA,
        "policy_id": policy_id,
        "model_digest": entry["model_digest"],
        "file_sha256": entry["file_sha256"],
        "committed_matches": entry["committed_matches"],
        "status": "bootstrap",
        "reason": "verified recovered V2 checkpoint used as initial tracking reference; not strength evidence",
        "promotions": [],
        "updated_at": time.time(),
    }


class StaleIncumbentError(RuntimeError):
    """Raised when a challenger tries to overwrite a newer incumbent."""


def promote_incumbent(best_path, history_dir, *, candidate_entry, previous, decision,
                      supporting, reason, expected_previous_id=None,
                      ruleset_version=RULESET_VERSION):
    """Atomically promote a candidate, guarding against stale challengers.

    ``expected_previous_id`` is the incumbent the candidate was evaluated
    against.  If the live best policy has since moved, the promotion is refused
    so an out-of-date challenge can never overwrite a newer champion.
    """
    current = read_json(best_path)
    if expected_previous_id is not None:
        if current is None or current.get("policy_id") != expected_previous_id:
            raise StaleIncumbentError(
                f"incumbent advanced from {expected_previous_id} to "
                f"{(current or {}).get('policy_id')}; refusing stale promotion")
    record = {
        "previous_policy": previous["policy_id"],
        "candidate_id": candidate_entry["id"],
        "decision": "promote",
        "supporting_receipts": supporting,
        "statistics": decision,
        "at": time.time(),
        "evaluator_version": EVALUATOR_VERSION,
        "ruleset_version": ruleset_version,
        "reason": reason,
    }
    new = {
        "schema": INCUMBENT_SCHEMA,
        "policy_id": candidate_entry["id"],
        "model_digest": candidate_entry["model_digest"],
        "file_sha256": candidate_entry["file_sha256"],
        "committed_matches": candidate_entry["committed_matches"],
        "status": "validated_direct_superiority",
        "reason": reason,
        "promotions": previous.get("promotions", []) + [record],
        "updated_at": time.time(),
    }
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    # Write the immutable history record first: if the process dies between the
    # two writes, best_policy.json is unchanged (recoverable), never a promotion
    # without an audit record.
    atomic_json(history_dir / f"{time.time_ns()}-{previous['policy_id']}.json", previous)
    atomic_json(best_path, new)
    return new


def record_rejection(history_dir, *, candidate_entry, previous, decision, reason, key=None,
                     ruleset_version=RULESET_VERSION, promotion_status=None):
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "previous_policy": previous["policy_id"],
        "candidate_id": candidate_entry["id"],
        "decision": "retain",
        "statistics": decision,
        "at": time.time(),
        "evaluator_version": EVALUATOR_VERSION,
        "ruleset_version": ruleset_version,
        "reason": reason,
    }
    if promotion_status is not None:
        record["promotion_status"] = promotion_status
    atomic_json(history_dir / f"{key or candidate_entry['id']}.json", record)
    return record


# ---------------------------------------------------------------------------
# Throughput observation (read-only)
# ---------------------------------------------------------------------------

_ITERATION_RE = re.compile(r"iteration (\d+) complete in ([0-9.]+)s")


def production_iteration_seconds(log_path, limit=8):
    try:
        text = Path(log_path).read_text(errors="ignore")
    except OSError:
        return []
    matches = _ITERATION_RE.findall(text)
    return [float(value) for _index, value in matches[-limit:]]


def production_throughput(log_path, *, matches_per_iteration=4096, limit=8):
    times = production_iteration_seconds(log_path, limit=limit)
    if not times or sum(times) <= 0:
        return None
    return matches_per_iteration * len(times) / sum(times)


def disk_headroom_bytes(path):
    return shutil.disk_usage(path).free


# ---------------------------------------------------------------------------
# Track B: held-out team generalization
# ---------------------------------------------------------------------------

HELDOUT_SCHEMA = "pa3-v2-heldout-dataset-v1"


def _team_records(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("teams", data.get("records", data.get("members")))
    if not isinstance(data, list) or not data:
        raise ValueError("held-out team dataset is empty or malformed")
    return data


def team_semantic_id(record):
    """Canonical, cosmetic-insensitive team identity.

    Prefers the dataset's own semantic fingerprint; otherwise hashes the six
    battle-relevant member records so a renamed file cannot masquerade as a
    disjoint team set.
    """
    if isinstance(record, dict) and isinstance(record.get("id"), str) and record["id"]:
        return record["id"]
    members = record.get("members", record) if isinstance(record, dict) else record
    canonical = []
    for member in members:
        if not isinstance(member, dict):
            canonical.append(member)
            continue
        canonical.append({
            "species": member.get("species"),
            "ability": member.get("ability"),
            "item": member.get("item"),
            "nature": member.get("nature"),
            "moves": sorted(member.get("moves", [])),
            "points": list(member.get("points", member.get("evs", []))),
            "ivs": list(member.get("ivs", [])),
            "gender": member.get("gender", ""),
            "level": member.get("level", 50),
        })
    return digest(sorted(canonical, key=lambda value: json.dumps(value, sort_keys=True)))


def build_heldout_manifest(heldout_path, training_teams_path):
    """Validate a held-out dataset and prove disjointness from the frozen pool."""
    heldout = _team_records(heldout_path)
    training = _team_records(training_teams_path)
    heldout_ids = {team_semantic_id(record) for record in heldout}
    training_ids = {team_semantic_id(record) for record in training}
    overlap = sorted(heldout_ids & training_ids)
    if overlap:
        raise ValueError(
            f"held-out dataset overlaps the frozen training pool ({len(overlap)} teams)"
        )
    return {
        "schema": HELDOUT_SCHEMA,
        "heldout_path": str(Path(heldout_path).resolve()),
        "heldout_sha256": sha256_file(heldout_path),
        "training_teams_sha256": sha256_file(training_teams_path),
        "heldout_teams": len(heldout_ids),
        "training_teams": len(training_ids),
        "overlap": 0,
        "namespace": "track-b-heldout",
    }


def heldout_status(root, *, training_teams_path, heldout_path=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "heldout_manifest.json"
    if heldout_path and Path(heldout_path).is_file():
        try:
            manifest = build_heldout_manifest(heldout_path, training_teams_path)
        except ValueError as error:
            atomic_json(root / "heldout_status.json",
                        {"status": "INVALID_HELDOUT_DATA", "reason": str(error), "at": time.time()})
            return {"status": "INVALID_HELDOUT_DATA", "reason": str(error)}
        atomic_json(manifest_path, manifest)
        atomic_json(root / "heldout_status.json",
                    {"status": "READY", "manifest": manifest, "at": time.time()})
        return {"status": "READY", "manifest": manifest}
    atomic_json(root / "heldout_status.json",
                {"status": "WAITING_FOR_VALID_HELDOUT_DATA",
                 "reason": "no verified team dataset disjoint from the frozen training pool",
                 "at": time.time()})
    return {"status": "WAITING_FOR_VALID_HELDOUT_DATA"}


# ---------------------------------------------------------------------------
# Internal relative ratings (diagnostic only)
# ---------------------------------------------------------------------------

def internal_ratings(receipts, *, iterations=200, k_factor=16.0):
    """Bradley-Terry/Elo-style internal relative ratings over observed matchups.

    Clearly labelled as an internal league diagnostic.  It is not Pokémon
    Showdown ladder Elo and is not a promotion criterion.
    """
    matches = []
    for receipt in receipts:
        if receipt.get("status") != "success":
            continue
        left, right = receipt.get("left_id"), receipt.get("right_id")
        score = receipt.get("score_mean")
        if left and right and left != right and score is not None:
            matches.append((left, right, float(score), int(receipt.get("matches", 0))))
    policies = sorted({left for left, _r, _s, _m in matches} | {right for _l, right, _s, _m in matches})
    if not policies:
        return {"status": "insufficient_coverage", "ratings": {}, "policies": []}
    rating = {policy: 0.0 for policy in policies}
    for _ in range(int(iterations)):
        for left, right, score, _games in matches:
            expected = 1.0 / (1.0 + 10 ** ((rating[right] - rating[left]) / 400.0))
            delta = k_factor * (score - expected)
            rating[left] += delta
            rating[right] -= delta
        mean = sum(rating.values()) / len(rating)
        for policy in rating:
            rating[policy] -= mean
    coverage = {policy: sum(games for left, right, _s, games in matches if policy in (left, right))
                for policy in policies}
    return {
        "status": "internal_relative_league_rating",
        "caveat": "internal diagnostic only; not ladder Elo and not promotion evidence",
        "ratings": {policy: round(rating[policy], 2) for policy in policies},
        "coverage_games": coverage,
    }
