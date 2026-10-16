#!/usr/bin/env python3
"""Resource-aware V2 evaluation worker.

Single-writer, CPU-only, low priority.  Processes scheduled league jobs against
the incumbent, the frozen baseline and available historical V2 anchors, then
applies conservative promotion rules.  It never touches PPO state and defers
work when disk or measured production throughput is at risk.
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

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch  # noqa: E402

from agent.evaluation.v2_league import (  # noqa: E402
    CONFIRMATORY_GAMES,
    CONFIRMATORY_SEED,
    DEFAULT_CAMPAIGN_START,
    EVALUATOR_VERSION,
    FULL_GAMES,
    LIGHT_GAMES,
    MASTER_PAIRS,
    MASTER_SEED,
    PROMOTION_MIN_PAIRS,
    RULESET_VERSION,
    V2_DATASET_MANIFEST_HASH,
    atomic_json,
    disk_headroom_bytes,
    evaluate_regression,
    evaluation_environment,
    initial_incumbent,
    mark_job,
    master_fixtures,
    match_environment,
    paired_difference,
    plan_panels,
    production_throughput,
    promote_incumbent,
    promotion_decision,
    read_json,
    record_rejection,
    run_matchup_cached,
    schedule_jobs,
    snapshot_entries,
)

DATA_DIR = ROOT / "engine" / "data"
TEAMS_PATH = ROOT / "engine" / "data" / "a10-training-teams.json"
ENGINE_PATH = ROOT / "engine" / "python" / "pa3_engine" / "pa3_engine.so"


def _ordinals(entries):
    ordered = sorted(entries, key=lambda entry: (entry["committed_matches"], entry["id"]))
    return {entry["id"]: index + 1 for index, entry in enumerate(ordered)}


def _effective_panels(tier, candidate_id, incumbent_id, baseline_id, anchors, *, smoke_games=None):
    panels = plan_panels(tier=tier, candidate_id=candidate_id, incumbent_id=incumbent_id,
                         baseline_id=baseline_id, anchors=anchors)
    if not smoke_games:
        return panels
    total = sum(panel["games"] for panel in panels)
    if total <= 0:
        return panels
    scaled, start = [], 0
    for panel in panels:
        games = max(2, int(round(panel["games"] * smoke_games / total)))
        games -= games % 2
        scaled.append({"opponent_id": panel["opponent_id"], "games": games, "fixture_start_pair": start // 2})
        start += games
    return scaled


def _slice(fixtures, start_pair, games):
    return fixtures[2 * start_pair:2 * (start_pair + games // 2)]


def run_job(*, eval_root, job, args, engine, environment, master, log_path):
    registry_path = eval_root / "registry.json"
    receipts_dir = eval_root / "receipts"
    jobs_dir = eval_root / "receipts" / "jobs"
    best_path = eval_root / "best_policy.json"
    history_dir = eval_root / "best_history" / "promotions"
    decisions_dir = eval_root / "decisions"

    registry, entries = snapshot_entries(registry_path)
    by_id = {entry["id"]: entry for entry in entries}
    candidate = by_id.get(job["policy_id"])
    if candidate is None:
        raise ValueError("scheduled candidate is not a live verified policy")
    incumbent = read_json(best_path)
    if incumbent is None:
        bootstrap = min(entries, key=lambda entry: entry["committed_matches"])
        incumbent = initial_incumbent(bootstrap["id"], bootstrap)
        atomic_json(best_path, incumbent)
    baseline = next((entry for entry in entries if entry.get("bootstrap")), None)
    if baseline is None:
        baseline = min(entries, key=lambda entry: (entry["committed_matches"], entry["id"]))
    anchors = [entry["id"] for entry in sorted(entries, key=lambda entry: entry["committed_matches"], reverse=True)
               if entry["committed_matches"] < candidate["committed_matches"]
               and entry["id"] not in (candidate["id"], incumbent["policy_id"], baseline["id"])][:2]
    panels = _effective_panels(job["tier"], candidate["id"], incumbent["policy_id"], baseline["id"],
                               anchors, smoke_games=args.smoke_games)

    started = time.time()
    matchups, bodies = [], []
    for panel in panels:
        fixtures = _slice(master, panel["fixture_start_pair"], panel["games"])
        receipt, cached = run_matchup_cached(
            registry_path, receipts_dir, engine, fixtures, candidate["id"], panel["opponent_id"],
            seed=MASTER_SEED, max_rounds=2000, environment=environment, batch_size=args.batch_size,
        )
        opponent = by_id[panel["opponent_id"]]
        matchups.append({
            "opponent_id": panel["opponent_id"],
            "opponent_clock": opponent["committed_matches"],
            "opponent_model_digest": opponent["model_digest"],
            "opponent_file_sha256": opponent["file_sha256"],
            "games": panel["games"],
            "matches": receipt["matches"],
            "wins": receipt["wins"],
            "losses": receipt["losses"],
            "draws": receipt["draws"],
            "score_mean": receipt["score_mean"],
            "paired_score_mean": receipt["paired_score_mean"],
            "paired_fixtures": receipt["paired_fixtures"],
            "score_ci95": receipt["score_ci95"],
            "identity": receipt["identity"],
            "fixture_manifest_hash": receipt["fixture_manifest_hash"],
            "fixture_seed": MASTER_SEED,
            "operational_errors": receipt["operational_errors"],
            "illegal_actions": receipt["illegal_actions"],
            "status": receipt["status"],
            "runtime_seconds": receipt["runtime_seconds"],
            "evaluator_backend": "batched-native",
            "device": "cpu",
            "cached": cached,
        })
        bodies.append(receipt)
    runtime = time.time() - started
    legal = all(row["status"] == "success" for row in matchups)
    baseline_receipt = next((body for body, row in zip(bodies, matchups)
                             if row["opponent_id"] == baseline["id"]), None)

    decision = {"decision": "invalid", "reason": "matchup invalidity"}
    regression = {"status": "insufficient_coverage"}
    confirmatory = None
    promoted = False
    preliminary = None
    if legal:
        incumbent_receipt = next((body for body, row in zip(bodies, matchups)
                                  if row["opponent_id"] == incumbent["policy_id"]), None)
        if incumbent["policy_id"] == candidate["id"]:
            decision = {"decision": "retain", "reason": "candidate is the current incumbent", "pairs": 0}
        elif incumbent_receipt is not None and incumbent_receipt["paired_fixtures"] >= PROMOTION_MIN_PAIRS:
            ordinals = _ordinals(entries)
            preliminary = promotion_decision(
                incumbent_receipt["outcomes"],
                comparison_index=ordinals[candidate["id"]],
                opponent_index=ordinals[incumbent["policy_id"]],
                stage_is_final=(job["tier"] != "full"),
            )
            if baseline_receipt is not None and incumbent["policy_id"] != baseline["id"]:
                previous = read_json(receipts_dir / "baseline" / f"{incumbent['policy_id']}.json")
                if previous is not None:
                    regression = evaluate_regression(
                        candidate_baseline_outcomes=baseline_receipt["outcomes"],
                        incumbent_baseline_outcomes=previous.get("outcomes", []),
                    )
            if job["tier"] != "full":
                decision = {"decision": "retain",
                            "reason": "PROMOTION_PENDING_EVIDENCE: lightweight evaluation is diagnostic only",
                            "statistics": preliminary, "pairs": incumbent_receipt["paired_fixtures"]}
            elif regression.get("status") == "regression_warning":
                decision = {"decision": "retain", "reason": "material regression against the frozen baseline",
                            "statistics": preliminary, "regression": regression}
            elif preliminary["decision"] != "promote":
                decision = {"decision": "retain", "reason": "direct paired evidence does not support superiority",
                            "statistics": preliminary, "regression": regression}
            else:
                confirm_fixtures = master_fixtures(engine, pairs=CONFIRMATORY_GAMES // 2, seed=CONFIRMATORY_SEED)
                confirm_receipt, _ = run_matchup_cached(
                    registry_path, receipts_dir, engine, confirm_fixtures, candidate["id"],
                    incumbent["policy_id"], seed=CONFIRMATORY_SEED, max_rounds=2000,
                    environment=environment, batch_size=args.batch_size,
                )
                confirmatory = promotion_decision(
                    confirm_receipt["outcomes"],
                    comparison_index=ordinals[candidate["id"]],
                    opponent_index=ordinals[incumbent["policy_id"]],
                    stage_is_final=True,
                )
                if confirmatory["decision"] == "promote":
                    support = [row["identity"] for row in matchups] + [confirm_receipt["identity"]]
                    promote_incumbent(
                        best_path, history_dir, candidate_entry=candidate, previous=incumbent,
                        decision={"primary": preliminary, "confirmatory": confirmatory, "regression": regression},
                        supporting=support,
                        reason="full-panel direct superiority confirmed on independent held-back fixtures",
                    )
                    promoted = True
                    decision = {"decision": "promote", "statistics": preliminary,
                                "confirmatory": confirmatory, "regression": regression}
                else:
                    decision = {"decision": "retain",
                                "reason": "confirmatory held-back evaluation did not reproduce superiority",
                                "statistics": preliminary, "confirmatory": confirmatory,
                                "regression": regression}
        else:
            decision = {"decision": "retain", "reason": "insufficient paired fixtures vs incumbent", "pairs": 0}

    if not promoted and legal and incumbent["policy_id"] != candidate["id"]:
        record_rejection(decisions_dir, candidate_entry=candidate, previous=incumbent,
                         decision={"statistics": decision, "regression": regression, "confirmatory": confirmatory},
                         reason=decision.get("reason", "inconclusive evidence retains the incumbent"),
                         key=f"{candidate['id']}")

    job_receipt = {
        "schema": "pa3-v2-eval-job-receipt-v1",
        "job_id": job["job_id"],
        "tier": job["tier"],
        "scheduled_boundaries": job["scheduled_boundaries"],
        "primary_boundary": job["primary_boundary"],
        "candidate_id": candidate["id"],
        "candidate_clock": candidate["committed_matches"],
        "candidate_source_sha": candidate["source_sha"],
        "candidate_model_digest": candidate["model_digest"],
        "candidate_file_sha256": candidate["file_sha256"],
        "precision_mode": candidate["precision_mode"],
        "precision_stability": candidate["precision_stability"],
        "precision_contract_version": candidate["precision_contract_version"],
        "observation_schema": candidate["observation_schema"],
        "action_schema": candidate["action_schema"],
        "dataset_manifest_hash": V2_DATASET_MANIFEST_HASH,
        "incumbent_id": incumbent["policy_id"],
        "baseline_id": baseline["id"],
        "fixture_seed": MASTER_SEED,
        "status": "success" if legal else "invalid",
        "games": sum(row["matches"] for row in matchups),
        "runtime_seconds": runtime,
        "evaluator_backend": "batched-native",
        "device": "cpu",
        "evaluator_version": EVALUATOR_VERSION,
        "ruleset_version": RULESET_VERSION,
        "decision": decision,
        "regression": regression,
        "confirmatory": confirmatory,
        "promoted": promoted,
        "matchups": matchups,
        "evaluated_at": time.time(),
    }
    jobs_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(jobs_dir / f"{job['job_id'].replace(':', '_')}.json", job_receipt)
    if baseline_receipt is not None:
        (receipts_dir / "baseline").mkdir(parents=True, exist_ok=True)
        atomic_json(receipts_dir / "baseline" / f"{candidate['id']}.json",
                    {"candidate_id": candidate["id"], "fixture_manifest_hash": baseline_receipt["fixture_manifest_hash"],
                     "environment": baseline_receipt.get("environment"), "outcomes": baseline_receipt["outcomes"]})
    return job_receipt


def resource_gate(eval_root, log_path, args):
    free = disk_headroom_bytes(eval_root)
    state_path = eval_root / "state" / "resource_state.json"
    state = read_json(state_path, {}) or {}
    baseline = args.baseline_gps or state.get("baseline_gps")
    recent = production_throughput(log_path) if log_path else None
    if baseline is None and recent:
        baseline = recent
    ratio = (recent / baseline) if (baseline and recent) else None
    record = {
        "baseline_gps": baseline,
        "recent_training_gps": recent,
        "ratio": ratio,
        "free_bytes": free,
        "reserve_bytes": int(args.reserve_gib * (1 << 30)),
        "throttled": bool(ratio is not None and ratio < 0.95),
        "deferred": bool(ratio is not None and ratio < 0.80),
        "caveat": "rolling production throughput; includes ordinary training variability",
        "updated_at": time.time(),
    }
    atomic_json(state_path, record)
    if free < args.reserve_gib * (1 << 30):
        return {"status": "resource_deferred", "reason": f"{args.reserve_gib:g}GiB disk reserve"}, record
    if record["deferred"]:
        return {"status": "resource_deferred", "reason": "production throughput below 80% of baseline"}, record
    return {"status": "ok"}, record


def work_once(*, eval_root, args, engine, environment, master, log_path):
    gate, record = resource_gate(eval_root, log_path, args)
    if gate["status"] != "ok":
        return {**gate, "resource": record}
    if (eval_root / "PAUSE").exists():
        return {"status": "paused"}
    jobs = schedule_jobs(eval_root / "schedule.json", statuses=("pending",))
    jobs = sorted(jobs, key=lambda job: (job["candidate_clock"], job["job_id"]))
    if not jobs:
        return {"status": "idle"}
    job = jobs[0]
    mark_job(eval_root / "schedule.json", job["job_id"], "running", started_at=time.time())
    try:
        receipt = run_job(eval_root=eval_root, job=job, args=args, engine=engine,
                          environment=environment, master=master, log_path=log_path)
    except Exception as error:  # noqa: BLE001 - worker must never crash the service loop
        mark_job(eval_root / "schedule.json", job["job_id"], "failed", error=str(error)[-2000:])
        atomic_json(eval_root / "state" / "worker_state.json",
                    {"status": "error", "job": job["job_id"], "error": str(error)[-2000:],
                     "updated_at": time.time(), "pid": os.getpid()})
        return {"status": "error", "job": job["job_id"], "error": str(error)[-2000:]}
    mark_job(eval_root / "schedule.json", job["job_id"], "completed",
             decided=receipt["decision"]["decision"], games=receipt["games"],
             finished_at=time.time())
    atomic_json(eval_root / "state" / "worker_state.json",
                {"status": "working", "last_job": job["job_id"], "last_decision": receipt["decision"],
                 "games": receipt["games"], "runtime_seconds": receipt["runtime_seconds"],
                 "updated_at": time.time(), "pid": os.getpid()})
    return {"status": "completed", "job": job["job_id"], "tier": job["tier"],
            "games": receipt["games"], "decision": receipt["decision"], "promoted": receipt["promoted"]}


def build_runtime(args):
    import pa3_engine
    torch.set_num_threads(int(args.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    engine = pa3_engine.NativeEngine(str(DATA_DIR), str(TEAMS_PATH), workers=int(args.engine_workers))
    environment = evaluation_environment(str(ENGINE_PATH), str(DATA_DIR), str(TEAMS_PATH),
                                         backend="batched", batch_size=int(args.batch_size),
                                         device="cpu", threads=int(args.threads),
                                         engine_workers=int(args.engine_workers))
    master = master_fixtures(engine, pairs=MASTER_PAIRS, seed=MASTER_SEED)
    return engine, environment, master


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="/home/minidc/eval-league/v2")
    parser.add_argument("--production-log", help="production rank0 log for throughput observation")
    parser.add_argument("--reserve-gib", type=float, default=12.0)
    parser.add_argument("--baseline-gps", type=float)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--engine-workers", type=int, default=2)
    parser.add_argument("--smoke-games", type=int)
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    eval_root = Path(args.eval_root)
    (eval_root / "state").mkdir(parents=True, exist_ok=True)
    (eval_root / "locks").mkdir(exist_ok=True)
    log_path = Path(args.production_log) if args.production_log else None

    lock_path = eval_root / "locks" / "worker.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "already_running"}))
            return 1
        engine = environment = master = None
        while True:
            if (eval_root / "PAUSE").exists():
                result = {"status": "paused"}
            else:
                if engine is None:
                    engine, environment, master = build_runtime(args)
                result = work_once(eval_root=eval_root, args=args, engine=engine,
                                   environment=environment, master=master, log_path=log_path)
            print(json.dumps(result, sort_keys=True), flush=True)
            if args.once:
                break
            time.sleep(1.0 if result["status"] == "completed" else float(args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
