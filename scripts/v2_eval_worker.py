#!/usr/bin/env python3
"""Resource-aware V2 evaluation worker (Evaluation League V2).

Single-writer, CPU-only, low priority.  Processes scheduled league jobs against
the incumbent, the frozen baseline and available historical V2 anchors.

V2 behaviour:

* Lightweight milestones are diagnostic only (``DIAGNOSTIC_ONLY``); they can
  never promote.
* A full milestone runs an incumbent-primary panel as *screening* and then, for
  promising candidates, a genuinely independent confirmation ladder.  Only the
  predeclared confirmation rule authorises promotion.
* The campaign-wide false-promotion budget is spent one comparison at a time
  from a preserved alpha ledger (``alpha_budget.json``); lightweight
  diagnostics never consume alpha.

The worker never touches PPO state and defers work when disk or measured
production throughput is at risk.
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
    CONFIRM_PAIRS,
    DEFAULT_CAMPAIGN_START,
    DIAGNOSTIC_ONLY,
    EVALUATOR_VERSION,
    FULL_INITIAL_GAMES,
    INCONCLUSIVE_BUDGET_EXHAUSTED,
    INCUMBENT_AMP_PAIRS,
    INCUMBENT_AMP_START_PAIR,
    INVALID,
    JOB_RECEIPT_SCHEMA_V2,
    LEAGUE_VERSION_V2,
    MASTER_SEED,
    PRIMARY_MAX_GAMES,
    PRIMARY_PAIRS,
    PROMOTED,
    PROMOTION_MIN_PAIRS,
    RETAINED_CONFIRMED_INFERIOR,
    RULESET_VERSION,
    RULESET_VERSION_V2,
    SCREEN_TRIGGER,
    V2_DATASET_MANIFEST_HASH,
    alpha_for_comparison,
    assert_panels_disjoint,
    atomic_json,
    commit_alpha,
    confirmation_rule,
    confirmation_seed_for,
    confirmation_stage_plan,
    disk_headroom_bytes,
    evaluate_regression,
    evaluation_environment,
    initial_incumbent,
    load_alpha_ledger,
    mark_job,
    master_fixtures,
    paired_scores,
    plan_full_panels,
    plan_light_panels,
    primary_screening,
    production_throughput,
    production_work_throughput,
    promote_incumbent,
    read_json,
    record_rejection,
    requeue_stale_jobs,
    run_matchup_cached,
    schedule_jobs,
    snapshot_entries,
    StaleIncumbentError,
)

DATA_DIR = ROOT / "engine" / "data"
TEAMS_PATH = ROOT / "engine" / "data" / "a10-training-teams.json"
ENGINE_PATH = ROOT / "engine" / "python" / "pa3_engine" / "pa3_engine.so"


def _ordinals(entries):
    ordered = sorted(entries, key=lambda entry: (entry["committed_matches"], entry["id"]))
    return {entry["id"]: index + 1 for index, entry in enumerate(ordered)}


def _slice(fixtures, start_pair, games):
    return fixtures[2 * start_pair:2 * (start_pair + games // 2)]


def _matchup_row(panel, receipt, opponent, seed, cached, fixtures):
    return {
        "opponent_id": panel["opponent_id"],
        "phase": panel.get("phase", "primary"),
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
        "fixture_seed": seed,
        "operational_errors": receipt["operational_errors"],
        "illegal_actions": receipt["illegal_actions"],
        "status": receipt["status"],
        "runtime_seconds": receipt["runtime_seconds"],
        "evaluator_backend": "batched-native",
        "device": "cpu",
        "cached": cached,
        "fixture_ids": [fixtures[0].fixture_id, fixtures[-1].fixture_id] if fixtures else None,
    }


def _run_panels(*, eval_root, job, candidate, by_id, engine, environment, master,
                registry_path, receipts_dir, panels, args, phase_label):
    matchups, bodies, signatures = [], [], set()
    for index, panel in enumerate(panels):
        atomic_json(eval_root / "state" / "active.json",
                    {"status": "running", "job": job["job_id"], "tier": job["tier"],
                     "phase": phase_label, "candidate": candidate["id"], "panel": index + 1,
                     "panels": len(panels), "opponent": panel["opponent_id"],
                     "games": panel["games"], "started_at": time.time(), "pid": os.getpid()})
        fixtures = _slice(master, panel["fixture_start_pair"], panel["games"])
        receipt, cached = run_matchup_cached(
            registry_path, receipts_dir, engine, fixtures, candidate["id"], panel["opponent_id"],
            seed=MASTER_SEED, max_rounds=2000, environment=environment, batch_size=args.batch_size,
        )
        opponent = by_id[panel["opponent_id"]]
        matchups.append(_matchup_row(panel, receipt, opponent, MASTER_SEED, cached, fixtures))
        bodies.append(receipt)
        signatures |= {_fixture_signature(fixture) for fixture in fixtures}
    return matchups, bodies, signatures


def _fixture_signature(fixture):
    return (fixture.team_a, fixture.team_b, tuple(fixture.seed), fixture.policy_side)


def light_job_decision(matchups):
    """Lightweight milestones are diagnostic only; they can never promote."""
    legal = bool(matchups) and all(row["status"] == "success" for row in matchups)
    if not legal:
        return {
            "promotion_status": INVALID,
            "promoted": False,
            "reason": "matchup invalidity",
            "decision": {"decision": "invalid", "reason": "matchup invalidity"},
        }
    return {
        "promotion_status": DIAGNOSTIC_ONLY,
        "promoted": False,
        "reason": "lightweight evaluation is diagnostic only and can never promote",
        "decision": {"decision": "diagnostic_only",
                     "reason": "lightweight evaluation is diagnostic only and can never promote",
                     "pairs": sum(row.get("paired_fixtures", 0) for row in matchups)},
    }


def _run_confirmation(*, eval_root, job, candidate, incumbent, engine, environment, receipts_dir,
                      registry_path, alpha_total, args):
    """Run the predeclared, genuinely independent confirmation ladder."""
    conf_seed = confirmation_seed_for(candidate["id"])
    conf_fixtures = master_fixtures(engine, pairs=CONFIRM_PAIRS, seed=conf_seed)
    stages = confirmation_stage_plan()
    pooled, rows = [], []
    promoted = False
    final = None
    for stage in stages:
        fixtures = conf_fixtures[2 * stage["first_pair"]:2 * stage["cumulative_pairs"]]
        atomic_json(eval_root / "state" / "active.json",
                    {"status": "confirmatory", "job": job["job_id"], "candidate": candidate["id"],
                     "opponent": incumbent["policy_id"], "stage": stage["stage"],
                     "cumulative_games": stage["cumulative_games"],
                     "started_at": time.time(), "pid": os.getpid()})
        receipt, cached = run_matchup_cached(
            registry_path, receipts_dir, engine, fixtures, candidate["id"], incumbent["policy_id"],
            seed=conf_seed, max_rounds=2000, environment=environment, batch_size=args.batch_size,
        )
        pooled.extend(receipt["outcomes"])
        result = confirmation_rule(paired_scores(pooled), alpha_total=alpha_total, stage=stage)
        result.update({"cumulative_games": stage["cumulative_games"],
                       "cumulative_pairs": stage["cumulative_pairs"],
                       "identity": receipt["identity"], "cached": cached,
                       "fixture_manifest_hash": receipt["fixture_manifest_hash"]})
        rows.append(result)
        final = result
        if result["decision"] == "promote":
            promoted = True
            break
        if result["decision"] == "retain_inferior":
            break
    return {
        "seed": conf_seed,
        "stages": rows,
        "final": final,
        "promoted": promoted,
        "games_consumed": sum(row["cumulative_games"] for row in rows[-1:]) if rows else 0,
        "fixtures": conf_fixtures,
    }


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

    tier = job["tier"]
    started = time.time()
    matchups, bodies = [], []
    primary_signatures = set()
    screen = {"status": "invalid", "reason": "matchup invalidity"}
    regression = {"status": "insufficient_coverage"}
    confirmation = None
    promotion_status = INVALID
    reason = "matchup invalidity"
    decision = {"decision": "invalid", "reason": reason}
    alpha_info = None
    promoted = False
    baseline_receipt = None
    escalated = False

    if tier != "full":
        panels = plan_light_panels(candidate_id=candidate["id"], incumbent_id=incumbent["policy_id"],
                                   baseline_id=baseline["id"], anchors=anchors)
        matchups, bodies, primary_signatures = _run_panels(
            eval_root=eval_root, job=job, candidate=candidate, by_id=by_id, engine=engine,
            environment=environment, master=master, registry_path=registry_path,
            receipts_dir=receipts_dir, panels=panels, args=args, phase_label="diagnostic")
        light = light_job_decision(matchups)
        promotion_status = light["promotion_status"]
        reason = light["reason"]
        decision = light["decision"]
    else:
        panels = plan_full_panels(candidate_id=candidate["id"], incumbent_id=incumbent["policy_id"],
                                  baseline_id=baseline["id"], anchors=anchors)
        matchups, bodies, primary_signatures = _run_panels(
            eval_root=eval_root, job=job, candidate=candidate, by_id=by_id, engine=engine,
            environment=environment, master=master, registry_path=registry_path,
            receipts_dir=receipts_dir, panels=panels, args=args, phase_label="primary")
        non_incumbent = [row for row in matchups if row["opponent_id"] != incumbent["policy_id"]]
        baseline_receipt = next((body for body, row in zip(bodies, matchups)
                                 if row["opponent_id"] == baseline["id"]), None)
        legal = all(row["status"] == "success" for row in matchups)
        incumbent_receipt = next((body for body, row in zip(bodies, matchups)
                                  if row["opponent_id"] == incumbent["policy_id"]), None)

        if not legal:
            promotion_status, reason = INVALID, "matchup invalidity"
            decision = {"decision": "invalid", "reason": reason}
        elif incumbent["policy_id"] == candidate["id"]:
            promotion_status, reason = RETAINED_CONFIRMED_INFERIOR, "candidate is the current incumbent"
            decision = {"decision": "retain", "reason": reason, "pairs": 0}
        elif incumbent_receipt is None or incumbent_receipt["paired_fixtures"] < PROMOTION_MIN_PAIRS:
            promotion_status = INCONCLUSIVE_BUDGET_EXHAUSTED
            reason = "insufficient paired fixtures vs incumbent"
            decision = {"decision": "retain", "reason": reason, "pairs": 0}
        else:
            if baseline_receipt is not None and incumbent["policy_id"] != baseline["id"]:
                previous = read_json(receipts_dir / "baseline" / f"{incumbent['policy_id']}.json")
                if previous is not None:
                    regression = evaluate_regression(
                        candidate_baseline_outcomes=baseline_receipt["outcomes"],
                        incumbent_baseline_outcomes=previous.get("outcomes", []),
                    )
            primary_outcomes = list(incumbent_receipt["outcomes"])
            screen = primary_screening(paired_scores(primary_outcomes))
            if screen["status"] == "amplify" and INCUMBENT_AMP_PAIRS > 0:
                escalated = True
                amp_panel = {"opponent_id": incumbent["policy_id"], "phase": "primary_amplify",
                             "games": PRIMARY_MAX_GAMES - FULL_INITIAL_GAMES,
                             "fixture_start_pair": INCUMBENT_AMP_START_PAIR}
                amp_matchups, amp_bodies, amp_signatures = _run_panels(
                    eval_root=eval_root, job=job, candidate=candidate, by_id=by_id, engine=engine,
                    environment=environment, master=master, registry_path=registry_path,
                    receipts_dir=receipts_dir, panels=[amp_panel], args=args,
                    phase_label="primary-amplify")
                if amp_matchups and amp_matchups[0]["status"] == "success":
                    matchups.extend(amp_matchups)
                    primary_signatures |= amp_signatures
                    primary_outcomes.extend(amp_bodies[0]["outcomes"])
                    screen = primary_screening(paired_scores(primary_outcomes))
                else:
                    legal = False
                    promotion_status, reason = INVALID, "primary amplification failed"

            if not legal:
                decision = {"decision": "invalid", "reason": reason}
            elif regression.get("status") == "regression_warning":
                promotion_status = RETAINED_CONFIRMED_INFERIOR
                reason = "material regression against the frozen baseline"
                decision = {"decision": "retain", "reason": reason,
                            "statistics": screen, "regression": regression}
            elif screen["status"] == "confirmed_inferior":
                promotion_status = RETAINED_CONFIRMED_INFERIOR
                reason = "primary evidence confirms inferiority against the incumbent"
                decision = {"decision": "retain", "reason": reason,
                            "statistics": screen, "regression": regression}
            elif screen["status"] != "promising":
                promotion_status = INCONCLUSIVE_BUDGET_EXHAUSTED
                reason = ("primary screening did not reach the escalation trigger "
                          "within the maximum primary budget")
                decision = {"decision": "retain", "reason": reason,
                            "statistics": screen, "regression": regression}
            else:
                ledger = load_alpha_ledger(eval_root)
                alpha_info = alpha_for_comparison(ledger)
                if alpha_info["exhausted"]:
                    promotion_status = INCONCLUSIVE_BUDGET_EXHAUSTED
                    reason = "campaign-wide promotion alpha budget exhausted"
                    decision = {"decision": "retain", "reason": reason,
                                "statistics": screen, "regression": regression}
                else:
                    confirmation = _run_confirmation(
                        eval_root=eval_root, job=job, candidate=candidate, incumbent=incumbent,
                        engine=engine, environment=environment, receipts_dir=receipts_dir,
                        registry_path=registry_path, alpha_total=alpha_info["alpha"], args=args)
                    assert_panels_disjoint(
                        [fixture for fixture in master[:2 * (INCUMBENT_AMP_START_PAIR + INCUMBENT_AMP_PAIRS)]],
                        confirmation["fixtures"])
                    final_stage = confirmation["final"]
                    if confirmation["promoted"]:
                        promotion_status = PROMOTED
                        commit_alpha(eval_root, candidate_id=candidate["id"],
                                     index=alpha_info["index"], alpha=alpha_info["alpha"],
                                     decision="promote")
                        support = [row["identity"] for row in matchups]
                        support += [row["identity"] for row in confirmation["stages"]]
                        try:
                            promote_incumbent(
                                best_path, history_dir, candidate_entry=candidate, previous=incumbent,
                                expected_previous_id=incumbent["policy_id"],
                                decision={"primary": screen, "confirmation": confirmation["stages"],
                                          "regression": regression, "alpha": alpha_info},
                                supporting=support, ruleset_version=RULESET_VERSION_V2,
                                reason=("adaptive primary screening + independent "
                                        "confirmation promotion (ruleset v2)"))
                            promoted = True
                        except StaleIncumbentError as error:
                            promotion_status = RETAINED_CONFIRMED_INFERIOR
                            reason = f"stale incumbent during promotion: {error}"
                        decision = {"decision": "promote" if promoted else "retain",
                                    "reason": reason if not promoted else "confirmed superiority",
                                    "statistics": screen,
                                    "confirmation": confirmation["stages"], "regression": regression}
                    else:
                        commit_alpha(eval_root, candidate_id=candidate["id"],
                                     index=alpha_info["index"], alpha=alpha_info["alpha"],
                                     decision="retain")
                        if final_stage is not None and final_stage.get("upper", 1.0) < 0.5:
                            promotion_status = RETAINED_CONFIRMED_INFERIOR
                            reason = "confirmation confirms inferiority against the incumbent"
                        else:
                            promotion_status = INCONCLUSIVE_BUDGET_EXHAUSTED
                            reason = ("confirmation budget exhausted without a "
                                      "superiority conclusion")
                        decision = {"decision": "retain", "reason": reason,
                                    "statistics": screen, "confirmation": confirmation["stages"],
                                    "regression": regression}

    runtime = time.time() - started
    if (not promoted and promotion_status != DIAGNOSTIC_ONLY and promotion_status != INVALID
            and incumbent["policy_id"] != candidate["id"]):
        record_rejection(decisions_dir, candidate_entry=candidate, previous=incumbent,
                         decision={"statistics": decision, "regression": regression,
                                   "confirmation": confirmation["stages"] if confirmation else None,
                                   "promotion_status": promotion_status,
                                   "alpha": alpha_info},
                         reason=reason, key=f"{candidate['id']}",
                         ruleset_version=RULESET_VERSION_V2, promotion_status=promotion_status)

    job_receipt = {
        "schema": JOB_RECEIPT_SCHEMA_V2,
        "job_id": job["job_id"],
        "tier": tier,
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
        "status": "success" if (matchups and all(row["status"] == "success" for row in matchups)) else "invalid",
        "games": sum(row["matches"] for row in matchups),
        "runtime_seconds": runtime,
        "evaluator_backend": "batched-native",
        "device": "cpu",
        "evaluator_version": EVALUATOR_VERSION,
        "league_version": LEAGUE_VERSION_V2,
        "ruleset_version": RULESET_VERSION_V2,
        "promotion_status": promotion_status,
        "escalated": escalated,
        "alpha_allocation": alpha_info,
        "primary_screening": screen,
        "confirmation": confirmation["stages"] if confirmation else None,
        "confirmation_seed": confirmation["seed"] if confirmation else None,
        "decision": decision,
        "regression": regression,
        "confirmatory": confirmation["final"] if confirmation else None,
        "promoted": promoted,
        "matchups": matchups,
        "evaluated_at": time.time(),
    }
    jobs_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(jobs_dir / f"{job['job_id'].replace(':', '_')}.json", job_receipt)
    atomic_json(eval_root / "state" / "active.json",
                {"status": "idle", "last_job": job["job_id"], "pid": os.getpid(),
                 "updated_at": time.time()})
    if baseline_receipt is not None:
        (receipts_dir / "baseline").mkdir(parents=True, exist_ok=True)
        atomic_json(receipts_dir / "baseline" / f"{candidate['id']}.json",
                    {"candidate_id": candidate["id"], "fixture_manifest_hash": baseline_receipt["fixture_manifest_hash"],
                     "environment": baseline_receipt.get("environment"), "outcomes": baseline_receipt["outcomes"]})
    return job_receipt


def production_log_age(log_path):
    try:
        return max(0.0, time.time() - Path(log_path).stat().st_mtime)
    except OSError:
        return None


def publish_gate_state(eval_root, gate, record):
    """Mirror the gate verdict into state/active.json so status tools cannot show a
    stale "idle" while work is actually being deferred."""
    path = eval_root / "state" / "active.json"
    current = read_json(path, {}) or {}
    current.update({
        "status": gate["status"],
        "gate_reason": gate.get("reason"),
        "gate": {key: record.get(key) for key in
                 ("metric", "ratio", "stale", "deferred", "throttled", "production_log_age_s")},
        "updated_at": time.time(),
    })
    atomic_json(path, current)


def resource_gate(eval_root, log_path, args):
    free = disk_headroom_bytes(eval_root)
    state_path = eval_root / "state" / "resource_state.json"
    state = read_json(state_path, {}) or {}
    recent_rows, detail = production_work_throughput(log_path, limit=8) if log_path else (None, {})
    baseline_rows = (getattr(args, "baseline_rows_per_s", None)
                     or state.get("baseline_rows_per_s"))
    if baseline_rows is None and recent_rows:
        baseline_rows = recent_rows
    ratio = (recent_rows / baseline_rows) if (baseline_rows and recent_rows) else None
    age = production_log_age(log_path) if log_path else None
    stale = bool(age is not None and age > float(getattr(args, "stale_after_s", 900.0)))
    mature = bool(detail.get("iterations", 0) >= 2)
    record = {
        "metric": "learner-rows-per-second-v1",
        "baseline_rows_per_s": baseline_rows,
        "recent_rows_per_s": recent_rows,
        "recent_rows_per_iteration": detail.get("rows_per_iteration"),
        "ratio": ratio,
        "mature_window": mature,
        "detail": detail,
        "baseline_gps_legacy": state.get("baseline_gps"),
        "recent_matches_per_s": detail.get("matches_per_second"),
        "free_bytes": free,
        "reserve_bytes": int(args.reserve_gib * (1 << 30)),
        "production_log_age_s": None if age is None else round(age, 1),
        "stale": stale,
        "throttled": bool(ratio is not None and ratio < 0.95 and not stale),
        "deferred": bool(ratio is not None and ratio < 0.80 and not stale and mature),
        "caveat": ("work-normalised (learner rows/s), invariant to game length; "
                   "gate is skipped while the production log is stale (training idle)"),
        "updated_at": time.time(),
    }
    atomic_json(state_path, record)
    if free < args.reserve_gib * (1 << 30):
        gate = {"status": "resource_deferred",
                "reason": f"{args.reserve_gib:g}GiB disk reserve"}
        publish_gate_state(eval_root, gate, record)
        return gate, record
    if stale:
        gate = {"status": "ok", "note": "production log stale: training idle, throughput gate skipped"}
        publish_gate_state(eval_root, gate, record)
        return gate, record
    if record["deferred"]:
        gate = {"status": "resource_deferred",
                "reason": (f"learner rows/s at {ratio:.3f} of baseline "
                           f"({recent_rows:.1f} vs {baseline_rows:.1f} rows/s over "
                           f"{detail.get('iterations', 0)} iterations)")}
        publish_gate_state(eval_root, gate, record)
        return gate, record
    gate = {"status": "ok"}
    publish_gate_state(eval_root, gate, record)
    return gate, record


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
             decided=receipt["decision"]["decision"], promotion_status=receipt["promotion_status"],
             games=receipt["games"], finished_at=time.time())
    atomic_json(eval_root / "state" / "worker_state.json",
                {"status": "working", "last_job": job["job_id"], "last_decision": receipt["decision"],
                 "last_promotion_status": receipt["promotion_status"],
                 "ruleset_version": receipt["ruleset_version"],
                 "games": receipt["games"], "runtime_seconds": receipt["runtime_seconds"],
                 "updated_at": time.time(), "pid": os.getpid()})
    return {"status": "completed", "job": job["job_id"], "tier": job["tier"],
            "games": receipt["games"], "promotion_status": receipt["promotion_status"],
            "decision": receipt["decision"], "promoted": receipt["promoted"]}


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
    master = master_fixtures(engine, pairs=PRIMARY_PAIRS, seed=MASTER_SEED)
    return engine, environment, master


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="/home/minidc/eval-league/v2")
    parser.add_argument("--production-log", help="production rank0 log for throughput observation")
    parser.add_argument("--reserve-gib", type=float, default=12.0)
    parser.add_argument("--baseline-gps", type=float,
                        help="legacy matches/s baseline (observability only)")
    parser.add_argument("--baseline-rows-per-s", type=float,
                        help="learner-rows/s baseline for the throughput gate; "
                             "adopted from the first observation when omitted")
    parser.add_argument("--stale-after-s", type=float, default=900.0,
                        help="skip the throughput gate when the production log has not "
                             "been written for this long (training idle)")
    parser.add_argument("--gate-log-interval", type=float, default=300.0,
                        help="re-log an unchanged deferral at most this often")
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
        requeued = requeue_stale_jobs(eval_root / "schedule.json")
        if requeued:
            print(json.dumps({"status": "requeued", "jobs": requeued}, sort_keys=True), flush=True)
        engine = environment = master = None
        last_gate = {"status": None, "printed_at": 0.0}
        while True:
            if (eval_root / "PAUSE").exists():
                result = {"status": "paused"}
            else:
                if engine is None:
                    engine, environment, master = build_runtime(args)
                result = work_once(eval_root=eval_root, args=args, engine=engine,
                                   environment=environment, master=master, log_path=log_path)
            now = time.time()
            deferral = str(result.get("status", "")).startswith("resource_deferred")
            if (not deferral
                    or result["status"] != last_gate["status"]
                    or (now - last_gate["printed_at"]) >= float(args.gate_log_interval)):
                print(json.dumps(result, sort_keys=True), flush=True)
                last_gate = {"status": result["status"], "printed_at": now}
            if args.once:
                break
            time.sleep(1.0 if result["status"] == "completed" else float(args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
