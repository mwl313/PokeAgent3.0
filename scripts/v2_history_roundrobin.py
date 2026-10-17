#!/usr/bin/env python3
"""Isolated historical Round Robin diagnostics (PA3 V2 AHL stages A-C).

The worker is CPU-only, low priority and owns a separate root, registry
snapshot, lock, status file, receipts, fixture seed domain and reports.  It
reuses ``run_matchup_cached`` unchanged and never writes into the formal V2
evaluation root or any training state.
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

try:
    os.nice(19)
except OSError:
    pass

import torch  # noqa: E402

from agent.evaluation.fixed_policy import fixture_manifest_hash  # noqa: E402
from agent.evaluation.historical_league import (  # noqa: E402
    BASIC_FIXTURE_PAIRS,
    DEFAULT_FORMAL_ROOT,
    DEFAULT_RESERVE_GIB,
    DEFAULT_RR_ROOT,
    FINE_FIXTURE_PAIRS,
    RR_PANEL_VERSION,
    RR_SEED,
    RR_SEED_DOMAIN,
    build_pair_plan,
    build_shadow_plan,
    canonical_pair,
    champion_validation,
    disk_free_bytes,
    formal_backlog,
    freeze_registry,
    load_pair_summaries,
    load_snapshot,
    milestone_hook,
    pair_game_count,
    protected_state,
    retire_shadow_plan,
    select_pairs,
    sha256_json,
    summarize_pair,
    write_gate_report,
    write_reports,
    write_status,
)
from agent.evaluation.historical_schedule import (  # noqa: E402
    load_state,
    schedule_for_snapshot,
    summary_plan,
)
from agent.evaluation.identity import sha256_file  # noqa: E402
from agent.evaluation.policy_archive import verify_entry  # noqa: E402
from agent.evaluation.league import evaluation_identity  # noqa: E402
from agent.evaluation.v2_league import (  # noqa: E402
    atomic_json,
    evaluation_environment,
    master_fixtures,
    read_json,
    run_matchup_cached,
)

DATA_DIR = ROOT / "engine" / "data"
TEAMS_PATH = DATA_DIR / "a10-training-teams.json"
ENGINE_PATH = ROOT / "engine" / "python" / "pa3_engine" / "pa3_engine.so"
MAX_ROUNDS = 2000


def _now() -> float:
    return time.time()


def _kst() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _runtime(args):
    import pa3_engine

    if not ENGINE_PATH.is_file():
        raise SystemExit(f"native engine is missing: {ENGINE_PATH}")
    torch.set_num_threads(int(args.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    engine = pa3_engine.NativeEngine(str(DATA_DIR), str(TEAMS_PATH),
                                     workers=int(args.engine_workers))
    environment = evaluation_environment(
        str(ENGINE_PATH), str(DATA_DIR), str(TEAMS_PATH), backend="batched",
        batch_size=int(args.batch_size), device="cpu", threads=int(args.threads),
        engine_workers=int(args.engine_workers))
    master = master_fixtures(engine, pairs=FINE_FIXTURE_PAIRS, seed=int(args.seed))
    return engine, environment, master


def _pair_cache_path(root: Path, pair_key: str) -> Path:
    return root / "receipts" / "pairs" / f"{sha256_json({'pair': pair_key})[:24]}.json"


def _existing_summary(root: Path, pair_key: str, fixture_pairs: int,
                      environment: dict) -> dict | None:
    value = read_json(_pair_cache_path(root, pair_key))
    if not value or value.get("schema") != "pa3-v2-history-pair-v1":
        return None
    if value.get("status") != "success":
        return None
    if int(value.get("fixture_pairs", -1)) != int(fixture_pairs):
        return None
    if value.get("environment") != environment:
        return None
    if value.get("panel_version") != RR_PANEL_VERSION:
        return None
    return value


def _ordered_ids(row, by_id):
    left, right = row["left_id"], row["right_id"]
    left_clock = int(by_id[left]["committed_matches"])
    right_clock = int(by_id[right]["committed_matches"])
    if left_clock > right_clock:
        return left, right
    if right_clock > left_clock:
        return right, left
    return tuple(sorted((left, right)))


def _run_pair(*, args, root, registry_path, receipts_dir, engine, environment,
              master, row, by_id):
    fixtures = master[:2 * int(row["fixture_pairs"])]
    left_id, right_id = _ordered_ids(row, by_id)
    started = _now()
    receipt, cached = run_matchup_cached(
        registry_path, receipts_dir, engine, fixtures, left_id, right_id,
        seed=int(args.seed), max_rounds=MAX_ROUNDS, environment=environment,
        batch_size=int(args.batch_size))
    oriented = dict(row, left_id=left_id, right_id=right_id)
    summary = summarize_pair(
        receipt, oriented, left_entry=by_id[left_id], right_entry=by_id[right_id],
        seed=int(args.seed))
    summary["runtime_seconds"] = round(_now() - started, 3)
    summary["cached"] = bool(cached)
    atomic_json(_pair_cache_path(root, row["pair_key"]), summary)
    return summary


def _base_status(args, meta, plan, selected, completed, games_run):
    return {
        "run_id": sha256_json({
            "snapshot": meta["snapshot_sha256"],
            "seed": int(args.seed),
            "panel": RR_PANEL_VERSION,
            "domain": RR_SEED_DOMAIN,
        })[:24],
        "state": "running",
        "root": str(args.root),
        "formal_root": str(args.formal_root),
        "champion_id": meta["champion_id"],
        "latest_id": meta["latest_id"],
        "registry_snapshot_sha256": meta["snapshot_sha256"],
        "source_registry_sha256": meta["source_registry_sha256"],
        "panel_version": RR_PANEL_VERSION,
        "seed_domain": RR_SEED_DOMAIN,
        "seed": int(args.seed),
        "arithmetic": plan["arithmetic"],
        "pairs_total": len(plan["pairs"]),
        "pairs_selected": len(selected),
        "pairs_completed": completed,
        "games_run": games_run,
        "games_selected_target": sum(row["games"] for row in selected),
        "max_games": args.max_games,
        "fixture_pairs_override": args.fixture_pairs,
        "formal_priority": formal_backlog(args.formal_root),
        "disk_free_bytes": disk_free_bytes(args.root),
        "reserve_gib": float(args.reserve_gib),
        "started_at": _now(),
        "started_at_kst": _kst(),
    }


def _reports(root, entries, champion_id):
    return write_reports(root, entries, champion_id)


def _harness_equivalence(*, args, root, registry_path, entries, engine):
    ids = {entry["id"] for entry in entries}
    formal_root = Path(args.formal_root)
    fixtures = master_fixtures(engine, pairs=BASIC_FIXTURE_PAIRS, seed=20261014)
    expected_hash = fixture_manifest_hash(fixtures)
    targets = []
    for path in sorted((formal_root / "receipts").glob("*.json")):
        receipt = read_json(path)
        if not receipt:
            continue
        if (receipt.get("matches") == 2 * BASIC_FIXTURE_PAIRS
                and receipt.get("seed") == 20261014
                and receipt.get("left_id") in ids and receipt.get("right_id") in ids
                and receipt.get("fixture_manifest_hash") == expected_hash
                and receipt.get("status") == "success"):
            targets.append((path, receipt))
    if not targets:
        raise RuntimeError("no known formal receipt matches the calibration panel")
    path, target = targets[0]
    environment = evaluation_environment(
        str(ENGINE_PATH), str(DATA_DIR), str(TEAMS_PATH), backend="batched",
        batch_size=32, device="cpu", threads=2, engine_workers=2)
    candidate = dict(next(entry for entry in entries if entry["id"] == target["left_id"]),
                     path=verify_entry(registry_path, next(
                         entry for entry in entries if entry["id"] == target["left_id"])))
    opponent = dict(next(entry for entry in entries if entry["id"] == target["right_id"]),
                    path=verify_entry(registry_path, next(
                        entry for entry in entries if entry["id"] == target["right_id"])))
    identity, body = evaluation_identity(
        candidate, opponent, fixtures, seed=20261014, max_rounds=MAX_ROUNDS,
        environment=environment)
    identity_match = identity == target["identity"]
    env_match = body.get("environment") == target.get("environment")
    fixture_match = body.get("fixture_manifest_hash") == target.get("fixture_manifest_hash")
    if not (identity_match and env_match and fixture_match):
        raise RuntimeError(
            "harness equivalence precheck failed: "
            f"identity={identity_match} environment={env_match} fixtures={fixture_match}")
    receipts_dir = root / "equivalence_receipts"
    receipt, cached = run_matchup_cached(
        registry_path, receipts_dir, engine, fixtures, target["left_id"],
        target["right_id"], seed=20261014, max_rounds=MAX_ROUNDS,
        environment=environment, batch_size=32)
    result = {
        "schema": "pa3-v2-history-harness-equivalence-v1",
        "status": "PASS" if (
            receipt["identity"] == target["identity"]
            and receipt["wins"] == target["wins"]
            and receipt["losses"] == target["losses"]
            and receipt["draws"] == target["draws"]
            and receipt["score_mean"] == target["score_mean"]
        ) else "FAIL",
        "formal_receipt": str(path),
        "formal_receipt_identity": target["identity"],
        "rerun_identity": receipt["identity"],
        "identity_match": receipt["identity"] == target["identity"],
        "environment_match": env_match,
        "fixture_manifest_match": fixture_match,
        "wdl_match": (
            receipt["wins"] == target["wins"]
            and receipt["losses"] == target["losses"]
            and receipt["draws"] == target["draws"]
        ),
        "score_match": receipt["score_mean"] == target["score_mean"],
        "formal_wdl": [target["wins"], target["draws"], target["losses"]],
        "rerun_wdl": [receipt["wins"], receipt["draws"], receipt["losses"]],
        "formal_score": target["score_mean"],
        "rerun_score": receipt["score_mean"],
        "games": receipt["matches"],
        "cached": bool(cached),
        "fixture_manifest_hash": expected_hash,
        "environment": environment,
        "checked_at": _now(),
        "checked_at_kst": _kst(),
    }
    atomic_json(root / "equivalence.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("harness equivalence rerun failed")
    return result


def command_freeze(args):
    entries, meta = freeze_registry(args.formal_root, args.root, verify=True)
    plan = build_pair_plan(entries, meta["champion_id"], meta["latest_id"])
    atomic_json(args.root / "pair_plan.json", plan)
    write_status(args.root, {
        "state": "frozen",
        "champion_id": meta["champion_id"],
        "latest_id": meta["latest_id"],
        "registry_snapshot_sha256": meta["snapshot_sha256"],
        "source_registry_sha256": meta["source_registry_sha256"],
        "policies": len(entries),
        "pairs_total": len(plan["pairs"]),
        "arithmetic": plan["arithmetic"],
        "panel_version": RR_PANEL_VERSION,
        "seed_domain": RR_SEED_DOMAIN,
        "formal_priority": formal_backlog(args.formal_root),
    })
    print(json.dumps({"status": "frozen", "policies": len(entries),
                      "arithmetic": plan["arithmetic"]}, sort_keys=True))
    return 0


def command_plan(args):
    entries, meta = load_snapshot(args.root)
    plan = build_pair_plan(entries, meta["champion_id"], meta["latest_id"])
    atomic_json(args.root / "pair_plan.json", plan)
    selected = select_pairs(plan, max_games=args.max_games,
                            fixture_pairs=args.fixture_pairs)
    print(json.dumps({"arithmetic": plan["arithmetic"],
                      "selected_pairs": len(selected),
                      "selected_games": sum(row["games"] for row in selected),
                      "selected": [
                          {key: row[key] for key in
                           ("pair_key", "tier", "games", "priority", "reason")}
                          for row in selected[:12]
                      ]}, indent=2, sort_keys=True))
    return 0


def command_harness(args):
    entries, meta = load_snapshot(args.root)
    engine, _environment, _master = _runtime(args)
    result = _harness_equivalence(
        args=args, root=args.root, registry_path=args.root / "registry.snapshot.json",
        entries=entries, engine=engine)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_run(args):
    entries, meta = load_snapshot(args.root)
    plan = build_pair_plan(entries, meta["champion_id"], meta["latest_id"])
    selected = select_pairs(plan, max_games=args.max_games,
                            fixture_pairs=args.fixture_pairs)
    if not selected:
        raise SystemExit("pair selection is empty")
    status_name = "status.json"
    slice_label = None
    if args.pair_slice:
        try:
            k_str, n_str = str(args.pair_slice).split("/")
            slice_k, slice_n = int(k_str), int(n_str)
        except Exception:
            raise SystemExit("--pair-slice must look like K/N")
        if not (1 <= slice_k <= slice_n):
            raise SystemExit("--pair-slice requires 1 <= K <= N")
        selected = [row for i, row in enumerate(selected) if i % slice_n == (slice_k - 1)]
        slice_label = f"{slice_k}of{slice_n}"
        status_name = f"status.slice-{slice_k}-of-{slice_n}.json"
        if not selected:
            print(json.dumps({"status": "empty_slice", "slice": args.pair_slice}))
            return 0
    by_id = {entry["id"]: entry for entry in entries}
    registry_path = args.root / "registry.snapshot.json"
    receipts_dir = args.root / "receipts"
    engine, environment, master = _runtime(args)
    if args.harness:
        _harness_equivalence(args=args, root=args.root, registry_path=registry_path,
                             entries=entries, engine=engine)
    status = _base_status(args, meta, plan, selected, 0, 0)
    write_status(args.root, status, status_name)
    summaries = load_pair_summaries(args.root)
    completed_ids = {summary["pair_key"] for summary in summaries if summary["status"] == "success"}
    games_run = sum(summary["games"] for summary in summaries)
    run_started = _now()
    stop_reason = None
    for index, row in enumerate(selected, 1):
        backlog = formal_backlog(args.formal_root)
        if backlog["pending_or_running"]:
            stop_reason = "deferred_formal_priority"
            break
        free = disk_free_bytes(args.root)
        if free < float(args.reserve_gib) * (1 << 30):
            stop_reason = "resource_deferred_disk_reserve"
            break
        if (args.root / "PAUSE").exists():
            stop_reason = "operator_pause"
            break
        if row["pair_key"] in completed_ids:
            continue
        status.update({
            "state": "running",
            "current_pair": index,
            "current_pair_key": row["pair_key"],
            "current_reason": row["reason"],
            "pairs_completed": len(completed_ids),
            "games_run": games_run,
            "formal_priority": backlog,
            "disk_free_bytes": free,
        })
        write_status(args.root, status, status_name)
        summary = _run_pair(
            args=args, root=args.root, registry_path=registry_path,
            receipts_dir=receipts_dir, engine=engine, environment=environment,
            master=master, row=row, by_id=by_id)
        completed_ids.add(row["pair_key"])
        games_run += summary["games"]
        status.update({
            "pairs_completed": len(completed_ids),
            "games_run": games_run,
            "last_pair": {
                "pair_key": summary["pair_key"],
                "status": summary["status"],
                "games": summary["games"],
                "left_score": summary["left_score"],
                "receipt_identity": summary["receipt_identity"],
                "cached": summary["cached"],
            },
        })
        write_status(args.root, status, status_name)
        if index % 5 == 0 and not args.no_reports:
            _reports(args.root, entries, meta["champion_id"])
    reports = None if args.no_reports else _reports(args.root, entries, meta["champion_id"])
    elapsed = _now() - run_started
    all_pairs = len(plan["pairs"])
    status.update({
        "state": "completed" if len(completed_ids) >= all_pairs else (
            stop_reason or "resumable"),
        "stop_reason": stop_reason,
        "pairs_completed": len(completed_ids),
        "games_run": games_run,
        "elapsed_seconds": round(elapsed, 3),
        "reports": reports,
        "full_grid_complete": len(completed_ids) >= all_pairs,
        "formal_priority": formal_backlog(args.formal_root),
        "disk_free_bytes": disk_free_bytes(args.root),
        "finished_at": _now(),
        "finished_at_kst": _kst(),
        "pair_slice": args.pair_slice,
    })
    write_status(args.root, status, status_name)
    print(json.dumps({"status": status["state"], "pairs_completed": len(completed_ids),
                      "pairs_total": all_pairs, "games_run": games_run,
                      "elapsed_seconds": round(elapsed, 3),
                      "full_grid_complete": status["full_grid_complete"],
                      "stop_reason": stop_reason,
                      "reports": reports}, indent=2, sort_keys=True))
    return 0


def command_report(args):
    entries, meta = load_snapshot(args.root)
    reports = _reports(args.root, entries, meta["champion_id"])
    summaries = load_pair_summaries(args.root)
    validation = champion_validation(meta["champion_id"], summaries, entries)
    status = read_json(args.root / "status.json", {}) or {}
    run_report = read_json(args.root / "run_report.json", {}) or {}
    complete = run_report.get("state") == "FULL_COMPLETE"
    status.update({
        "state": "completed" if complete else status.get("state", "reported"),
        "pairs_completed": run_report.get("pairs_completed", status.get("pairs_completed", 0)),
        "pairs_total": run_report.get("pairs_total", status.get("pairs_total", 0)),
        "games_run": run_report.get("games_total", status.get("games_run", 0)),
        "full_grid_complete": complete,
        "reports": reports,
        "run_report_state": run_report.get("state"),
        "rating_schema": run_report.get("rating_schema"),
        "champion_validation_summary": {
            "valid_opponents": validation["valid_opponents"],
            "total_games": validation["total_games"],
            "weighted_score": validation["weighted_score"],
        },
    })
    write_status(args.root, status)
    print(json.dumps({"reports": reports,
                      "state": run_report.get("state"),
                      "pairs_completed": run_report.get("pairs_completed"),
                      "games_total": run_report.get("games_total"),
                      "valid_opponents": validation["valid_opponents"],
                      "total_games": validation["total_games"],
                      "weighted_score": validation["weighted_score"]},
                     indent=2, sort_keys=True))
    return 0


def command_incremental(args):
    entries, meta = load_snapshot(args.root)
    summaries = load_pair_summaries(args.root)
    hook = milestone_hook(
        entries, summaries, promotions_dir=Path(args.formal_root) / "best_history" / "promotions",
        interval=args.full_interval)
    atomic_json(args.root / "incremental_plan.json", hook)
    print(json.dumps(hook, indent=2, sort_keys=True))
    return 0


def command_shadow(args):
    entries, meta = load_snapshot(args.root)
    summaries = load_pair_summaries(args.root)
    fixture_identity = {}
    if summaries:
        latest = max(summaries, key=lambda row: row.get("evaluated_at") or "")
        fixture_identity = {
            "fixture_manifest_hash": latest.get("fixture_manifest_hash"),
            "environment": latest.get("environment"),
            "panel_version": latest.get("panel_version"),
        }
    plan = build_shadow_plan(
        entries, summaries, reference_id=meta["champion_id"], root=args.root,
        promotions_dir=Path(args.formal_root) / "best_history" / "promotions",
        fixture_identity=fixture_identity, seed=int(args.seed))
    print(json.dumps({
        "status": plan["status"],
        "shadow_only": plan["shadow_only"],
        "history_rows_target": plan["history_rows_target"],
        "reference_policy_id": plan["reference_policy_id"],
        "opponents": len(plan["opponents"]),
        "weight_sum": sum(row["weight"] for row in plan["opponents"]),
        "plan_sha256": plan["plan_sha256"],
        "path": str(args.root / "opponent_plan.json"),
    }, indent=2, sort_keys=True))
    return 0


def command_schedule(args):
    state = load_state(args.root)
    payload = schedule_for_snapshot(
        args.root, args.formal_root,
        promotions_dir=Path(args.formal_root) / "best_history" / "promotions",
        state=state)
    print(json.dumps(summary_plan(payload), indent=2, sort_keys=True))
    return 0


def command_retire_shadow(args):
    result = retire_shadow_plan(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_gate(args):
    entries, meta = load_snapshot(args.root)
    summaries = load_pair_summaries(args.root)
    status = read_json(args.root / "status.json", {}) or {}
    run_report = read_json(args.root / "run_report.json", {}) or {}
    equivalence = read_json(args.root / "equivalence.json", {}) or {}
    schedule = read_json(args.root / "milestone_schedule.json", {}) or {}
    shadow = read_json(args.root / "opponent_plan.json", {}) or {}
    plan = build_pair_plan(entries, meta["champion_id"], meta["latest_id"])
    protected = protected_state(args.root, args.formal_root)
    backlog = formal_backlog(args.formal_root)
    free = disk_free_bytes(args.root)
    complete = run_report.get("state") == "FULL_COMPLETE"
    stage_a_status = (
        "FULL_COMPLETE" if complete
        else "PARTIAL_RESUMABLE" if summaries
        else "IMPLEMENTED_NOT_RUN")
    stage_a = {
        "status": stage_a_status,
        "registry_snapshot_verified": bool(meta.get("source_registry_sha256")),
        "policies_before_dedup": meta.get("policies_before_dedup"),
        "policies_after_dedup": meta.get("policies_after_dedup"),
        "duplicates": len(meta.get("duplicates", [])),
        "arithmetic": plan["arithmetic"],
        "harness_equivalence": equivalence.get("status", "NOT_RUN"),
        "harness_identity_match": equivalence.get("identity_match"),
        "harness_games": equivalence.get("games"),
        "pairs_completed": run_report.get("pairs_completed", len(summaries)),
        "pairs_total": run_report.get("pairs_total", len(plan["pairs"])),
        "games_run": run_report.get("games_total",
                                    sum(row["games"] for row in summaries)),
        "full_grid_complete": complete,
        "run_report_state": run_report.get("state"),
        "run_report_validation": run_report.get("validation"),
        "rating_schema": run_report.get("rating_schema"),
        "resumable": not complete,
        "formal_alpha_spent": 0,
        "formal_alpha_unchanged": not any(
            row["path"].endswith("alpha_budget.json")
            for row in protected["critical_mismatches"]),
        "formal_writes": len(protected["critical_mismatches"]),
        "schedule_hash_changed_by_formal_discovery": protected["schedule"]["hash_changed"],
        "formal_priority_observed": backlog["pending_or_running"],
        "cpu_only": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "nice": os.nice(0),
    }
    stage_b = {
        "status": schedule.get("state", "IMPLEMENTED_NOT_TRIGGERED"),
        "action": schedule.get("action"),
        "latest_clock": schedule.get("latest_clock"),
        "tier_a_interval": schedule.get("tier_a_interval", 200000),
        "tier_a_first": schedule.get("tier_a_first", 2800000),
        "tier_b_first": schedule.get("tier_b_first", 3800000),
        "tier_b_interval": schedule.get("tier_b_interval", 1000000),
        "tier_b_policy_cap": schedule.get("tier_b_policy_cap", 24),
        "next_tier_a": schedule.get("next_tier_a"),
        "next_tier_b": schedule.get("next_tier_b"),
        "due_jobs": len(schedule.get("due", [])),
        "new_jobs": len(schedule.get("new_jobs", [])),
        "resumed_jobs": len(schedule.get("resumed_jobs", [])),
        "stable_anchor_ids": schedule.get("stable_anchor_ids", []),
        "executed": False,
        "formal_priority_observed": backlog["pending_or_running"],
    }
    retired = shadow.get("status") == "OBSOLETE"
    stage_c = {
        "status": "SHADOW_PLAN_RETIRED" if retired else (
            "SHADOW_PLAN_READY" if shadow else "NOT_GENERATED"),
        "shadow_only": bool(shadow.get("shadow_only")),
        "activatable": False if retired else bool(shadow.get("activatable")),
        "history_rows_target": 0,
        "history_probability": 0.0,
        "opponents": len(shadow.get("opponents", [])),
        "retired_at": shadow.get("retired_at"),
        "provenance": shadow.get("provenance"),
        "activation": "FORBIDDEN",
        "active_training_changed": False,
    }
    stage_d = {
        "status": "NO_GO",
        "reason": ("owner approval, reviewed V2 source/profile/opponent migration, "
                   "isolated A/B pilot and rollback proof are all still required"),
        "training_code_modified": False,
        "ppo_resumed": False,
        "migration_receipt_created": False,
    }
    report = write_gate_report(
        args.root, stage_a=stage_a, stage_b=stage_b, stage_c=stage_c, stage_d=stage_d,
        formal_priority=backlog,
        disk={"free_bytes": free,
              "reserve_bytes": int(float(args.reserve_gib) * (1 << 30)),
              "ok": free >= float(args.reserve_gib) * (1 << 30)},
        protected=protected)
    print(json.dumps({"go_no_go": report["go_no_go"], "stage_a": stage_a,
                      "stage_b": stage_b, "stage_c": stage_c,
                      "protected_unchanged": protected["unchanged"]},
                     indent=2, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_RR_ROOT))
    parser.add_argument("--formal-root", default=str(DEFAULT_FORMAL_ROOT))
    parser.add_argument("--seed", type=int, default=RR_SEED)
    parser.add_argument("--fixture-pairs", type=int, default=None,
                        help="override fixture pairs for every selected pair")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--full-interval", type=int, default=200000)
    parser.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--engine-workers", type=int, default=2)
    parser.add_argument("--pair-slice", default=None,
                        help="K/N: only run slice K of N of the selected pair list "
                             "(disjoint slices let N workers share one receipt store)")
    parser.add_argument("--lock-name", default="worker.lock",
                        help="lock file name under <root>/locks; parallel workers MUST use "
                             "distinct names")
    parser.add_argument("--no-reports", action="store_true",
                        help="skip matrix/ratings regeneration (the coordinator runs "
                             "'report' once at the end)")
    parser.add_argument("--harness", action="store_true",
                        help="run the formal-receipt equivalence check before the pairs")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("plan")
    sub.add_parser("harness")
    sub.add_parser("run")
    sub.add_parser("report")
    sub.add_parser("incremental")
    sub.add_parser("shadow")
    sub.add_parser("schedule")
    sub.add_parser("retire-shadow")
    sub.add_parser("gate")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.root = Path(args.root)
    args.formal_root = Path(args.formal_root)
    (args.root / "locks").mkdir(parents=True, exist_ok=True)
    (args.root / "receipts" / "pairs").mkdir(parents=True, exist_ok=True)
    lock_path = args.root / "locks" / args.lock_name
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"status": "already_running"}))
            return 1
        handlers = {
            "freeze": command_freeze,
            "plan": command_plan,
            "harness": command_harness,
            "run": command_run,
            "report": command_report,
            "incremental": command_incremental,
            "shadow": command_shadow,
            "schedule": command_schedule,
            "retire-shadow": command_retire_shadow,
            "gate": command_gate,
        }
        return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
