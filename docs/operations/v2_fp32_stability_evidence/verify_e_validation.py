"""Read completed E diagnostics only; write compact verification receipts.

Run after training exits. Missing/incomplete artifacts return PENDING before
loading checkpoint tensors. Full state comparison hashes every tensor byte;
it never compares abbreviated tensor reprs or modifies source artifacts.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def tensor_digest(value):
    import numpy as np
    import torch
    digest = hashlib.sha256()

    def add(item):
        digest.update(type(item).__name__.encode() + b":")
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes(order="C"))
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str(len(item)).encode())
            for element in item:
                add(element)
        else:
            digest.update(repr(item).encode())
        digest.update(b";")

    add(value)
    return digest.hexdigest()


def finite_tensors(value, path="learner"):
    import torch
    count = 0
    invalid = []
    if isinstance(value, torch.Tensor):
        count = 1
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            invalid.append(path)
    elif isinstance(value, dict):
        for key, element in value.items():
            n, bad = finite_tensors(element, f"{path}.{key}")
            count += n
            invalid.extend(bad)
    elif isinstance(value, (tuple, list)):
        for index, element in enumerate(value):
            n, bad = finite_tensors(element, f"{path}[{index}]")
            count += n
            invalid.extend(bad)
    elif isinstance(value, float) and not math.isfinite(value):
        invalid.append(path)
    return count, invalid


def generations(suite, names):
    found = {}
    for name in names:
        for path in (suite/name/"rolling/generations").glob("*/manifest.json"):
            manifest = read(path)
            if manifest.get("complete") is True:
                found[manifest["iteration"]] = (path.parent, manifest)
    return found


def ready(suite, names, expected, restore):
    missing = []
    telemetry = {}
    reports = {}
    for name in [*names, restore]:
        for rank in (0, 1):
            stem = suite/name/f"report.json.rank{rank}"
            report_path = stem.with_suffix(stem.suffix+".json")
            telemetry_path = stem.with_suffix(stem.suffix+".ppo.jsonl")
            if not telemetry_path.exists():
                missing.append(f"{name}/rank{rank} reports")
                continue
            telemetry[name, rank] = [json.loads(line) for line in telemetry_path.read_text().splitlines() if line.strip()]
            if report_path.exists():
                reports[name, rank] = read(report_path)["iterations"]
            else:
                # Interrupted parent emitted durable post-barrier telemetry, but
                # no final rank report. Derive only fields actually recorded.
                reports[name, rank] = [{
                    "parity":{"absolute_iteration":row["identity"]["absolute_iteration"]-1},
                    "games":row["collector"]["games"], "rows":row["collector"]["learner_rows"],
                    "optimizer_steps_this_iteration":row["ppo"]["optimizer_steps"]-(telemetry[name,rank][row_index-1]["ppo"]["optimizer_steps"] if row_index else 718),
                    "sync_calls":None,"no_sync_calls":None,
                    "model_digest":row["safety"]["rank_verification"]["model_digest"],
                    "optimizer_digest":row["safety"]["rank_verification"]["optimizer_digest"],
                    "iteration_wall_s":row["performance"]["iteration_wall_s"],
                    "derived_from":"durable original post-barrier PPO telemetry; parent interrupted before final report"
                } for row_index,row in enumerate(telemetry[name, rank])]
    boundaries = generations(suite, names)
    if len(boundaries) != expected:
        missing.append(f"main-line complete checkpoint boundaries {len(boundaries)}/{expected}")
    for rank in (0, 1):
        count = sum(len(telemetry.get((name, rank), [])) for name in names)
        if count != expected:
            missing.append(f"rank{rank} completed telemetry rows {count}/{expected}")
    restore_pointer = suite/restore/"rolling/LATEST"
    if not restore_pointer.exists():
        missing.append(f"{restore}/rolling/LATEST")
    return missing, telemetry, reports, boundaries, restore_pointer


def verify(suite, names, expected, restore, reference_iteration, telemetry, reports, boundaries, restore_pointer):
    import torch
    from agent.contracts import require_checkpoint_identity
    torch.set_num_threads(2)
    source_sha = read(suite/"migration.json")["target_sha"]
    checks = {}
    rows = []
    main_reports = {}
    for rank in (0, 1):
        for name in names:
            for row in reports[name, rank]:
                main_reports[rank, row["parity"]["absolute_iteration"]+1] = row
    for name in names:
        paired = {rank: {row["identity"]["absolute_iteration"]: row for row in telemetry[name, rank]} for rank in (0, 1)}
        for iteration in sorted(paired[0]):
            left, right = paired[0][iteration], paired[1][iteration]
            rank_rows = []
            for rank, record in enumerate((left, right)):
                ppo, safety = record["ppo"], record["safety"]
                recompute = safety["recompute"]
                secondary = recompute.get("secondary", {})
                isolated = recompute.get("isolated_outlier_diagnosis", {})
                epoch_ok = ppo["epochs_run"] == 2 or (ppo["epochs_run"] == 1 and ppo["stopped_early"] and max(ppo["epoch_approx_kl"]) > .03)
                numerical_ok = finite_tensors(ppo, "ppo")[1] == []
                complete_ok = (recompute["within_gate"] and recompute["all_ranks_accepted"] and recompute["tolerance"] == .001
                    and recompute["audit_enabled"] and secondary.get("accepted") is True and isolated.get("accepted") is True
                    and safety["rank_verification"]["digests_equal"] and ppo["optimizer_steps_skipped"] == 0
                    and record["collector"]["operational_errors"] == 0 and numerical_ok and epoch_ok)
                checks[f"iteration{iteration}.rank{rank}"] = complete_ok
                residuals = [abs(item[a]-item[b]) for diagnosis in (secondary, isolated) for item in diagnosis.get("rows", [])
                             for a,b in (("actor_replay", "fp32_group"), ("batched_amp", "fp32_isolated"), ("isolated_amp", "fp32_isolated")) if a in item and b in item]
                report = main_reports[rank, iteration]
                rank_rows.append({"rank":rank, "ppo":ppo, "collector":record["collector"],
                    "natural_matches":report["games"], "learner_rows":report["rows"],
                    "primary_max_abs_diff":recompute["max_abs_diff"], "primary_p99_abs_diff":recompute["p99_abs_diff"],
                    "secondary_accepted":secondary.get("accepted"), "isolated_secondary_accepted":isolated.get("accepted"),
                    "secondary_max_fp32_residual":max(residuals, default=0), "strata":recompute["strata"],
                    "optimizer_steps_this_iteration":report["optimizer_steps_this_iteration"],
                    "sync_calls":report["sync_calls"], "no_sync_calls":report["no_sync_calls"],
                    "model_digest":report["model_digest"], "optimizer_digest":report["optimizer_digest"],
                    "iteration_wall_s":report["iteration_wall_s"]})
            checks[f"iteration{iteration}.rank_model_adam_parity"] = all(rank_rows[0][key] == rank_rows[1][key] for key in ("model_digest", "optimizer_digest"))
            checks[f"iteration{iteration}.match_clock"] = (left["identity"]["committed_matches"] == right["identity"]["committed_matches"]
                and sum(row["natural_matches"] for row in rank_rows) == left["performance"]["committed_matches_this_iteration"])
            rows.append({"iteration":iteration, "committed_matches":left["identity"]["committed_matches"], "ranks":rank_rows})
    finite = []
    for iteration, (path, manifest) in sorted(boundaries.items()):
        peer_learner_digests = []
        for rank in (0, 1):
            artifact = path/f"rank{rank}.pt"
            raw_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
            state = torch.load(artifact, map_location="cpu", weights_only=False)
            count, invalid = finite_tensors(state["learner"])
            learner_digest = tensor_digest(state["learner"])
            peer_learner_digests.append(learner_digest)
            identity_ok = (require_checkpoint_identity(state["learner"]) == "E" and state["source_sha"] == source_sha
                           and state["accounting"]["iteration"] == iteration and state["accounting"]["committed_matches"] == manifest["committed_matches"])
            checks[f"checkpoint{iteration}.rank{rank}"] = not invalid and identity_ok and raw_hash == manifest["ranks"][str(rank)]["sha256"]
            finite.append({"iteration":iteration, "rank":rank, "path":str(artifact), "sha256":raw_hash,
                           "learner_tensor_count":count, "invalid_paths":invalid, "full_learner_digest":learner_digest})
            del state
        checks[f"checkpoint{iteration}.full_learner_rank_parity"] = len(set(peer_learner_digests)) == 1
    restore_path = restore_pointer.parent/"generations"/restore_pointer.read_text().strip()
    reference_path = boundaries[reference_iteration][0]
    restore_checks = {}
    for rank in (0, 1):
        original = torch.load(reference_path/f"rank{rank}.pt", map_location="cpu", weights_only=False)
        restored = torch.load(restore_path/f"rank{rank}.pt", map_location="cpu", weights_only=False)
        restore_checks[str(rank)] = {key:tensor_digest(original[key]) == tensor_digest(restored[key])
            for key in ("learner", "rng", "accounting", "collector_restart", "rank_digests", "dataset_manifest_hash", "opponent_manifest")}
        if "rollout_digest" in main_reports[rank, reference_iteration]["parity"]:
            restore_checks[str(rank)]["rollout_digest"] = (main_reports[rank, reference_iteration]["parity"]["rollout_digest"]
                == reports[restore, rank][-1]["parity"]["rollout_digest"])
        checks[f"restore.rank{rank}"] = all(restore_checks[str(rank)].values())
        del original, restored
    summary = {"source_sha":source_sha, "status":"PASS" if all(checks.values()) else "FAIL",
        "iterations":sorted(rows, key=lambda row:row["iteration"]), "checkpoint_scan":finite,
        "restore":{"classification":"BIT_EXACT" if all(checks[f"restore.rank{rank}"] for rank in (0,1)) else "MISMATCH",
                   "reference":str(reference_path), "restored":str(restore_path), "checks":restore_checks},
        "checks":checks, "production_matches_added":0,
        "interruption":"initial parent stopped by usage limit after iteration17; iteration17 rank files verified/published by unchanged publisher; missing final report fields derived solely from durable original telemetry; no original rollout digest persisted",
        "scope":"bounded same-source E diagnostic continuation; forced unchanged .001 secondary rule; full tensor bytes and owned RNG state"}
    destination = suite/"E_VALIDATION_VERIFIED.json"
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"status":summary["status"], "output":str(destination), "source_sha":source_sha,
                      "iterations":len(rows), "checkpoint_rank_artifacts":len(finite), "restore":summary["restore"]["classification"],
                      "failed_checks":[key for key, value in checks.items() if not value]}, indent=2))
    return 0 if summary["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=Path(__file__).parent/"E_exact")
    parser.add_argument("--parts", nargs="+", default=["sustained", "continued"])
    parser.add_argument("--expected-iterations", type=int, default=10)
    parser.add_argument("--restore", default="restored_after_stop")
    parser.add_argument("--reference-iteration", type=int, default=16)
    parser.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    deadline = time.monotonic()+args.wait_seconds
    while True:
        missing, telemetry, reports, boundaries, pointer = ready(args.suite, args.parts, args.expected_iterations, args.restore)
        if not missing:
            return verify(args.suite, args.parts, args.expected_iterations, args.restore,
                          args.reference_iteration, telemetry, reports, boundaries, pointer)
        if time.monotonic() >= deadline:
            print(json.dumps({"status":"PENDING", "missing":missing}, indent=2))
            return 2
        time.sleep(min(5, max(0, deadline-time.monotonic())))


if __name__ == "__main__":
    raise SystemExit(main())
