#!/usr/bin/env python3
"""Real-policy PA3-8M end-to-end benchmark with stage accounting.

This is the Phase 0 measurement instrument of the optimization master plan:
same script for every before/after comparison, explicit configuration in a run
manifest, CUDA-event timing for the GPU section, explicit synchronization at
wall boundaries, and the report schema from the plan (section 5.3).

Modes:
    collect  - actor-only collection (comparable to the historical 16.4 games/s)
    full     - collection + GAE + 4 PPO epochs + checkpoint (honest all-in)

Usage:
    PYTHONPATH=engine/python:. .venv/bin/python scripts/bench_pa3_end_to_end.py \
        --games 2048 --envs 1024 --workers 16 --repeats 3 \
        --observations perview --report runs/perf/baseline.json

Nothing here starts training; it is a bounded measurement of the collection and
PPO machinery on the frozen training pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine", "python"))

import pa3_engine  # noqa: E402
from agent.model.config import PA3Config  # noqa: E402
from agent.model.pa3_model import PA3Model  # noqa: E402
from agent.ppo.config import PPOConfig  # noqa: E402
from agent.ppo.learner import PPOLearner  # noqa: E402
from agent.train.native_collector import NativeCollector, NativeCollectorConfig  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2048, help="natural matches per repeat")
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["collect", "full"], default="collect")
    parser.add_argument("--observations", choices=["perview", "fixed"], default="perview")
    parser.add_argument("--candidate-wire", choices=["tuples", "packed"], default="tuples")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--inference-mode", action="store_true")
    parser.add_argument("--data", default=os.path.join(ROOT, "engine", "data"))
    parser.add_argument("--teams", default=os.path.join(ROOT, "engine", "data", "training-teams.json"))
    parser.add_argument("--report", default=os.path.join(ROOT, "runs", "perf", "report.json"))
    parser.add_argument("--manifest", default="", help="run_manifest path (defaults next to --report)")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--tag", default="", help="free-form label recorded in the report")
    parser.add_argument("--profile-trace", default="",
                        help="write a PyTorch profiler Chrome trace of the learner update to this path")
    parser.add_argument("--recompute-tolerance", type=float, default=None,
                        help="declared mixed-precision parity gate (default 1e-4 fp32 / 1e-3 fp16)")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover - diagnostics only
        return None


def git_diff_hash():
    """Identify a dirty tree precisely instead of only flagging it."""
    try:
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout
        untracked = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                   capture_output=True, check=True).stdout
        return hashlib.sha256(diff + untracked).hexdigest()
    except Exception:  # pragma: no cover - diagnostics only
        return None


def gpu_inventory():
    rows = []
    query = "index,pci.bus_id,name,driver_version,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
        for line in out.strip().splitlines():
            index, bus, name, driver, power, temp, util, used, total = [part.strip() for part in line.split(",")]
            rows.append({
                "index": int(index), "pci_bus_id": bus, "name": name, "driver_version": driver,
                "power_limit_w": float(power), "temperature_c": int(temp), "utilization_pct": int(util),
                "memory_used_mib": int(used), "memory_total_mib": int(total),
            })
    except Exception as error:  # pragma: no cover - diagnostics only
        rows.append({"error": str(error)})
    return rows


def dataset_sha():
    manifest = os.path.join(ROOT, "data", "teams", "mb-mc-v3-userteam-all-train", "manifest.json")
    return {"dataset_id": "mb-mc-v3-userteam-all-train",
            "dataset_manifest_sha256": sha256_file(manifest) if os.path.exists(manifest) else None}


def build_runner(args, seed_offset):
    torch.manual_seed(args.seed + seed_offset)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Events, allocations and the RNG generator must live on the rank's own
        # GPU; without this, rank 1 would record events on device 0.
        torch.cuda.set_device(device)
    engine = pa3_engine.NativeEngine(args.data, args.teams, workers=args.workers)
    model = PA3Model(PA3Config())
    learner = PPOLearner(model, PPOConfig(), device=device) if args.mode == "full" else None
    config = NativeCollectorConfig(
        envs=args.envs, workers=args.workers, seed=args.seed + seed_offset,
        device=str(device), observation_mode=args.observations,
        amp=args.precision == "fp16", inference_mode=args.inference_mode,
        candidate_wire=args.candidate_wire,
    )
    collector = NativeCollector(engine, model, config, device=device)
    model_sha = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        model_sha.update(name.encode())
        model_sha.update(parameter.detach().cpu().numpy().tobytes())
    return device, collector, learner, model_sha.hexdigest()


def run_repeat(args, repeat):
    device, collector, learner, model_sha = build_runner(args, repeat)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_event = end_event = None
    if device.type == "cuda":
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
    wall_start = time.perf_counter()
    buffer = collector.collect(args.games)
    collect_wall = time.perf_counter() - wall_start
    ppo_wall = 0.0
    checkpoint_wall = 0.0
    update = None
    recompute_diff = None
    if args.mode == "full" and learner is not None:
        # Correctness gate before any weight update: the sampled joint
        # log-probability must be reproducible from the stored tables.
        recompute_diff = recompute_check(learner, buffer, device, amp=(args.precision == "fp16"))
    if args.mode == "full" and learner is not None:
        wall = time.perf_counter()
        batch = learner.prepare_batch(buffer)
        profile_extra: dict = {}
        if args.profile_trace:
            os.makedirs(os.path.dirname(os.path.abspath(args.profile_trace)), exist_ok=True)
            from torch.profiler import ProfilerActivity, profile as torch_profile

            with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                update = learner.update(batch, committed_matches=collector.stats.games).as_dict()
            prof.export_chrome_trace(args.profile_trace)
            kernel_totals: dict[str, float] = {}
            for entry in prof.key_averages():
                if entry.device_type == torch.autograd.DeviceType.CUDA:
                    kernel_totals[entry.key] = float(entry.device_time_total)
            profile_extra = {
                "trace_path": os.path.abspath(args.profile_trace),
                "top_cuda_kernels": sorted(kernel_totals.items(), key=lambda kv: -kv[1])[:15],
            }
        else:
            update = learner.update(batch, committed_matches=collector.stats.games).as_dict()
        profile_extra["prepare_profile"] = dict(learner.prepare_profile)
        profile_extra["update_profile"] = dict(learner.profile)
        ppo_wall = time.perf_counter() - wall
        if args.checkpoint:
            # Real crash-safe checkpoint write inside the measured all-in window
            # (atomic temp file -> fsync -> rename).
            wall = time.perf_counter()
            target = os.path.abspath(args.checkpoint)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            temporary = f"{target}.tmp-{os.getpid()}"
            torch.save(learner.state_dict(), temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            checkpoint_wall = time.perf_counter() - wall
    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize(device)
    stats = collector.stats.as_dict()
    total_wall = collect_wall + ppo_wall + checkpoint_wall
    report = {
        "repeat": repeat,
        "mode": args.mode,
        "observation_mode": args.observations,
        "precision": args.precision,
        "inference_mode": bool(args.inference_mode),
        "model_sha256": model_sha,
        "games_target": args.games,
        "games_natural_complete": stats["games"],
        "cohort_overshoot": stats["games"] - args.games,
        "decisions": stats["decisions"],
        "operational_errors": stats["operational_errors"],
        "wins": stats["wins"], "losses": stats["losses"], "draws": stats["draws"],
        "actor_collect_wall_s": collect_wall,
        "ppo_update_wall_s": ppo_wall,
        "checkpoint_write_wall_s": checkpoint_wall,
        "checkpoint_path": os.path.abspath(args.checkpoint) if args.checkpoint else None,
        "bounded_ppo_games_per_s": stats["games"] / max(collect_wall + ppo_wall, 1e-9),
        "all_in_wall_s": total_wall,
        "all_in_committed_games_per_s": stats["games"] / max(total_wall, 1e-9),
        "report_includes": ["collect", "gae", "prepare", "4_epoch_update", "scheduler",
                            "grad_scaler"] + (["checkpoint_write"] if args.checkpoint else []),
        "actor_games_per_s": stats["games"] / max(collect_wall, 1e-9),
        "decisions_per_s": stats["decisions"] / max(collect_wall, 1e-9),
        "stage_seconds": {
            "reset": stats["reset_seconds"], "native_observation": stats["observation_seconds"],
            "request_info": stats["request_info_seconds"],
            "parse_convert": stats["parse_seconds"], "h2d": stats["h2d_seconds"],
            "candidates": stats["candidate_seconds"], "candidate_table": stats["table_seconds"],
            "model": stats["model_seconds"],
            "readback": stats["readback_seconds"], "rust_step": stats["step_seconds"],
            "prefix_specs": stats["spec_seconds"], "assemble_submissions": stats["assemble_seconds"],
            "buffer_record": stats["record_seconds"], "unaccounted": stats["unaccounted_seconds"],
        },
        "candidate_calls": stats["candidate_calls"],
        "candidates_returned": stats["candidates_returned"],
        "learner_rows": stats["learner_rows"],
        "cohorts": stats["cohorts"],
        "rounds": stats["rounds"],
        "max_open_games": stats["max_open_games"],
        "cpu_seconds": stats.get("cpu_seconds"),
        "rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0,
        "gpu_event_ms": start_event.elapsed_time(end_event) if start_event is not None else None,
    }
    if device.type == "cuda":
        report["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / (1 << 30)
        report["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / (1 << 30)
    if update is not None:
        report["ppo_update"] = {
            key: update[key] for key in (
                "epochs_run", "policy_loss", "value_loss", "entropy", "uniform_kl", "approx_kl",
                "epoch_approx_kl", "ratio_mean", "clip_fraction", "grad_norm", "grad_norm_max",
                "optimizer_steps", "optimizer_steps_skipped", "stopped_early", "learning_rate",
                "committed_matches", "rows", "actor_rows",
            )
        }
        if profile_extra:
            report["learner_profile"] = profile_extra
    if recompute_diff is not None:
        report["logprob_recompute"] = recompute_diff
        report["logprob_recompute_max_abs_diff"] = recompute_diff["max_abs_diff"]
        tolerance = args.recompute_tolerance if args.recompute_tolerance is not None else (
            1e-3 if args.precision == "fp16" else 1e-4)
        report["logprob_recompute_tolerance"] = tolerance
        report["logprob_recompute_within_gate"] = bool(
            recompute_diff["max_abs_diff"] <= tolerance
        )
    del collector, learner
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def recompute_check(learner, buffer, device, limit=1024, amp=False, per_stratum=96):
    """Stratified sampled/recomputed log-probability parity gate.

    A first-N-rows check cannot be extrapolated to the whole iteration, so rows
    are sampled per (request kind, branch count, actor active) stratum and every
    stratum's row count and max absolute difference is reported.
    """
    strata: dict[tuple, list] = {}
    for row in buffer.rows:
        key = (int(row.request_kind), int(row.branch_count), bool(row.actor_active))
        strata.setdefault(key, []).append(row)
    sampled = []
    for _key, group in sorted(strata.items()):
        if len(group) <= per_stratum:
            sampled.extend(group)
        else:
            step = len(group) / per_stratum
            sampled.extend(group[int(index * step)] for index in range(per_stratum))
    sampled = sampled[: max(limit, 1)]
    batch = buffer.to_batch(sampled, device=device)
    # Recompute under the same forward precision that sampled the rollout, as
    # the plan requires for mixed-precision parity.
    context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp and device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    )
    with torch.no_grad(), context:
        encoded = learner.model.encode(batch.observation)
        evaluation = learner.model.evaluate_encoded(encoded, batch.candidates, selected=batch.candidates.selected)
    stored = torch.tensor([row.old_logprob for row in sampled], dtype=torch.float32, device=device)
    difference = (evaluation.request_logprob.float() - stored).abs()
    per_row = difference.detach().cpu().tolist()
    stratum_report: dict[str, dict] = {}
    for index, row in enumerate(sampled):
        key = (f"kind{int(row.request_kind)}_branches{int(row.branch_count)}"
               f"_actor{int(bool(row.actor_active))}")
        entry = stratum_report.setdefault(key, {"rows": 0, "max_abs_diff": 0.0})
        entry["rows"] += 1
        entry["max_abs_diff"] = max(entry["max_abs_diff"], float(per_row[index]))
    tolerance = 1e-3 if amp else 1e-4
    return {
        "rows": len(sampled),
        "total_rollout_rows": len(buffer.rows),
        "sampled_fraction": len(sampled) / max(len(buffer.rows), 1),
        "strata": stratum_report,
        "max_abs_diff": float(difference.max().item()) if len(sampled) else 0.0,
        "mean_abs_diff": float(difference.mean().item()) if len(sampled) else 0.0,
        "tolerance": tolerance,
        "within_gate": bool(float(difference.max().item()) <= tolerance) if len(sampled) else True,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    manifest_path = args.manifest or os.path.join(os.path.dirname(os.path.abspath(args.report)), "run_manifest.json")
    command = " ".join([os.path.basename(sys.executable), *sys.argv])
    manifest = {
        "run_id": f"bench-{int(time.time())}-{args.tag or args.mode}",
        "command": command,
        "git_sha": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "git_diff_hash": git_diff_hash() if git_output("status", "--porcelain") else None,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "python": sys.version.split()[0],
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "gpus": gpu_inventory(),
        **dataset_sha(),
        "config": {
            "games": args.games, "envs": args.envs, "workers": args.workers,
            "repeats": args.repeats, "seed": args.seed, "device": args.device,
            "mode": args.mode, "observations": args.observations, "precision": args.precision,
            "candidate_wire": args.candidate_wire,
            "inference_mode": bool(args.inference_mode),
        },
        "readiness_note": "readiness_check is 10/16 PASS; this benchmark does not change it",
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    repeats = []
    for repeat in range(args.repeats):
        result = run_repeat(args, repeat)
        repeats.append(result)
        print(json.dumps({k: result[k] for k in (
            "repeat", "games_natural_complete", "actor_games_per_s", "all_in_committed_games_per_s",
            "rss_peak_gib", "gpu_peak_reserved_gib", "gpu_event_ms")}, sort_keys=True))
    summary_keys = ["actor_games_per_s", "all_in_committed_games_per_s", "decisions_per_s"]
    summary = {}
    for key in summary_keys:
        values = sorted(result[key] for result in repeats)
        summary[key] = {"min": values[0], "median": values[len(values) // 2], "max": values[-1]}
    stage_totals = {}
    for stage in repeats[0]["stage_seconds"]:
        values = sorted(result["stage_seconds"][stage] for result in repeats)
        shares = sorted(
            result["stage_seconds"][stage] / max(result["actor_collect_wall_s"], 1e-9)
            for result in repeats
        )
        # The share is the median of each repeat's own stage/collect ratio, so it
        # never pairs a stage median with another repeat's wall.
        stage_totals[stage] = {"median_s": values[len(values) // 2],
                               "median_share_of_collect": shares[len(shares) // 2]}
    report = {**manifest, "repeats": repeats, "summary": summary, "stage_medians": stage_totals}
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps({"summary": summary, "stage_medians": stage_totals}, indent=2, sort_keys=True))
    if any(result["operational_errors"] for result in repeats):
        raise SystemExit("operational errors during benchmark")


if __name__ == "__main__":
    main()
