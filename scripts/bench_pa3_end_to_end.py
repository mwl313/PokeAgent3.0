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
    update = None
    recompute_diff = None
    if args.mode == "full" and learner is not None:
        # Correctness gate before any weight update: the sampled joint
        # log-probability must be reproducible from the stored tables.
        recompute_diff = recompute_check(learner, buffer, device, amp=(args.precision == "fp16"))
    if args.mode == "full" and learner is not None:
        wall = time.perf_counter()
        batch = learner.prepare_batch(buffer)
        update = learner.update(batch, committed_matches=collector.stats.games).as_dict()
        ppo_wall = time.perf_counter() - wall
    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize(device)
    stats = collector.stats.as_dict()
    total_wall = collect_wall + ppo_wall
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
        "all_in_wall_s": total_wall,
        "all_in_committed_games_per_s": stats["games"] / max(total_wall, 1e-9),
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
                "epoch_approx_kl", "ratio_mean", "clip_fraction", "grad_norm", "optimizer_steps",
                "stopped_early", "learning_rate", "committed_matches", "rows", "actor_rows",
            )
        }
    if recompute_diff is not None:
        report["logprob_recompute_max_abs_diff"] = recompute_diff
        tolerance = args.recompute_tolerance if args.recompute_tolerance is not None else (
            1e-3 if args.precision == "fp16" else 1e-4)
        report["logprob_recompute_tolerance"] = tolerance
        report["logprob_recompute_within_gate"] = bool(recompute_diff <= tolerance)
    del collector, learner
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def recompute_check(learner, buffer, device, limit=1024, amp=False):
    rows = list(buffer.rows)[:limit]
    batch = buffer.to_batch(rows, device=device)
    # Recompute under the same forward precision that sampled the rollout, as
    # the plan requires for mixed-precision parity.
    context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp and device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
    )
    with torch.no_grad(), context:
        encoded = learner.model.encode(batch.observation)
        evaluation = learner.model.evaluate_encoded(encoded, batch.candidates, selected=batch.candidates.selected)
    stored = torch.tensor([row.old_logprob for row in rows], dtype=torch.float32, device=device)
    return float((evaluation.request_logprob.float() - stored).abs().max().item())


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
        stage_totals[stage] = {"median_s": values[len(values) // 2],
                               "share_of_wall": values[len(values) // 2] / max(repeats[0]["all_in_wall_s"], 1e-9)}
    report = {**manifest, "repeats": repeats, "summary": summary, "stage_medians": stage_totals}
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps({"summary": summary, "stage_medians": stage_totals}, indent=2, sort_keys=True))
    if any(result["operational_errors"] for result in repeats):
        raise SystemExit("operational errors during benchmark")


if __name__ == "__main__":
    main()
