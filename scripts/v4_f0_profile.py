#!/usr/bin/env python3
"""F0: bounded torch.profiler pass over the single-GPU PPO minibatch.

Collects a real fixture, wraps one 4,096-row minibatch update (microbatch
1,024) in a CPU+CUDA profiler after a warmup, and writes a compact JSON:
top CUDA kernels by device time, top CPU ops, and GPU-busy vs wall.

No full Chrome trace is committed; `--trace` optionally writes one under
`runs/perf/v4/` (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine", "python"))

import pa3_engine  # noqa: E402
from agent.model import PA3Config, build_model  # noqa: E402
from agent.ppo import PPOConfig, PPOLearner  # noqa: E402
from agent.train.native_collector import NativeCollector, NativeCollectorConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--microbatch", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-cache", choices=["none", "cpu", "cuda"], default="none")
    parser.add_argument("--compact-candidates", action="store_true")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("runs/perf/v4/f0_learner_kernels.json"))
    parser.add_argument("--trace", default="")
    return parser.parse_args()


def kernel_window(events) -> tuple[float, float]:
    """Union of actual kernel intervals and their enclosing span, in us."""
    intervals = sorted((float(e["ts"]), float(e["ts"]) + float(e["dur"]))
                       for e in events if float(e.get("dur", 0)) > 0)
    if not intervals:
        return 0.0, 0.0
    begin, end = intervals[0]
    first = begin
    busy = 0.0
    for left, right in intervals[1:]:
        if left > end:
            busy += end - begin
            begin, end = left, right
        else:
            end = max(end, right)
    return busy + end - begin, end - first


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(17)
    engine = pa3_engine.NativeEngine(
        os.path.join(ROOT, "engine", "data"),
        os.path.join(ROOT, "engine", "data", "training-teams.json"),
        workers=args.workers,
    )
    model = build_model(PA3Config(), device=device)
    collector = NativeCollector(
        engine,
        model,
        NativeCollectorConfig(
            envs=args.envs, workers=args.workers, seed=17, device=str(device),
            observation_mode="fixed", candidate_wire="packed", amp=True, inference_mode=True,
        ),
        device=device,
    )
    buffer = collector.collect(args.games)
    learner = PPOLearner(
        model,
        PPOConfig(
            global_minibatch_size=4096,
            microbatch_size=args.microbatch,
            ppo_epochs=1,
            sample_weighted_ddp_reduction=False,
        ),
        device=device,
        amp=True,
    )
    plan = learner.prepare_streaming(
        buffer, cache_device=None if args.batch_cache == "none" else args.batch_cache,
        compact_candidates=args.compact_candidates,
    )
    # Warmup: one full minibatch minus the final partial chunk.
    learner.update_streaming(plan, committed_matches=args.games,
                             generator=torch.Generator().manual_seed(1))
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    from torch.profiler import ProfilerActivity, profile

    wall_start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        learner.update_streaming(plan, committed_matches=args.games,
                                 generator=torch.Generator().manual_seed(2))
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - wall_start
    import collections
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        trace_path = args.trace or os.path.join(tmp, "f0_trace.json")
        prof.export_chrome_trace(trace_path)
        with open(trace_path) as handle:
            trace = json.load(handle)
    kernel_events = [event for event in trace["traceEvents"] if event.get("cat") == "kernel"]
    kernel_union_us, kernel_span_us = kernel_window(kernel_events)
    kernel_totals: dict[str, dict[str, float]] = collections.OrderedDict()
    kernel_busy_us = 0.0
    for event in kernel_events:
        name = event.get("name", "unknown").split("(")[0][:120]
        duration = float(event.get("dur", 0.0))
        kernel_busy_us += duration
        entry = kernel_totals.setdefault(name, {"calls": 0.0, "device_time_ms": 0.0})
        entry["calls"] += 1
        entry["device_time_ms"] += duration / 1000.0
    leaf_kernel_rows = sorted(
        ({"key": name, **values} for name, values in kernel_totals.items()),
        key=lambda row: -row["device_time_ms"],
    )
    leaf_kernel_rows = [
        {"key": row["key"], "calls": int(row["calls"]), "device_time_ms": row["device_time_ms"]}
        for row in leaf_kernel_rows[:20]
    ]

    cuda_rows = []
    cpu_rows = []
    cuda_total_us = 0.0
    for entry in prof.key_averages():
        if entry.device_time_total > 0:
            cuda_rows.append({
                "key": entry.key,
                "calls": int(entry.count),
                "device_time_ms": float(entry.device_time_total) / 1000.0,
            })
            cuda_total_us += float(entry.device_time_total)
        if entry.cpu_time_total > 0:
            cpu_rows.append({
                "key": entry.key,
                "calls": int(entry.count),
                "cpu_time_ms": float(entry.cpu_time_total) / 1000.0,
            })
    cuda_rows.sort(key=lambda row: -row["device_time_ms"])
    cpu_rows.sort(key=lambda row: -row["cpu_time_ms"])
    payload = {
        "rows": len(buffer.rows),
        "microbatch": args.microbatch,
        "batch_cache": args.batch_cache,
        "compact_candidates": args.compact_candidates,
        "profiled_wall_s": wall,
        "gpu_busy_s": cuda_total_us / 1e6,
        "gpu_busy_leaf_kernel_s": kernel_union_us / 1e6,
        "gpu_kernel_span_s": kernel_span_us / 1e6,
        "leaf_kernels": leaf_kernel_rows,
        "gpu_busy_note": (
            "gpu_busy_s sums aggregated ops and leaf kernels (double counts); "
            "gpu_busy_leaf_kernel_s is the interval union of trace cat=kernel "
            "events. gpu_busy_fraction uses the first-to-last kernel span; "
            "profile wall additionally includes profiler setup/teardown. "
            "Busy is execution presence, not SM/TensorCore utilization."
        ),
        "gpu_idle_s": max(kernel_span_us - kernel_union_us, 0.0) / 1e6,
        "gpu_busy_fraction": kernel_union_us / max(kernel_span_us, 1e-9),
        "gpu_busy_fraction_of_profile_wall": (kernel_union_us / 1e6) / max(wall, 1e-9),
        "peak_vram_mib": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
        "top_cuda_kernels": cuda_rows[:20],
        "top_cpu_ops": cpu_rows[:20],
        "profile_stage": dict(learner.profile),
    }
    if args.trace:
        prof.export_chrome_trace(args.trace)
        payload["trace_path"] = args.trace
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        {
            "profiled_wall_s": round(wall, 2),
            "gpu_busy_s": round(payload["gpu_busy_s"], 2),
            "gpu_busy_leaf_kernel_s": round(payload["gpu_busy_leaf_kernel_s"], 2),
            "gpu_busy_fraction": round(payload["gpu_busy_fraction"], 3),
            "peak_vram_mib": round(payload["peak_vram_mib"]),
            "top_kernels": cuda_rows[:6],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
