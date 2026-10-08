#!/usr/bin/env python3
"""Two-rank, real-policy PA3-8M actor benchmark (Phase 6).

Unlike `run_actor_pair.py` (random policy), both ranks run the PA3-8M policy
sampled on their own GPU with their own native engine group, pinned to their
GPU-local NUMA node. This measures the horizontal scaling of the *real*
collection path; the DDP learner is a separate step and is not claimed here.

Each rank writes a JSON metrics file; the parent sums the wall-aligned
throughput. Ranks are the documented 1,024 environments / 16 workers each.

Usage:
    PYTHONPATH=engine/python:. .venv/bin/python scripts/run_actor_pair_real.py \
        --games 2048 --envs 1024 --workers 16 --report runs/perf/dual-real.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKS = [
    {"rank": 0, "numa": 0, "cpus": "0-15", "gpu": "cuda:0", "expected_bus": "05:00.0"},
    {"rank": 1, "numa": 1, "cpus": "20-35", "gpu": "cuda:1", "expected_bus": "84:00.0"},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2048, help="natural matches per rank")
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--observations", choices=["perview", "fixed"], default="fixed")
    parser.add_argument("--candidate-wire", choices=["tuples", "packed"], default="packed")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--inference-mode", action="store_true", default=True)
    parser.add_argument("--report", default=os.path.join(ROOT, "runs", "perf", "dual-real.json"))
    parser.add_argument("--check-hardware", action="store_true", default=True)
    return parser.parse_args()


def pci_bus_map():
    query = "index,pci.bus_id"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return {int(line.split(",")[0]): line.split(",")[1].strip().split(":")[-2] + ":" + line.split(",")[1].strip().split(":")[-1]
                for line in out.strip().splitlines()}
    except Exception:
        return {}


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    bus = pci_bus_map()
    for rank in RANKS:
        actual = bus.get(rank["rank"], "")
        if args.check_hardware and actual and not actual.endswith(rank["expected_bus"]):
            raise SystemExit(f"GPU {rank['rank']} bus {actual} != expected {rank['expected_bus']}")
    numactl = shutil.which("numactl")
    procs = []
    started = time.perf_counter()
    for rank in RANKS:
        metrics_path = os.path.join(os.path.dirname(os.path.abspath(args.report)),
                                    f"dual-real-rank{rank['rank']}.json")
        command = [
            sys.executable, os.path.join(ROOT, "scripts", "bench_pa3_end_to_end.py"),
            "--games", str(args.games), "--envs", str(args.envs), "--workers", str(args.workers),
            "--repeats", "1", "--seed", str(args.seed + rank["rank"]),
            "--mode", "collect", "--device", rank["gpu"],
            "--observations", args.observations, "--candidate-wire", args.candidate_wire,
            "--precision", args.precision,
            "--report", metrics_path, "--tag", f"dual-rank{rank['rank']}",
        ]
        if args.inference_mode:
            command.append("--inference-mode")
        if numactl:
            command = [numactl, f"--cpunodebind={rank['numa']}", f"--membind={rank['numa']}", "--"] + command
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(ROOT, "engine", "python") + os.pathsep + ROOT
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        procs.append((rank, metrics_path, subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)))
    results = []
    for rank, metrics_path, proc in procs:
        output, _ = proc.communicate()
        if proc.returncode != 0:
            raise SystemExit(f"rank {rank['rank']} failed:\n{output.decode()[-2000:]}")
        with open(metrics_path) as handle:
            report = json.load(handle)
        repeats = report["repeats"][0]
        results.append({"rank": rank["rank"], "numa": rank["numa"], "cpus": rank["cpus"],
                        "gpu": rank["gpu"], "games": repeats["games_natural_complete"],
                        "wall_s": repeats["actor_collect_wall_s"],
                        "games_per_s": repeats["actor_games_per_s"],
                        "decisions_per_s": repeats["decisions_per_s"],
                        "operational_errors": repeats["operational_errors"],
                        "rss_peak_gib": repeats["rss_peak_gib"],
                        "gpu_peak_reserved_gib": repeats.get("gpu_peak_reserved_gib"),
                        "stage_seconds": repeats["stage_seconds"]})
        print(f"rank {rank['rank']} numa{rank['numa']} {rank['gpu']}: {repeats['games_natural_complete']} games, "
              f"{repeats['actor_games_per_s']:.2f} games/s, op_errors={repeats['operational_errors']}, "
              f"rss={repeats['rss_peak_gib']:.2f} GiB")
    wall = time.perf_counter() - started
    total_games = sum(item["games"] for item in results)
    summary = {
        "ranks": results,
        "total_games": total_games,
        "environments": args.envs * len(RANKS),
        "wall_s": wall,
        "aggregate_games_per_s": total_games / max(wall, 1e-9),
        "sum_rank_games_per_s": sum(item["games_per_s"] for item in results),
        "single_rank_games_per_s": results[0]["games_per_s"],
        "scaling_vs_one_rank": (total_games / max(wall, 1e-9)) / max(results[0]["games_per_s"], 1e-9),
        "mode": "real_policy_collect",
        "ddp_learner": "not part of this measurement",
        "config": vars(args),
    }
    with open(args.report, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({k: summary[k] for k in
                      ("total_games", "aggregate_games_per_s", "sum_rank_games_per_s",
                       "scaling_vs_one_rank")}, indent=2))


if __name__ == "__main__":
    main()
