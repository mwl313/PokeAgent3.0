#!/usr/bin/env python3
"""Run the documented two-actor 2,048-environment collection topology.

Rank 0 -> GPU0 / NUMA0 (CPUs 0-19,40-59), workers pinned to 0-15.
Rank 1 -> GPU1 / NUMA1 (CPUs 20-39,60-79), workers pinned to 20-35.

Each rank is a separate OS process owning one native environment group of
1,024 environments with 16 workers, matching the engine spec. When `numactl`
exists the process is bound with `--cpunodebind/--membind`; the actor also
applies `sched_setaffinity` for the worker threads. GPU/torch work is not part
of this script: the engine is CPU-only and the policy is a placeholder here.

Usage:
    PYTHONPATH=engine/python python3 scripts/run_actor_pair.py --games 512
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKS = [
    {"rank": 0, "numa": 0, "cpus": "0-15"},
    {"rank": 1, "numa": 1, "cpus": "20-35"},
]
# Documented machine profile: GPU0 PCI 05:00.0 on NUMA0, GPU1 PCI 84:00.0 on
# NUMA1 (POKEAGENT_3_0_FULL_SPEC.md, hardware table). The check below compares
# these against the live /sys topology and refuses to run when they disagree.
EXPECTED_GPU_BUS = {0: "05:00.0", 1: "84:00.0"}


def parse_cpu_spec(text):
    """Parse `0-15,40-59` into an ordered list of CPU ids."""
    cpus = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-")
            cpus.extend(range(int(low), int(high) + 1))
        else:
            cpus.append(int(part))
    return cpus


def read_text(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


def pci_numa_node(bus):
    """NUMA node of a PCI function, or None when the kernel does not report."""
    value = read_text(f"/sys/bus/pci/devices/0000:{bus}/numa_node")
    if value is None or value == "" or int(value) < 0:
        return None
    return int(value)


def node_cpus(node):
    value = read_text(f"/sys/devices/system/node/node{node}/cpulist")
    return None if value is None else set(parse_cpu_spec(value))


def gpu_bus_ids():
    """[index] -> PCI bus id from nvidia-smi, or {} when unavailable."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {}
    try:
        out = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,pci.bus_id",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    buses = {}
    for line in out.splitlines():
        index, bus = [field.strip() for field in line.split(",")]
        # nvidia-smi prints an 8-digit domain (`00000000:05:00.0`); the sysfs
        # device names use the 4-digit form (`0000:05:00.0`).
        parts = bus.split(":")
        buses[int(index)] = ":".join(parts[-2:]) if len(parts) >= 3 else bus
    return buses


def check_hardware(args):
    """Cross-check the documented GPU/NUMA/CPU map against live state.

    Returns a list of human-readable validation lines. Any disagreement is
    reported and, unless `--allow-hardware-mismatch` is passed, aborts before
    any worker starts, so a benchmark can never silently run on the wrong
    NUMA node and be reported as the documented topology.
    """
    problems = []
    lines = []
    buses = gpu_bus_ids()
    numactl = shutil.which("numactl")
    lines.append(f"numactl: {numactl or 'missing (worker sched_setaffinity only)'}")
    if buses:
        for index, bus in sorted(buses.items()):
            lines.append(f"gpu{index}: pci {bus} numa {pci_numa_node(bus)}")
    else:
        problems.append("nvidia-smi GPU/PCI map unavailable")
    with open("/proc/cpuinfo") as handle:
        online = {int(m.group(1)) for m in re.finditer(r"^processor\s*:\s*(\d+)", handle.read(), re.M)}
    for rank in RANKS:
        cpus = set(parse_cpu_spec(rank["cpus"]))
        node_set = node_cpus(rank["numa"])
        gpu_bus = buses.get(rank["rank"])
        gpu_node = pci_numa_node(gpu_bus) if gpu_bus else None
        expected_bus = EXPECTED_GPU_BUS[rank["rank"]]
        lines.append(
            f"rank {rank['rank']}: gpu {gpu_bus or '?'} (expected {expected_bus}) "
            f"pci-numa {gpu_node} rank-numa {rank['numa']} workers {rank['cpus']}"
        )
        if gpu_bus is not None and gpu_bus != expected_bus:
            problems.append(
                f"rank {rank['rank']} GPU bus {gpu_bus} != documented {expected_bus}"
            )
        if gpu_node is not None and gpu_node != rank["numa"]:
            problems.append(
                f"rank {rank['rank']} GPU NUMA {gpu_node} != documented NUMA {rank['numa']}"
            )
        if node_set is not None:
            missing = cpus - node_set
            if missing:
                problems.append(
                    f"rank {rank['rank']} workers {sorted(missing)} outside NUMA "
                    f"{rank['numa']} cpulist"
                )
        missing_online = cpus - online
        if missing_online:
            problems.append(
                f"rank {rank['rank']} workers {sorted(missing_online)} are not online CPUs"
            )
    return lines, problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=1024, help="environments per rank")
    parser.add_argument("--workers", type=int, default=16, help="workers per rank")
    parser.add_argument("--games", type=int, default=2048, help="games per rank")
    parser.add_argument("--policy", default="first")
    parser.add_argument(
        "--observations",
        default="perview",
        choices=["fixed", "perview"],
        help="observation payload path passed to the actor",
    )
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--python", default=os.path.join(ROOT, ".venv", "bin", "python"))
    parser.add_argument(
        "--teams",
        default=os.path.join(ROOT, "engine/data/training-teams.json"),
        help="team catalogue for reset sampling (development benchmarks may pass a cohort)",
    )
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-hardware-mismatch",
        action="store_true",
        help="continue even when the live GPU/NUMA/CPU map differs from the spec",
    )
    parser.add_argument(
        "--report",
        default="",
        help="write the combined run metrics as JSON to this path",
    )
    parser.add_argument(
        "--skip-hardware-check",
        action="store_true",
        help="skip the startup GPU/NUMA/CPU map validation",
    )
    args = parser.parse_args()

    if not args.skip_hardware_check:
        lines, problems = check_hardware(args)
        for line in lines:
            print(f"hardware: {line}")
        if problems:
            for problem in problems:
                print(f"hardware: PROBLEM {problem}", file=sys.stderr)
            if not args.allow_hardware_mismatch:
                raise SystemExit(
                    "hardware/NUMA validation failed; pass --allow-hardware-mismatch "
                    "to run anyway (results would not describe the documented topology)"
                )

    python = args.python if os.path.exists(args.python) else sys.executable
    numactl = shutil.which("numactl")
    procs = []
    started = time.perf_counter()
    report = {
        "envs_per_rank": args.envs,
        "workers_per_rank": args.workers,
        "ranks": [],
    }
    for rank in RANKS:
        command = [
            python,
            os.path.join(ROOT, "engine/python/pa3_actor.py"),
            "--data",
            os.path.join(ROOT, "engine/data"),
            "--teams",
            args.teams,
            "--envs",
            str(args.envs),
            "--workers",
            str(args.workers),
            "--games",
            str(args.games),
            "--policy",
            args.policy,
            "--observations",
            args.observations,
            "--seed",
            str(args.seed + rank["rank"]),
            "--pin",
            rank["cpus"],
        ]
        if args.no_reset:
            command.append("--no-reset")
        if numactl:
            command = [
                numactl,
                f"--cpunodebind={rank['numa']}",
                f"--membind={rank['numa']}",
                "--",
            ] + command
        if args.dry_run:
            print(" ".join(command))
            continue
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(ROOT, "engine/python")
        procs.append((rank, subprocess.Popen(command, stdout=subprocess.PIPE, env=env)))
    if args.dry_run:
        return
    wall = 0.0
    total_games = 0
    for rank, proc in procs:
        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            raise SystemExit(f"rank {rank['rank']} failed with {proc.returncode}")
        metrics = json.loads(stdout.decode())
        total_games += metrics["games"]
        wall = max(wall, metrics["wall_seconds"])
        report["ranks"].append(
            {
                "rank": rank["rank"],
                "numa": rank["numa"],
                "cpus": rank["cpus"],
                "metrics": metrics,
            }
        )
        print(
            f"rank {rank['rank']} numa{rank['numa']}: {metrics['games']} games, "
            f"{metrics['games_per_second']:.1f} games/s, "
            f"{metrics['transitions_per_second']:.0f} transitions/s, "
            f"op_errors={metrics['operational_errors']}, "
            f"obs={metrics['observe_ms_per_round']:.3f} ms/round, "
            f"step={metrics['step_ms_per_round']:.3f} ms/round, "
            f"max_rss={metrics.get('max_rss_mb', float('nan')):.1f} MB"
        )
    elapsed = time.perf_counter() - started
    report["total_games"] = total_games
    report["environments"] = args.envs * len(RANKS)
    report["wall_seconds"] = elapsed
    print(
        f"combined: {total_games} natural games over 2x{args.envs} envs, "
        f"{total_games / max(elapsed, 1e-9):.1f} games/s wall, "
        f"max rank wall {wall:.2f}s"
    )
    if args.report:
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
