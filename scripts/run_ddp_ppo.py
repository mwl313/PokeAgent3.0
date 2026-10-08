#!/usr/bin/env python3
"""Two-rank, real-policy PPO with NCCL DDP (v3 G1).

Each rank owns one GPU/NUMA node, its own native engine group and collector, and
a copy of the frozen PA3-8M policy. After the local rollout drains, the ranks
run a DDP update whose policy/entropy/KL terms are normalised by the **global
valid actor rows** and whose value term uses the **global valid rows**
(`agent.ppo.ddp`); `no_sync` covers the forward and backward of every
non-final microbatch, and only the final microbatch all-reduces gradients.
The rank-local minibatch is `global_minibatch / world_size` (2048 with the
default 4096 global batch).

Usage (parent):
    PYTHONPATH=engine/python:. .venv/bin/python scripts/run_ddp_ppo.py \
        --games 2048 --envs 1024 --workers 16 --microbatch 256 \
        --report runs/perf/v3_ddp_2k.json

Unchanged contracts: frozen 1,137-team pool, PA3-8M architecture, 4 PPO epochs,
global minibatch 4096, FP32 probabilities/losses, FP16 autocast, no host or
driver changes. This is a bounded benchmark/smoke, never 100M-match training.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKS = [{"rank": 0, "numa": 0, "cpus": "0-15", "gpu": 0},
         {"rank": 1, "numa": 1, "cpus": "20-35", "gpu": 1}]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2048, help="natural matches per rank")
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--minibatch", type=int, default=4096, help="global minibatch (both ranks)")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--data", default=os.path.join(ROOT, "engine", "data"))
    parser.add_argument("--teams", default=os.path.join(ROOT, "engine", "data", "training-teams.json"))
    parser.add_argument("--report", default=os.path.join(ROOT, "runs", "perf", "v3_ddp.json"))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def worker(args) -> None:
    """One rank: local rollout, then the DDP update."""
    import torch.distributed as dist
    import time as _time

    def note(message: str) -> None:
        print(f"[rank{os.environ.get('RANK')}] {message}", file=sys.stderr, flush=True)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        init_method=f"tcp://127.0.0.1:{args.port}",
    )
    note(f"process group ready at {_time.time():.1f}")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "engine", "python"))

    import pa3_engine
    from agent.model import PA3Config, build_model
    from agent.ppo import PPOConfig, PPOLearner
    from agent.ppo.ddp import DDPCommunication
    from agent.train.native_collector import NativeCollector, NativeCollectorConfig

    torch.manual_seed(args.seed + rank)
    engine = pa3_engine.NativeEngine(args.data, args.teams, workers=args.workers)
    model = build_model(PA3Config())
    ppo_config = PPOConfig(
        global_minibatch_size=args.minibatch,
        microbatch_size=args.microbatch,
        per_rank_minibatch_size=args.minibatch // world_size,
        grad_accumulation_per_rank=max(1, (args.minibatch // world_size) // args.microbatch),
    )
    learner = PPOLearner(model, ppo_config, device=device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False, find_unused_parameters=True,
    )
    learner.attach_ddp(ddp_model, world_size)
    communication = DDPCommunication(ddp_model, ppo_config.grad_accumulation_per_rank)
    collector = NativeCollector(
        engine, model,
        NativeCollectorConfig(
            envs=args.envs, workers=args.workers, seed=args.seed + rank, device=str(device),
            observation_mode="fixed", candidate_wire="packed",
            amp=True, inference_mode=True,
        ),
        device=device,
    )

    started = time.perf_counter()
    buffer = collector.collect(args.games)
    collect_wall = time.perf_counter() - started
    note(f"collected {collector.stats.games} games / {len(buffer.rows)} rows in {collect_wall:.1f}s")
    plan = learner.prepare_streaming(buffer)
    note("streaming plan ready")
    update_started = time.perf_counter()
    report = learner.update_ddp(
        plan, committed_matches=int(collector.stats.games),
        generator=torch.Generator(device="cpu").manual_seed(args.seed + rank),
        communication=communication,
    )
    update_wall = time.perf_counter() - update_started
    note(f"ddp update done in {update_wall:.1f}s")

    checkpoint_wall = 0.0
    if args.checkpoint and rank == 0:
        wall = time.perf_counter()
        target = os.path.abspath(args.checkpoint)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        temporary = f"{target}.tmp-{os.getpid()}"
        torch.save(learner.state_dict(), temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        checkpoint_wall = time.perf_counter() - wall

    # Rank parity: broadcast rank 0's parameter digest and compare.
    digest = torch.zeros(1, dtype=torch.float64, device=device)
    for parameter in model.parameters():
        digest += parameter.detach().double().sum()
    gathered = [torch.zeros_like(digest) for _ in range(world_size)]
    dist.all_gather(gathered, digest)
    digests_equal = all(torch.allclose(gathered[0], value, atol=1e-6) for value in gathered)
    note(f"digest gather done, equal={digests_equal}")

    stats = collector.stats.as_dict()
    metrics = {
        "rank": rank, "numa": RANKS[rank]["numa"], "gpu": local_rank,
        "games": stats["games"], "decisions": stats["decisions"], "rows": len(buffer.rows),
        "collect_wall_s": collect_wall, "update_wall_s": update_wall,
        "checkpoint_wall_s": checkpoint_wall,
        "operational_errors": stats["operational_errors"],
        "rss_peak_gib": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 1048576.0,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1 << 30),
        "digest": float(digest.item()), "digests_equal": bool(digests_equal),
        "report": report.as_dict(),
        "profile": learner.profile,
        "prepare_profile": learner.prepare_profile,
    }
    payload = json.dumps(metrics)
    if rank == 0:
        print(payload)
    dist.barrier()
    dist.destroy_process_group()


def parent(args) -> None:
    port = args.port or free_port()
    numactl = shutil.which("numactl")
    procs = []
    started = time.perf_counter()
    for entry in RANKS:
        command = [
            sys.executable, os.path.abspath(__file__), "--worker",
            "--games", str(args.games), "--envs", str(args.envs),
            "--workers", str(args.workers), "--microbatch", str(args.microbatch),
            "--minibatch", str(args.minibatch), "--seed", str(args.seed),
            "--data", args.data, "--teams", args.teams, "--port", str(port),
            "--report", args.report, "--checkpoint", args.checkpoint,
        ]
        if numactl:
            command = [numactl, f"--cpunodebind={entry['numa']}", f"--membind={entry['numa']}", "--"] + command
        env = dict(os.environ)
        env.update({
            "RANK": str(entry["rank"]), "LOCAL_RANK": str(entry["rank"]),
            "WORLD_SIZE": str(len(RANKS)),
            "PYTHONPATH": os.path.join(ROOT, "engine", "python") + os.pathsep + ROOT,
            "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "0",
            # Localhost-only 2-process NCCL; without an explicit interface the
            # transport selection can stall on this host.
            "NCCL_SOCKET_IFNAME": "lo",
            "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        })
        procs.append((entry, subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=ROOT)))
    results = []
    for entry, proc in procs:
        output, _ = proc.communicate()
        text = output.decode()
        if proc.returncode != 0:
            raise SystemExit(f"rank {entry['rank']} failed:\n{text[-3000:]}")
        for line in reversed(text.strip().splitlines()):
            if line.startswith("{"):
                results.append(json.loads(line))
                break
        else:
            raise SystemExit(f"rank {entry['rank']} produced no metrics:\n{text[-2000:]}")
    wall = time.perf_counter() - started
    total_games = sum(item["games"] for item in results)
    report = {
        "mode": "real_policy_ddp_ppo",
        "ranks": results,
        "total_games": total_games,
        "launcher_wall_s": wall,
        "all_in_committed_games_per_s": total_games / max(wall, 1e-9),
        "sum_rank_update_wall_s": sum(item["update_wall_s"] for item in results),
        "digests_equal": all(item["digests_equal"] for item in results),
        "config": vars(args),
    }
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps({key: report[key] for key in
                      ("total_games", "all_in_committed_games_per_s", "digests_equal")}, indent=2))


def main():
    args = parse_args()
    if args.worker:
        worker(args)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        parent(args)


if __name__ == "__main__":
    main()
