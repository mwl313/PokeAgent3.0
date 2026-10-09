#!/usr/bin/env python3
"""Two-rank, real-policy PPO with the fixed-step NCCL DDP protocol (v4 D1).

Each rank owns one GPU/NUMA node, its own native engine group and collector,
and a copy of the frozen PA3-8M policy. The distributed update uses the
protocol implemented by ``PPOLearner.update_ddp``:

* global actor/value denominators A/V from an all-reduce of the per-minibatch
  counts, with ``world_size * (S_actor/A + value_coef*S_value/V)`` per rank so
  DDP's gradient average equals the single-process global objective;
* a fixed micro-step count per minibatch on every rank (graph-connected zero
  micros when a rank has no real rows), so the collective sequence never
  diverges;
* one global epoch-KL decision, one shared finiteness/step/skip decision and a
  single global natural-match LR clock (all-reduce SUM of completed games).

The parent spawns the two workers with per-rank log files, drains both
concurrently, and merges the per-rank JSON. No host, driver, power-cap or
service change is made; NCCL environment variables are set for the child
processes only.

Usage (parent):
    PYTHONPATH=engine/python:. .venv/bin/python scripts/run_ddp_ppo.py \
        --games 2048 --envs 1024 --workers 16 --microbatch 256 \
        --report runs/perf/v4_ddp_2k.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import resource
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
    # v5 A/B promotion (2026-10-09): dual default is now micro 1024
    # (median 26.25 -> 32.50 games/s, all gates PASS; docs/perf/V5_MICROBATCH_AB.md).
    parser.add_argument("--microbatch", type=int, default=1024)
    parser.add_argument("--minibatch", type=int, default=4096, help="global minibatch (both ranks)")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--epochs", type=int, default=0, help="override ppo_epochs (0 = config default)")
    parser.add_argument("--executor", choices=["ddp", "manual"], default="ddp")
    parser.add_argument("--data", default=os.path.join(ROOT, "engine", "data"))
    parser.add_argument("--teams", default=os.path.join(ROOT, "engine", "data", "training-teams.json"))
    parser.add_argument("--report", default=os.path.join(ROOT, "runs", "perf", "v4_ddp.json"))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--timeout", type=float, default=900.0, help="process-group/task timeout seconds")
    parser.add_argument("--debug", action="store_true", help="enable TORCH_DISTRIBUTED_DEBUG=DETAIL")
    parser.add_argument(
        "--recompute-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stratified sampled-vs-recomputed logprob parity gate before the update (v4 gate)",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def state_digest(model, optimizer) -> tuple[str, str]:
    """Full named-tensor SHA256 for the model and the Adam moments."""
    model_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        model_hash.update(name.encode())
        model_hash.update(str(tensor.dtype).encode())
        model_hash.update(str(tuple(tensor.shape)).encode())
        model_hash.update(tensor.numpy().tobytes())
    optimizer_hash = hashlib.sha256()
    named = {id(parameter): name for name, parameter in model.named_parameters()}
    for parameter, entry in optimizer.state.items():
        name = named.get(id(parameter), "unknown")
        optimizer_hash.update(name.encode())
        for key in ("step", "exp_avg", "exp_avg_sq"):
            value = entry.get(key)
            if value is None:
                continue
            tensor = torch.as_tensor(value).detach().cpu().contiguous()
            optimizer_hash.update(key.encode())
            optimizer_hash.update(tensor.to(torch.float64).numpy().tobytes())
    return model_hash.hexdigest(), optimizer_hash.hexdigest()


def worker(args) -> None:
    """One rank: local rollout, fixed-step DDP update, per-rank metrics JSON."""
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{args.port}",
        timeout=datetime.timedelta(seconds=args.timeout),
    )

    def note(message: str) -> None:
        print(f"[rank{rank}] {message}", file=sys.stderr, flush=True)

    note(f"process group ready at {time.time():.1f}")
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
    overrides = dict(
        global_minibatch_size=args.minibatch,
        microbatch_size=args.microbatch,
        per_rank_minibatch_size=args.minibatch // world_size,
        grad_accumulation_per_rank=max(1, (args.minibatch // world_size) // args.microbatch),
    )
    if args.epochs:
        overrides["ppo_epochs"] = args.epochs
    ppo_config = PPOConfig(**overrides)
    learner = PPOLearner(model, ppo_config, device=device)
    if args.executor == "manual":
        # No DDP wrapper: local FP32 gradients are summed explicitly.
        learner.attach_ddp(None, world_size)
        communication = None
    else:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=True,
        )
        learner.attach_ddp(ddp_model, world_size)
        communication = DDPCommunication(ddp_model)
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
    if not buffer.rows:
        raise SystemExit(f"rank {rank} collected zero rows; refusing to run the DDP update")
    note(f"collected {collector.stats.games} games / {len(buffer.rows)} rows in {collect_wall:.1f}s")

    recompute = None
    recompute_wall = 0.0
    if args.recompute_gate:
        scripts_dir = os.path.join(ROOT, "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from bench_pa3_end_to_end import recompute_check

        gate_started = time.perf_counter()
        recompute = recompute_check(learner, buffer, device, amp=True)
        recompute_wall = time.perf_counter() - gate_started
        note(
            f"recompute gate: {recompute['rows']} rows max|diff|="
            f"{recompute['max_abs_diff']:.3e} (tol {recompute['tolerance']:.1e}) "
            f"within={recompute['within_gate']}"
        )

    # One absolute LR clock: the sum of every rank's completed natural matches.
    games_tensor = torch.tensor([float(collector.stats.games)], device=device)
    dist.all_reduce(games_tensor, op=dist.ReduceOp.SUM)
    global_games = int(games_tensor.item())
    note(f"global committed matches this iteration: {global_games}")

    plan = learner.prepare_streaming_ddp(buffer)
    note("streaming plan ready (global advantage normalization)")
    update_started = time.perf_counter()
    update_kwargs = dict(
        committed_matches=global_games,
        generator=torch.Generator(device="cpu").manual_seed(args.seed + rank),
    )
    if args.executor == "manual":
        report = learner.update_manual_allreduce(plan, **update_kwargs)
    else:
        report = learner.update_ddp(plan, communication=communication, **update_kwargs)
    update_wall = time.perf_counter() - update_started
    note(
        f"ddp update done in {update_wall:.1f}s "
        f"(epochs {report.epochs_run}, steps {report.optimizer_steps}, "
        f"skipped {report.optimizer_steps_skipped})"
    )

    checkpoint_wall = 0.0
    if args.checkpoint and rank == 0:
        wall = time.perf_counter()
        target = os.path.abspath(args.checkpoint)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        temporary = f"{target}.tmp-{os.getpid()}"
        torch.save(
            {
                **learner.state_dict(),
                "global_committed_matches": global_games,
                "rank_states": {
                    "rank0_rows": len(buffer.rows),
                    "rank0_games": int(collector.stats.games),
                },
            },
            temporary,
        )
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        checkpoint_wall = time.perf_counter() - wall

    model_digest, optimizer_digest = state_digest(model, learner.optimizer)
    digests = [None] * world_size
    dist.all_gather_object(digests, {"model": model_digest, "optimizer": optimizer_digest})
    digests_equal = all(entry == digests[0] for entry in digests)
    note(f"digest gather done, equal={digests_equal}")

    stats = collector.stats.as_dict()
    metrics = {
        "rank": rank,
        "numa": RANKS[rank]["numa"],
        "gpu": local_rank,
        "games": stats["games"],
        "decisions": stats["decisions"],
        "rows": len(buffer.rows),
        "global_committed_matches": global_games,
        "collect_wall_s": collect_wall,
        "recompute_wall_s": recompute_wall,
        "recompute": recompute,
        "update_wall_s": update_wall,
        "checkpoint_wall_s": checkpoint_wall,
        "operational_errors": stats["operational_errors"],
        "rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1 << 30),
        "model_digest": model_digest,
        "optimizer_digest": optimizer_digest,
        "digests_equal": bool(digests_equal),
        "executor": args.executor,
        "sync_calls": communication.sync_calls if communication is not None else None,
        "no_sync_calls": communication.no_sync_calls if communication is not None else None,
        "report": report.as_dict(),
        "profile": learner.profile,
        "prepare_profile": learner.prepare_profile,
    }
    rank_report = f"{os.path.abspath(args.report)}.rank{rank}.json"
    os.makedirs(os.path.dirname(rank_report), exist_ok=True)
    with open(rank_report, "w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    note(f"rank metrics written to {rank_report}")

    dist.barrier()
    dist.destroy_process_group()


def parent(args) -> None:
    port = args.port or free_port()
    numactl = shutil.which("numactl")
    procs = []
    logs = []
    started = time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    for entry in RANKS:
        command = [
            sys.executable, os.path.abspath(__file__), "--worker",
            "--games", str(args.games), "--envs", str(args.envs),
            "--workers", str(args.workers), "--microbatch", str(args.microbatch),
            "--minibatch", str(args.minibatch), "--seed", str(args.seed),
            "--epochs", str(args.epochs), "--timeout", str(args.timeout),
            "--data", args.data, "--teams", args.teams, "--port", str(port),
            "--report", args.report, "--checkpoint", args.checkpoint,
            "--executor", args.executor,
        ]
        if not args.recompute_gate:
            command.append("--no-recompute-gate")
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
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        })
        if args.debug:
            env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
            env["TORCH_NCCL_DESYNC_DEBUG"] = "1"
            env["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "2048"
        log_path = f"{os.path.abspath(args.report)}.rank{entry['rank']}.log"
        log = open(log_path, "wb")
        logs.append(log)
        procs.append((entry, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)))

    failures = []
    deadline = time.time() + args.timeout + 300.0
    try:
        while any(proc.poll() is None for _, proc in procs):
            if time.time() > deadline:
                failures.append("launcher watchdog timeout")
                break
            time.sleep(1.0)
        for entry, proc in procs:
            if proc.poll() is None:
                proc.terminate()
            code = proc.wait(timeout=60)
            if code != 0:
                failures.append(f"rank {entry['rank']} exited with {code}")
    finally:
        for log in logs:
            log.close()
    if failures:
        tails = []
        for entry, _ in procs:
            path = f"{os.path.abspath(args.report)}.rank{entry['rank']}.log"
            if os.path.exists(path):
                with open(path, "r", errors="replace") as handle:
                    tails.append(f"--- rank {entry['rank']} log tail ---\n{handle.read()[-3000:]}")
        raise SystemExit("; ".join(failures) + "\n" + "\n".join(tails))

    results = []
    for entry in RANKS:
        path = f"{os.path.abspath(args.report)}.rank{entry['rank']}.json"
        if not os.path.exists(path):
            raise SystemExit(f"rank {entry['rank']} produced no metrics JSON at {path}")
        with open(path) as handle:
            results.append(json.load(handle))
    wall = time.perf_counter() - started
    total_games = sum(item["games"] for item in results)
    global_games = results[0]["global_committed_matches"]
    report = {
        "mode": "real_policy_ddp_ppo_fixed_step",
        "ranks": results,
        "total_games": total_games,
        "global_committed_matches": global_games,
        "launcher_wall_s": wall,
        "all_in_committed_games_per_s": total_games / max(wall, 1e-9),
        "sum_rank_update_wall_s": sum(item["update_wall_s"] for item in results),
        "digests_equal": all(item["digests_equal"] for item in results),
        "model_digests": [item["model_digest"] for item in results],
        "optimizer_digests": [item["optimizer_digest"] for item in results],
        "config": {key: value for key, value in vars(args).items() if key != "worker"},
    }
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(
        {key: report[key] for key in
         ("total_games", "global_committed_matches", "all_in_committed_games_per_s", "digests_equal")},
        indent=2,
    ))


def main():
    args = parse_args()
    if args.worker:
        worker(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()


# v5 promotion note (2026-10-09): the dual default microbatch is 1024
# (A/B: median 26.25 -> 32.50 games/s at equal 2,048 total games, all gates
# PASS; see docs/perf/V5_MICROBATCH_AB.md). PA3_TRAINING_CONFIG.yaml's
# `microbatch_per_rank: 256` (docs/spec/fullspec-1.1-minidc-20261006/, mirrored
# by configs/train.yaml) should be updated to 1024 at the next spec revision.
