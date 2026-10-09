#!/usr/bin/env python3
"""D0: real 4096-row single-GPU vs two-GPU DDP update parity (NCCL).

The parent builds (once) a real both-seat fixture with the native collector and
saves it. Two GPU workers then run the fixed-step DDP update, each on one
complete-trajectory half of the same 4,096 rows. Rank 0 additionally runs the
single-process reference on the union of both halves --- one 4,096-row
minibatch, same 4 epochs* / microbatch / LR clock / precision --- and writes the
weight, Adam-moment, optimizer-step, LR, epoch-KL and gradient-norm deltas.

*Default 3 epochs to keep the bounded gate short; the protocol is identical to
the 4-epoch contract (one global minibatch per epoch).

This is a correctness gate, not a throughput panel; the D2 A/B benchmark uses
its own raw JSON.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKS = [{"rank": 0, "numa": 0, "gpu": 0}, {"rank": 1, "numa": 1, "gpu": 1}]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--microbatch", type=int, default=256)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--executor", choices=["ddp", "manual"], default="ddp")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--fixture", default=os.path.join(ROOT, "runs", "perf", "v4", "d0_fixture.pt"))
    parser.add_argument("--fixture-games", type=int, default=200)
    parser.add_argument("--fixture-envs", type=int, default=128)
    parser.add_argument("--fixture-workers", type=int, default=8)
    parser.add_argument("--out", default=os.path.join(ROOT, "runs", "perf", "v4", "d0_dual_parity.json"))
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def build_fixture(args) -> None:
    """Collect and save the shared real-row fixture (single process, GPU0)."""
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "engine", "python"))
    import pa3_engine
    from agent.model import PA3Config, build_model
    from agent.train.native_collector import NativeCollector, NativeCollectorConfig

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engine = pa3_engine.NativeEngine(
        os.path.join(ROOT, "engine", "data"),
        os.path.join(ROOT, "engine", "data", "training-teams.json"),
        workers=args.fixture_workers,
    )
    model = build_model(PA3Config(), device=device)
    collector = NativeCollector(
        engine,
        model,
        NativeCollectorConfig(
            envs=args.fixture_envs,
            workers=args.fixture_workers,
            seed=args.seed,
            device=str(device),
            observation_mode="fixed",
            candidate_wire="packed",
            amp=True,
            inference_mode=True,
        ),
        device=device,
    )
    started = time.perf_counter()
    buffer = collector.collect(args.fixture_games)
    wall = time.perf_counter() - started
    print(
        f"fixture: {collector.stats.games} games / {len(buffer.rows)} rows "
        f"in {wall:.1f}s (target {args.rows} rows)"
    )
    torch.save(buffer, args.fixture)
    print(f"fixture saved to {args.fixture}")


def trajectory_groups(buffer):
    groups: dict[tuple, list] = {}
    order = []
    for row in buffer.rows:
        key = (row.match_id, row.side)
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(row)
    return [groups[key] for key in order]


def two_shards(groups, per_rank: int):
    """Two complete-trajectory shards, each inside one per-rank minibatch.

    GAE chains never cross ranks (each rank owns whole games) and the fast
    first implementation's parity contract requires every rank to fit in one
    local minibatch, which this greedy packing guarantees.
    """
    shard0: list = []
    shard1: list = []
    for group in groups:
        if len(shard0) + len(group) <= per_rank:
            shard0.extend(group)
        elif len(shard1) + len(group) <= per_rank:
            shard1.extend(group)
    return shard0, shard1


def learner_config(rows: int, microbatch: int, epochs: int, per_rank: bool):
    from agent.ppo import PPOConfig

    return PPOConfig(
        global_minibatch_size=rows,
        microbatch_size=microbatch,
        per_rank_minibatch_size=rows // 2 if per_rank else rows,
        grad_accumulation_per_rank=max(1, (rows // 2 if per_rank else rows) // microbatch),
        ppo_epochs=epochs,
        sample_weighted_ddp_reduction=False,
    )


def reference_update(fixture, rows, args, device):
    from agent.model import PA3Config, build_model
    from agent.ppo import PPOLearner

    torch.manual_seed(args.seed)
    model = build_model(PA3Config(), device=device)
    learner = PPOLearner(
        model,
        learner_config(args.rows, args.microbatch, args.epochs, per_rank=False),
        device=device,
        amp=args.precision == "fp16",
    )
    batch = learner.prepare_batch(fixture, rows=rows)
    report = learner.update(
        batch,
        committed_matches=args.rows,
        generator=torch.Generator(device="cpu").manual_seed(args.seed + 7),
    )
    return learner, report, model


def worker(args) -> None:
    import torch.distributed as dist
    from agent.model import PA3Config, build_model
    from agent.ppo import PPOLearner
    from agent.ppo.ddp import DDPCommunication
    from agent.train.native_collector import NativeCollectorConfig  # noqa: F401

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
    try:
        fixture = torch.load(args.fixture, weights_only=False)
        groups = trajectory_groups(fixture)
        per_rank = args.rows // 2
        shard0, shard1 = two_shards(groups, per_rank)
        rows = shard0 + shard1
        shard = shard0 if rank == 0 else shard1
        if min(len(shard0), len(shard1)) < int(per_rank * 0.75):
            raise SystemExit(
                f"fixture shards too unbalanced: {len(shard0)} / {len(shard1)}; "
                "collect a larger fixture"
            )

        reference = None
        if rank == 0:
            learner, report, model = reference_update(fixture, rows, args, device)
            reference = {
                "state": copy.deepcopy(learner.model.state_dict()),
                "moments": {
                    name: {
                        key: entry[key].detach().clone()
                        for key in ("exp_avg", "exp_avg_sq")
                        if key in entry
                    }
                    for name, entry in (
                        (name, learner.optimizer.state.get(parameter, {}))
                        for name, parameter in learner.model.named_parameters()
                    )
                },
                "steps": learner.optimizer_steps,
                "skipped": report.optimizer_steps_skipped,
                "lr": learner.scheduler.learning_rate,
                "epoch_kl": list(report.epoch_approx_kl),
                "grad_norm": report.grad_norm,
                "grad_norm_max": report.grad_norm_max,
                "policy_loss": report.policy_loss,
                "value_loss": report.value_loss,
                "rows": len(rows),
            }
            del learner, model
        dist.barrier()

        torch.manual_seed(args.seed)
        model = build_model(PA3Config(), device=device)
        learner = PPOLearner(
            model,
            learner_config(args.rows, args.microbatch, args.epochs, per_rank=True),
            device=device,
            amp=args.precision == "fp16",
        )
        if args.executor == "manual":
            # No DDP wrapper: local grads are summed explicitly in FP32.
            learner.attach_ddp(None, world_size)
            communication = None
            ddp_model = None
        else:
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            learner.attach_ddp(ddp_model, world_size)
            communication = DDPCommunication(ddp_model)
        plan = learner.prepare_streaming_ddp(fixture, rows=shard)
        dist.barrier()
        started = time.perf_counter()
        update_kwargs = dict(
            committed_matches=args.rows,
            generator=torch.Generator(device="cpu").manual_seed(args.seed + 7),
        )
        if args.executor == "manual":
            report = learner.update_manual_allreduce(plan, **update_kwargs)
        else:
            report = learner.update_ddp(plan, communication=communication, **update_kwargs)
        update_wall = time.perf_counter() - started

        payload = {
            "rank": rank,
            "executor": args.executor,
            "rows": len(shard),
            "update_wall_s": update_wall,
            "steps": learner.optimizer_steps,
            "skipped": report.optimizer_steps_skipped,
            "lr": learner.scheduler.learning_rate,
            "epoch_kl": list(report.epoch_approx_kl),
            "grad_norm": report.grad_norm,
            "grad_norm_max": report.grad_norm_max,
            "policy_loss": report.policy_loss,
            "value_loss": report.value_loss,
            "sync_calls": communication.sync_calls if communication is not None else None,
            "no_sync_calls": communication.no_sync_calls if communication is not None else None,
            "manual_allreduce_seconds": learner.profile.get("manual_allreduce_seconds"),
            "manual_flat_bytes": learner.profile.get("manual_flat_bytes"),
        }
        if rank == 0:
            state = learner.model.state_dict()
            weight_delta = max(
                float((state[name] - reference["state"][name]).abs().max())
                for name in state
            )
            moment_delta = 0.0
            reference_moments = reference["moments"]
            for name, parameter in learner.model.named_parameters():
                entry = learner.optimizer.state.get(parameter, {})
                reference_entry = reference_moments[name]
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in entry and key in reference_entry:
                        moment_delta = max(
                            moment_delta,
                            float((entry[key] - reference_entry[key]).abs().max()),
                        )
            payload["weight_delta"] = weight_delta
            payload["moment_delta"] = moment_delta
            payload["reference"] = {
                key: value for key, value in reference.items() if key not in ("state", "moments")
            }
            payload["steps_match"] = learner.optimizer_steps == reference["steps"]
            payload["lr_match"] = abs(learner.scheduler.learning_rate - reference["lr"]) < 1e-12
            payload["epoch_kl_delta"] = (
                max(
                    abs(left - right)
                    for left, right in zip(report.epoch_approx_kl, reference["epoch_kl"])
                )
                if report.epoch_approx_kl and reference["epoch_kl"]
                else None
            )
            payload["grad_norm_relative_delta"] = abs(
                report.grad_norm - reference["grad_norm"]
            ) / max(abs(reference["grad_norm"]), 1e-12)
            with open(args.out, "w") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)

        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload)
        if rank == 0:
            with open(f"{args.out}.ranks.json", "w") as handle:
                json.dump(gathered, handle, indent=2, sort_keys=True)
        dist.barrier()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def parent(args) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not os.path.exists(args.fixture):
        build_fixture(args)
    port = args.port or free_port()
    numactl = shutil.which("numactl")
    procs = []
    logs = []
    for entry in RANKS:
        command = [
            sys.executable, os.path.abspath(__file__), "--worker",
            "--rows", str(args.rows), "--epochs", str(args.epochs),
            "--microbatch", str(args.microbatch), "--seed", str(args.seed),
            "--fixture", args.fixture, "--out", args.out, "--port", str(port),
            "--timeout", str(args.timeout), "--precision", args.precision,
            "--executor", args.executor,
        ]
        if numactl:
            command = [numactl, f"--cpunodebind={entry['numa']}", f"--membind={entry['numa']}", "--"] + command
        env = dict(os.environ)
        env.update({
            "RANK": str(entry["rank"]), "LOCAL_RANK": str(entry["rank"]),
            "WORLD_SIZE": str(len(RANKS)),
            "PYTHONPATH": os.path.join(ROOT, "engine", "python") + os.pathsep + ROOT,
            "NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "0",
            "NCCL_SOCKET_IFNAME": "lo", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        })
        log_path = f"{os.path.abspath(args.out)}.rank{entry['rank']}.log"
        log = open(log_path, "wb")
        logs.append(log)
        procs.append((entry, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)))
    deadline = time.time() + args.timeout + 300.0
    failures = []
    try:
        while any(proc.poll() is None for _, proc in procs):
            if time.time() > deadline:
                failures.append("watchdog timeout")
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
            path = f"{os.path.abspath(args.out)}.rank{entry['rank']}.log"
            if os.path.exists(path):
                with open(path, "r", errors="replace") as handle:
                    tails.append(handle.read()[-4000:])
        raise SystemExit("; ".join(failures) + "\n" + "\n".join(tails))
    with open(args.out) as handle:
        summary = json.load(handle)
    print(json.dumps(
        {
            key: summary[key]
            for key in (
                "rows", "weight_delta", "moment_delta", "steps_match", "lr_match",
                "epoch_kl_delta", "grad_norm_relative_delta", "update_wall_s",
                "sync_calls", "no_sync_calls",
            )
            if key in summary
        },
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
