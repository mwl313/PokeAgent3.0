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
        --games 2048 --envs 1024 --workers 16 --microbatch 1024 \
        --report runs/perf/v6_ddp_2k.json

Persistent measurement (same model/optimizer, new rollout each iteration):
    PYTHONPATH=engine/python:. .venv/bin/python scripts/run_ddp_ppo.py \
        --games 1024 --envs 1024 --iterations 3 --batch-cache cuda \
        --report runs/perf/persistent_ppo.json

The launcher rate includes process startup/shutdown. Per-iteration rates
include preparation, parity checks, optional checkpoints, state digests and
cleanup. The steady-state rate excludes iteration 1 and uses total games /
total measured wall for iterations 2..N; it is not a mean of per-rank rates.
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2048, help="natural matches per rank")
    parser.add_argument(
        "--iterations", type=int, default=1,
        help="persistent collect/update iterations; steady-state summary excludes iteration 1",
    )
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=16)
    # v5 A/B promotion (2026-10-09): dual default is now micro 1024
    # (median 26.25 -> 32.50 games/s, all gates PASS; docs/perf/V5_MICROBATCH_AB.md).
    # PPOConfig and both YAML mirrors now agree with that measured promotion.
    parser.add_argument("--microbatch", type=int, default=1024)
    parser.add_argument("--minibatch", type=int, default=4096, help="global minibatch (both ranks)")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--epochs", type=int, default=0, help="override ppo_epochs (0 = config default)")
    parser.add_argument("--executor", choices=["ddp", "manual"], default="ddp")
    parser.add_argument(
        "--batch-cache", choices=["none", "cpu", "cuda"], default="none",
        help="materialize the iteration's learner rows once on the selected device",
    )
    parser.add_argument("--trim-observation-padding", action=argparse.BooleanOptionalAction,
                        default=False, help="skip CPU-proven empty trailing transformer tokens")
    parser.add_argument("--compact-candidates", action=argparse.BooleanOptionalAction, default=False,
                        help="remove verified empty candidate padding, never legal candidates")
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
    parser.add_argument(
        "--columnar-store",
        action="store_true",
        help="v5b T1: SoA observation store (must pass the row-SHA neutrality gate)",
    )
    parser.add_argument(
        "--rolling-slots",
        action="store_true",
        help="v5c T2: rolling slot refill (gated by the rolling-slot equivalence tests)",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    return args


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


def publish_verified_checkpoint(learner, path, metadata, *, digests_equal: bool) -> None:
    """Replace the prior checkpoint only after the distributed state gate."""
    if not digests_equal:
        raise RuntimeError("refusing to publish a checkpoint with divergent rank digests")
    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = f"{target}.tmp-{os.getpid()}"
    try:
        torch.save({**learner.state_dict(), **metadata}, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def stop_workers(processes, timeout: float = 10.0) -> None:
    """Terminate/reap only the child process objects started by this launcher."""
    for process in processes:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:  # child exited between poll and signal
                pass
    for process in processes:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.wait(timeout=timeout)


def source_provenance() -> dict:
    """Record the checkout actually executed, including uncommitted changes."""
    def git(*command):
        return subprocess.check_output(
            ["git", "-C", ROOT, *command], text=True, stderr=subprocess.DEVNULL,
        ).rstrip("\n")

    try:
        status = git("status", "--porcelain", "--untracked-files=normal")
        return {
            "git_sha": git("rev-parse", "HEAD"),
            "git_branch": git("branch", "--show-current"),
            "git_dirty": bool(status),
            "git_status": status.splitlines(),
            "tracked_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "-C", ROOT, "diff", "HEAD", "--binary"])
            ).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_dirty": None}


STAGE_WALL_KEYS = (
    "iteration_setup_wall_s", "collect_wall_s", "recompute_wall_s",
    "lr_clock_wall_s", "prepare_wall_s", "update_wall_s", "checkpoint_wall_s",
    "digest_wall_s", "cleanup_wall_s", "iteration_barrier_wall_s",
)


def summarize_iterations(rank_results: list[dict]) -> tuple[list[dict], dict | None]:
    """Use each iteration's slowest rank, never the sum of rank durations.

    Stage maxima are diagnostic only: different ranks can be critical in
    different stages, so their sum is not the iteration's measured makespan.
    The first iteration includes cold CUDA/engine work and is excluded from
    the persistent steady-state summary. Both policies and optimizer moments
    continue evolving across iterations; these are not independent repeats.
    """
    lengths = {len(rank["iterations"]) for rank in rank_results}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("all ranks must report the same nonzero iteration count")
    summaries = []
    for index in range(next(iter(lengths))):
        rows = [rank["iterations"][index] for rank in rank_results]
        clocks = {row["global_committed_matches"] for row in rows}
        if len(clocks) != 1:
            raise ValueError("ranks disagree about the global committed-match clock")
        wall = max(row["iteration_wall_s"] for row in rows)
        games = sum(row["games"] for row in rows)
        summaries.append({
            "iteration": index + 1,
            "total_games": games,
            "total_rows": sum(row["rows"] for row in rows),
            "global_committed_matches": clocks.pop(),
            "iteration_wall_s": wall,
            "committed_games_per_s": games / max(wall, 1e-9),
            "digests_equal": all(row["digests_equal"] for row in rows),
            "operational_errors": sum(row["operational_errors"] for row in rows),
            "max_rank_stage_wall_s": {
                key: max(row.get(key, 0.0) for row in rows) for key in STAGE_WALL_KEYS
            },
        })
    steady = summaries[1:]
    steady_state = None
    if steady:
        wall = sum(row["iteration_wall_s"] for row in steady)
        games = sum(row["total_games"] for row in steady)
        steady_state = {
            "first_iteration": 2,
            "iterations": len(steady),
            "total_games": games,
            "total_rows": sum(row["total_rows"] for row in steady),
            "wall_s": wall,
            "committed_games_per_s": games / max(wall, 1e-9),
        }
    return summaries, steady_state


def worker(args) -> None:
    """One rank: local rollout, fixed-step DDP update, per-rank metrics JSON."""
    import torch.distributed as dist

    worker_started = time.perf_counter()
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
    model.encoder.trim_padding = args.trim_observation_padding
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
        # DDP performs this broadcast in its constructor. The manual executor
        # must also start from identical weights despite rank-local RNG seeds.
        for value in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(value.detach(), src=0)
        learner.attach_ddp(None, world_size)
        communication = None
    else:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=True,
        )
        learner.attach_ddp(ddp_model, world_size)
        communication = DDPCommunication(ddp_model)
    recompute_check = None
    if args.recompute_gate:
        scripts_dir = os.path.join(ROOT, "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from bench_pa3_end_to_end import recompute_check

    # Keep all expensive resources and the shuffle RNG alive. A fresh collector
    # is lightweight and avoids reusing cumulative stats, completed trajectories,
    # observation stores or row cursors. Each iteration gets a distinct engine
    # and policy-sampling seed; iteration 1 is identical to the legacy seed.
    shuffle_generator = torch.Generator(device="cpu").manual_seed(args.seed + rank)
    global_games = 0
    iterations = []
    torch.cuda.synchronize(device)
    startup_wall = time.perf_counter() - worker_started
    for index in range(args.iterations):
        iteration_started = time.perf_counter()
        timings = {key: 0.0 for key in STAGE_WALL_KEYS}

        def finish_stage(key: str, started: float) -> None:
            torch.cuda.synchronize(device)
            timings[key] = time.perf_counter() - started

        stage_started = time.perf_counter()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        iteration_seed = args.seed + rank + index * world_size
        collector = NativeCollector(
            engine, model,
            NativeCollectorConfig(
                envs=args.envs, workers=args.workers, seed=iteration_seed, device=str(device),
                observation_mode="fixed", candidate_wire="packed",
                amp=True, inference_mode=True,
                columnar_observation_store=args.columnar_store,
                rolling_slots=args.rolling_slots,
            ),
            device=device,
        )
        if communication is not None:
            communication.sync_calls = 0
            communication.no_sync_calls = 0
        finish_stage("iteration_setup_wall_s", stage_started)

        stage_started = time.perf_counter()
        buffer = collector.collect(args.games)
        finish_stage("collect_wall_s", stage_started)
        if not buffer.rows:
            raise SystemExit(f"rank {rank} collected zero rows; refusing to run the DDP update")
        note(
            f"iteration {index + 1}/{args.iterations}: collected {collector.stats.games} games / "
            f"{len(buffer.rows)} rows in {timings['collect_wall_s']:.1f}s"
        )

        recompute = None
        if recompute_check is not None:
            stage_started = time.perf_counter()
            recompute = recompute_check(learner, buffer, device, amp=True)
            finish_stage("recompute_wall_s", stage_started)
            note(
                f"recompute gate: {recompute['rows']} rows max|diff|="
                f"{recompute['max_abs_diff']:.3e} (tol {recompute['tolerance']:.1e}) "
                f"within={recompute['within_gate']}"
            )
            if not recompute["within_gate"]:
                raise RuntimeError("sampled-vs-recomputed logprob gate failed; refusing PPO update")

        # The scheduler consumes the absolute global clock, not this iteration's
        # count. Resetting it would silently repeat the LR warmup every iteration.
        stage_started = time.perf_counter()
        games_tensor = torch.tensor([collector.stats.games], dtype=torch.int64, device=device)
        dist.all_reduce(games_tensor, op=dist.ReduceOp.SUM)
        iteration_games = int(games_tensor.item())
        global_games += iteration_games
        finish_stage("lr_clock_wall_s", stage_started)

        stage_started = time.perf_counter()
        plan = learner.prepare_streaming_ddp(
            buffer, cache_device=None if args.batch_cache == "none" else args.batch_cache,
            compact_candidates=args.compact_candidates,
        )
        finish_stage("prepare_wall_s", stage_started)
        stage_started = time.perf_counter()
        optimizer_steps_before = learner.optimizer_steps
        update_kwargs = dict(committed_matches=global_games, generator=shuffle_generator)
        if args.executor == "manual":
            report = learner.update_manual_allreduce(plan, **update_kwargs)
        else:
            report = learner.update_ddp(plan, communication=communication, **update_kwargs)
        finish_stage("update_wall_s", stage_started)
        optimizer_steps_this_iteration = learner.optimizer_steps - optimizer_steps_before
        note(
            f"update done in {timings['update_wall_s']:.1f}s "
            f"(epochs {report.epochs_run}, steps {optimizer_steps_this_iteration}, "
            f"cumulative steps {report.optimizer_steps}, "
            f"skipped {report.optimizer_steps_skipped}, global matches {global_games})"
        )

        stage_started = time.perf_counter()
        model_digest, optimizer_digest = state_digest(model, learner.optimizer)
        digests = [None] * world_size
        dist.all_gather_object(digests, {"model": model_digest, "optimizer": optimizer_digest})
        digests_equal = all(entry == digests[0] for entry in digests)
        finish_stage("digest_wall_s", stage_started)
        if not digests_equal:
            raise RuntimeError("ranks disagree on model/optimizer digests after update")

        stage_started = time.perf_counter()
        if args.checkpoint and rank == 0:
            publish_verified_checkpoint(
                learner, args.checkpoint,
                {
                    "global_committed_matches": global_games,
                    "iteration": index + 1,
                    "rank_states": {
                        "rank0_rows": len(buffer.rows),
                        "rank0_games": int(collector.stats.games),
                    },
                },
                digests_equal=digests_equal,
            )
        finish_stage("checkpoint_wall_s", stage_started)

        stats = collector.stats.as_dict()
        metrics = {
            "iteration": index + 1,
            "seed": iteration_seed,
            "rank": rank,
            "numa": RANKS[rank]["numa"],
            "gpu": local_rank,
            "games": stats["games"],
            "decisions": stats["decisions"],
            "rows": len(buffer.rows),
            "iteration_global_committed_matches": iteration_games,
            "global_committed_matches": global_games,
            "optimizer_steps_this_iteration": optimizer_steps_this_iteration,
            "optimizer_steps_cumulative": learner.optimizer_steps,
            "recompute": recompute,
            "operational_errors": stats["operational_errors"],
            # ru_maxrss is a process-lifetime peak; the CUDA peak is reset per
            # iteration and includes the live model/optimizer allocation.
            "rss_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0,
            "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1 << 30),
            "model_digest": model_digest,
            "optimizer_digest": optimizer_digest,
            "digests_equal": bool(digests_equal),
            "executor": args.executor,
            "execution_options": {
                "batch_cache": args.batch_cache,
                "compact_candidates": args.compact_candidates,
                "trim_observation_padding": args.trim_observation_padding,
                "microbatch": args.microbatch,
            },
            "sync_calls": communication.sync_calls if communication is not None else None,
            "no_sync_calls": communication.no_sync_calls if communication is not None else None,
            "report": report.as_dict(),
            "profile": dict(learner.profile),
            "prepare_profile": dict(learner.prepare_profile),
            "collector_stats": stats,
        }
        stage_started = time.perf_counter()
        del plan, buffer, collector
        finish_stage("cleanup_wall_s", stage_started)
        stage_started = time.perf_counter()
        dist.barrier()
        finish_stage("iteration_barrier_wall_s", stage_started)
        metrics.update(timings)
        metrics["iteration_wall_s"] = time.perf_counter() - iteration_started
        metrics["unaccounted_wall_s"] = max(
            metrics["iteration_wall_s"] - sum(timings.values()), 0.0,
        )
        iterations.append(metrics)
        note(f"iteration {index + 1} complete in {metrics['iteration_wall_s']:.1f}s, digests equal")

    # One-iteration consumers retain all original rank keys. For persistent
    # runs, counters/durations below are totals and report/profile are the last
    # update; the complete series is always available in `iterations`.
    metrics = dict(iterations[-1])
    for key in ("games", "decisions", "rows", "operational_errors", *STAGE_WALL_KEYS, "iteration_wall_s"):
        metrics[key] = sum(item[key] for item in iterations)
    for key in ("sync_calls", "no_sync_calls"):
        metrics[key] = sum(item[key] for item in iterations) if communication is not None else None
    metrics.update({
        "iterations": iterations,
        "optimizer_steps_total": sum(item["optimizer_steps_this_iteration"] for item in iterations),
        "startup_wall_s": startup_wall,
        "worker_wall_s": time.perf_counter() - worker_started,
        "digests_equal": all(item["digests_equal"] for item in iterations),
    })
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
    provenance = source_provenance()
    started = time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    failures = []
    try:
        for entry in RANKS:
            command = [
                sys.executable, os.path.abspath(__file__), "--worker",
                "--games", str(args.games), "--envs", str(args.envs),
                "--iterations", str(args.iterations), "--batch-cache", args.batch_cache,
                "--workers", str(args.workers), "--microbatch", str(args.microbatch),
                "--minibatch", str(args.minibatch), "--seed", str(args.seed),
                "--epochs", str(args.epochs), "--timeout", str(args.timeout),
                "--data", args.data, "--teams", args.teams, "--port", str(port),
                "--report", args.report, "--checkpoint", args.checkpoint,
                "--executor", args.executor,
            ]
            if not args.recompute_gate:
                command.append("--no-recompute-gate")
            command.append("--trim-observation-padding" if args.trim_observation_padding
                           else "--no-trim-observation-padding")
            command.append("--compact-candidates" if args.compact_candidates
                           else "--no-compact-candidates")
            if args.columnar_store:
                command.append("--columnar-store")
            if args.rolling_slots:
                command.append("--rolling-slots")
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

        deadline = time.time() + args.timeout * args.iterations + 300.0
        while any(proc.poll() is None for _, proc in procs):
            # A failed correctness gate must not leave its peer blocked in a
            # collective until the full process-group timeout expires.
            if any(proc.poll() not in (None, 0) for _, proc in procs):
                break
            if time.time() > deadline:
                failures.append("launcher watchdog timeout")
                break
            time.sleep(1.0)
        stop_workers([proc for _, proc in procs])
        for entry, proc in procs:
            code = proc.returncode
            if code != 0:
                failures.append(f"rank {entry['rank']} exited with {code}")
    except BaseException:
        # Includes partial spawn failure and user interruption. No process
        # group or broad process-name matching is used: only our Popen handles.
        stop_workers([proc for _, proc in procs])
        raise
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
    expected_execution = {
        "batch_cache": args.batch_cache,
        "compact_candidates": args.compact_candidates,
        "trim_observation_padding": args.trim_observation_padding,
        "microbatch": args.microbatch,
    }
    if any(result["execution_options"] != expected_execution for result in results):
        raise RuntimeError("worker execution options differ from the requested benchmark")
    wall = time.perf_counter() - started
    total_games = sum(item["games"] for item in results)
    global_games = results[0]["global_committed_matches"]
    iteration_summaries, steady_state = summarize_iterations(results)
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
        "iterations": iteration_summaries,
        "steady_state": steady_state,
        "steady_state_committed_games_per_s": (
            steady_state["committed_games_per_s"] if steady_state is not None else None
        ),
        "provenance": {
            **provenance,
            "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "numa_binding": bool(numactl),
        },
        "measurement": {
            "launcher_all_in": "parent spawn through rank report reads; includes startup and shutdown",
            "iteration_wall": "slowest rank's measured wall, including prepare/recompute/checkpoint/digest/cleanup/barriers",
            "steady_state": "weighted games/wall over iterations 2..N; excludes only the first iteration",
            "persistent_state": "engine, model, optimizer, scaler and shuffle RNG; distinct rollout seed each iteration",
            "rank_top_level": "summed games/rows/durations; final update report/profile; full series in iterations",
        },
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
