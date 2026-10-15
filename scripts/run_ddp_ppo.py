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
from pathlib import Path

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKS = [{"rank": 0, "numa": 0, "cpus": "0-15", "gpu": 0},
         {"rank": 1, "numa": 1, "cpus": "20-35", "gpu": 1}]


def _publish_completed_rolling_boundary(args, store, published_keys):
    """Publish a pair only after both ranks expose the same drained boundary.

    Workers atomically replace their per-rank envelope after each iteration.  A
    parent-side poll therefore must treat mismatched or partially-written files
    as transient and never make either one resumable on its own.
    """
    if not args.rolling_checkpoint_root or not args.experimental_checkpoint:
        return None
    from agent.ppo.experimental_checkpoint import load_envelope
    paths = {rank: Path(f"{os.path.abspath(args.experimental_checkpoint)}.rank{rank}") for rank in (0, 1)}
    if any(not path.is_file() for path in paths.values()):
        return None
    try:
        # Inspect and publish the same immutable byte snapshot. Re-reading a
        # worker's replaceable boundary file can otherwise publish iteration
        # N+1 bytes with iteration N metadata.
        import io
        payloads = {rank: path.read_bytes() for rank, path in paths.items()}
        envelopes = {rank: torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
                     for rank, payload in payloads.items()}
        a0, a1 = (envelopes[0].get("accounting", {}), envelopes[1].get("accounting", {}))
        # Rows and natural-game counts are rank-local.  The boundary identity
        # is the global iteration/clock (and cumulative optimizer step count).
        if any(a0.get(field) != a1.get(field) for field in ("iteration", "committed_matches", "optimizer_steps")):
            return None
        if any(not e.get("collector_drain", {}).get("complete") or
               int(e.get("collector_drain", {}).get("in_flight_battles", -1)) != 0
               for e in envelopes.values()):
            return None
        if envelopes[0].get("rank_digests") != envelopes[1].get("rank_digests"):
            return None
        if envelopes[0].get("source_sha") != envelopes[1].get("source_sha") or \
           envelopes[0].get("dataset_manifest_hash") != envelopes[1].get("dataset_manifest_hash") or \
           envelopes[0].get("opponent_manifest") != envelopes[1].get("opponent_manifest"):
            return None
        committed = int(a0.get("committed_matches", -1))
        iteration = int(a0.get("iteration", -1))
        if committed < 0 or iteration < 0:
            return None
        key = (iteration, committed)
        if key in published_keys:
            return None
        team_receipts = []
        if args.team_stats_root:
            from agent.analysis.team_stats import reconcile, run_identity
            if any(not e.get('team_stats_receipts') for e in envelopes.values()):
                raise RuntimeError('checkpoint missing required team statistics')
            team_receipts = [ref for e in envelopes.values() for ref in e['team_stats_receipts']]
            reconcile(team_receipts, committed_matches=committed,
                      run_id=run_identity(args.team_stats_root))
        generation_name = f"iter-{committed:012d}"
        for ready in store.generations_ready():
            try:
                if (ready / "manifest.json").read_text(encoding="utf-8").find(f'"generation": "{generation_name}"') >= 0:
                    published_keys.add(key)
                    return None
            except OSError:
                continue
        manifest = store.publish(
            generation_name,
            payloads,
            {"schema": "ppo-experimental-envelope-v1", "iteration_complete": True,
             "world_size": 2, "iteration": iteration, "committed_matches": committed,
             "dataset_manifest": os.path.abspath(args.dataset_manifest),
             "dataset_manifest_hash": envelopes[0].get("dataset_manifest_hash"),
             "source_sha": envelopes[0].get("source_sha"),
             "team_stats_receipts": team_receipts,
             "precision_mode": args.precision_mode,
             "ppo_epochs": int(args.epochs or envelopes[0].get("experiment", {}).get("ppo_epochs", 2))},
        )
        published_keys.add(key)
        return manifest
    except Exception as exc:
        from agent.ppo.storage_manager import StorageBudgetError
        if isinstance(exc, StorageBudgetError):
            raise
        # A worker may be replacing the file or still completing its peer.
        if isinstance(exc, (OSError, ValueError, KeyError, TypeError, EOFError)):
            return None
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="", help="strict validated runtime profile")
    parser.add_argument("--print-effective-config", action="store_true", help="resolve profile and exit")
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
    parser.add_argument("--experimental-lr-scale", type=float, default=1.0,
                        help="opt-in multiplier for the absolute match-clock LR schedule")
    parser.add_argument("--experimental-allow-source-sha", action="store_true",
                        help="explicitly migrate a trusted A3 checkpoint from its recorded source SHA")
    parser.add_argument("--experimental-source-migration", default="",
                        help="trusted local, exact-source A10 recovery migration receipt")
    parser.add_argument("--experimental-allow-opponent-migration", action="store_true",
                        help="explicitly migrate a current-only A4 envelope to the declared historical pool")
    parser.add_argument("--historical-checkpoint", action="append", default=[],
                        help="trusted frozen historical envelope rank0 path (repeatable)")
    parser.add_argument("--historical-probability", type=float, default=0.0)
    parser.add_argument("--executor", choices=["ddp", "manual"], default="ddp")
    parser.add_argument(
        "--batch-cache", choices=["none", "cpu", "cuda"], default="cuda",
        help="materialize the iteration's learner rows once on the selected device",
    )
    parser.add_argument("--trim-observation-padding", action=argparse.BooleanOptionalAction,
                        default=False, help="skip CPU-proven empty trailing transformer tokens")
    parser.add_argument("--compact-candidates", action=argparse.BooleanOptionalAction, default=True,
                        help="remove verified empty candidate padding, never legal candidates")
    parser.add_argument("--data", default=os.path.join(ROOT, "engine", "data"))
    parser.add_argument("--teams", default=os.path.join(ROOT, "engine", "data", "training-teams.json"))
    parser.add_argument("--report", default=os.path.join(ROOT, "runs", "perf", "v4_ddp.json"))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--team-stats-root", default="", help="opt-in checkpoint-bound V2 natural-match statistics")
    parser.add_argument("--checkpoint-first-iteration", action="store_true",
                        help="also publish the first scratch boundary without changing the regular interval")
    parser.add_argument('--stop-file', default='', help='cooperative stop: checkpoint and exit at the next verified iteration boundary')
    parser.add_argument(
        "--rolling-checkpoint-root", default="",
        help="opt-in generation-based publication root; requires experimental checkpoint envelopes",
    )
    parser.add_argument(
        "--experimental-checkpoint", default="",
        help="opt-in iteration-boundary envelope prefix (one file per rank)",
    )
    parser.add_argument(
        "--checkpoint-interval-matches", type=int, default=100000,
        help="full experimental checkpoint interval in global committed matches",
    )
    parser.add_argument(
        "--resume-latest-from", default="",
        help="resume from the newest verified generation under this checkpoint root",
    )
    parser.add_argument(
        "--experimental-resume", default="",
        help="opt-in envelope prefix to restore (one file per rank)",
    )
    parser.add_argument(
        "--dataset-manifest",
        default=os.path.join(ROOT, "data/teams/mb-mc-v3-userteam-all-train/manifest.json"),
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="process-group/task timeout seconds")
    parser.add_argument("--debug", action="store_true", help="enable TORCH_DISTRIBUTED_DEBUG=DETAIL")
    parser.add_argument("--deterministic-diagnostic", action="store_true",
                        help="opt-in deterministic PyTorch kernels for diagnosis only")
    parser.add_argument(
        "--recompute-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stratified sampled-vs-recomputed logprob parity gate before the update (v4 gate)",
    )
    parser.add_argument("--precision-mode", choices=("A", "B", "C", "D"), default="A",
                        help="experimental policy precision contract; A is the A9 FP32 reference")
    parser.add_argument('--experimental-exact-token-lookup',action='store_true',
                        help='opt-in V100 exact-order CUDA backward; bounded non-production diagnostics only')
    parser.add_argument("--precision-stability", choices=("default", "fp16-reduction-off"), default="default",
                        help="opt-in numerical stability setting for precision experiments")
    parser.add_argument("--recompute-audit", action="store_true",
                        help="run isolated-row diagnostics in addition to the production gate")
    parser.add_argument("--recompute-secondary", action="store_true",
                        help="bounded original-actor-group and FP32 verification of AMP outliers")
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
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    if args.profile:
        from agent.runtime.config import resolve_profile, launcher_values
        resolved = resolve_profile(args.profile)
        # Reject explicit contradictions instead of silently overwriting them.
        controlled = {"epochs": str(resolved["profile"]["training"]["ppo_epochs"]),
                      "experimental-lr-scale": str(resolved["profile"]["training"]["lr_scale"]),
                      "precision-mode": resolved["profile"]["precision"]["mode"],
                      "precision-stability": resolved["profile"]["precision"]["stability"],
                      "historical-probability": str(resolved["profile"]["training"]["historical_probability"]),
                      "microbatch": str(resolved["profile"]["training"]["microbatch"]),
                      "minibatch": str(resolved["profile"]["training"]["global_minibatch"]),
                      "envs": str(resolved["profile"]["training"]["envs"]),
                      "workers": str(resolved["profile"]["training"]["workers"]),
                      "checkpoint-interval-matches": str(resolved["profile"]["training"].get("checkpoint_interval_matches", 100000)),
                      "teams": str(resolved["profile"]["dataset"].get("teams", args.teams)),
                      "data": str(resolved["profile"]["dataset"].get("data", args.data)),
                      "dataset-manifest": str(resolved["profile"]["dataset"].get("manifest", args.dataset_manifest))}
        for index, token in enumerate(raw_argv):
            if not token.startswith("--") or token in {"--profile", "--print-effective-config"}: continue
            name, _, value = token[2:].partition("=")
            if not value and index + 1 < len(raw_argv) and not raw_argv[index + 1].startswith("--"):
                value = raw_argv[index + 1]
            if name in controlled and value and value != controlled[name]:
                raise ValueError(f"profile conflict: --{name}={value} != {controlled[name]}")
        if "--no-recompute-gate" in raw_argv:
            raise ValueError("profile conflict: recompute gate cannot be disabled")
        for key, value in launcher_values(resolved).items():
            setattr(args, key if key not in {"minibatch", "microbatch"} else key, value)
        args.effective_profile = resolved
        if args.print_effective_config:
            print(json.dumps({**resolved, "launcher_values": launcher_values(resolved)}, indent=2, sort_keys=True))
            raise SystemExit(0)
    elif args.print_effective_config:
        parser.error("--print-effective-config requires --profile")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.checkpoint_interval_matches < 1:
        parser.error("--checkpoint-interval-matches must be positive")
    if args.experimental_resume and args.rolling_slots:
        parser.error("experimental resume requires static collector slots")
    if args.experimental_lr_scale <= 0.0:
        parser.error("--experimental-lr-scale must be positive")
    if args.experimental_lr_scale != 1.0 and not (args.experimental_resume or args.experimental_checkpoint):
        parser.error("experimental LR scale requires an experimental checkpoint or resume")
    if args.team_stats_root and (not args.rolling_checkpoint_root or not args.experimental_checkpoint):
        parser.error('--team-stats-root requires generation-based experimental checkpoints')
    if args.stop_file and not args.experimental_checkpoint:
        parser.error('--stop-file requires experimental checkpoint publication')
    if args.profile and resolved['profile']['version'] == 'pa3-v2-production-v1' and not args.team_stats_root:
        parser.error('V2 production profile requires checkpoint-bound --team-stats-root')
    if not 0.0 <= args.historical_probability <= 1.0:
        parser.error("--historical-probability must be between 0 and 1")
    if args.historical_probability and not args.historical_checkpoint:
        parser.error("historical probability requires --historical-checkpoint")
    if args.resume_latest_from:
        if args.experimental_resume:
            parser.error("--resume-latest-from and --experimental-resume are mutually exclusive")
        from agent.ppo.storage_manager import CheckpointGenerationStore
        latest = CheckpointGenerationStore(args.resume_latest_from).latest()
        if latest is None:
            parser.error("--resume-latest-from has no complete verified generation")
        args.experimental_resume = str(latest)
    if args.experimental_exact_token_lookup:
        if not args.deterministic_diagnostic or args.precision_mode!='D' or args.precision_stability!='fp16-reduction-off':
            parser.error('exact token lookup requires deterministic ModeD/reduction-off')
        if args.iterations>5 or args.profile and resolved['profile']['version']=='pa3-v2-production-v1':
            parser.error('exact token lookup is limited to bounded non-production runs')
    return args


def lookup_backend(args):
    return 'exact-row-v100-v1' if args.experimental_exact_token_lookup else 'gather'


def validate_lookup_restore(experiment,args):
    if experiment.get('token_lookup_backend','gather')!=lookup_backend(args):
        raise RuntimeError('checkpoint token lookup backend mismatch')


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def validate_secondary_restore(experiment, args):
    if bool(experiment.get("recompute_secondary", False)) != bool(args.recompute_secondary):
        raise RuntimeError("checkpoint recompute-secondary contract mismatch; reviewed migration required")


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


def _compact_digest(value) -> str:
    """Stable compact digest for parity evidence without dumping rollout data."""
    digest = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    else:
        digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def rollout_digest(buffer) -> str:
    digest = hashlib.sha256()
    for row in buffer.rows:
        digest.update(repr((row.match_id, row.side, row.team_ids, row.request_index,
                           row.turn, int(row.request_kind), row.branch_slots,
                           row.action_ids, row.candidate_mask, row.selected,
                           row.selected_actions, row.old_logprob, row.value,
                           row.reward, row.done, row.actor_active, row.seed_ref)).encode())
    return digest.hexdigest()


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


def pin_deterministic_workspace(args):
    """Pin the original V2 CUDA numeric environment before CUDA initializes."""
    if args.deterministic_diagnostic:
        configured=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        if configured not in (None, ':4096:8'):
            raise RuntimeError('deterministic V2 execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'


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
            "cublas_workspace_config": os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
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
    "digest_wall_s", "team_stats_wall_s", "cleanup_wall_s", "iteration_barrier_wall_s",
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
    pin_deterministic_workspace(args)
    """One rank: local rollout, fixed-step DDP update, per-rank metrics JSON."""
    import torch.distributed as dist

    if args.precision_stability == "fp16-reduction-off":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if args.precision_stability == "default":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    effective_reduction = bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)

    worker_started = time.perf_counter()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if args.deterministic_diagnostic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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
    from agent.ppo.experimental_checkpoint import (
        build_envelope, capture_rng_state, hash_file, hash_json, load_envelope, restore_envelope,
        save_envelope, validate_epoch_override,
    )
    from agent.train.native_collector import NativeCollector, NativeCollectorConfig

    resume_envelope = None
    if args.experimental_resume:
        resume_root = Path(os.path.abspath(args.experimental_resume))
        resume_path = (resume_root / f"rank{rank}.pt") if resume_root.is_dir() else Path(f"{resume_root}.rank{rank}")
        if not os.path.isfile(resume_path):
            raise RuntimeError(f"experimental resume artifact is missing: {resume_path}")
        resume_envelope = load_envelope(resume_path)
    torch.manual_seed(args.seed + rank)
    import random
    import numpy as np
    random.seed(args.seed + rank)
    np.random.seed((args.seed + rank) % (2**32))
    engine = pa3_engine.NativeEngine(args.data, args.teams, workers=args.workers)
    model = build_model(PA3Config())
    if args.experimental_exact_token_lookup:
        from agent.model.exact_row_lookup import prepare_exact_lookup
        prepare_exact_lookup(device)
        model.scorer.experimental_exact_row_lookup=True
    model.configure_precision(args.precision_mode)
    model.encoder.trim_padding = args.trim_observation_padding
    historical_models = {}
    historical_manifest = []
    for hist_index, hist_path in enumerate(args.historical_checkpoint):
        hist_path = os.path.abspath(hist_path)
        if not os.path.isfile(hist_path):
            raise RuntimeError(f"historical checkpoint is missing: {hist_path}")
        payload = torch.load(hist_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "learner" not in payload:
            raise RuntimeError(f"historical checkpoint is not an experimental envelope: {hist_path}")
        from agent.contracts import require_checkpoint_identity
        require_checkpoint_identity(payload['learner'], args.precision_mode)
        historical = build_model(PA3Config(), device=device)
        historical.configure_precision(args.precision_mode)
        historical.load_state_dict(payload["learner"]["model"], strict=True)
        historical.eval()
        for parameter in historical.parameters():
            parameter.requires_grad_(False)
        identity = f"historical_{hist_index}"
        historical_models[identity] = historical
        model_hash = hashlib.sha256()
        for name, value in historical.state_dict().items():
            model_hash.update(name.encode())
            model_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        historical_manifest.append({"id": identity, "path": hist_path, "sha256": hash_file(hist_path),
                                    "model_digest": model_hash.hexdigest()})
    overrides = dict(
        global_minibatch_size=args.minibatch,
        microbatch_size=args.microbatch,
        per_rank_minibatch_size=args.minibatch // world_size,
        grad_accumulation_per_rank=max(1, (args.minibatch // world_size) // args.microbatch),
    )
    saved_epochs = (
        int(resume_envelope["learner"]["config"]["ppo"]["ppo_epochs"])
        if resume_envelope is not None else 4
    )
    if args.epochs:
        overrides["ppo_epochs"] = args.epochs
    elif resume_envelope is not None:
        overrides["ppo_epochs"] = saved_epochs
    if (args.experimental_resume or args.experimental_checkpoint) and overrides.get("ppo_epochs", 4) not in (2, 4):
        raise RuntimeError("experimental continuation permits only PPO epochs 2 or 4")
    ppo_config = PPOConfig(**overrides)
    lr_scale = float(args.experimental_lr_scale)
    if resume_envelope is not None:
        saved_scale_value = resume_envelope.get("experiment", {}).get("lr_scale")
        if saved_scale_value is not None and abs(float(saved_scale_value) - lr_scale) > 1e-12:
            raise RuntimeError("experimental resume LR scale mismatch; use an explicit matching arm")
    learner = PPOLearner(model, ppo_config, device=device, lr_scale=lr_scale)

    source_sha = source_provenance().get("git_sha")
    source_migration = resume_envelope.get("experiment", {}).get("source_migration") if resume_envelope else None
    dataset_manifest_hash = hash_file(args.dataset_manifest)
    from agent.contracts import OBSERVATION_SCHEMA,ACTION_SCHEMA,CONTRACT_HASH
    action_schema_hash = hash_json({
        "observation": OBSERVATION_SCHEMA,
        "actions": ACTION_SCHEMA,
        "candidates": "branch-candidates-v2",
        "semantic_hash": CONTRACT_HASH,
    })
    opponent_manifest = {"current": "current", "historical": historical_manifest,
                         "historical_probability": float(args.historical_probability)}
    model_config_hash = hash_json(model.config.to_dict())
    collector_contract = {
        "envs": args.envs, "workers": args.workers, "data": os.path.abspath(args.data),
        "teams": os.path.abspath(args.teams), "observation_mode": "fixed",
        "candidate_wire": "packed", "columnar_store": bool(args.columnar_store),
        "rolling_slots": bool(args.rolling_slots), "both_current_sides": True,
        "precision_mode": args.precision_mode,
    }
    collector_config_hash = hash_json(collector_contract)
    legacy_collector_contract = dict(collector_contract)
    legacy_collector_contract.pop("precision_mode", None)
    legacy_collector_contract_hash = hash_json(legacy_collector_contract)
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
    if resume_envelope is not None:
        saved_precision = resume_envelope.get("experiment", {}).get("precision_mode")
        if saved_precision is not None and saved_precision != args.precision_mode:
            raise RuntimeError(
                f"precision contract mismatch: checkpoint={saved_precision} requested={args.precision_mode}"
            )
        saved_stability = resume_envelope.get("experiment", {}).get("precision_stability")
        if saved_stability is None:
            legacy_ok = bool(args.experimental_allow_source_sha and resume_envelope.get("source_sha") in {
                "30b10e28d608f869811a4525afaa8101613c7adc",
                "0059b34179790610d0f39e832297be8645790a80",
                "575a706e2fed072e30d9f0dda0f8a3327fcb48aa",
            })
            if args.precision_mode == "D" and args.precision_stability != "default" and not legacy_ok:
                raise RuntimeError("checkpoint lacks required precision-stability metadata for Mode D")
            if legacy_ok:
                note("explicit trusted legacy migration: missing precision-stability metadata")
        elif saved_stability != args.precision_stability:
            raise RuntimeError(
                f"precision stability mismatch: checkpoint={saved_stability} requested={args.precision_stability}"
            )
        if args.precision_stability == "fp16-reduction-off" and effective_reduction:
            raise RuntimeError("effective CUDA FP16 reduction setting does not match D1 contract")
        saved_experiment=resume_envelope.get('experiment',{})
        validate_lookup_restore(saved_experiment,args)
        validate_secondary_restore(saved_experiment,args)
        if 'cublas_workspace_config' in saved_experiment:
            if saved_experiment['cublas_workspace_config']!=os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
                raise RuntimeError('checkpoint CUDA workspace execution contract mismatch')
        elif resume_envelope.get('source_sha')==source_sha:
            raise RuntimeError('same-source checkpoint missing CUDA workspace execution identity')
        validate_epoch_override(resume_envelope, ppo_config, ppo_config.ppo_epochs)
        recorded_source_sha = resume_envelope.get("source_sha")
        expected_source_sha = source_sha
        if recorded_source_sha != source_sha:
            trusted_migrations = {
                "30b10e28d608f869811a4525afaa8101613c7adc",
                "0059b34179790610d0f39e832297be8645790a80",
                "575a706e2fed072e30d9f0dda0f8a3327fcb48aa",
            }
            if args.experimental_source_migration:
                from agent.ppo.source_migration import validate_recovery_migration
                source_migration = validate_recovery_migration(
                    args.experimental_source_migration, ROOT, recorded_source_sha, source_sha)
                migration_receipt = json.loads(Path(args.experimental_source_migration).read_text())
                if source_migration.get('scope')=='v2-bounded-data-path-validation-only':
                    if args.iterations>5 or args.profile and args.effective_profile['profile']['version']=='pa3-v2-production-v1':
                        raise RuntimeError('V2 data-path migration is bounded validation only; production cutover is not approved')
                if str(resume_root.resolve()) != migration_receipt.get("checkpoint"):
                    raise RuntimeError("migration receipt checkpoint path mismatch")
                for peer in (0, 1):
                    if hash_file(resume_root / f"rank{peer}.pt") != migration_receipt["rank_artifact_hashes"][str(peer)]:
                        raise RuntimeError("migration checkpoint artifact checksum mismatch")
            elif not args.experimental_allow_source_sha or recorded_source_sha not in trusted_migrations:
                raise RuntimeError("source SHA mismatch; pass --experimental-allow-source-sha only for the recorded A3 source")
            expected_source_sha = recorded_source_sha
            note(f"explicit checkpoint migration: {recorded_source_sha} -> {source_sha}")
        restore_opponents = opponent_manifest
        if resume_envelope.get("opponent_manifest") != opponent_manifest:
            old_manifest = resume_envelope.get("opponent_manifest", {})
            if not args.experimental_allow_opponent_migration or old_manifest != {"current": "current", "historical": []}:
                raise RuntimeError("opponent manifest mismatch; explicit migration is required")
            note("explicit opponent-manifest migration from current-only A4 envelope")
            restore_opponents = old_manifest
        if bool(resume_envelope.get('experiment',{}).get('deterministic_diagnostic',False)) != bool(args.deterministic_diagnostic):
            raise RuntimeError('experimental resume deterministic execution contract mismatch')
        restore_envelope(
            learner, resume_envelope, dataset_manifest_hash=dataset_manifest_hash,
            action_schema_hash=action_schema_hash, source_sha=expected_source_sha,
            opponent_manifest=restore_opponents, expected_rank=rank,
            expected_world_size=world_size, expected_executor=args.executor,
            model_config_hash=model_config_hash, generators=None,
        )
        restart = resume_envelope.get("collector_restart", {})
        if restart.get("config_hash") != collector_config_hash:
            if restart.get("config_hash") == legacy_collector_contract_hash:
                note("explicit precision-contract migration from legacy A9/A5 collector envelope")
            else:
                raise RuntimeError("experimental resume collector contract mismatch")
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
    if resume_envelope is not None:
        # Restore the explicitly owned shuffle stream after the DDP reducer has
        # been recreated. The envelope has one generator per rank process.
        from agent.ppo.experimental_checkpoint import restore_rng_state
        restore_rng_state(resume_envelope["rng"], [shuffle_generator])
    initial_model_digest, initial_optimizer_digest = state_digest(model, learner.optimizer)
    if resume_envelope is not None:
        expected_digests = resume_envelope["rank_digests"][str(rank)]
        if {"model": initial_model_digest, "optimizer": initial_optimizer_digest} != expected_digests:
            raise RuntimeError("restored actual model/optimizer digests disagree with checkpoint")
        if learner.scheduler.state_dict() != resume_envelope["learner"]["scheduler"] or \
                learner.scaler.state_dict() != resume_envelope["learner"]["scaler"] or \
                learner.optimizer_steps != resume_envelope["learner"]["optimizer_steps"]:
            raise RuntimeError("restored scheduler/GradScaler/optimizer clock mismatch")
        if not torch.equal(shuffle_generator.get_state(), resume_envelope["rng"]["generators"][0]) or \
                not torch.equal(torch.get_rng_state(), resume_envelope["rng"]["torch_cpu"]) or \
                any(not torch.equal(actual, saved) for actual, saved in
                    zip(torch.cuda.get_rng_state_all(), resume_envelope["rng"]["torch_cuda"])):
            raise RuntimeError("restored owned/CPU/CUDA RNG mismatch")
        note(f"resume state verified: matches={learner.scheduler.matches} "
             f"optimizer_steps={learner.optimizer_steps} model={initial_model_digest} "
             f"optimizer={initial_optimizer_digest}")
    initial_scheduler_snapshot = dict(learner.scheduler.state_dict())
    initial_scaler_snapshot = dict(learner.scaler.state_dict())
    initial_cpu_rng_digest = _compact_digest(torch.get_rng_state())
    initial_cuda_rng_digest = _compact_digest(torch.cuda.get_rng_state(device))
    initial_shuffle_rng_digest = _compact_digest(shuffle_generator.get_state())
    global_games = int(resume_envelope["accounting"]["committed_matches"]) if resume_envelope else 0
    iteration_offset = int(resume_envelope["accounting"]["iteration"]) if resume_envelope else 0
    from agent.ppo.checkpoint_schedule import CommittedMatchCheckpointSchedule
    checkpoint_schedule = CommittedMatchCheckpointSchedule.from_clock(
        global_games, args.checkpoint_interval_matches)
    team_receipts = []
    stats_attempt = None
    if args.team_stats_root:
        from agent.analysis.team_stats import TeamAccumulator, stage as stage_team_stats, load as load_team_stats, run_identity
        dataset_identity = dict(teams=hash_file(args.teams), manifest=dataset_manifest_hash,
                                team_count=int(engine.team_count()))
        if dataset_identity != dict(teams='ecabd47a8d000ba8dd9f3010c09992505fe21e95efe244e2c8a3d75ac3f70b1f',
                                    manifest='1aa64507e17f4a1f819d59834dbd314e96a1a1b6d13f94a791bd3f51fca309bb', team_count=1220):
            raise RuntimeError('V2 team statistics require the authoritative frozen 1220-team dataset')
        identities = [None] * world_size
        dist.all_gather_object(identities, dataset_identity)
        if any(identity != identities[0] for identity in identities):
            raise RuntimeError('ranks disagree on actual native dataset identity')
        import uuid
        attempt = [uuid.uuid4().hex if rank == 0 else None]
        dist.broadcast_object_list(attempt, src=0)
        stats_attempt = attempt[0]
        if resume_envelope:
            team_receipts = resume_envelope.get('team_stats_receipts', [])
            if not team_receipts and global_games:
                raise RuntimeError('cannot resume team statistics without prior checkpoint receipts')
            for ref in team_receipts:
                value = load_team_stats(ref)
                if value['rank'] != rank or value['run_id'] != run_identity(args.team_stats_root):
                    raise RuntimeError('team statistics resume identity mismatch')
            from agent.analysis.team_stats import reconcile
            rank_receipts = [None] * world_size
            dist.all_gather_object(rank_receipts, team_receipts)
            reconcile([ref for refs in rank_receipts for ref in refs],
                      committed_matches=global_games, run_id=run_identity(args.team_stats_root))
    iterations = []
    torch.cuda.synchronize(device)
    startup_wall = time.perf_counter() - worker_started
    for index in range(args.iterations):
        absolute_iteration = iteration_offset + index
        iteration_start_matches = global_games
        team_accumulator = TeamAccumulator(int(engine.team_count())) if args.team_stats_root else None
        iteration_started = time.perf_counter()
        timings = {key: 0.0 for key in STAGE_WALL_KEYS}

        def finish_stage(key: str, started: float) -> None:
            torch.cuda.synchronize(device)
            timings[key] = time.perf_counter() - started

        stage_started = time.perf_counter()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        iteration_seed = args.seed + rank + absolute_iteration * world_size
        collector = NativeCollector(
            engine, model,
            NativeCollectorConfig(
                envs=args.envs, workers=args.workers, seed=iteration_seed, device=str(device),
                observation_mode="fixed", candidate_wire="packed",
                amp=True, inference_mode=True,
                precision_mode=args.precision_mode,
                recompute_provenance=args.recompute_secondary,
                columnar_observation_store=args.columnar_store,
                rolling_slots=args.rolling_slots,
                historical_probability=args.historical_probability,
            ),
            historical_models=historical_models,
            device=device,
            team_stats_sink=team_accumulator,
        )
        if communication is not None:
            communication.sync_calls = 0
            communication.no_sync_calls = 0
        finish_stage("iteration_setup_wall_s", stage_started)

        stage_started = time.perf_counter()
        buffer = collector.collect(args.games)
        finish_stage("collect_wall_s", stage_started)
        collected_rollout_digest = rollout_digest(buffer)
        if not buffer.rows:
            raise SystemExit(f"rank {rank} collected zero rows; refusing to run the DDP update")
        note(
            f"iteration {index + 1}/{args.iterations}: collected {collector.stats.games} games / "
            f"{len(buffer.rows)} rows in {timings['collect_wall_s']:.1f}s"
        )

        recompute = None
        if recompute_check is not None:
            stage_started = time.perf_counter()
            evidence_path = (f"{os.path.abspath(args.report)}.rank{rank}."
                             f"iteration{absolute_iteration + 1}.recompute.pt")
            recompute = recompute_check(learner, buffer, device, amp=True, audit=args.recompute_audit,
                                        secondary=args.recompute_secondary, evidence_path=evidence_path)
            finish_stage("recompute_wall_s", stage_started)
            note(
                f"recompute gate: {recompute['rows']} rows max|diff|="
                f"{recompute['max_abs_diff']:.3e} (tol {recompute['tolerance']:.1e}) "
                f"within={recompute['within_gate']}"
            )
            if recompute.get("secondary"):
                note(f"recompute secondary audit: {json.dumps(recompute['secondary'], sort_keys=True)}")
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
        if args.team_stats_root:
            if report.optimizer_steps_skipped or not collector.drain_complete or collector.in_flight_battles != 0:
                raise RuntimeError('team statistics refuse skipped-step or undrained iteration')
            if team_accumulator.matches != collector.stats.games:
                raise RuntimeError('natural match statistics disagree with collector')
            receipt = team_accumulator.receipt(
                run_id=run_identity(args.team_stats_root), rank=rank,
                iteration=absolute_iteration + 1, attempt_id=stats_attempt, source_sha=source_sha,
                dataset_manifest_sha256=dataset_manifest_hash, team_file_sha256=hash_file(args.teams),
                start_matches=iteration_start_matches, end_matches=global_games)
            team_receipts.append(stage_team_stats(args.team_stats_root, receipt))
        finish_stage('team_stats_wall_s', stage_started)

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
        stop_signal = [bool(args.stop_file and Path(args.stop_file).exists()) if rank == 0 else None]
        dist.broadcast_object_list(stop_signal, src=0)
        stop_requested = bool(stop_signal[0])
        if stop_requested and not args.experimental_checkpoint:
            raise RuntimeError('cooperative stop requires a resumable checkpoint destination')
        checkpoint_due = bool(args.experimental_checkpoint and
            (checkpoint_schedule.due(global_games) or args.checkpoint_first_iteration and absolute_iteration == 0 or stop_requested))
        if checkpoint_due:
            if args.rolling_slots:
                raise RuntimeError("experimental envelope requires static collector restart")
            if not collector.drain_complete or collector.in_flight_battles != 0:
                raise RuntimeError(
                    "experimental envelope requires the native collector to report a drained boundary"
                )
            envelope = build_envelope(
                learner,
                source_sha=source_sha,
                dataset_manifest_hash=dataset_manifest_hash,
                action_schema_hash=action_schema_hash,
                opponent_manifest=opponent_manifest,
                experiment={
                    "name": "ddp_ppo_iteration_boundary",
                    "ppo_epochs": ppo_config.ppo_epochs,
                    "parent_resume": bool(resume_envelope),
                    "lr_scale": lr_scale,
                    "source_migration_allowed": bool(args.experimental_allow_source_sha),
                    "source_migration": source_migration,
                    "recompute_secondary": bool(args.recompute_secondary),
                    "precision_mode": args.precision_mode,
                    "cublas_workspace_config": os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                    "token_lookup_backend": lookup_backend(args),
                    "deterministic_diagnostic": bool(args.deterministic_diagnostic),
                    "precision_stability": args.precision_stability,
                    "precision_contract_version": f"pa3-precision-{args.precision_mode}-v2",
                    "fp16_reduced_precision_reduction": effective_reduction,
                },
                rank=rank, world_size=world_size, executor=args.executor,
                iteration=absolute_iteration + 1,
                committed_matches=global_games,
                natural_games=int(collector.stats.games),
                learner_rows=len(buffer.rows),
                collector_restart={
                    "seed": args.seed, "iteration": absolute_iteration + 1,
                    "cohorts": int(collector.stats.cohorts),
                    "match_counter": int(getattr(collector, "_match_counter", 0)),
                    "slot_counter": int(getattr(collector, "_slot_counter", 0)),
                    "config_hash": collector_config_hash,
                },
                rng_generators=[shuffle_generator],
                rank_digests={str(index): value for index, value in enumerate(digests)},
                in_flight_battles=int(collector.in_flight_battles),
                iteration_complete=bool(collector.drain_complete),
                drain_status={
                    "complete": bool(collector.drain_complete),
                    "in_flight_battles": int(collector.in_flight_battles),
                    "source": "NativeCollector.collect static drain invariant",
                },
            )
            if args.team_stats_root:
                envelope['team_stats_receipts'] = list(team_receipts)
            save_envelope(
                f"{os.path.abspath(args.experimental_checkpoint)}.rank{rank}",
                envelope,
            )
            checkpoint_schedule.consume(global_games)
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
                "token_lookup_backend": lookup_backend(args),
                "compact_candidates": args.compact_candidates,
                "trim_observation_padding": args.trim_observation_padding,
                "microbatch": args.microbatch,
                "experimental_lr_scale": lr_scale,
                "historical_probability": args.historical_probability,
            },
            "sync_calls": communication.sync_calls if communication is not None else None,
            "no_sync_calls": communication.no_sync_calls if communication is not None else None,
            "report": report.as_dict(),
            "scheduled_learning_rate": learner.scheduler.lr_for_matches(global_games) / lr_scale,
            "effective_learning_rate": learner.scheduler.learning_rate,
            "experimental_lr_scale": lr_scale,
            "profile": dict(learner.profile),
            "prepare_profile": dict(learner.prepare_profile),
            "collector_stats": stats,
            "parity": {
                "initial_model_digest": initial_model_digest,
                "initial_optimizer_digest": initial_optimizer_digest,
                "initial_scheduler": initial_scheduler_snapshot,
                "initial_scaler": initial_scaler_snapshot,
                "cpu_rng_digest": initial_cpu_rng_digest,
                "cuda_rng_digest": initial_cuda_rng_digest,
                "shuffle_rng_digest": initial_shuffle_rng_digest,
                "collector_seed": iteration_seed,
                "absolute_iteration": absolute_iteration,
                "rollout_digest": collected_rollout_digest,
                "optimizer_steps": learner.optimizer_steps,
            },
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
        # Best-effort observability only. Rank-specific JSONL lives beside
        # the existing report and never participates in PPO/DDP collectives.
        # Persist only after model/optimizer parity and the barrier have passed.
        try:
            from agent.ppo.telemetry import append_training_update
            append_training_update(
                f"{os.path.abspath(args.report)}.rank{rank}.ppo.jsonl",
                iteration=absolute_iteration + 1,
                committed_matches=global_games,
                rank=rank,
                update=report,
                collector=stats,
                iteration_matches=iteration_games,
                iteration_wall_s=metrics["iteration_wall_s"],
                timing=timings,
                recompute=recompute,
                verification={"model_digest": model_digest, "optimizer_digest": optimizer_digest,
                              "digests_equal": bool(digests_equal), "source_sha": source_sha,
                              "source_migration": source_migration},
            )
        except (OSError, ValueError, TypeError) as exc:
            note(f"WARNING: nonfatal PPO telemetry write failure: {exc}")
        iterations.append(metrics)
        note(f"iteration {index + 1} complete in {metrics['iteration_wall_s']:.1f}s, digests equal")
        if stop_requested:
            note('cooperative stop requested; completed boundary saved, exiting both ranks')
            break

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
                "--experimental-lr-scale", str(args.experimental_lr_scale),
                "--historical-probability", str(args.historical_probability),
                "--data", args.data, "--teams", args.teams, "--port", str(port),
                "--report", args.report, "--checkpoint", args.checkpoint,
                "--team-stats-root", args.team_stats_root,
                '--stop-file', args.stop_file,
                "--experimental-checkpoint", args.experimental_checkpoint,
                "--experimental-resume", args.experimental_resume,
                "--checkpoint-interval-matches", str(args.checkpoint_interval_matches),
                "--rolling-checkpoint-root", args.rolling_checkpoint_root,
                "--dataset-manifest", args.dataset_manifest,
                "--executor", args.executor,
                "--precision-mode", args.precision_mode,
                "--precision-stability", args.precision_stability,
                "--experimental-source-migration", args.experimental_source_migration,
            ]
            if args.experimental_allow_source_sha:
                command.append("--experimental-allow-source-sha")
            if args.deterministic_diagnostic:
                command.append("--deterministic-diagnostic")
            if args.experimental_exact_token_lookup:
                command.append('--experimental-exact-token-lookup')
            if args.checkpoint_first_iteration:
                command.append('--checkpoint-first-iteration')
            if args.experimental_allow_opponent_migration:
                command.append("--experimental-allow-opponent-migration")
            for historical_path in args.historical_checkpoint:
                command.extend(["--historical-checkpoint", historical_path])
            if not args.recompute_gate:
                command.append("--no-recompute-gate")
            if args.recompute_audit:
                command.append("--recompute-audit")
            if args.recompute_secondary:
                command.append("--recompute-secondary")
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
        rolling_store = None
        published_rolling_keys = set()
        if args.rolling_checkpoint_root:
            if not args.experimental_checkpoint:
                raise RuntimeError("--rolling-checkpoint-root requires --experimental-checkpoint")
            from agent.ppo.storage_manager import CheckpointGenerationStore
            rolling_store = CheckpointGenerationStore(os.path.abspath(args.rolling_checkpoint_root))
        while any(proc.poll() is None for _, proc in procs):
            # A failed correctness gate must not leave its peer blocked in a
            # collective until the full process-group timeout expires.
            if any(proc.poll() not in (None, 0) for _, proc in procs):
                break
            if time.time() > deadline:
                failures.append("launcher watchdog timeout")
                break
            if rolling_store is not None:
                _publish_completed_rolling_boundary(args, rolling_store, published_rolling_keys)
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
    if args.rolling_checkpoint_root:
        from agent.ppo.storage_manager import CheckpointGenerationStore
        store = CheckpointGenerationStore(os.path.abspath(args.rolling_checkpoint_root))
        _publish_completed_rolling_boundary(args, store, set())
    expected_execution = {
        "batch_cache": args.batch_cache,
        "compact_candidates": args.compact_candidates,
        "trim_observation_padding": args.trim_observation_padding,
        "microbatch": args.microbatch,
        "experimental_lr_scale": args.experimental_lr_scale,
        "historical_probability": args.historical_probability,
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
    pin_deterministic_workspace(args)
    if args.worker:
        worker(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
