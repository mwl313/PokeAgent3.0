#!/usr/bin/env python3
"""One gloo rank of the D1 fixed-step DDP protocol check (CPU).

The pytest wrapper ``test_ddp_collective_protocol.py`` launches two of these
and asserts on the JSON they produce. Kept as a standalone script so the child
process has an explicit, pytest-independent import path (the same pattern as
``scripts/run_ddp_ppo.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT, REPO_ROOT / "engine" / "python"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch  # noqa: E402

from tests_integration_helpers import collect_rows, small_config  # noqa: E402


CASES = [
    {"name": "uneven_trajectories", "mode": "uneven"},
    {"name": "empty_rank", "mode": "empty"},
]


def learner_config(**overrides):
    from agent.ppo import PPOConfig

    base = dict(
        global_minibatch_size=64,
        microbatch_size=16,
        per_rank_minibatch_size=32,
        grad_accumulation_per_rank=2,
        ppo_epochs=1,
        sample_weighted_ddp_reduction=False,
    )
    base.update(overrides)
    return PPOConfig(**base)


def reference_update(buffer, rows, seed: int):
    from agent.model import build_model
    from agent.ppo import PPOLearner

    model = build_model(small_config())
    learner = PPOLearner(model, learner_config(), device="cpu", amp=False)
    batch = learner.prepare_batch(buffer, rows=rows)
    report = learner.update(
        batch,
        committed_matches=1_000,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    return learner, report


def moments_by_name(learner):
    state = learner.optimizer.state
    return {
        name: state.get(parameter, {})
        for name, parameter in learner.model.named_parameters()
    }


def run_case(rank: int, world_size: int, case: dict, seed: int, cache_device=None) -> dict:
    from agent.model import build_model
    from agent.ppo import PPOLearner
    from agent.ppo.ddp import DDPCommunication

    buffer = collect_rows(seed)
    rows = list(buffer.rows)
    # Shard by complete (match_id, side) trajectories: GAE chains never cross
    # ranks in the real runs (each rank owns whole games), and a partial chain
    # would legitimately produce different advantages.
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row.match_id, row.side), []).append(row)
    ordered = [groups[key] for key in sorted(groups)]
    # Keep every rank inside one local minibatch (per_rank_minibatch=32 rows):
    # the fast first implementation's parity contract (plan §5.2). The global
    # minibatch (64) still holds the complete union of both shards.
    shard0: list = []
    shard1: list = []
    for index, group in enumerate(ordered):
        if shard1:
            break
        if len(shard0) + len(group) > 28:
            shard1 = list(group if case["mode"] == "uneven" else [])
            break
        shard0.extend(group)
    shard = shard0 if rank == 0 else shard1

    model = build_model(small_config())
    learner = PPOLearner(model, learner_config(), device="cpu", amp=False)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, find_unused_parameters=True
    )
    learner.attach_ddp(ddp_model, world_size)
    communication = DDPCommunication(ddp_model)
    plan = learner.prepare_streaming_ddp(buffer, rows=shard, cache_device=cache_device)
    assert (plan.cached_batch is not None) == (cache_device == "cpu" and bool(shard))
    fallback = buffer.to_batch(rows[:1], device="cpu") if not shard else None
    report = learner.update_ddp(
        plan,
        committed_matches=1_000,
        generator=torch.Generator().manual_seed(seed + 1),
        communication=communication,
        fallback_batch=fallback,
    )

    reference_rows = shard0 + shard1
    reference_learner, reference_report = reference_update(buffer, reference_rows, seed)
    reference_parameters = dict(reference_learner.model.named_parameters())
    reference_state = reference_learner.model.state_dict()
    model_deltas = {
        name: float((value - reference_state[name]).abs().max())
        for name, value in learner.model.state_dict().items()
    }
    worst_name = max(model_deltas, key=model_deltas.get)
    worst = learner.model.state_dict()[worst_name]
    reference_worst = reference_state[worst_name]
    worst_index = int((worst - reference_worst).abs().flatten().argmax())
    reference_batch = reference_learner.prepare_batch(buffer, rows=reference_rows)
    index_of = {id(row): index for index, row in enumerate(reference_rows)}
    positions = [index_of[id(row)] for row in shard]
    advantage_delta = (
        float(
            (
                plan.advantages
                - reference_batch.advantages[torch.tensor(positions, dtype=torch.long)]
            ).abs().max()
        )
        if len(shard)
        else 0.0
    )
    optimizer_delta = 0.0
    reference_moments = moments_by_name(reference_learner)
    moments = moments_by_name(learner)
    for name, entry in moments.items():
        reference_entry = reference_moments[name]
        for key in ("exp_avg", "exp_avg_sq"):
            if key in entry and key in reference_entry:
                optimizer_delta = max(
                    optimizer_delta,
                    float((entry[key] - reference_entry[key]).abs().max()),
                )
    digest = torch.cat(
        [
            value.detach().float().reshape(-1)
            for value in learner.model.state_dict().values()
        ]
    )
    return {
        "rank": rank,
        "rows": len(shard),
        "total_rows": len(reference_rows),
        "advantage_delta": advantage_delta,
        "max_weight_delta": max(model_deltas.values()),
        "worst_parameter": worst_name,
        "worst_parameter_delta": model_deltas[worst_name],
        "worst_initial": float(worst.flatten()[worst_index]),
        "worst_reference": float(reference_worst.flatten()[worst_index]),
        "grad_norm": report.grad_norm,
        "reference_grad_norm": reference_report.grad_norm,
        "policy_loss": report.policy_loss,
        "reference_policy_loss": reference_report.policy_loss,
        "max_optimizer_delta": optimizer_delta,
        "optimizer_steps": learner.optimizer_steps,
        "reference_optimizer_steps": reference_learner.optimizer_steps,
        "skipped": report.optimizer_steps_skipped,
        "epoch_kl": list(report.epoch_approx_kl),
        "reference_epoch_kl": list(reference_report.epoch_approx_kl),
        "learning_rate": learner.scheduler.learning_rate,
        "reference_learning_rate": reference_learner.scheduler.learning_rate,
        "sync_calls": communication.sync_calls,
        "no_sync_calls": communication.no_sync_calls,
        "profile_sync_steps": learner.profile.get("ddp_sync_steps"),
        "state_sha": hashlib.sha256(digest.numpy().tobytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--cache-device", choices=("none", "cpu"), default="none")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{args.port}",
        rank=args.rank,
        world_size=args.world_size,
    )
    try:
        results = {
            case["name"]: run_case(
                args.rank, args.world_size, case, args.seed,
                None if args.cache_device == "none" else args.cache_device,
            )
            for case in CASES
        }
        gathered = [None] * args.world_size
        dist.all_gather_object(gathered, results)
        if args.rank == 0:
            pathlib.Path(args.out).write_text(json.dumps(gathered, indent=2))
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
