#!/usr/bin/env python3
"""C1: single-GPU PPO oracle on real PA3-8M rows.

Collects real both-seat rows with the native collector, then pins the
single-process objective the distributed executors must reproduce:

* FP32 raw-gradient parity of the largest fitting microbatch (the reference)
  against smaller microbatch splits under exact row-weighted accumulation;
* FP16 autocast + GradScaler parity with a separately declared tolerance;
* Adam multi-step parity (1/3 consecutive updates): weights, moments, step
  counter, scaler, LR and epoch KL;
* checkpoint save/resume continuing identical updates.

The 4096-row single backward is deliberately not attempted: the v3 sweep
measured ~9.25 GiB at micro 1024 and ~17.98 GiB at micro 2048, so a single
4096-row pass exceeds the documented 28 GiB soft budget. The reference is the
largest fitting split and is labelled as such in the JSON; no number here is
extrapolated from it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=384)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument(
        "--splits", type=int, nargs="+", default=[64, 128, 256, 512, 1024]
    )
    parser.add_argument("--reference-microbatch", type=int, default=2048)
    parser.add_argument("--adam-steps", type=int, default=3)
    parser.add_argument("--data", default=os.path.join(ROOT, "engine", "data"))
    parser.add_argument(
        "--teams", default=os.path.join(ROOT, "engine", "data", "training-teams.json")
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("runs/perf/v4/c1_single_gpu_oracle.json"),
    )
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def gpu_inventory() -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,power.limit,temperature.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def model_digest(model: PA3Model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def group_gradients(model: PA3Model) -> dict[str, torch.Tensor]:
    groups: dict[str, list[torch.Tensor]] = {"encoder": [], "scorer": [], "value_head": []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("encoder"):
            target = "encoder"
        elif name.startswith("scorer"):
            target = "scorer"
        elif name.startswith("value_head"):
            target = "value_head"
        else:
            target = "other"
            groups.setdefault(target, [])
        groups[target].append(parameter.grad.detach().float().reshape(-1))
    return {
        group: torch.cat(chunks) if chunks else torch.zeros(0, device="cpu")
        for group, chunks in groups.items()
    }


def compare_gradients(
    reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for group, reference_grad in reference.items():
        candidate_grad = candidate[group]
        difference = candidate_grad - reference_grad
        reference_norm = float(reference_grad.norm())
        scale = max(reference_norm, 1e-12)
        comparison[group] = {
            "reference_norm": reference_norm,
            "candidate_norm": float(candidate_grad.norm()),
            "delta_norm": float(difference.norm()),
            "relative_norm_delta": float(difference.norm()) / scale,
            "max_abs_delta": float(difference.abs().max()) if difference.numel() else 0.0,
            "max_abs_reference": float(reference_grad.abs().max()) if reference_grad.numel() else 0.0,
            "cosine": (
                float(
                    torch.nn.functional.cosine_similarity(
                        reference_grad.unsqueeze(0), candidate_grad.unsqueeze(0)
                    )
                )
                if reference_grad.numel() and reference_norm > 0
                else None
            ),
        }
    return comparison


def accumulated_loss_and_grads(
    learner: PPOLearner,
    batch,
    microbatch: int,
    *,
    unscaled: bool,
) -> tuple[float, float, dict[str, torch.Tensor], float]:
    """Exact row-weighted accumulation over one fixed row ordering.

    Returns (objective loss, sum of per-micro scaled losses, gradients,
    wall seconds). ``microbatch <= 0`` means one pass over the whole batch.
    """
    device = learner.device
    batch = batch.to(device)
    actor_total = float((batch.actor_mask & batch.row_valid).sum().item())
    valid_total = float(batch.row_valid.sum().item())
    ranges = (
        [(0, len(batch))]
        if microbatch <= 0 or microbatch >= len(batch)
        else [(begin, min(begin + microbatch, len(batch))) for begin in range(0, len(batch), microbatch)]
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    learner.optimizer.zero_grad(set_to_none=True)
    accumulated = torch.zeros((), dtype=torch.float32, device=device)
    for begin, end in ranges:
        micro = batch.select(
            torch.arange(begin, end, dtype=torch.long, device=batch.old_logprob.device)
        )
        terms = learner._forward_terms(micro)
        actor_scale = terms["actor_count"] / max(actor_total, 1.0)
        value_scale = terms["value_count"] / max(valid_total, 1.0)
        loss = (
            terms["loss_unscaled"] * actor_scale
            + learner.config.value_coefficient * terms["value_mean"] * value_scale
        )
        if unscaled:
            loss.backward()
        else:
            learner.scaler.scale(loss).backward()
        accumulated = accumulated + loss.detach()
    if not unscaled:
        # Manual unscale: `GradScaler.unscale_` records per-optimizer state and
        # would refuse a second call without an intervening `update()`. No step
        # is taken here, so dividing by the current (fixed) scale is exact.
        scale = float(learner.scaler.get_scale())
        for parameter in learner.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(scale)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    gradients = {
        group: tensor.cpu() for group, tensor in group_gradients(learner.model).items()
    }
    peak_vram_mib = (
        float(torch.cuda.max_memory_reserved(device)) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    return float(accumulated), wall, gradients, peak_vram_mib


def _nonfinite_gradients(model: PA3Model) -> int:
    count = 0
    for parameter in model.parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            count += 1
    return count


def _config(rows: int, microbatch: int, *, epochs: int, amp: bool) -> PPOConfig:
    return PPOConfig(
        global_minibatch_size=rows,
        microbatch_size=microbatch,
        per_rank_minibatch_size=rows,
        grad_accumulation_per_rank=max(1, rows // microbatch),
        ppo_epochs=epochs,
        amp_grad_scaler=amp,
        sample_weighted_ddp_reduction=False,
        exact_row_weighted_accumulation=True,
    )


def collect(args) -> tuple[torch.device, PA3Model, object, object, dict]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    engine = pa3_engine.NativeEngine(args.data, args.teams, workers=args.workers)
    model = PA3Model(PA3Config())
    learner = PPOLearner(
        model,
        _config(args.rows, args.reference_microbatch, epochs=1, amp=False),
        device=device,
    )
    collector_config = NativeCollectorConfig(
        envs=args.envs,
        workers=args.workers,
        seed=args.seed,
        device=str(device),
        observation_mode="fixed",
        amp=False,
        inference_mode=True,
        candidate_wire="packed",
        collect_both_sides_when_current_self_play=True,
    )
    collector = NativeCollector(engine, model, collector_config, device=device)
    started = time.perf_counter()
    buffer = collector.collect(args.games)
    wall = time.perf_counter() - started
    stats = collector.stats
    collection = {
        "target_games": args.games,
        "collection_wall_seconds": wall,
        "games_per_second": stats.games / wall if wall > 0 else None,
        "collector": {
            "cohorts": stats.cohorts,
            "rounds": stats.rounds,
            "games": stats.games,
            "decisions": stats.decisions,
            "learner_rows": stats.learner_rows,
            "operational_errors": stats.operational_errors,
            "wins": stats.wins,
            "losses": stats.losses,
            "draws": stats.draws,
            "reward_sum": stats.reward_sum,
        },
        "rollout_rows": len(buffer),
        "natural_matches": buffer.natural_match_count(),
        "rows_per_match": len(buffer) / max(buffer.natural_match_count(), 1),
        "model_sha256": model_digest(model),
        "device": str(device),
        "gpu": gpu_inventory(),
        "torch": torch.__version__,
    }
    return device, model, learner, buffer, collection


def main() -> int:
    args = parse_args()
    device, model, learner, buffer, collection = collect(args)
    prepared = learner.prepare_batch(buffer)
    if len(prepared) < args.rows:
        raise SystemExit(
            f"only {len(prepared)} rows collected; need {args.rows}. Increase --games."
        )
    selected = prepared.select(torch.arange(args.rows, dtype=torch.long))
    selected = selected.to(device)
    actor_rows = int((selected.actor_mask & selected.row_valid).sum().item())
    batch_info = {
        "rows": len(selected),
        "actor_rows": actor_rows,
        "value_rows": int(selected.row_valid.sum().item()),
        "matches": int(selected.match_ids.unique().numel()),
        "sides_present": sorted(int(v) for v in selected.sides.unique().tolist()),
        "policy_ids": sorted(set(selected.policy_ids)),
        "illegal_action_rows": 0,
    }

    # --- FP32 raw-gradient parity -------------------------------------------
    initial_state = {k: v.detach().clone() for k, v in learner.model.state_dict().items()}
    fp32_reference = {}
    fp32_splits = {}
    reference_microbatch = args.reference_microbatch
    reference_loss, reference_wall, reference_grads, reference_vram = accumulated_loss_and_grads(
        learner, selected, reference_microbatch, unscaled=True
    )
    fp32_reference = {
        "microbatch": reference_microbatch,
        "objective": reference_loss,
        "wall_seconds": reference_wall,
        "gradients": reference_grads,
        "peak_vram_mib": reference_vram,
        "nonfinite_gradients": _nonfinite_gradients(learner.model),
    }
    for microbatch in args.splits:
        loss, wall, gradients, vram = accumulated_loss_and_grads(
            learner, selected, microbatch, unscaled=True
        )
        fp32_splits[microbatch] = {
            "microbatch": microbatch,
            "objective": loss,
            "loss_delta_vs_reference": loss - reference_loss,
            "wall_seconds": wall,
            "peak_vram_mib": vram,
            "nonfinite_gradients": _nonfinite_gradients(learner.model),
            "gradient_comparison": compare_gradients(reference_grads, gradients),
        }

    # --- FP16 autocast + GradScaler parity ----------------------------------
    fp16_model = copy.deepcopy(model)
    fp16_learner = PPOLearner(
        fp16_model,
        _config(args.rows, reference_microbatch, epochs=1, amp=True),
        device=device,
        amp=True,
    )
    fp16_batch = selected.to(device)
    fp16_reference_loss, fp16_reference_wall, fp16_reference_grads, fp16_reference_vram = (
        accumulated_loss_and_grads(fp16_learner, fp16_batch, reference_microbatch, unscaled=False)
    )
    fp16_splits = {}
    for microbatch in sorted(set(args.splits)):
        loss, wall, gradients, vram = accumulated_loss_and_grads(
            fp16_learner, fp16_batch, microbatch, unscaled=False
        )
        fp16_splits[microbatch] = {
            "microbatch": microbatch,
            "objective": loss,
            "objective_delta_vs_reference": loss - fp16_reference_loss,
            "wall_seconds": wall,
            "peak_vram_mib": vram,
            "nonfinite_gradients": _nonfinite_gradients(fp16_learner.model),
            "gradient_comparison": compare_gradients(fp16_reference_grads, gradients),
        }
    fp16_grad_scale = fp16_learner.scaler.get_scale()

    # --- Adam multi-step parity ----------------------------------------------
    def update_steps(model_seed: PA3Model, microbatch: int, steps: int, amp: bool):
        torch.manual_seed(args.seed)
        step_model = copy.deepcopy(model_seed)
        config = _config(args.rows, microbatch, epochs=1, amp=amp)
        step_learner = PPOLearner(step_model, config, device=device, amp=amp)
        step_batch = step_learner.prepare_batch(buffer)
        step_batch = step_batch.select(torch.arange(args.rows, dtype=torch.long)).to(device)
        losses = []
        epoch_kls = []
        skipped = 0
        for step in range(steps):
            report = step_learner.update(
                step_batch,
                committed_matches=step * 128,
                generator=torch.Generator().manual_seed(700 + step),
            )
            losses.append(float(report.policy_loss))
            epoch_kls.append(float(report.epoch_approx_kl[-1]) if report.epoch_approx_kl else None)
            skipped += int(report.optimizer_steps_skipped)
        return step_learner, losses, epoch_kls, skipped

    full_learner, full_losses, full_kls, full_skipped = update_steps(
        model, reference_microbatch, args.adam_steps, amp=False
    )
    split_learner, split_losses, split_kls, split_skipped = update_steps(
        model, min(args.splits), args.adam_steps, amp=False
    )
    adam_comparison = {}
    for name, left in full_learner.model.state_dict().items():
        right = split_learner.model.state_dict()[name]
        adam_comparison[name] = {
            "max_abs_delta": float((left - right).abs().max()),
            "relative_norm_delta": float((left - right).norm())
            / max(float(left.norm()), 1e-12),
        }
    full_opt = full_learner.optimizer.state_dict()
    split_opt = split_learner.optimizer.state_dict()
    moment_comparison = {}
    for full_param, split_param in zip(
        full_opt["param_groups"][0]["params"], split_opt["param_groups"][0]["params"]
    ):
        left, right = full_opt["state"][full_param], split_opt["state"][split_param]
        moment_comparison[str(full_param)] = {
            "step": [int(left["step"]), int(right["step"])],
            "exp_avg_max_abs_delta": float((left["exp_avg"] - right["exp_avg"]).abs().max()),
            "exp_avg_sq_max_abs_delta": float(
                (left["exp_avg_sq"] - right["exp_avg_sq"]).abs().max()
            ),
        }
    adam = {
        "steps": args.adam_steps,
        "reference_microbatch": reference_microbatch,
        "split_microbatch": min(args.splits),
        "optimizer_steps": [full_learner.optimizer_steps, split_learner.optimizer_steps],
        "skipped_steps": [full_skipped, split_skipped],
        "learning_rate": [
            full_learner.optimizer.param_groups[0]["lr"],
            split_learner.optimizer.param_groups[0]["lr"],
        ],
        "scheduler_matches": [full_learner.scheduler.matches, split_learner.scheduler.matches],
        "policy_loss": [full_losses, split_losses],
        "epoch_approx_kl": [full_kls, split_kls],
        "parameter_comparison": adam_comparison,
        "moment_comparison": moment_comparison,
    }

    # --- Checkpoint save/resume ---------------------------------------------
    torch.manual_seed(args.seed)
    checkpoint_model = copy.deepcopy(model)
    checkpoint_learner = PPOLearner(
        checkpoint_model,
        _config(args.rows, min(args.splits), epochs=1, amp=False),
        device=device,
    )
    checkpoint_batch = checkpoint_learner.prepare_batch(buffer)
    checkpoint_batch = checkpoint_batch.select(torch.arange(args.rows, dtype=torch.long)).to(device)
    checkpoint_learner.update(
        checkpoint_batch,
        committed_matches=0,
        generator=torch.Generator().manual_seed(11),
    )
    checkpoint_state = copy.deepcopy(checkpoint_learner.state_dict())
    checkpoint_learner.update(
        checkpoint_batch,
        committed_matches=128,
        generator=torch.Generator().manual_seed(12),
    )
    reference_after = copy.deepcopy(checkpoint_learner.state_dict())

    resumed_model = copy.deepcopy(model)
    resumed = PPOLearner(
        resumed_model, _config(args.rows, min(args.splits), epochs=1, amp=False), device=device
    )
    resumed.load_state_dict(checkpoint_state)
    resumed_batch = resumed.prepare_batch(buffer)
    resumed_batch = resumed_batch.select(torch.arange(args.rows, dtype=torch.long)).to(device)
    resumed.update(
        resumed_batch, committed_matches=128, generator=torch.Generator().manual_seed(12)
    )
    resumed_after = copy.deepcopy(resumed.state_dict())
    resume_comparison = {
        name: float((reference_after["model"][name] - resumed_after["model"][name]).abs().max())
        for name in reference_after["model"]
    }
    optimizer_moment_delta = 0.0
    for key in reference_after["optimizer"]["state"]:
        left = reference_after["optimizer"]["state"][key]
        right = resumed_after["optimizer"]["state"][key]
        optimizer_moment_delta = max(
            optimizer_moment_delta,
            float((left["exp_avg"] - right["exp_avg"]).abs().max()),
            float((left["exp_avg_sq"] - right["exp_avg_sq"]).abs().max()),
        )

    payload = {
        "gate": "C1",
        "git_head": git_output("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git_output("status", "--porcelain", "--untracked-files=no")),
        "seed": args.seed,
        "collection": collection,
        "batch": batch_info,
        "model_parameter_count": int(learner.model.parameter_count()),
        "reference_boundary": {
            "note": (
                "single 4096-row backward exceeds the 28 GiB soft VRAM budget; the "
                "reference is the largest fitting split, documented rather than extrapolated"
            ),
            "reference_microbatch": reference_microbatch,
        },
        "fp32": {
            "tolerance": {"relative_norm_delta_max": 1e-4, "max_abs_delta": 1e-5},
            "reference": {
                key: value
                for key, value in fp32_reference.items()
                if key != "gradients"
            },
            "splits": fp32_splits,
        },
        "fp16": {
            "tolerance": {"relative_norm_delta_max": 1e-3, "max_abs_delta": 1e-4},
            "grad_scale_after": fp16_grad_scale,
            "reference": {
                "microbatch": reference_microbatch,
                "objective": fp16_reference_loss,
                "wall_seconds": fp16_reference_wall,
                "peak_vram_mib": fp16_reference_vram,
                "nonfinite_gradients": _nonfinite_gradients(fp16_learner.model),
            },
            "splits": fp16_splits,
        },
        "adam": adam,
        "checkpoint": {
            "model_max_abs_delta": max(resume_comparison.values()) if resume_comparison else 0.0,
            "optimizer_moment_max_abs_delta": optimizer_moment_delta,
            "optimizer_steps": [checkpoint_learner.optimizer_steps, resumed.optimizer_steps],
            "scheduler_matches": [checkpoint_learner.scheduler.matches, resumed.scheduler.matches],
        },
        "fixture_restored": all(
            torch.equal(initial_state[name], learner.model.state_dict()[name].detach())
            for name in initial_state
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "fp32": {
                    micro: {
                        "relative_norm_delta_max": max(
                            entry["gradient_comparison"][group]["relative_norm_delta"]
                            for group in entry["gradient_comparison"]
                        ),
                        "max_abs_delta": max(
                            entry["gradient_comparison"][group]["max_abs_delta"]
                            for group in entry["gradient_comparison"]
                        ),
                    }
                    for micro, entry in fp32_splits.items()
                },
                "fp16": {
                    micro: {
                        "relative_norm_delta_max": max(
                            entry["gradient_comparison"][group]["relative_norm_delta"]
                            for group in entry["gradient_comparison"]
                        ),
                        "max_abs_delta": max(
                            entry["gradient_comparison"][group]["max_abs_delta"]
                            for group in entry["gradient_comparison"]
                        ),
                    }
                    for micro, entry in fp16_splits.items()
                },
                "adam_max_weight_delta": max(
                    entry["max_abs_delta"] for entry in adam_comparison.values()
                ),
                "checkpoint_model_delta": payload["checkpoint"]["model_max_abs_delta"],
                "batch": batch_info,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
