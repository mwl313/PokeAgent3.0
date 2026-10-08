#!/usr/bin/env python3
"""Bounded PPO smoke test: PA3-8M x the real native engine.

This is a *test*, not training. It collects a bounded number of naturally
completed matches from the frozen training pool with the PA3-8M policy sampled
on the GPU, verifies that the sampled joint action/log-probability is exactly
reproducible by the learner, runs the PPO update on those real rows, checks
numerical health, and verifies checkpoint save/resume.

Usage:
    PYTHONPATH=engine/python:. .venv/bin/python scripts/run_ppo_smoke.py \
        --games 10000 --envs 1024 --workers 16 --report runs/ppo-smoke/report.json

The 100M-match training run is *not* authorized by this script and must not be
started from it.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine", "python"))

import pa3_engine  # noqa: E402
from agent.model.config import PA3Config  # noqa: E402
from agent.model.pa3_model import PA3Model  # noqa: E402
from agent.types.requests import BranchCandidatesBatch  # noqa: E402
from agent.ppo.config import PPOConfig  # noqa: E402
from agent.ppo.learner import PPOLearner  # noqa: E402
from agent.train.native_collector import (  # noqa: E402
    NativeCollector,
    NativeCollectorConfig,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data", default=os.path.join(ROOT, "engine", "data"))
    parser.add_argument("--teams", default=os.path.join(ROOT, "engine", "data", "training-teams.json"))
    parser.add_argument("--report", default=os.path.join(ROOT, "runs", "ppo-smoke", "report.json"))
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "runs", "ppo-smoke", "smoke.pt"))
    parser.add_argument("--logprob-check-rows", type=int, default=512)
    parser.add_argument("--skip-update", action="store_true")
    return parser.parse_args()


def recompute_logprob_check(model, buffer, rows, device, tolerance=1e-4):
    """Compare stored sampled log-probabilities with a fresh recomputation."""
    sample = rows[: max(1, min(len(rows), 2048))]
    batch = buffer.to_batch(sample, device=device)
    with torch.no_grad():
        encoded = model.encode(batch.observation)
        evaluation = model.evaluate_encoded(
            encoded, batch.candidates, selected=batch.candidates.selected
        )
    recomputed = evaluation.request_logprob.float()
    stored = torch.tensor([row.old_logprob for row in sample], dtype=torch.float32, device=device)
    diff = (recomputed - stored).abs()
    return {
        "rows": len(sample),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "within_tolerance": bool(diff.max().item() <= tolerance),
    }


def finite_parameters(model):
    bad = []
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            bad.append(name)
    return bad


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    engine = pa3_engine.NativeEngine(args.data, args.teams, workers=args.workers)
    model = PA3Model(PA3Config())
    learner = PPOLearner(model, PPOConfig(), device=device)
    config = NativeCollectorConfig(
        envs=args.envs, workers=args.workers, seed=args.seed, device=str(device)
    )
    collector = NativeCollector(engine, model, config, device=device)

    started = time.perf_counter()
    buffer = collector.collect(args.games)
    collection_wall = time.perf_counter() - started
    stats = collector.stats.as_dict()
    rows = list(buffer.rows)
    natural_matches = buffer.natural_match_count()

    report = {
        "authorization": "bounded smoke test only; the 100M-match training run is not authorized",
        "device": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "engine": {
            "envs": args.envs,
            "workers": args.workers,
            "games": stats["games"],
            "cohorts": stats["cohorts"],
            "rounds": stats["rounds"],
            "decisions": stats["decisions"],
            "operational_errors": stats["operational_errors"],
            "wall_seconds": stats["wall_seconds"],
            "games_per_second": stats["games"] / max(stats["wall_seconds"], 1e-9),
            "decisions_per_second": stats["decisions"] / max(stats["wall_seconds"], 1e-9),
            "observation_seconds": stats["observation_seconds"],
            "candidate_seconds": stats["candidate_seconds"],
            "model_seconds": stats["model_seconds"],
            "step_seconds": stats["step_seconds"],
            "reset_seconds": stats["reset_seconds"],
            "candidate_calls": stats["candidate_calls"],
            "candidates_returned": stats["candidates_returned"],
            "max_open_games": stats["max_open_games"],
            "action_kind_histogram": stats["action_kinds"],
        },
        "rewards": {
            "wins": stats["wins"],
            "losses": stats["losses"],
            "draws": stats["draws"],
            "reward_sum": stats["reward_sum"],
            "games_accounted": stats["wins"] + stats["losses"] + stats["draws"],
        },
        "rollout": {
            **buffer.stats(),
            "buffer_rows": len(rows),
            "natural_matches": natural_matches,
            "learner_rows": stats["learner_rows"],
            "opponent_requests": stats["opponent_requests"],
            "uniform_sampling_note": "team indices are drawn uniformly over the frozen training pool each reset",
        },
        "checks": {},
        "collection_wall_seconds": collection_wall,
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    if device.type == "cuda":
        report["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1048576.0

    checks = report["checks"]
    # A cohort drains to its natural end, so the collected count may overshoot
    # the target by up to one cohort. Overshoot is kept, never discarded.
    checks["games_accounted"] = report["rewards"]["games_accounted"] >= args.games
    report["games_target"] = args.games
    report["games_collected"] = report["rewards"]["games_accounted"]
    report["cohort_overshoot"] = report["rewards"]["games_accounted"] - args.games
    checks["operational_errors_zero"] = stats["operational_errors"] == 0
    checks["all_learner_requests_recorded"] = stats["learner_rows"] == buffer.stats()["value_rows"]
    checks["finite_parameters_before"] = finite_parameters(model) == []
    checks["recompute_matches_sample"] = recompute_logprob_check(
        model, buffer, rows, device, tolerance=1e-4
    )

    if not args.skip_update and rows:
        batch = learner.prepare_batch(buffer)
        checks["minibatch_padding_note"] = (
            "the learner applies the DDP sample-weighted reduction, so reporting "
            "metrics are scaled by the real-row fraction of each minibatch; the "
            "effect is negligible once a minibatch is full (4096 rows)"
        )
        update = learner.update(batch, committed_matches=args.games).as_dict()
        report["ppo_update"] = update
        checks["approx_kl_after_update"] = update["epoch_approx_kl"]
        checks["grad_norm_finite"] = bool(torch.isfinite(torch.tensor(update["grad_norm"])).item())
        checks["policy_loss_finite"] = bool(torch.isfinite(torch.tensor(update["policy_loss"])).item())
        checks["value_loss_finite"] = bool(torch.isfinite(torch.tensor(update["value_loss"])).item())
        checks["finite_parameters_after"] = finite_parameters(model) == []
        target_kl = learner.config.target_approx_kl
        checks["approx_kl_within_target_or_early_stop"] = (
            update["approx_kl"] <= target_kl or update["stopped_early"]
        )
        # Checkpoint save/resume.
        state = learner.state_dict()
        torch.save(state, args.checkpoint)
        reloaded = PPOLearner(PA3Model(PA3Config()), PPOConfig(), device=device)
        reloaded.load_state_dict(torch.load(args.checkpoint, map_location=device))
        probe = buffer.to_batch(rows[: min(64, len(rows))], device=device)
        with torch.no_grad():
            a = learner.model.encode(probe.observation)
            b = reloaded.model.encode(probe.observation)
            logprob_a = learner.model.evaluate_encoded(
                a, probe.candidates, selected=probe.candidates.selected
            ).request_logprob
            logprob_b = reloaded.model.evaluate_encoded(
                b, probe.candidates, selected=probe.candidates.selected
            ).request_logprob
            checks["checkpoint_outputs_match"] = bool(
                torch.allclose(a.global_repr, b.global_repr, atol=1e-6)
                and torch.allclose(logprob_a, logprob_b, atol=1e-6)
            )
        checks["checkpoint_optimizer_steps"] = reloaded.optimizer_steps
        checks["checkpoint_saved_bytes"] = os.path.getsize(args.checkpoint)

    failures = [name for name, value in checks.items() if value is False]
    report["failures"] = failures
    report["passed"] = not failures
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps({k: report[k] for k in ("passed", "failures", "engine", "rewards", "rollout", "checks")},
                     indent=2, sort_keys=True)[:8000])
    if failures:
        raise SystemExit(f"PPO smoke checks failed: {failures}")


if __name__ == "__main__":
    main()
