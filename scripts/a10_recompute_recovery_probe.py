#!/usr/bin/env python3
"""One frozen-policy collection and bounded precision audit. No PPO updates."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "engine/python"), str(ROOT / "scripts")]
import torch
import pa3_engine
from agent.model import PA3Config, build_model
from agent.ppo import PPOLearner, PPOConfig
from agent.ppo.experimental_checkpoint import load_envelope
from agent.train.native_collector import NativeCollector, NativeCollectorConfig
from bench_pa3_end_to_end import recompute_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=20261646)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--games", type=int, default=2048)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(torch.device(args.device))
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    payload = load_envelope(args.checkpoint)
    learner = PPOLearner(build_model(PA3Config()), PPOConfig(**payload["learner"]["config"]["ppo"]),
                         device=args.device, lr_scale=0.5)
    learner.load_state_dict(payload["learner"])
    learner.model.configure_precision("D")
    run = Path("/home/minidc/Desktop/PokeAgent3.0-A")
    engine = pa3_engine.NativeEngine(str(run / "engine/data"), str(run / "engine/data/a10-training-teams.json"), workers=16)
    collector = NativeCollector(engine, learner.model, NativeCollectorConfig(
        envs=1024, workers=16, seed=args.seed, device=args.device,
        observation_mode="fixed", candidate_wire="packed", amp=True, inference_mode=True,
        precision_mode="D", recompute_provenance=True), device=args.device)
    started = time.perf_counter()
    buffer = collector.collect(args.games)
    collect_wall = time.perf_counter() - started
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    result = recompute_check(learner, buffer, torch.device(args.device), amp=True, audit=True,
                             secondary=True, evidence_path=args.report + ".recompute.pt")
    result.update(collector=collector.stats.as_dict(), collect_wall_s=collect_wall,
                  actor_groups=len(buffer.actor_groups), seed=args.seed,
                  checkpoint=args.checkpoint,
                  rng_unchanged=torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state()))
    # Operational report; this diagnostic does not increment production clock.
    from agent.evaluation.a10_league import atomic
    atomic(args.report, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
