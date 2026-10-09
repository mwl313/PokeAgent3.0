#!/usr/bin/env python3
"""Compare real PA3-8M FP16 outputs/gradients across isolated source roots.

Run this file once with --make-fixture and the reference root, then with the
same fixture under the candidate root. No reference Python modules are shared
between the processes. The small fixture includes rows across the rollout.
"""
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--fixture", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--make-fixture", action="store_true")
parser.add_argument("--compact-candidates", action="store_true")
parser.add_argument("--trim-observation-padding", action="store_true")
args = parser.parse_args()
sys.path[:0] = [args.root, os.path.join(args.root, "engine", "python")]

import torch
from agent.model import PA3Config, build_model
from agent.ppo import PPOConfig, PPOLearner

torch.set_num_threads(1)
torch.cuda.set_device(0)
torch.manual_seed(20261009)
model = build_model(PA3Config())
model.encoder.trim_padding = args.trim_observation_padding
learner = PPOLearner(model, PPOConfig(), device="cuda:0", amp=True)
if args.make_fixture:
    import pa3_engine
    from agent.train.native_collector import NativeCollector, NativeCollectorConfig
    engine = pa3_engine.NativeEngine(
        os.path.join(args.root, "engine/data"),
        os.path.join(args.root, "engine/data/training-teams.json"), workers=4,
    )
    collector = NativeCollector(engine, model, NativeCollectorConfig(
        envs=32, workers=4, seed=20261009, device="cuda:0",
        observation_mode="fixed", candidate_wire="packed", amp=True, inference_mode=True,
    ), device="cuda:0")
    buffer = collector.collect(32)
    buffer.compute_gae()
    indices = torch.linspace(0, len(buffer.rows) - 1, 128).long()
    batch = buffer.to_batch([buffer.rows[int(i)] for i in indices])
    batch = batch.with_advantages(batch.raw_advantages)
    torch.save({"batch": batch, "model": model.cpu().state_dict()}, args.fixture)
    model.cuda(0)
fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
model.load_state_dict(fixture["model"])
batch = fixture["batch"]
if args.compact_candidates:
    occupied = batch.candidates.mask.any(dim=0).any(dim=0).nonzero().flatten()
    width = max(1, int(occupied.max()) + 1)
    batch.candidates = batch.candidates.trim_padding(width)
batch = batch.to("cuda:0")
terms = learner._forward_terms(batch)
loss = terms["loss_unscaled"] + learner.config.value_coefficient * terms["value_mean"]
learner.scaler.scale(loss).backward()
learner.scaler.unscale_(learner.optimizer)
result = {
    "outputs": {key: value.detach().cpu() for key, value in terms.items()},
    "gradients": {name: p.grad.detach().cpu() for name, p in model.named_parameters()
                  if p.grad is not None},
}
assert all(torch.isfinite(value).all() for value in result["gradients"].values())
torch.save(result, args.out)
print(json.dumps({"rows": len(batch), "loss": float(loss.detach()),
                  "gradients": len(result["gradients"])}))
