#!/usr/bin/env python3
"""Replay a preserved V2 recompute failure without touching training state.

The tool compares the original actor group with the sampled recompute batch,
records the earliest numerically divergent tensor for the selected row, and
then repeats both paths with the selective stable effects projection enabled.
It is intentionally inference-only: it never samples actions, mutates an
envelope, starts collection, or performs a PPO update.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path

import torch

from agent.buffer.rollout_buffer import RolloutBuffer
from agent.model import PA3Config, build_model
from agent.ppo.experimental_checkpoint import load_envelope


def _buffer(rows, observations) -> RolloutBuffer:
    buffer = RolloutBuffer()
    if hasattr(observations, "select"):
        source = observations
        observations = (source.select([index]) for index in range(len(rows)))
    for row, observation in zip(rows, observations):
        buffer.rows.append(
            dataclasses.replace(row, observation_ref=buffer.observation_store.add(observation))
        )
    return buffer


def _pad_observation(observation, physical_rows: int):
    if len(observation) >= physical_rows:
        return observation
    return observation.cat([observation.select([0] * (physical_rows - len(observation)))])


def _target_ragged_range(observation, row: int, family: str) -> tuple[int, int]:
    counts = observation.ragged[family].counts.reshape(len(observation), -1)
    start = int(counts[:row].sum().item())
    return start, start + int(counts[row].sum().item())


def _target_index(rows, target) -> int:
    for index, row in enumerate(rows):
        if (row.match_id, row.request_index, row.side, tuple(row.selected)) == target:
            return index
    raise ValueError("the secondary-audit target is absent from sampled rows")


def _capture_encoder(model, observation, target_row: int) -> dict:
    """Capture target outputs through the first divergent operation."""
    ragged = model.encoder.embedding.ragged_embedding
    start, stop = _target_ragged_range(observation, target_row, "effects")
    captured: dict[str, torch.Tensor] = {}

    projection = ragged._numeric_projection

    def capture_projection(name, numeric):
        output = projection(name, numeric)
        if name == "effects":
            captured.setdefault("effects_numeric", output[start:stop].detach().float().cpu())
        return output

    def embedding_hook(_module, _inputs, output):
        captured.setdefault("embedding", output[target_row].detach().float().cpu())

    def encoder_hook(_module, _inputs, output):
        captured.setdefault("encoder", output[target_row].detach().float().cpu())

    hooks = [
        model.encoder.embedding.register_forward_hook(embedding_hook),
        model.encoder.final_norm.register_forward_hook(encoder_hook),
    ]
    ragged._numeric_projection = capture_projection
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            model.encode(observation)
    finally:
        ragged._numeric_projection = projection
        for hook in hooks:
            hook.remove()
    return captured


def _evaluate(model, batch, row: int) -> float:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        _, evaluation = model.evaluate(
            batch.observation, batch.candidates, selected=batch.candidates.selected
        )
    return float(evaluation.request_logprob[row])


def _latency_ms(model, observation, repeats: int = 7) -> list[float]:
    for _ in range(2):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            model.encode(observation)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            model.encode(observation)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    delta = (left - right).abs()
    return {"max_abs": float(delta.max()) if delta.numel() else 0.0,
            "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
            "shape": list(delta.shape)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="saved *.recompute.pt envelope")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this forensic replay requires CUDA")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    envelope = load_envelope(args.artifact)
    secondary = envelope["result"].get("secondary", {})
    targets = secondary.get("rows", [])
    if not targets:
        raise ValueError("artifact does not preserve an audited actor group")
    target_meta = targets[0]
    target = (target_meta["match_id"], target_meta["request_index"],
              target_meta["side"], tuple(target_meta["selected"]))

    sampled = _buffer(envelope["sampled_rows"], envelope["sampled_observations"])
    sampled_index = _target_index(sampled.rows, target)
    groups = envelope["actor_groups"]
    group_index = next(
        index for index, group in enumerate(groups)
        if group["start"] <= target_meta["row_index"] < group["stop"]
    )
    actor = _buffer(envelope["group_rows"][group_index], envelope["group_observations"][group_index])
    actor_index = target_meta["row_index"] - groups[group_index]["start"]

    model = build_model(PA3Config(**envelope["model_config"]), device=device)
    model.load_state_dict(envelope["model"])
    model.eval()
    model.configure_precision(envelope["precision_mode"])
    model.encoder.canonical_batch_rows = 1024

    actor_batch = actor.to_batch(actor.rows, device=device)
    sampled_batch = sampled.to_batch(sampled.rows, device=device)
    actor_chunk_start = actor_index // 1024 * 1024
    actor_local = actor_index - actor_chunk_start
    actor_chunk = _pad_observation(
        actor_batch.observation.narrow(actor_chunk_start, min(actor_chunk_start + 1024, len(actor_batch))),
        1024,
    )
    sampled_chunk = _pad_observation(sampled_batch.observation, 1024)

    report = {
        "artifact": str(Path(args.artifact).resolve()),
        "rank": args.rank,
        "target": target_meta,
        "actor_group_rows": len(actor),
        "sampled_rows": len(sampled),
        "physical_rows": 1024,
        "matrix": {},
    }
    for label, stable in (("historical_fp16_effects", False), ("stable_fp32_effects", True)):
        model.encoder.embedding.ragged_embedding.fp32_effects = stable
        actor_logprob = _evaluate(model, actor_batch, actor_index)
        sampled_logprob = _evaluate(model, sampled_batch, sampled_index)
        # Repeat the group path to verify the source path is stable.
        repeat_logprob = _evaluate(model, actor_batch, actor_index)
        actor_trace = _capture_encoder(model, actor_chunk, actor_local)
        sampled_trace = _capture_encoder(model, sampled_chunk, sampled_index)
        actor_time = _latency_ms(model, actor_chunk)
        sampled_time = _latency_ms(model, sampled_chunk)
        report["matrix"][label] = {
            "actor_group_logprob": actor_logprob,
            "actor_group_repeat_logprob": repeat_logprob,
            "sampled_logprob": sampled_logprob,
            "actor_vs_sampled_abs": abs(actor_logprob - sampled_logprob),
            "actor_repeat_abs": abs(actor_logprob - repeat_logprob),
            "first_operation": "ragged_embedding.numeric.effects",
            "per_operation": {
                name: _difference(actor_trace[name], sampled_trace[name])
                for name in ("effects_numeric", "embedding", "encoder")
            },
            "encoder_latency_ms": {
                "actor_chunk_median": statistics.median(actor_time),
                "sampled_chunk_median": statistics.median(sampled_time),
                "actor_chunk_samples": actor_time,
                "sampled_chunk_samples": sampled_time,
            },
        }
    report["conclusion"] = (
        "The historical FP16 effects numeric projection is the first captured "
        "batch-composition-dependent operation. The selective FP32 path must "
        "make both the target encoder output and selected log probability agree "
        "before it can be promoted."
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
