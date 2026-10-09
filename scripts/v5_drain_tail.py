#!/usr/bin/env python3
"""W2: read-only collection drain-tail measurement (rolling-slot precursor).

Collects real both-seat games with `NativeCollectorConfig.telemetry` on/off and
writes a compact JSON: per-round active/open env counts, finished-but-idle
slot-seconds, game-length distribution (P50/P95/P99/max), cohort 50%->100% tail
walls, plus a content digest of the produced rows so the telemetry-on and
telemetry-off runs can be proven identical.

Telemetry is read-only: with `--no-telemetry` the collector is byte-for-byte
the training path and the digest must match the telemetry-on run.

Usage:
    PYTHONPATH=engine/python:. .venv/bin/python scripts/v5_drain_tail.py \
        --games 2048 --envs 1024 --workers 16 --telemetry \
        --out runs/perf/v5/tail_2k.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine", "python"))

import pa3_engine  # noqa: E402
from agent.model import PA3Config, build_model  # noqa: E402
from agent.train.native_collector import NativeCollector, NativeCollectorConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--telemetry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rolling-slots", action="store_true")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/perf/v5/tail.json"))
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return float(ordered[index])


def row_digest(buffer, chunk: int = 2048) -> dict:
    """Content digest over every stored row (chunked, no full-iteration blowup)."""
    digest = hashlib.sha256()
    rows = list(buffer.rows)
    for begin in range(0, len(rows), chunk):
        batch = buffer.to_batch(rows[begin:begin + chunk], device="cpu")
        for name, tensor in (
            ("old_logprob", batch.old_logprob.float()),
            ("values", batch.values.float()),
            ("rewards", batch.rewards.float()),
            ("dones", batch.dones.to(torch.uint8)),
            ("actor_mask", batch.actor_mask.to(torch.uint8)),
            ("row_valid", batch.row_valid.to(torch.uint8)),
            ("match_ids", batch.match_ids.to(torch.int64)),
            ("sides", batch.sides.to(torch.int64)),
            ("request_index", batch.request_index.to(torch.int64)),
            ("turns", batch.turns.to(torch.int64)),
            ("request_kind", batch.request_kind.to(torch.int64)),
            ("selected", batch.candidates.selected.to(torch.int64)),
            ("candidate_mask", batch.candidates.mask.to(torch.uint8)),
            ("action_ids", batch.candidates.action_ids.to(torch.int64)),
            ("token_mask", batch.observation.token_mask.to(torch.uint8)),
            ("categories", batch.observation.categories.to(torch.int64)),
            ("floats", batch.observation.floats.float()),
            ("flags", batch.observation.flags.to(torch.int64)),
        ):
            digest.update(name.encode())
            digest.update(tensor.contiguous().numpy().tobytes())
    return {"rows": len(rows), "sha256": digest.hexdigest()}


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    engine = pa3_engine.NativeEngine(
        os.path.join(ROOT, "engine", "data"),
        os.path.join(ROOT, "engine", "data", "training-teams.json"),
        workers=args.workers,
    )
    model = build_model(PA3Config(), device=device)
    collector = NativeCollector(
        engine,
        model,
        NativeCollectorConfig(
            envs=args.envs,
            workers=args.workers,
            seed=args.seed,
            device=str(device),
            observation_mode="fixed",
            candidate_wire="packed",
            amp=True,
            inference_mode=True,
            telemetry=args.telemetry,
            rolling_slots=args.rolling_slots,
        ),
        device=device,
    )
    model_sha = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        model_sha.update(name.encode())
        model_sha.update(parameter.detach().cpu().contiguous().numpy().tobytes())

    started = time.perf_counter()
    buffer = collector.collect(args.games)
    wall = time.perf_counter() - started
    digest = row_digest(buffer)

    payload: dict = {
        "config": {
            "games_target": args.games,
            "envs": args.envs,
            "workers": args.workers,
            "seed": args.seed,
            "telemetry": bool(args.telemetry),
            "rolling_slots": bool(args.rolling_slots),
            "device": str(device),
        },
        "model_sha256": model_sha.hexdigest(),
        "rows": digest,
        "wall_seconds": wall,
        "games_per_second": collector.stats.games / max(wall, 1e-9),
        "collector_stats": collector.stats.as_dict(),
    }
    if args.telemetry:
        telemetry = collector.telemetry
        game_rounds = [float(value) for value in telemetry.get("game_rounds", [])]
        round_seconds = [float(value) for value in telemetry.get("round_seconds", [])]
        active_envs = [float(value) for value in telemetry.get("active_envs", [])]
        open_envs = [float(value) for value in telemetry.get("open_envs", [])]
        cohorts = telemetry.get("cohorts", [])
        tails = [float(c["tail_seconds_50_to_100"]) for c in cohorts if c.get("tail_seconds_50_to_100")]
        slot_seconds = float(telemetry.get("slot_seconds", 0.0))
        idle_slot_seconds = float(telemetry.get("idle_slot_seconds", 0.0))
        payload["telemetry"] = {
            "rounds": telemetry.get("rounds", 0),
            "round_seconds": round_seconds,
            "active_envs": active_envs,
            "open_envs": open_envs,
            "game_rounds": {
                "count": len(game_rounds),
                "p50": percentile(game_rounds, 0.50),
                "p95": percentile(game_rounds, 0.95),
                "p99": percentile(game_rounds, 0.99),
                "max": max(game_rounds) if game_rounds else 0.0,
                "mean": statistics.mean(game_rounds) if game_rounds else 0.0,
            },
            "active_envs_p50": percentile(active_envs, 0.50),
            "active_envs_p05": percentile(active_envs, 0.05),
            "cohorts": cohorts,
            "idle_slot_seconds": idle_slot_seconds,
            "slot_seconds": slot_seconds,
            "idle_slot_fraction": idle_slot_seconds / max(slot_seconds, 1e-9),
            "tail_seconds_total": sum(tails),
            "tail_share_of_wall": sum(tails) / max(wall, 1e-9),
            "rolling_slot_upper_bound_seconds": sum(tails),
            "rolling_slot_note": (
                "upper bound: sum of each cohort's 50%->100% wall assumes rolling "
                "refills could remove the entire second-half drain tail; it is an "
                "optimistic ceiling, not a promise"
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        {
            "out": str(args.out),
            "games": collector.stats.games,
            "rows": digest["rows"],
            "row_sha256": digest["sha256"][:16],
            "wall_s": round(wall, 2),
            "games_per_s": round(payload["games_per_second"], 2),
            "telemetry": bool(args.telemetry),
            "tail_share_of_wall": payload.get("telemetry", {}).get("tail_share_of_wall"),
            "idle_slot_fraction": payload.get("telemetry", {}).get("idle_slot_fraction"),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
