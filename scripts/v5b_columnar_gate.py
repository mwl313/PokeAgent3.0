#!/usr/bin/env python3
"""v5b T1/T2 gate: columnar observation store must be row-identical.

Collects the same input twice (old per-row dict store vs SoA columnar store),
compares the full-row SHA256 (reusing the W2 `row_digest` utility), the
collector statistics and the model manifest, and additionally checks the
store-level `get`/`stacked`/`stacked_indices` round trips against the inline
store for the same indices.

Usage:
    PYTHONPATH=engine/python:. .venv/bin/python scripts/v5b_columnar_gate.py \
        --games 512 --envs 256 --workers 8 \
        --out runs/perf/v5b/columnar_gate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine", "python"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import pa3_engine  # noqa: E402
from agent.model import PA3Config, build_model  # noqa: E402
from agent.train.native_collector import NativeCollector, NativeCollectorConfig  # noqa: E402
from v5_drain_tail import row_digest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/perf/v5b/columnar_gate.json"))
    return parser.parse_args()


def collect(args, columnar: bool):
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
            columnar_observation_store=columnar,
        ),
        device=device,
    )
    buffer = collector.collect(args.games)
    model_sha = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        model_sha.update(name.encode())
        model_sha.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return collector, buffer, model_sha.hexdigest()


def store_roundtrip(store, indices) -> dict:
    batch = store.stacked_indices(indices)
    direct = batch.token_mask.to(torch.uint8).numpy()
    checks = {}
    for index, position in enumerate(indices):
        single = store.get(int(position))
        checks[str(int(position))] = bool(
            np.array_equal(single.categories[0].numpy(), batch.categories[index].numpy())
            and np.array_equal(single.floats[0].numpy(), batch.floats[index].numpy())
        )
    stacked = store.stacked()
    return {
        "gather_rows": int(direct.shape[0]),
        "per_row_match": all(checks.values()),
        "rows_checked": len(checks),
        "stacked_rows": len(stacked),
    }


def main() -> int:
    args = parse_args()
    old_collector, old_buffer, old_model_sha = collect(args, columnar=False)
    new_collector, new_buffer, new_model_sha = collect(args, columnar=True)
    old_digest = row_digest(old_buffer)
    new_digest = row_digest(new_buffer)

    indices = [0, len(new_buffer) // 2, max(len(new_buffer) - 1, 0)]
    roundtrip = store_roundtrip(new_buffer.observation_store, indices)
    payload = {
        "config": {
            "games": args.games,
            "envs": args.envs,
            "workers": args.workers,
            "seed": args.seed,
        },
        "old": {
            "rows": old_digest["rows"],
            "row_sha256": old_digest["sha256"],
            "model_sha256": old_model_sha,
            "stats": {key: old_collector.stats.as_dict()[key]
                      for key in ("games", "decisions", "learner_rows", "wins", "losses", "draws", "reward_sum", "operational_errors")},
        },
        "columnar": {
            "rows": new_digest["rows"],
            "row_sha256": new_digest["sha256"],
            "model_sha256": new_model_sha,
            "stats": {key: new_collector.stats.as_dict()[key]
                      for key in ("games", "decisions", "learner_rows", "wins", "losses", "draws", "reward_sum", "operational_errors")},
            "store_type": type(new_buffer.observation_store).__name__,
            "store_roundtrip": roundtrip,
        },
        "gates": {
            "row_count_equal": old_digest["rows"] == new_digest["rows"],
            "row_sha_equal": old_digest["sha256"] == new_digest["sha256"],
            "stats_equal": old_collector.stats.as_dict()["games"] == new_collector.stats.as_dict()["games"]
            and old_collector.stats.as_dict()["decisions"] == new_collector.stats.as_dict()["decisions"],
            "model_sha_equal": old_model_sha == new_model_sha,
            "store_roundtrip_ok": bool(roundtrip["per_row_match"]),
        },
    }
    payload["gates"]["all_pass"] = all(payload["gates"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "gates": payload["gates"],
                      "old_rows": old_digest["rows"], "new_rows": new_digest["rows"]}, indent=2))
    return 0 if payload["gates"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
