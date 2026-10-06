#!/usr/bin/env python3
"""Development benchmark: packed observation -> numpy -> torch tensor cost.

Measures the Python-side consumption of the native `observe_encoded_batch`
payload on a fixture-cohort batch. The engine's Rust-side packing cost is
reported separately by the actor; this script answers whether decoding and
tensor conversion add a meaningful multiple on top of it.

Run:
    PYTHONPATH=engine/python .venv/bin/python engine/python/bench_observation.py

No Pokemon Showdown execution and no training-run side effects.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pa3_engine  # noqa: E402
from pa3_engine.observation import parse_batch, parse_view  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "engine", "data")
FIXTURE_TEAMS = os.path.join(ROOT, "engine", "benchmarks", "fixture_teams.json")


def build_batch(engine, envs, seed=20261007):
    import random

    rng = random.Random(seed)
    teams = engine.team_count()
    team_a = [rng.randrange(teams) for _ in range(envs)]
    team_b = [rng.randrange(teams) for _ in range(envs)]
    seeds = [tuple(rng.randrange(1 << 16) for _ in range(4)) for _ in range(envs)]
    roles = [(0, 1) if i % 2 == 0 else (1, 0) for i in range(envs)]
    return engine.reset_batch(team_a, team_b, seeds, roles)


def collect(engine, handles, rounds):
    """Advance with the first legal completion until `rounds` observation sets."""
    batches = []
    for _ in range(rounds):
        selected = []
        for handle in handles:
            for side in (0, 1):
                kind = engine.request_kind(handle[0], handle[1], side)
                if kind in (0, 1, 2):
                    selected.append((handle, side))
        if not selected:
            break
        hs = [h for h, _ in selected]
        sides = [s for _, s in selected]
        blobs = engine.observe_encoded_batch(hs, sides)
        fixed, ragged = engine.observe_fixed_batch(hs, sides)
        batches.append((selected, blobs, fixed, ragged))
        per_env = {}
        for (handle, side) in selected:
            prefix = []
            while True:
                candidates = engine.candidates(handle[0], handle[1], side, prefix)
                if not candidates:
                    break
                prefix.append(candidates[0])
            per_env.setdefault(handle, []).append((side, prefix))
        specs = [(h[0], h[1], actions) for h, actions in per_env.items()]
        engine.step_batch(specs)
    return batches


def stack_numpy(blobs):
    """Stack a batch of blobs into dense arrays (padding ragged rows)."""
    views = [parse_view(blob) for blob in blobs]
    categories = np.stack([v["categories"] for v in views])
    floats = np.stack([v["floats"] for v in views])
    flags = np.stack([v["flags"] for v in views])
    masks = np.stack([v["token_mask"] for v in views])
    return categories, floats, flags, masks


def decode_zero_copy(fixed, ragged, count):
    batch = parse_batch(fixed, ragged, count)
    return (
        batch["categories"],
        batch["floats"],
        batch["flags"],
        batch["token_mask"],
    )


def stack_torch(blobs):
    import torch

    categories, floats, flags, masks = stack_numpy(blobs)
    return (
        torch.from_numpy(categories.copy()).long(),
        torch.from_numpy(floats.copy()).float(),
        torch.from_numpy(flags.copy()).float(),
        torch.from_numpy(masks.copy()).bool(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--teams", default=FIXTURE_TEAMS)
    args = parser.parse_args()

    engine = pa3_engine.NativeEngine(DATA, args.teams, workers=16)
    handles = build_batch(engine, args.envs)

    start = time.perf_counter()
    batches = collect(engine, handles, args.rounds)
    collect_s = time.perf_counter() - start
    total_views = sum(len(blobs) for _, blobs, _, _ in batches)
    total_bytes = sum(len(blob) for _, blobs, _, _ in batches for blob in blobs)

    start = time.perf_counter()
    for _, blobs, _, _ in batches:
        stack_numpy(blobs)
    numpy_s = time.perf_counter() - start

    start = time.perf_counter()
    for _, _, fixed, ragged in batches:
        decode_zero_copy(fixed, ragged, len(fixed) // pa3_engine.OBSERVATION_FIXED_BYTES)
    zerocopy_s = time.perf_counter() - start

    torch_s = None
    if not args.skip_torch:
        try:
            start = time.perf_counter()
            for _, blobs, _, _ in batches:
                stack_torch(blobs)
            torch_s = time.perf_counter() - start
        except ImportError:
            torch_s = None

    rounds = max(len(batches), 1)
    print(f"views: {total_views} over {rounds} rounds ({total_views / rounds:.1f}/round)")
    print(f"payload: {total_bytes / 1e6:.2f} MB ({total_bytes / max(total_views, 1) / 1024:.1f} kB/view)")
    print(f"collect (native round-trip): {1000 * collect_s / rounds:.2f} ms/round")
    print(f"numpy decode+stack:          {1000 * numpy_s / rounds:.3f} ms/round")
    print(f"fixed batch zero-copy view:  {1000 * zerocopy_s / rounds:.3f} ms/round")
    if torch_s is not None:
        print(f"torch tensor conversion:     {1000 * torch_s / rounds:.3f} ms/round")
    else:
        print("torch tensor conversion:     skipped")
    print(f"engine observe-only share:   see actor metrics (batch_execution_report.md)")


if __name__ == "__main__":
    main()
