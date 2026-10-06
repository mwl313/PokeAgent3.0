"""Native batch/PyO3 binding smoke test (development only, no reference calls).

Build the extension first:
    bash scripts/build_python.sh
Then run:
    PYTHONPATH=engine/python python3 engine/python/test_binding.py --envs 64
Use `--envs 2048 --workers 16` to exercise a documented actor-process group.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import pa3_engine  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "engine", "data")
TEAMS = os.path.join(DATA, "training-teams.json")


def joint_action(engine, handle, side):
    """First feasible complete joint action for this side's request."""
    branches = engine.request_branches(handle[0], handle[1], side)
    prefix = []
    for _ in branches:
        candidates = engine.candidates(handle[0], handle[1], side, prefix)
        if not candidates:
            raise RuntimeError("request has no feasible completion")
        prefix.append(candidates[0])
    return prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-turns", type=int, default=400)
    args = parser.parse_args()

    started = time.perf_counter()
    engine = pa3_engine.NativeEngine(DATA, TEAMS, workers=args.workers)
    teams = engine.team_count()
    load = time.perf_counter() - started
    print(f"loaded dex + {teams} teams in {load:.2f}s, {engine.env_count()} envs")

    team_a, team_b, seeds, roles = [], [], [], []
    for i in range(args.envs):
        a = i % teams
        b = (i * 7 + 3) % teams
        team_a.append(a)
        team_b.append(b)
        seeds.append(((i * 11 + 1) & 0xFFFF, (i * 13 + 5) & 0xFFFF, (i * 17 + 7) & 0xFFFF, i & 0xFFFF))
        roles.append((0, 1))
    handles = engine.reset_batch(team_a, team_b, seeds, roles)
    assert len(handles) == args.envs

    finished = [False] * args.envs
    operational = {}
    obs_count = 0
    steps = 0
    request_cache = {}
    start = time.perf_counter()
    for turn in range(args.max_turns):
        pending = []
        for env, handle in enumerate(handles):
            if finished[env]:
                continue
            needs = []
            for side in (0, 1):
                kind = engine.request_kind(handle[0], handle[1], side)
                if kind in (0, 1, 2):
                    needs.append((side, joint_action(engine, handle, side)))
            if needs:
                pending.append((handle[0], handle[1], needs))
        if not pending:
            break
        results = engine.step_batch(pending)
        steps += len(pending)
        # Observations for the envs that just moved (both player views).
        view_handles = [(entry[0], entry[1]) for entry in pending]
        sides = [0] * len(pending) + [1] * len(pending)
        blobs = engine.observe_encoded_batch(view_handles + view_handles, sides)
        for blob in blobs:
            view = pa3_engine.parse_view(blob)
            assert view["schema_version"] == pa3_engine.SCHEMA_VERSION
            assert view["token_mask"].shape == (pa3_engine.OBSERVATION_TOKENS,)
            obs_count += 1
        by_slot = {entry[0]: i for i, entry in enumerate(pending)}
        for slot, generation, _ in pending:
            index = by_slot[slot]
            accepted, terminated, truncated, winner, error, k0, k1 = results[index]
            assert accepted
            if error is not None:
                # Explicit operational errors are engine coverage gaps, not
                # bridge failures: the batch path must survive them.
                operational[error] = operational.get(error, 0) + 1
                finished[slot] = True
                continue
            assert not truncated
            if terminated:
                assert winner in (0, 1, None)
                finished[slot] = True
        if all(finished):
            break
    elapsed = time.perf_counter() - start
    done = sum(finished)
    print(
        f"{done}/{args.envs} natural battles in {elapsed:.2f}s "
        f"({done / elapsed:.1f} games/s), {steps} batch submissions, "
        f"{obs_count} packed observations"
    )
    if operational:
        top = sorted(operational.items(), key=lambda kv: -kv[1])[:10]
        print(f"{len(operational)} distinct operational gaps over {sum(operational.values())} envs:")
        for message, count in top:
            print(f"  {count:5d}  {message}")
    if done != args.envs:
        print("WARNING: not every battle terminated naturally")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
