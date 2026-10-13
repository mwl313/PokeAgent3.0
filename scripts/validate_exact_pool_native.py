"""Bounded model-free frozen-pool smoke; natural completion is not oracle parity."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import pa3_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rounds", type=int, default=1500)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / "engine/data"
    pool = data / "a10-training-teams.json"
    digest = hashlib.sha256(pool.read_bytes()).hexdigest()
    engine = pa3_engine.NativeEngine(str(data), str(pool), workers=16)
    count = engine.team_count()
    assert count == 1220
    a, b, seeds = [], [], []
    for team in range(count):
        opponent = (team * 7 + 3) % count
        a.extend([team, opponent]); b.extend([opponent, team])
        seeds.extend([(2026, 10, 14, team + 1)] * 2)
    handles = engine.reset_batch(a, b, seeds, [(0, 1)] * len(a))
    rng = random.Random(20261014)
    finished = set(); completed = []; errors = []; retries = 0
    started = time.perf_counter()
    for round_no in range(args.max_rounds):
        specs = []; order = []
        for env, h in enumerate(handles):
            if env in finished:
                continue
            choices = []
            for side in (0, 1):
                if engine.request_kind(*h, side) not in (0, 1, 2):
                    continue
                prefix = []
                for _ in engine.request_branches(*h, side):
                    candidates = engine.candidates(*h, side, prefix)
                    assert candidates
                    prefix.append(candidates[rng.randrange(len(candidates))])
                choices.append((side, prefix))
            assert choices, f"no request at env {env}"
            specs.append((*h, choices)); order.append(env)
        if not specs:
            break
        results = engine.step_batch(specs)
        for env, result in zip(order, results):
            accepted, terminated, truncated, winner, error, *_ = result
            if error or truncated:
                errors.append({"env": env, "team_a": a[env], "team_b": b[env],
                               "seed": seeds[env], "round": round_no, "result": result,
                               "snapshot": engine.snapshot(*handles[env]).hex()})
                finished.add(env)
            elif terminated:
                completed.append({"env": env, "winner": winner, "rounds": round_no + 1})
                finished.add(env)
            elif not accepted:
                retries += 1
    elapsed = time.perf_counter() - started
    receipt = {"schema": "pa3-exact-pool-smoke-v1", "dataset_sha256": digest,
               "team_count": count, "envs": len(handles), "natural_completed": len(completed),
               "retries": retries, "errors": errors, "unfinished": len(handles) - len(finished),
               "seconds": elapsed, "games_per_second": len(completed) / elapsed,
               "completed": completed, "model_or_optimizer_used": False,
               "limitation": "Finite model-free execution smoke, not Showdown parity or exhaustive mechanics proof."}
    assert digest == hashlib.sha256(pool.read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("completed", "errors")}))
    if errors or receipt["unfinished"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
