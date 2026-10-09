"""Batch actor for the documented 1,024-environment / 16-worker group.

One actor process owns one native environment group and never crosses into
Python per effect, per Pokémon or per environment: each round builds the
observations and legal-action masks for every environment that needs a
decision, calls the policy once for the whole round, and submits one
`step_batch` call. Pokémon Showdown is never executed here.

Accounting follows the engine spec: transitions, policy decisions, natural
completed games, observation / legal-action / bridge / step costs and worker
scaling. Operational errors (unported mechanics) are counted separately and are
never reported as draws or wins.
"""
import argparse
import json
import random
import resource
import time

import pa3_engine
from pa3_engine.observation import parse_batch


class Request:
    """One pending player decision: observation, branches and mask queries."""

    __slots__ = (
        "env",
        "side",
        "kind",
        "observation",
        "observation_index",
        "branches",
        "engine",
        "handle",
        "counter",
        "timer",
        "prefix",
    )

    def __init__(
        self, engine, env, handle, side, kind, observation, observation_index, branches, counter, timer
    ):
        self.engine = engine
        self.env = env
        self.handle = handle
        self.side = side
        self.kind = kind
        self.observation = observation
        self.observation_index = observation_index
        self.branches = branches
        self.counter = counter
        self.timer = timer
        self.prefix = []

    def observation_view(self):
        """Zero-copy numpy slices of this request's observation, if needed."""
        batch = self.observation
        if batch is None:
            return None
        index = self.observation_index
        return {
            "schema_version": int(batch["schema_version"][index]),
            "token_mask": batch["token_mask"][index],
            "categories": batch["categories"][index],
            "floats": batch["floats"][index],
            "flags": batch["flags"][index],
            "effect_counts": batch["effect_counts"][index],
        }

    def candidates(self, prefix):
        """Legal next actions for the branch after `prefix` (the native mask)."""
        start = time.perf_counter()
        self.counter["calls"] += 1
        candidates = self.engine.candidates(
            self.handle[0], self.handle[1], self.side, list(prefix)
        )
        self.counter["returned"] += len(candidates)
        self.timer["candidates_seconds"] += time.perf_counter() - start
        return candidates

    def complete(self, pick):
        """Pick a joint action branch by branch using the native masks."""
        prefix = []
        while len(prefix) < len(self.branches):
            candidates = self.candidates(prefix)
            if not candidates:
                raise RuntimeError(f"request without legal completion at branch {len(prefix)}")
            prefix.append(pick(prefix, candidates))
        return prefix

    def completions(self):
        """Exhaustive legal joint actions (complete mask, bounded by the engine)."""
        out = []
        prefix = []

        def rec():
            candidates = self.candidates(prefix)
            if not candidates:
                if prefix:
                    out.append(list(prefix))
                return
            for candidate in candidates:
                prefix.append(candidate)
                rec()
                prefix.pop()

        rec()
        return out


class Policy:
    """One batched policy call per round; environment loops never call it."""

    def __init__(self, mode, seed):
        self.mode = mode
        self.random = random.Random(seed)

    def __call__(self, requests):
        actions = []
        for request in requests:
            if self.mode == "random":
                action = request.complete(lambda _prefix, candidates: self.random.choice(candidates))
            else:
                action = request.complete(lambda _prefix, candidates: candidates[0])
            if not action:
                raise RuntimeError("empty joint action")
            actions.append(action)
        return actions

    def choose_batch(self, engine, requests, counter=None, timer=None):
        """One `candidates_batch` crossing per branch level for the whole round.

        Legal sets match the per-request `Request.complete` walk exactly. Draw
        order is depth-major (all requests' branch 0, then branch 1, ...) rather
        than request-major, so a seeded random policy stays deterministic for a
        given code revision without consuming draws in the old order.
        """
        active = list(requests)
        while active:
            specs = [
                (request.handle[0], request.handle[1], request.side, list(request.prefix))
                for request in active
            ]
            start = time.perf_counter()
            results = engine.candidates_batch(specs)
            elapsed = time.perf_counter() - start
            if counter is not None:
                counter["calls"] += 1
                counter["returned"] += sum(len(row) for row in results)
            if timer is not None:
                timer["candidates_seconds"] += elapsed
            nxt = []
            for request, candidates in zip(active, results):
                if not candidates:
                    raise RuntimeError(
                        f"request without legal completion at branch {len(request.prefix)}"
                    )
                pick = self.random.choice(candidates) if self.mode == "random" else candidates[0]
                request.prefix.append(pick)
                if len(request.prefix) < len(request.branches):
                    nxt.append(request)
            active = nxt
        return [list(request.prefix) for request in requests]


class ActorRunner:
    def __init__(
        self, data_dir, teams_json, envs=1024, workers=16, seed=20261006, pin=None,
        observations="perview",
    ):
        if pin:
            try:
                import os

                os.sched_setaffinity(0, set(pin))
            except (AttributeError, OSError) as error:  # pragma: no cover - host specific
                print(f"warning: cpu affinity not applied: {error}")
        self.engine = pa3_engine.NativeEngine(data_dir, teams_json, workers=workers)
        self.envs = envs
        self.seed = seed
        self.observations = observations
        self.random = random.Random(seed)
        self.counter = {"calls": 0, "returned": 0}
        self.timer = {"candidates_seconds": 0.0}
        self.metrics = {
            "envs": envs,
            "workers": workers,
            "cohorts": 0,
            "rounds": 0,
            "transitions": 0,
            "decisions": 0,
            "games": 0,
            "operational_errors": 0,
            "reset_seconds": 0.0,
            "observe_seconds": 0.0,
            "decode_seconds": 0.0,
            "candidates_seconds": 0.0,
            "step_seconds": 0.0,
            "policy_seconds": 0.0,
            "observations": 0,
            "candidate_calls": 0,
            "candidates_returned": 0,
            "operational_gaps": {},
        }
        self._reset()

    def _reset(self):
        started = time.perf_counter()
        teams = self.engine.team_count()
        team_a, team_b, seeds, roles = [], [], [], []
        for index in range(self.envs):
            team_a.append(self.random.randrange(teams))
            team_b.append(self.random.randrange(teams))
            seeds.append(tuple(self.random.randrange(1 << 16) for _ in range(4)))
            roles.append((0, 1) if index % 2 == 0 else (1, 0))
        self.handles = self.engine.reset_batch(team_a, team_b, seeds, roles)
        self.finished = [False] * self.envs
        self.pending = self.envs
        self.metrics["reset_seconds"] += time.perf_counter() - started
        self.metrics["cohorts"] += 1

    def _collect_round(self, policy):
        pending = []
        for index in range(self.envs):
            if self.finished[index]:
                continue
            handle = self.handles[index]
            for side in (0, 1):
                pending.append((index, handle, side))
        if not pending:
            return None
        info = self.engine.request_info_batch(
            [(handle[0], handle[1], side) for _, handle, side in pending]
        )
        requests = []
        handles = []
        sides = []
        branches = []
        for (index, handle, side), (kind, branch_slots) in zip(pending, info):
            if kind in (0, 1, 2):
                requests.append((index, handle, side, kind))
                handles.append((handle[0], handle[1]))
                sides.append(side)
                branches.append(branch_slots)
        if not requests:
            return None
        start = time.perf_counter()
        if self.observations == "fixed":
            fixed, ragged = self.engine.observe_fixed_batch(handles, sides)
            blobs = None
        else:
            blobs = self.engine.observe_encoded_batch(handles, sides)
            fixed = ragged = None
        self.metrics["observe_seconds"] += time.perf_counter() - start
        self.metrics["observations"] += len(handles)
        start = time.perf_counter()
        batch = parse_batch(fixed, ragged, len(handles)) if fixed is not None else None
        self.metrics["decode_seconds"] += time.perf_counter() - start
        payload = []
        for offset, (index, handle, side, kind) in enumerate(requests):
            request = Request(
                self.engine,
                index,
                handle,
                side,
                kind,
                batch if batch is not None else blobs[offset],
                offset if batch is not None else 0,
                branches[offset],
                self.counter,
                self.timer,
            )
            payload.append(request)
        start = time.perf_counter()
        chosen = policy.choose_batch(self.engine, payload, self.counter, self.timer)
        self.metrics["policy_seconds"] += time.perf_counter() - start
        self.metrics["candidates_seconds"] = self.timer["candidates_seconds"]
        self.metrics["candidate_calls"] = self.counter["calls"]
        self.metrics["candidates_returned"] = self.counter["returned"]
        submissions = {}
        for request, action in zip(payload, chosen):
            submissions.setdefault(request.env, []).append((request.side, action))
            self.metrics["decisions"] += 1
        specs = []
        order = []
        for index in sorted(submissions):
            handle = self.handles[index]
            specs.append((handle[0], handle[1], submissions[index]))
            order.append(index)
        start = time.perf_counter()
        results = self.engine.step_batch(specs)
        self.metrics["step_seconds"] += time.perf_counter() - start
        self.metrics["rounds"] += 1
        self.metrics["transitions"] += len(specs)
        for index, result in zip(order, results):
            accepted, terminated, truncated, winner, error, kind0, kind1 = result
            if not accepted or truncated:
                raise RuntimeError(f"invalid batch result for env {index}: {result}")
            if error is not None:
                self.metrics["operational_errors"] += 1
                self.metrics["operational_gaps"][error] = (
                    self.metrics["operational_gaps"].get(error, 0) + 1
                )
                self.finished[index] = True
                self.pending -= 1
            elif terminated:
                if winner not in (0, 1, None):
                    raise RuntimeError(f"invalid winner for env {index}: {winner}")
                self.metrics["games"] += 1
                self.finished[index] = True
                self.pending -= 1
        return True

    def run(self, target_games, policy, recycle=True):
        """Collect until `target_games` natural completions (op errors excluded).

        A cohort drains naturally: every environment plays its match to a
        natural end and only then is the whole group reset. This matches the
        documented collection contract (freeze the policy, drain, then update)
        and keeps the reset cost visible in its own timer instead of hiding it
        inside the stepping cost.
        """
        started = time.perf_counter()
        while self.metrics["games"] < target_games and self.pending > 0:
            if self._collect_round(policy) is None:
                break
            if self.pending == 0 and recycle and self.metrics["games"] < target_games:
                self._reset()
        self.metrics["wall_seconds"] = time.perf_counter() - started
        # Peak resident set of this actor process after its first full cohort.
        self.metrics["max_rss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.metrics["max_rss_mb"] = self.metrics["max_rss_kb"] / 1024.0
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.metrics["cpu_seconds"] = usage.ru_utime + usage.ru_stime
        self.metrics["cpu_fraction_of_one_core"] = self.metrics["cpu_seconds"] / max(self.metrics["wall_seconds"], 1e-9)
        return self.metrics


def parse_pin(text):
    if not text:
        return None
    cpus = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            cpus.extend(range(int(low), int(high) + 1))
        else:
            cpus.append(int(part))
    return cpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="engine/data")
    parser.add_argument("--teams", default="engine/data/training-teams.json")
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--policy", choices=["first", "random"], default="first")
    parser.add_argument("--pin", default="", help="CPU list, e.g. 0-15")
    parser.add_argument(
        "--observations",
        choices=["fixed", "perview"],
        default="perview",
        help="fixed-stride batch payload or per-view packed blobs",
    )
    parser.add_argument("--no-reset", action="store_true", help="stop after one cohort")
    args = parser.parse_args()
    runner = ActorRunner(
        args.data,
        args.teams,
        envs=args.envs,
        workers=args.workers,
        seed=args.seed,
        pin=parse_pin(args.pin),
        observations=args.observations,
    )
    target = args.envs if args.no_reset else args.games
    metrics = runner.run(target, Policy(args.policy, args.seed), recycle=not args.no_reset)
    wall = max(metrics["wall_seconds"], 1e-9)
    metrics["games_per_second"] = metrics["games"] / wall
    metrics["transitions_per_second"] = metrics["transitions"] / wall
    metrics["decisions_per_second"] = metrics["decisions"] / wall
    rounds = max(metrics["rounds"], 1)
    metrics["observe_ms_per_round"] = 1000 * metrics["observe_seconds"] / rounds
    metrics["decode_ms_per_round"] = 1000 * metrics["decode_seconds"] / rounds
    metrics["candidate_ms_per_round"] = 1000 * metrics["candidates_seconds"] / rounds
    metrics["step_ms_per_round"] = 1000 * metrics["step_seconds"] / rounds
    metrics["reset_ms_per_cohort"] = 1000 * metrics["reset_seconds"] / max(metrics["cohorts"], 1)
    metrics["policy_ms_per_round"] = 1000 * metrics["policy_seconds"] / rounds
    accounted = (metrics["observe_seconds"] + metrics["decode_seconds"] + metrics["candidates_seconds"]
                 + metrics["step_seconds"] + metrics["policy_seconds"] + metrics["reset_seconds"])
    metrics["accounted_seconds"] = accounted
    metrics["unaccounted_ms_per_round"] = 1000 * max(metrics["wall_seconds"] - accounted, 0.0) / rounds
    top_gaps = sorted(metrics["operational_gaps"].items(), key=lambda kv: -kv[1])[:12]
    metrics["operational_gaps"] = dict(top_gaps)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
