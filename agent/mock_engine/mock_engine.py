"""Deterministic mock of the future ``NativeEngine`` surface.

The mock mirrors the shape of ``engine/python/pa3_engine.NativeEngine``:

* ``team_count()``
* ``reset_batch(team_a, team_b, seeds, roles) -> handles``
* ``request_info_batch(specs) -> [(kind, branch_slots)]``
* ``candidates_batch(specs) -> [[action tuples]]`` (native masks)
* ``observe_payload_batch(handles, sides) -> [native-shaped payload]``
* ``step_batch(specs) -> [(accepted, terminated, truncated, winner, error, …)]``

Everything is fake: no Rust, no Showdown, no match is ever a training datum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch

from agent.buffer.rollout_buffer import RolloutBuffer
from agent.model.pa3_model import PA3Model
from agent.types.actions import (
    ActionKind,
    AtomicAction,
    CandidateSet,
    RequestKind,
    Resource,
    branch_slots_for_request,
)
from agent.types.observation import ObservationBatch, ObservationLayout
from agent.types.requests import ActionRef, BranchCandidatesBatch, RequestRow

NO_SLOT = 255


@dataclass
class _MockEnv:
    handle: tuple[int, int]
    team_a: int
    team_b: int
    seed: int
    roles: tuple[int, int]
    request_index: int = 0
    turn: int = 1
    submissions: dict[int, list[tuple[int, int, int, int, int, int]]] = field(default_factory=dict)
    terminated: bool = False
    winner: Optional[int] = None


class MockNativeEngine:
    """A tiny deterministic stand-in for the native batch engine."""

    def __init__(
        self,
        num_teams: int = 32,
        seed: int = 20261006,
        requests_per_match: int = 4,
        layout: Optional[ObservationLayout] = None,
    ) -> None:
        if requests_per_match < 2:
            raise ValueError("a mock match needs at least a preview and one turn")
        self.num_teams = int(num_teams)
        self.seed = int(seed)
        self.requests_per_match = int(requests_per_match)
        self.layout = layout or ObservationLayout()
        self._environments: dict[tuple[int, int], _MockEnv] = {}
        self._next_generation = 0
        self._rng = np.random.default_rng(self.seed)

    # -- catalogue ---------------------------------------------------------
    def team_count(self) -> int:
        return self.num_teams

    def observations_per_match(self) -> int:
        """Rows each side records per match (the collector's row count)."""
        return self.requests_per_match

    # -- lifecycle ---------------------------------------------------------
    def reset_batch(
        self,
        team_a: Sequence[int],
        team_b: Sequence[int],
        seeds: Sequence[Sequence[int]],
        roles: Sequence[Sequence[int]],
    ) -> list[tuple[int, int]]:
        if not (len(team_a) == len(team_b) == len(seeds) == len(roles)):
            raise ValueError("reset_batch arguments must share a length")
        handles = []
        for index, (a, b, seed, role) in enumerate(zip(team_a, team_b, seeds, roles)):
            self._next_generation += 1
            handle = (self._next_generation, index)
            self._environments[handle] = _MockEnv(
                handle=handle,
                team_a=int(a) % self.num_teams,
                team_b=int(b) % self.num_teams,
                seed=int(seed[0]) if hasattr(seed, "__len__") else int(seed),
                roles=(int(role[0]), int(role[1])),
            )
            handles.append(handle)
        return handles

    def _env(self, handle: tuple[int, int]) -> _MockEnv:
        try:
            return self._environments[handle]
        except KeyError as error:  # pragma: no cover - defensive
            raise KeyError(f"unknown handle {handle}") from error

    def environment(self, handle: tuple[int, int]) -> _MockEnv:
        """Public per-environment state (the collector needs roles/team ids)."""
        return self._env(handle)

    # -- requests ----------------------------------------------------------
    def request_info(self, handle: tuple[int, int], side: int) -> tuple[int, tuple[int, ...]]:
        env = self._env(handle)
        if env.terminated:
            return (int(RequestKind.FINISHED), ())
        if side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        if env.request_index == 0:
            return (int(RequestKind.PREVIEW), branch_slots_for_request(RequestKind.PREVIEW))
        kind = RequestKind.NORMAL
        return (int(kind), branch_slots_for_request(kind))

    def request_info_batch(
        self, specs: Sequence[tuple[int, int, int]]
    ) -> list[tuple[int, tuple[int, ...]]]:
        return [self.request_info((s[0], s[1]), s[2]) for s in specs]

    def observe_payload(self, handle: tuple[int, int], side: int) -> dict[str, Any]:
        """Native-shaped observation payload for one player view."""
        env = self._env(handle)
        rng = np.random.default_rng(
            (self.seed * 1_000_003 + env.handle[0] * 7919 + env.request_index * 104_729 + side)
            % (2**63 - 1)
        )
        layout = self.layout
        tokens = layout.TOKENS
        token_mask = np.zeros(tokens, dtype=bool)
        token_mask[: layout.ACTIVE_TOKENS] = True
        categories = rng.integers(
            0, 512, size=(tokens, layout.CATEGORY_SLOTS), dtype=np.uint16
        )
        floats = rng.normal(0.0, 1.0, size=(tokens, layout.FLOAT_SLOTS)).astype(np.float32)
        floats[~token_mask] = 0.0
        flags = rng.random((tokens, layout.FLAG_SLOTS)) < 0.5
        known = np.ones((tokens, layout.CATEGORY_SLOTS), dtype=bool)
        known[layout.ACTIVE_TOKENS :] = False
        float_known = np.ones((tokens, layout.FLOAT_SLOTS), dtype=bool)
        float_known[layout.ACTIVE_TOKENS :] = False
        flag_known = np.ones((tokens, layout.FLAG_SLOTS), dtype=bool)
        flag_known[layout.ACTIVE_TOKENS :] = False
        payload = {
            "schema_version": 1,
            "token_mask": token_mask,
            "categories": categories,
            "category_known": known,
            "floats": floats,
            "float_known": float_known,
            "flags": flags,
            "flag_known": flag_known,
            "role_ids": layout.default_roles().numpy(),
            "side_ids": layout.default_sides().numpy(),
        }
        # Target locations are encoded -1..4 in the packed engine blob.
        payload["floats"][:, 0] = float(side)
        return payload

    def observe_payload_batch(
        self, handles: Sequence[tuple[int, int]], sides: Sequence[int]
    ) -> list[dict[str, Any]]:
        return [self.observe_payload(h, s) for h, s in zip(handles, sides)]

    def observe_batch(
        self, handles: Sequence[tuple[int, int]], sides: Sequence[int]
    ) -> list[ObservationBatch]:
        return [
            ObservationBatch.from_native_payload(self.observe_payload(h, s), self.layout)
            for h, s in zip(handles, sides)
        ]

    # -- candidates ---------------------------------------------------------
    def candidates(
        self, handle: tuple[int, int], side: int, prefix: Sequence[int]
    ) -> list[tuple[int, int, int, int, int, int]]:
        env = self._env(handle)
        kind, slots = self.request_info(handle, side)
        branch = len(prefix)
        if branch >= len(slots):
            return []
        slot = slots[branch]
        if kind == int(RequestKind.PREVIEW):
            total = 6
            counts = (6, 6, 4, 3)
            count = counts[min(branch, len(counts) - 1)]
            actions = [
                AtomicAction(
                    kind=int(ActionKind.PICK),
                    own_slot=slot,
                    move_slot=NO_SLOT,
                    target_location=0,
                    switch_destination=(int(prefix[branch - 1][4]) + index + 1) % total
                    if branch
                    else index % total,
                    resource=int(Resource.NONE),
                )
                for index in range(count)
            ]
        else:
            count = 4 if branch == 0 else 1
            if env.request_index == 2 and side == 1:
                count = 1  # deterministic all-singleton request for side P2
            actions = [
                AtomicAction(
                    kind=int(ActionKind.MOVE),
                    own_slot=slot,
                    move_slot=index % 4,
                    target_location=index % 3,
                    switch_destination=NO_SLOT,
                    resource=int(Resource.MEGA) if index == 3 else int(Resource.NONE),
                )
                for index in range(count)
            ]
        return [action.as_tuple() for action in actions]

    def candidate_mask(
        self, handle: tuple[int, int], side: int, prefix: Sequence[int]
    ) -> list[bool]:
        """Native mask: some structurally present candidates are illegal."""
        candidates = self.candidates(handle, side, prefix)
        env = self._env(handle)
        mask = [True] * len(candidates)
        if len(mask) >= 4 and env.request_index % 2 == 0 and side == 0:
            mask[-1] = False  # illegal action must never be sampled
        if len(mask) >= 3 and env.request_index % 2 == 1 and side == 1:
            mask[-2] = False
        return mask

    def candidates_batch(
        self, specs: Sequence[tuple[int, int, int, Sequence[int]]]
    ) -> list[list[tuple[int, int, int, int, int, int]]]:
        return [self.candidates((s[0], s[1]), s[2], list(s[3])) for s in specs]

    # -- stepping ------------------------------------------------------------
    def step_batch(
        self, specs: Sequence[tuple[int, int, Sequence[tuple[int, Sequence]]]]
    ) -> list[tuple[bool, bool, bool, Optional[int], Optional[str], int, int]]:
        """Apply one committed simultaneous choice per environment."""
        results = []
        for slot, generation, submissions in specs:
            env = self._env((slot, generation))
            if env.terminated:
                raise ValueError("step_batch on a terminated environment")
            for side, action in submissions:
                env.submissions[int(side)] = [tuple(int(v) for v in a) for a in action]
            complete = len(env.submissions) == 2
            if not complete:
                results.append((True, False, False, None, None, 0, 0))
                continue
            env.submissions.clear()
            env.request_index += 1
            env.turn += 1
            terminated = env.request_index >= self.requests_per_match - 1
            if terminated:
                env.terminated = True
                env.winner = (env.team_a + env.seed + env.request_index) % 2
            next_kind = int(RequestKind.FINISHED if terminated else RequestKind.NORMAL)
            results.append(
                (
                    True,
                    terminated,
                    False,
                    env.winner if terminated else None,
                    None,
                    next_kind,
                    next_kind,
                )
            )
        return results


class MockRolloutCollector:
    """Collect fake rollouts from :class:`MockNativeEngine` with a real model."""

    def __init__(
        self,
        engine: MockNativeEngine,
        model: PA3Model,
        policy_id: str = "current",
        opponent_policy_id: str = "current",
        layout: Optional[ObservationLayout] = None,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.engine = engine
        self.model = model.to(device)
        self.policy_id = policy_id
        self.opponent_policy_id = opponent_policy_id
        self.layout = layout or engine.layout
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def _policy_for_side(self, side: int, roles: tuple[int, int]) -> str:
        learner = 0 if roles[0] == 0 else 1
        return self.policy_id if side == learner else self.opponent_policy_id

    def collect(
        self,
        envs: int = 4,
        target_matches: int = 2,
        buffer: Optional[RolloutBuffer] = None,
        max_rounds: int = 64,
    ) -> RolloutBuffer:
        buffer = buffer if buffer is not None else RolloutBuffer(layout=self.layout)
        rng = np.random.default_rng(self.engine.seed)
        team_a = [int(rng.integers(0, self.engine.team_count())) for _ in range(envs)]
        team_b = [int(rng.integers(0, self.engine.team_count())) for _ in range(envs)]
        seeds = [[int(rng.integers(0, 1 << 16)) for _ in range(4)] for _ in range(envs)]
        roles = [(0, 1) if index % 2 == 0 else (1, 0) for index in range(envs)]
        handles = self.engine.reset_batch(team_a, team_b, seeds, roles)
        self.model.eval()

        completed = 0
        rows_by_env: dict[int, dict[int, list]] = {i: {0: [], 1: []} for i in range(envs)}
        active = set(range(envs))
        rounds = 0
        while active and completed < target_matches and rounds < max_rounds:
            rounds += 1
            submissions: dict[int, list[tuple[int, list]]] = {}
            for index in sorted(active):
                handle = handles[index]
                env = self.engine.environment(handle)
                for side in (0, 1):
                    kind, slots = self.engine.request_info(handle, side)
                    if not RequestKind(kind).is_decision:
                        continue
                    payload = self.engine.observe_payload(handle, side)
                    observation = ObservationBatch.from_native_payload(
                        payload, self.layout, device=self.device
                    )
                    candidates = self._candidate_sets(handle, side, slots)
                    request = RequestRow(
                        observation=observation,
                        kind=RequestKind(kind),
                        branch_slots=tuple(slots),
                        branches=candidates,
                    )
                    batch = BranchCandidatesBatch.from_requests(
                        [request], device=self.device
                    )
                    sampled = self.model.sample(
                        observation,
                        batch,
                        temperature=1.0,
                        generator=self.generator,
                    )
                    selected = sampled.prefix(request.branch_count)[0].tolist()
                    policy_id = self._policy_for_side(side, env.roles)
                    if policy_id == self.policy_id:
                        buffer.record(
                            observation=observation,
                            request=request,
                            selected=selected,
                            old_logprob=float(sampled.request_logprob[0]),
                            value=float(
                                self.model.value(
                                    self.model.encode(observation)
                                )[0].detach()
                            ),
                            match_id=index,
                            side=side,
                            policy_id=policy_id,
                            opponent_policy_id=self.opponent_policy_id,
                            team_ids=(env.team_a, env.team_b),
                            request_index=env.request_index,
                            turn=env.turn,
                            reward=0.0,
                            done=False,
                        )
                        rows_by_env[index][side].append(buffer.rows[-1])
                    action = [
                        request.branches[branch].actions[selected[branch]].as_tuple()
                        for branch in range(len(selected))
                    ]
                    submissions.setdefault(index, []).append((side, action))
            specs = [
                (handles[index][0], handles[index][1], submissions[index])
                for index in sorted(submissions)
            ]
            if not specs:
                break
            results = self.engine.step_batch(specs)
            for index, result in zip(sorted(submissions), results):
                accepted, terminated, truncated, winner, error, _, _ = result
                if not accepted or truncated or error is not None:
                    raise RuntimeError(f"mock step failed for env {index}: {result}")
                if terminated:
                    completed += 1
                    active.discard(index)
                    for side in (0, 1):
                        rows = rows_by_env[index][side]
                        if not rows:
                            continue
                        if winner is None:
                            reward = 0.0
                        else:
                            reward = 1.0 if winner == side else -1.0
                        rows[-1].reward = reward
                        rows[-1].done = True
        self.model.train()
        return buffer

    def _candidate_sets(
        self, handle: tuple[int, int], side: int, slots: Sequence[int]
    ) -> tuple[CandidateSet, ...]:
        branches = []
        prefix: list[int] = []
        for _ in slots:
            actions = self.engine.candidates(handle, side, prefix)
            mask = self.engine.candidate_mask(handle, side, prefix)
            branches.append(CandidateSet.from_tuples(actions, mask))
            # Only the legal prefix can be extended; the engine's own prefix
            # feasibility is what makes branch j+1 conditional on branch j.
            legal = [index for index, flag in enumerate(mask) if flag]
            prefix.append(actions[legal[0]])
        return tuple(branches)


def collect_mock_rollout(
    engine: MockNativeEngine,
    model: PA3Model,
    envs: int = 4,
    target_matches: int = 2,
    policy_id: str = "current",
    opponent_policy_id: str = "current",
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> RolloutBuffer:
    collector = MockRolloutCollector(
        engine,
        model,
        policy_id=policy_id,
        opponent_policy_id=opponent_policy_id,
        device=device,
        seed=seed,
    )
    return collector.collect(envs=envs, target_matches=target_matches)
