"""PA3-8M rollout collector over the compiled native engine.

This is the first real (non-mock) training-path integration. It drives
``pa3_engine.NativeEngine`` exactly like the development actor does -- one
``request_info_batch``, one ``observe_encoded_batch``, one ``candidates_batch``
per branch level, and one ``step_batch`` per round -- but the policy is the
PA3-8M entity transformer sampled on the GPU.

Correctness notes:

* the native branch mask is prefix-dependent, so candidate tables are queried
  *after* each sampled branch and the model walks the levels sequentially
  (``PA3Model.sample_levels``); the same per-level tables are stored in the
  rollout buffer so the learner's recomputation is exact,
* illegal candidates can never be sampled (their probability is exactly zero),
* only the learner seat's requests become current-policy rows; the opponent
  seat uses the same frozen policy version but its rows are not PPO rows,
* every terminated match pays exactly one terminal reward (+1/-1/0) on the last
  request of the learner side's sequence,
* operational errors are counted and raised, never converted into games.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from agent.buffer.rollout_buffer import RolloutBuffer
from agent.model.pa3_model import PA3Model, assemble_level_result
from agent.types.actions import (
    AtomicAction,
    CandidateSet,
    RequestKind,
)
from agent.types.observation import ObservationBatch, ObservationLayout
from agent.types.requests import ActionRef, BranchCandidatesBatch, RequestRow

@dataclass
class NativeCollectorConfig:
    envs: int = 1024
    workers: int = 16
    seed: int = 20261006
    learner_seat_every_other_env: bool = True
    max_rounds_per_cohort: int = 2000
    candidate_capacity: int = 64
    branch_capacity: int = 4
    record_rows: bool = True
    device: str = "cuda"
    prefer_cuda: bool = True


@dataclass
class NativeCollectorStats:
    cohorts: int = 0
    rounds: int = 0
    games: int = 0
    decisions: int = 0
    learner_rows: int = 0
    opponent_requests: int = 0
    candidate_calls: int = 0
    candidates_returned: int = 0
    operational_errors: int = 0
    observation_seconds: float = 0.0
    candidate_seconds: float = 0.0
    model_seconds: float = 0.0
    step_seconds: float = 0.0
    reset_seconds: float = 0.0
    wall_seconds: float = 0.0
    reward_sum: float = 0.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    action_kinds: dict = field(default_factory=dict)
    team_ids_seen: int = 0
    max_open_games: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class NativeCollector:
    """Collect natural self-play matches with PA3-8M on the real engine."""

    def __init__(
        self,
        engine,
        model: PA3Model,
        config: Optional[NativeCollectorConfig] = None,
        layout: Optional[ObservationLayout] = None,
        policy_id: str = "current",
        device: Optional[torch.device] = None,
    ) -> None:
        self.engine = engine
        self.config = config or NativeCollectorConfig()
        self.layout = layout or ObservationLayout()
        self.policy_id = str(policy_id)
        if device is None:
            device = torch.device(
                "cuda" if self.config.prefer_cuda and torch.cuda.is_available() else "cpu"
            )
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.buffer = RolloutBuffer(layout=self.layout)
        self.generator = torch.Generator(device=self.device.type).manual_seed(self.config.seed)
        self.stats = NativeCollectorStats()
        self._match_counter = 0
        self._team_count = int(self.engine.team_count())

    # -- helpers ----------------------------------------------------------
    def _reset_cohort(self) -> tuple[list, list, list]:
        import random

        rng = random.Random(self.config.seed + self.stats.cohorts)
        team_a, team_b, seeds, roles = [], [], [], []
        for index in range(self.config.envs):
            team_a.append(rng.randrange(self._team_count))
            team_b.append(rng.randrange(self._team_count))
            seeds.append(tuple(rng.randrange(1 << 16) for _ in range(4)))
            # Half of the environments give the learner seat to side 0 and half
            # to side 1; the role map decides which side is the learner.
            roles.append((0, 1) if index % 2 == 0 else (1, 0))
        start = time.perf_counter()
        handles = self.engine.reset_batch(team_a, team_b, seeds, roles)
        self.stats.reset_seconds += time.perf_counter() - start
        self.stats.cohorts += 1
        self.stats.team_ids_seen += len(set(team_a) | set(team_b))
        return handles, list(zip(team_a, team_b)), roles

    def _learner_side(self, roles: Sequence[int]) -> int:
        return 0 if int(roles[0]) == 0 else 1

    def _level_table(
        self, candidate_rows: Sequence[Sequence[Sequence[int]]]
    ) -> BranchCandidatesBatch:
        """Build a single-branch table for one level from native candidate tuples."""
        batch = len(candidate_rows)
        capacity = self.config.candidate_capacity
        action_ids = torch.zeros((batch, 1, capacity, 6), dtype=torch.long)
        mask = torch.zeros((batch, 1, capacity), dtype=torch.bool)
        entity = torch.zeros((batch, 1, capacity), dtype=torch.long)
        move = torch.full((batch, 1, capacity), -1, dtype=torch.long)
        for row_index, candidates in enumerate(candidate_rows):
            if not candidates:
                raise RuntimeError("request level without a legal candidate")
            if len(candidates) > capacity:
                raise RuntimeError(
                    f"candidate capacity {capacity} exceeded ({len(candidates)}); "
                    "candidate capacity must be resolved from the full scope, never truncated"
                )
            for index, packed in enumerate(candidates):
                action = AtomicAction.from_tuple(packed)
                ref = ActionRef.resolve(self.layout, action)
                action_ids[row_index, 0, index] = torch.tensor(ref.tuple, dtype=torch.long)
                mask[row_index, 0, index] = True
                entity[row_index, 0, index] = int(ref.entity_token)
                move[row_index, 0, index] = int(ref.move_token)
        return BranchCandidatesBatch(
            action_ids=action_ids.to(self.device),
            mask=mask.to(self.device),
            entity_token=entity.to(self.device),
            move_token=move.to(self.device),
            branch_valid=torch.ones((batch, 1), dtype=torch.bool, device=self.device),
            selected=torch.full((batch, 1), -1, dtype=torch.long, device=self.device),
        )

    def _record(
        self,
        observation: ObservationBatch,
        request: RequestRow,
        sampled,
        row_index: int,
        value: float,
        match_id: int,
        side: int,
        team_ids: tuple[int, int],
        request_index: int,
        turn: int,
    ) -> None:
        if not self.config.record_rows:
            return
        selected = [int(sampled.selected[row_index, level].item()) for level in range(request.branch_count)]
        self.buffer.record(
            observation=observation,
            request=request,
            selected=selected,
            old_logprob=float(sampled.request_logprob[row_index].detach()),
            value=value,
            match_id=match_id,
            side=side,
            policy_id=self.policy_id,
            opponent_policy_id=self.policy_id,
            team_ids=team_ids,
            request_index=request_index,
            turn=turn,
            reward=0.0,
            done=False,
        )
        self.stats.learner_rows += 1

    # -- one collection round ---------------------------------------------
    def _collect_round(self, handles, roles, team_ids, match_ids, request_index, turn, finished):
        pending = []
        for env in range(len(handles)):
            if finished[env]:
                continue
            for side in (0, 1):
                pending.append((env, handles[env], side))
        if not pending:
            return False
        info = self.engine.request_info_batch(
            [(handle[0], handle[1], side) for _, handle, side in pending]
        )
        decision = []
        for (env, handle, side), (kind, slots) in zip(pending, info):
            if int(kind) in (0, 1, 2):
                decision.append((env, handle, side, int(kind), tuple(int(s) for s in slots)))
        if not decision:
            return False
        start = time.perf_counter()
        blobs = self.engine.observe_encoded_batch(
            [(handle[0], handle[1]) for _, handle, _, _, _ in decision],
            [side for _, _, side, _, _ in decision],
        )
        observation_seconds = time.perf_counter() - start
        self.stats.observation_seconds += observation_seconds
        from pa3_engine.observation import parse_view

        observations = [
            ObservationBatch.from_native_payload(parse_view(blob), layout=self.layout)
            for blob in blobs
        ]
        # Group requests by branch count so a batched sequential sample is
        # well-formed for every row in the group.
        groups: dict[int, list[int]] = {}
        for index, entry in enumerate(decision):
            groups.setdefault(len(entry[4]), []).append(index)
        submissions: dict[int, list[tuple[int, list]]] = {}
        for branch_count, indices in sorted(groups.items()):
            subset_obs = observations[indices[0]].cat([observations[i] for i in indices[1:]]) \
                if len(indices) > 1 else observations[indices[0]]
            subset_obs = subset_obs.to(self.device)
            start = time.perf_counter()
            with torch.no_grad():
                encoded = self.model.encode(subset_obs)
                hidden = self.model.scorer.initial_state(
                    encoded.tokens.shape[0], device=encoded.tokens.device, dtype=encoded.tokens.dtype
                )
            steps = []
            candidate_rows_per_level: list[list[Sequence[Sequence[int]]]] = [[] for _ in range(branch_count)]
            prefixes: list[list[tuple]] = [[] for _ in indices]
            for level in range(branch_count):
                specs = []
                for offset, request_index_in_decision in enumerate(indices):
                    env, handle, side, _kind, _slots = decision[request_index_in_decision]
                    specs.append((handle[0], handle[1], side, list(prefixes[offset])))
                start_candidates = time.perf_counter()
                candidate_lists = self.engine.candidates_batch(specs)
                self.stats.candidate_seconds += time.perf_counter() - start_candidates
                self.stats.candidate_calls += 1
                self.stats.candidates_returned += sum(len(row) for row in candidate_lists)
                for offset, candidates in enumerate(candidate_lists):
                    candidate_rows_per_level[level].append(candidates)
                    self.stats.action_kinds[int(AtomicAction.from_tuple(candidates[0]).kind)] = (
                        self.stats.action_kinds.get(int(AtomicAction.from_tuple(candidates[0]).kind), 0) + 1
                    )
                start_model = time.perf_counter()
                with torch.no_grad():
                    step = self.model.sample_level_step(
                        encoded,
                        hidden,
                        self._level_table(candidate_lists),
                        temperature=1.0,
                        generator=self.generator,
                    )
                self.stats.model_seconds += time.perf_counter() - start_model
                steps.append(step)
                hidden = step.hidden
                for offset, candidates in enumerate(candidate_lists):
                    # The next level's mask is conditional on this sampled pick.
                    prefixes[offset].append(tuple(candidates[int(step.pick[offset])]))
            sampled = assemble_level_result(steps)
            start_model = time.perf_counter()
            with torch.no_grad():
                values = self.model.value(encoded)
            self.stats.model_seconds += time.perf_counter() - start_model
            for offset, decision_index in enumerate(indices):
                env, handle, side, kind, slots = decision[decision_index]
                branches = tuple(
                    CandidateSet.from_tuples(list(candidate_rows_per_level[level][offset]))
                    for level in range(branch_count)
                )
                request = RequestRow(
                    observation=subset_obs.select([offset]),
                    kind=RequestKind(kind),
                    branch_slots=slots,
                    branches=branches,
                )
                action = [tuple(value) for value in prefixes[offset]]
                if side == self._learner_side(roles[env]):
                    self._record(
                        observation=subset_obs.select([offset]),
                        request=request,
                        sampled=sampled,
                        row_index=offset,
                        value=float(values[offset].detach()),
                        match_id=match_ids[env],
                        side=side,
                        team_ids=team_ids[env],
                        request_index=request_index[env],
                        turn=turn[env],
                    )
                    self._row_cursor[(env, side)] = len(self.buffer.rows) - 1
                else:
                    self.stats.opponent_requests += 1
                submissions.setdefault(env, []).append((side, action))
                self.stats.decisions += 1
        specs = []
        order = sorted(submissions)
        for env in order:
            handle = handles[env]
            specs.append((handle[0], handle[1], submissions[env]))
        start = time.perf_counter()
        results = self.engine.step_batch(specs)
        self.stats.step_seconds += time.perf_counter() - start
        self.stats.rounds += 1
        for env, result in zip(order, results):
            accepted, terminated, truncated, winner, error, _, _ = result
            if not accepted or truncated:
                raise RuntimeError(f"invalid batch result for env {env}: {result}")
            if error is not None:
                self.stats.operational_errors += 1
                raise RuntimeError(f"operational error in env {env}: {error}")
            if terminated:
                learner = self._learner_side(roles[env])
                if winner is None:
                    reward = 0.0
                    self.stats.draws += 1
                elif int(winner) == learner:
                    reward = 1.0
                    self.stats.wins += 1
                else:
                    reward = -1.0
                    self.stats.losses += 1
                row_index = self._row_cursor.get((env, learner))
                if row_index is not None:
                    row = self.buffer.rows[row_index]
                    row.reward = reward
                    row.done = True
                self.stats.reward_sum += reward
                self.stats.games += 1
                finished[env] = True
                match_ids[env] = self._match_counter
                self._match_counter += 1
                request_index[env] = 0
                turn[env] = 0
            else:
                request_index[env] += 1
                turn[env] += 1
        return True

    # -- collection -------------------------------------------------------
    def collect(self, target_games: int) -> RolloutBuffer:
        self._row_cursor: dict[tuple[int, int], int] = {}
        self.model.eval()
        started = time.perf_counter()
        while self.stats.games < target_games:
            handles, teams, roles = self._reset_cohort()
            finished = [False] * len(handles)
            match_ids = [self._match_counter + index for index in range(len(handles))]
            self._match_counter += len(handles)
            request_index = [0] * len(handles)
            turn = [0] * len(handles)
            self._row_cursor = {}
            rounds = 0
            while not all(finished):
                rounds += 1
                if rounds > self.config.max_rounds_per_cohort:
                    raise RuntimeError("cohort did not drain within the round budget")
                progressed = self._collect_round(
                    handles, roles, teams, match_ids, request_index, turn, finished
                )
                if not progressed:
                    raise RuntimeError("no actionable request while a cohort is open")
                open_games = sum(1 for value in finished if not value)
                self.stats.max_open_games = max(self.stats.max_open_games, open_games)
        self.stats.wall_seconds = time.perf_counter() - started
        return self.buffer
