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
    # "perview": one packed PyBytes per request (legacy). "fixed": one fixed
    # stride buffer plus a ragged sidecar for the whole decision batch.
    observation_mode: str = "perview"
    # "tuples": legacy Vec<Vec<ActionTuple>> crossing. "packed": one
    # (counts, actions) byte crossing decoded as a structured numpy view.
    candidate_wire: str = "tuples"
    # Actor-only inference flags (A/B measured; probabilities stay FP32).
    amp: bool = False
    inference_mode: bool = False
    # configs/train.yaml `collect_both_sides_when_current_self_play: true`: when
    # both seats run the frozen current policy, both sides' requests are
    # current-policy learner rows. With no history pool yet this doubles the
    # learner rows per natural match (and the per-match PPO work) for the same
    # game, matching the documented collection contract.
    collect_both_sides_when_current_self_play: bool = True
    # Read-only drain-tail telemetry (v5 W2). When False the collector takes
    # exactly the same code path and produces the same rows/statistics as
    # before; when True it additionally records per-round active/open env
    # counts, idle slot-seconds, game lengths and cohort 50%->100% tail walls.
    telemetry: bool = False
    # v5b T1: SoA observation store (one growing array per field) instead of
    # the per-row dict store. Row values/order/digests must stay identical;
    # the row-SHA gate proves it before any promotion.
    columnar_observation_store: bool = False
    # v5c T2: rolling slot refill. Finished slots immediately start a new
    # natural match (same frozen iteration policy) until the iteration quota is
    # met; the remaining open games then drain (overshoot preserved). Contract
    # gates: match counted once, terminal accounting, both seats, row schema.
    rolling_slots: bool = False


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
    request_info_seconds: float = 0.0
    parse_seconds: float = 0.0
    h2d_seconds: float = 0.0
    candidate_seconds: float = 0.0
    table_seconds: float = 0.0
    readback_seconds: float = 0.0
    spec_seconds: float = 0.0
    assemble_seconds: float = 0.0
    model_seconds: float = 0.0
    step_seconds: float = 0.0
    record_seconds: float = 0.0
    reset_seconds: float = 0.0
    wall_seconds: float = 0.0
    unaccounted_seconds: float = 0.0
    cpu_seconds: float = 0.0
    cpu_fraction_of_one_core: float = 0.0
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
        if self.config.columnar_observation_store:
            from agent.buffer.rollout_buffer import ColumnarObservationStore

            self.buffer = RolloutBuffer(
                layout=self.layout, observation_store=ColumnarObservationStore()
            )
        else:
            self.buffer = RolloutBuffer(layout=self.layout)
        self.generator = torch.Generator(device=self.device.type).manual_seed(self.config.seed)
        self.stats = NativeCollectorStats()
        self._match_counter = 0
        self._team_count = int(self.engine.team_count())
        # Flag-gated, read-only drain-tail telemetry (v5 W2). Empty when off.
        self.telemetry: dict = {}
        self._round_active_envs = 0
        self._slot_counter = 0

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

    def _reset_slots(self, count: int, rng) -> tuple[list, list, list]:
        """Reset `count` slots from a persistent team RNG (rolling refill).

        Same team/seed/role distributions as `_reset_cohort`; role alternation
        continues across refills via a running slot counter so the learner seat
        stays balanced over the iteration.
        """
        team_a, team_b, seeds, roles = [], [], [], []
        for index in range(count):
            team_a.append(rng.randrange(self._team_count))
            team_b.append(rng.randrange(self._team_count))
            seeds.append(tuple(rng.randrange(1 << 16) for _ in range(4)))
            slot_index = self._slot_counter + index
            roles.append((0, 1) if slot_index % 2 == 0 else (1, 0))
        start = time.perf_counter()
        handles = self.engine.reset_batch(team_a, team_b, seeds, roles)
        self.stats.reset_seconds += time.perf_counter() - start
        self.stats.cohorts += 1
        self.stats.team_ids_seen += len(set(team_a) | set(team_b))
        self._slot_counter += count
        return handles, list(zip(team_a, team_b)), roles

    def _learner_side(self, roles: Sequence[int]) -> int:
        return 0 if int(roles[0]) == 0 else 1

    def _records_side(self, side: int, roles: Sequence[int]) -> bool:
        """Whether this seat's requests become current-policy learner rows.

        Both seats run the same frozen current policy while the history pool is
        empty, so the documented contract records both. Historical-opponent rows
        stay excluded (`collect_historical_opponent_rows: false`).
        """
        if self.config.collect_both_sides_when_current_self_play:
            return True
        return side == self._learner_side(roles)

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

    def _level_table_packed(self, counts, records, offsets) -> BranchCandidatesBatch:
        """Vectorized single-branch table from the packed candidate wire.

        No per-candidate Python object is created: the `[B, 1, P, 6]` action
        block, the mask and the entity/move token references are all computed
        with numpy from the flat `u8` records.
        """
        import numpy as np

        from pa3_engine.observation import packed_candidate_rows

        batch = len(counts)
        total = len(records)
        capacity = max(int(counts.max()) if batch else 0, 1)
        action_ids = np.zeros((batch, 1, capacity, 6), dtype=np.int64)
        mask = np.zeros((batch, 1, capacity), dtype=bool)
        if total:
            row, position = packed_candidate_rows({"counts": counts, "records": records, "offsets": offsets})
            action_ids[row, 0, position] = records.astype(np.int64)
            mask[row, 0, position] = True
        kind = action_ids[..., 0]
        own_slot = action_ids[..., 1]
        move_slot = action_ids[..., 2]
        destination = action_ids[..., 4]
        # Same token mapping as ActionRef.resolve, vectorized.
        slot_for_entity = np.where(kind == 0, destination, own_slot)
        valid_entity = (slot_for_entity >= 0) & (slot_for_entity < 6) & mask
        entity = np.where(valid_entity, self.layout.POKEMON_START + slot_for_entity, -1)
        valid_move = (kind == 1) & (own_slot >= 0) & (own_slot < 6) & (move_slot >= 0) & (move_slot < 4) & mask
        move_token = np.where(
            valid_move, self.layout.MOVES_START + own_slot * 4 + move_slot, -1
        )
        return BranchCandidatesBatch(
            action_ids=torch.as_tensor(action_ids).to(self.device),
            mask=torch.as_tensor(mask).to(self.device),
            entity_token=torch.as_tensor(entity.astype(np.int64)).to(self.device),
            move_token=torch.as_tensor(move_token.astype(np.int64)).to(self.device),
            branch_valid=torch.ones((batch, 1), dtype=torch.bool, device=self.device),
            selected=torch.full((batch, 1), -1, dtype=torch.long, device=self.device),
        )

    @staticmethod
    def _packed_action(packed, index: int, pick: int) -> tuple:
        """One packed record as the engine's action tuple (target re-signed)."""
        import numpy as np

        record = packed["records"][int(packed["offsets"][index]) + int(pick)]
        return (int(record[0]), int(record[1]), int(record[2]), int(np.int8(record[3])),
                int(record[4]), int(record[5]))

    @staticmethod
    def _prefix_actions(prefix_buf, offset: int, branch_count: int) -> list:
        """Packed prefix slab row as engine action tuples."""
        import numpy as np

        return [
            (int(record[0]), int(record[1]), int(record[2]), int(np.int8(record[3])),
             int(record[4]), int(record[5]))
            for record in prefix_buf[offset, :branch_count]
        ]

    def _packed_candidates(self, packed, index: int) -> list:
        """All packed records of one request as engine action tuples."""
        import numpy as np

        start = int(packed["offsets"][index])
        stop = int(packed["offsets"][index + 1])
        return [ (int(record[0]), int(record[1]), int(record[2]), int(np.int8(record[3])),
                  int(record[4]), int(record[5]))
                 for record in packed["records"][start:stop] ]

    def _packed_row_inputs(self, packed_by_level, offset: int, branch_count: int):
        """Per-level record arrays plus vectorized entity/move token references."""
        import numpy as np

        records = []
        entities = []
        moves = []
        for level in range(branch_count):
            packed = packed_by_level[level]
            start = int(packed["offsets"][offset])
            stop = int(packed["offsets"][offset + 1])
            record = packed["records"][start:stop]
            records.append(record)
            kind = record[:, 0].astype(np.int64)
            own_slot = record[:, 1].astype(np.int64)
            move_slot = record[:, 2].astype(np.int64)
            destination = record[:, 4].astype(np.int64)
            slot_for_entity = np.where(kind == 0, destination, own_slot)
            valid_entity = (slot_for_entity >= 0) & (slot_for_entity < 6)
            entities.append(np.where(valid_entity, self.layout.POKEMON_START + slot_for_entity, -1))
            valid_move = ((kind == 1) & (own_slot >= 0) & (own_slot < 6)
                          & (move_slot >= 0) & (move_slot < 4))
            moves.append(np.where(valid_move, self.layout.MOVES_START + own_slot * 4 + move_slot, -1))
        return records, entities, moves

    def _record(
        self,
        observation: ObservationBatch,
        request: RequestRow,
        selected: Sequence[int],
        old_logprob: float,
        value: float,
        match_id: int,
        side: int,
        team_ids: tuple[int, int],
        request_index: int,
        turn: int,
    ) -> None:
        if not self.config.record_rows:
            return
        self.buffer.record(
            observation=observation,
            request=request,
            selected=list(selected),
            old_logprob=float(old_logprob),
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

    def _inference_context(self):
        """Actor forward context: optional fp16 autocast + inference_mode."""
        from contextlib import nullcontext

        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.config.amp and self.device.type == "cuda"
            else nullcontext()
        )
        grad = torch.inference_mode() if self.config.inference_mode else torch.no_grad()
        return context, grad

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
        start_info = time.perf_counter()
        info = self.engine.request_info_batch(
            [(handle[0], handle[1], side) for _, handle, side in pending]
        )
        self.stats.request_info_seconds += time.perf_counter() - start_info
        decision = []
        for (env, handle, side), (kind, slots) in zip(pending, info):
            if int(kind) in (0, 1, 2):
                decision.append((env, handle, side, int(kind), tuple(int(s) for s in slots)))
        if not decision:
            return False
        start = time.perf_counter()
        handle_rows = [(handle[0], handle[1]) for _, handle, _, _, _ in decision]
        side_rows = [side for _, _, side, _, _ in decision]
        from pa3_engine.observation import parse_batch, parse_view

        # "fixed": one fixed-stride buffer plus a ragged sidecar for the whole
        # decision batch, decoded as a single structured numpy view.
        # "perview": one packed PyBytes per request (legacy path).
        batch_observations = None
        observations = None
        if self.config.observation_mode == "fixed":
            fixed_bytes, ragged_bytes = self.engine.observe_fixed_batch(handle_rows, side_rows)
            self.stats.observation_seconds += time.perf_counter() - start
            start = time.perf_counter()
            batch_observations = ObservationBatch.from_native_payload(
                parse_batch(fixed_bytes, ragged_bytes, len(decision)), layout=self.layout
            )
            self.stats.parse_seconds += time.perf_counter() - start
        else:
            blobs = self.engine.observe_encoded_batch(handle_rows, side_rows)
            self.stats.observation_seconds += time.perf_counter() - start
            start = time.perf_counter()
            observations = [
                ObservationBatch.from_native_payload(parse_view(blob), layout=self.layout)
                for blob in blobs
            ]
            self.stats.parse_seconds += time.perf_counter() - start
        # Group requests by branch count so a batched sequential sample is
        # well-formed for every row in the group.
        groups: dict[int, list[int]] = {}
        for index, entry in enumerate(decision):
            groups.setdefault(len(entry[4]), []).append(index)
        submissions: dict[int, list[tuple[int, list]]] = {}
        # Batch-level compact observation (one conversion for the whole round)
        # for the packed rollout fast path.
        compact_batch = batch_observations.to_compact_numpy() if batch_observations is not None else None
        for branch_count, indices in sorted(groups.items()):
            start = time.perf_counter()
            if batch_observations is not None:
                subset_obs = batch_observations.select(indices).to(self.device)
            else:
                subset_obs = observations[indices[0]].cat([observations[i] for i in indices[1:]]) \
                    if len(indices) > 1 else observations[indices[0]]
                subset_obs = subset_obs.to(self.device)
            self.stats.h2d_seconds += time.perf_counter() - start
            amp_context, grad_context = self._inference_context()
            start = time.perf_counter()
            with grad_context, amp_context:
                encoded = self.model.encode(subset_obs)
                hidden = self.model.scorer.initial_state(
                    encoded.tokens.shape[0], device=encoded.tokens.device, dtype=encoded.tokens.dtype
                )
            self.stats.model_seconds += time.perf_counter() - start
            steps = []
            picks_by_level: list[list[int]] = []
            packed_by_level: list = []
            candidate_rows_per_level: list[list[Sequence[Sequence[int]]]] = [[] for _ in range(branch_count)]
            prefixes: list[list[tuple]] = [[] for _ in indices]
            # Packed walk state: one [B, branch, 6] uint8 slab, updated with the
            # sampled rows. No per-request Python prefix list is built.
            prefix_buf = None
            walk_handles = walk_sides = None
            if self.config.candidate_wire == "packed":
                import numpy as np

                prefix_buf = np.zeros((len(indices), branch_count, 6), dtype=np.uint8)
                walk_handles = [(decision[i][1][0], decision[i][1][1]) for i in indices]
                walk_sides = [decision[i][2] for i in indices]
            for level in range(branch_count):
                start_spec = time.perf_counter()
                specs = []
                if prefix_buf is None:
                    for offset, request_index_in_decision in enumerate(indices):
                        env, handle, side, _kind, _slots = decision[request_index_in_decision]
                        specs.append((handle[0], handle[1], side, list(prefixes[offset])))
                self.stats.spec_seconds += time.perf_counter() - start_spec
                start_candidates = time.perf_counter()
                packed = None
                candidate_lists = None
                if self.config.candidate_wire == "packed":
                    import numpy as np

                    from pa3_engine.observation import parse_packed_candidates

                    prefix_bytes = np.ascontiguousarray(prefix_buf[:, :level, :]).tobytes()
                    counts_bytes, actions_bytes, _max_count = self.engine.candidates_packed_walk(
                        walk_handles, walk_sides, level, prefix_bytes
                    )
                    packed = parse_packed_candidates(counts_bytes, actions_bytes, len(indices))
                    self.stats.candidates_returned += int(packed["counts"].sum())
                    for offset in range(len(indices)):
                        kind = int(packed["records"][int(packed["offsets"][offset])][0])
                        self.stats.action_kinds[kind] = self.stats.action_kinds.get(kind, 0) + 1
                else:
                    candidate_lists = self.engine.candidates_batch(specs)
                    self.stats.candidates_returned += sum(len(row) for row in candidate_lists)
                    for offset, candidates in enumerate(candidate_lists):
                        candidate_rows_per_level[level].append(candidates)
                        # Raw packed tuples carry kind in field 0; no object is
                        # built for the histogram.
                        kind = int(candidates[0][0])
                        self.stats.action_kinds[kind] = self.stats.action_kinds.get(kind, 0) + 1
                self.stats.candidate_seconds += time.perf_counter() - start_candidates
                self.stats.candidate_calls += 1
                packed_by_level.append(packed)
                with grad_context, amp_context:
                    start_table = time.perf_counter()
                    level_table = (
                        self._level_table_packed(packed["counts"], packed["records"], packed["offsets"])
                        if packed is not None else self._level_table(candidate_lists)
                    )
                    self.stats.table_seconds += time.perf_counter() - start_table
                    start_model = time.perf_counter()
                    step = self.model.sample_level_step(
                        encoded,
                        hidden,
                        level_table,
                        temperature=1.0,
                        generator=self.generator,
                    )
                    self.stats.model_seconds += time.perf_counter() - start_model
                steps.append(step)
                hidden = step.hidden
                start_readback = time.perf_counter()
                if packed is not None:
                    import numpy as np

                    # One bulk D2H per level, then a vectorized prefix update.
                    picks_np = step.pick.detach().to("cpu").numpy().astype(np.int64)
                    picks_by_level.append([int(value) for value in picks_np])
                    selected_rows = packed["records"][packed["offsets"][:-1] + picks_np]
                    prefix_buf[:, level, :] = selected_rows
                else:
                    picks = [int(value) for value in step.pick.tolist()]
                    picks_by_level.append(picks)
                    for offset, candidates in enumerate(candidate_lists):
                        # The next level's mask is conditional on this sampled pick.
                        prefixes[offset].append(tuple(candidates[picks[offset]]))
                self.stats.readback_seconds += time.perf_counter() - start_readback
            sampled = assemble_level_result(steps)
            start_model = time.perf_counter()
            with grad_context, amp_context:
                values = self.model.value(encoded)
            self.stats.model_seconds += time.perf_counter() - start_model
            start_readback = time.perf_counter()
            old_logprobs = [float(value) for value in
                            sampled.request_logprob.detach().float().cpu().tolist()]
            value_list = [float(value) for value in values.detach().float().cpu().tolist()]
            self.stats.readback_seconds += time.perf_counter() - start_readback
            for offset, decision_index in enumerate(indices):
                env, handle, side, kind, slots = decision[decision_index]
                action = (
                    self._prefix_actions(prefix_buf, offset, branch_count)
                    if prefix_buf is not None else [tuple(value) for value in prefixes[offset]]
                )
                if self._records_side(side, roles[env]):
                    start_record = time.perf_counter()
                    selected_prefix = [picks_by_level[level][offset] for level in range(branch_count)]
                    if packed_by_level[0] is not None and compact_batch is not None:
                        # Fast path: no typed ActionRef/CandidateSet objects and
                        # no per-row torch copy — compact numpy rows only.
                        records, entities, moves = self._packed_row_inputs(
                            packed_by_level, offset, branch_count
                        )
                        # compact_batch is indexed by the round-wide decision
                        # index, while `offset` is the index inside this branch
                        # group. Using the group index here silently pairs a row
                        # with another request's observation.
                        decision_offset = indices[offset]
                        self.buffer.record_packed(
                            compact_batch,
                            observation_index=decision_offset,
                            branch_records=records,
                            entity_token=entities,
                            move_token=moves,
                            selected=selected_prefix,
                            old_logprob=old_logprobs[offset],
                            value=value_list[offset],
                            match_id=match_ids[env],
                            side=side,
                            policy_id=self.policy_id,
                            opponent_policy_id=self.policy_id,
                            team_ids=team_ids[env],
                            request_index=request_index[env],
                            turn=turn[env],
                            request_kind=RequestKind(kind),
                            branch_slots=slots,
                            actor_active=len(slots) > 0 and any(
                                int(packed_by_level[level]["counts"][offset]) >= 2
                                for level in range(branch_count)
                            ),
                        )
                        self.stats.learner_rows += 1
                    else:
                        # Typed path (legacy wire or per-view observations).
                        row_observation = subset_obs.select([offset])
                        if packed_by_level[0] is not None:
                            branches = tuple(
                                CandidateSet.from_tuples(self._packed_candidates(packed_by_level[level], offset))
                                for level in range(branch_count)
                            )
                        else:
                            branches = tuple(
                                CandidateSet.from_tuples(list(candidate_rows_per_level[level][offset]))
                                for level in range(branch_count)
                            )
                        request = RequestRow(
                            observation=row_observation,
                            kind=RequestKind(kind),
                            branch_slots=slots,
                            branches=branches,
                        )
                        self._record(
                            observation=row_observation,
                            request=request,
                            selected=selected_prefix,
                            old_logprob=old_logprobs[offset],
                            value=value_list[offset],
                            match_id=match_ids[env],
                            side=side,
                            team_ids=team_ids[env],
                            request_index=request_index[env],
                            turn=turn[env],
                        )
                    self.stats.record_seconds += time.perf_counter() - start_record
                    self._row_cursor[(env, side)] = len(self.buffer.rows) - 1
                else:
                    self.stats.opponent_requests += 1
                submissions.setdefault(env, []).append((side, action))
                self.stats.decisions += 1
        specs = []
        order = sorted(submissions)
        start_assemble = time.perf_counter()
        for env in order:
            handle = handles[env]
            specs.append((handle[0], handle[1], submissions[env]))
        self.stats.assemble_seconds += time.perf_counter() - start_assemble
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
                # One terminal reward per recorded side's own trajectory; a
                # draw pays 0.0 to both seats.
                if winner is None:
                    learner_reward = 0.0
                    self.stats.draws += 1
                elif int(winner) == learner:
                    learner_reward = 1.0
                    self.stats.wins += 1
                else:
                    learner_reward = -1.0
                    self.stats.losses += 1
                for side in (0, 1):
                    if not self._records_side(side, roles[env]):
                        continue
                    row_index = self._row_cursor.get((env, side))
                    if row_index is None:
                        continue
                    if winner is None:
                        reward = 0.0
                    elif int(winner) == side:
                        reward = 1.0
                    else:
                        reward = -1.0
                    row = self.buffer.rows[row_index]
                    row.reward = reward
                    row.done = True
                self.stats.reward_sum += learner_reward
                self.stats.games += 1
                finished[env] = True
                match_ids[env] = self._match_counter
                self._match_counter += 1
                request_index[env] = 0
                turn[env] = 0
            else:
                request_index[env] += 1
                turn[env] += 1
        if self.config.telemetry:
            # Number of environments that produced at least one decision this
            # round (read-only; the training path is untouched when disabled).
            self._round_active_envs = len(submissions)
        return True

    # -- collection -------------------------------------------------------
    def collect(self, target_games: int) -> RolloutBuffer:
        import resource

        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        self._row_cursor: dict[tuple[int, int], int] = {}
        self.model.eval()
        started = time.perf_counter()
        telemetry = self.config.telemetry
        if telemetry:
            self.telemetry = {
                "rounds": 0,
                "round_seconds": [],
                "active_envs": [],
                "open_envs": [],
                "idle_slot_seconds": 0.0,
                "slot_seconds": 0.0,
                "game_rounds": [],
                "cohorts": [],
            }
        if self.config.rolling_slots:
            self._collect_rolling(target_games)
        while not self.config.rolling_slots and self.stats.games < target_games:
            handles, teams, roles = self._reset_cohort()
            cohort_started = time.perf_counter()
            finished = [False] * len(handles)
            match_ids = [self._match_counter + index for index in range(len(handles))]
            self._match_counter += len(handles)
            request_index = [0] * len(handles)
            turn = [0] * len(handles)
            self._row_cursor = {}
            rounds = 0
            env_rounds = [0] * len(handles)
            cohort_t50 = None
            while not all(finished):
                round_started = time.perf_counter()
                rounds += 1
                if rounds > self.config.max_rounds_per_cohort:
                    raise RuntimeError("cohort did not drain within the round budget")
                finished_before = list(finished)
                progressed = self._collect_round(
                    handles, roles, teams, match_ids, request_index, turn, finished
                )
                if not progressed:
                    raise RuntimeError("no actionable request while a cohort is open")
                open_games = sum(1 for value in finished if not value)
                self.stats.max_open_games = max(self.stats.max_open_games, open_games)
                if telemetry:
                    round_seconds = time.perf_counter() - round_started
                    finished_count_before = sum(1 for value in finished_before if value)
                    newly_finished = [
                        index
                        for index, value in enumerate(finished)
                        if value and not finished_before[index]
                    ]
                    for index in newly_finished:
                        self.telemetry["game_rounds"].append(env_rounds[index] + 1)
                    for index, value in enumerate(finished):
                        if not value:
                            env_rounds[index] += 1
                    self.telemetry["rounds"] += 1
                    self.telemetry["round_seconds"].append(round_seconds)
                    self.telemetry["active_envs"].append(int(self._round_active_envs))
                    self.telemetry["open_envs"].append(len(handles) - finished_count_before)
                    self.telemetry["idle_slot_seconds"] += finished_count_before * round_seconds
                    self.telemetry["slot_seconds"] += len(handles) * round_seconds
                    finished_now = len(handles) - open_games
                    if cohort_t50 is None and finished_now * 2 >= len(handles):
                        cohort_t50 = time.perf_counter() - cohort_started
            if telemetry:
                cohort_wall = time.perf_counter() - cohort_started
                self.telemetry["cohorts"].append(
                    {
                        "size": len(handles),
                        "rounds": rounds,
                        "wall_seconds": cohort_wall,
                        "games_finished": sum(1 for value in finished if value),
                        "tail_seconds_50_to_100": (
                            cohort_wall - cohort_t50 if cohort_t50 is not None else None
                        ),
                    }
                )
        self.stats.wall_seconds = time.perf_counter() - started
        accounted = (self.stats.reset_seconds + self.stats.observation_seconds + self.stats.parse_seconds
                     + self.stats.h2d_seconds + self.stats.request_info_seconds
                     + self.stats.candidate_seconds + self.stats.table_seconds + self.stats.model_seconds
                     + self.stats.readback_seconds + self.stats.spec_seconds + self.stats.assemble_seconds
                     + self.stats.step_seconds + self.stats.record_seconds)
        self.stats.unaccounted_seconds = max(self.stats.wall_seconds - accounted, 0.0)
        usage_after = resource.getrusage(resource.RUSAGE_SELF)
        self.stats.cpu_seconds = ((usage_after.ru_utime - usage_before.ru_utime)
                                  + (usage_after.ru_stime - usage_before.ru_stime))
        self.stats.cpu_fraction_of_one_core = self.stats.cpu_seconds / max(self.stats.wall_seconds, 1e-9)
        return self.buffer

    def _collect_rolling(self, target_games: int) -> None:
        """Rolling-slot collection (v5c T2): refill finished slots immediately."""
        import random

        telemetry = self.config.telemetry
        if telemetry:
            self.telemetry = {
                "rounds": 0,
                "round_seconds": [],
                "active_envs": [],
                "open_envs": [],
                "idle_slot_seconds": 0.0,
                "slot_seconds": 0.0,
                "game_rounds": [],
                "cohorts": [],
            }
        rng = random.Random(self.config.seed)
        size = max(1, int(self.config.envs))
        initial_games = self.stats.games
        loop_started = time.perf_counter()
        handles, teams, roles = self._reset_slots(size, rng)
        match_ids = [self._match_counter + index for index in range(size)]
        self._match_counter += size
        request_index = [0] * size
        turn = [0] * size
        finished = [False] * size
        env_rounds = [0] * size
        self._row_cursor = {}
        rounds = 0
        round_budget = self.config.max_rounds_per_cohort * max(1, (target_games // size) + 2)
        t50 = None
        while True:
            open_games = sum(1 for value in finished if not value)
            quota_done = self.stats.games - initial_games >= target_games
            if open_games == 0 and quota_done:
                break
            if rounds >= round_budget:
                raise RuntimeError("rolling slots did not drain within the round budget")
            if not quota_done:
                refill = [index for index, value in enumerate(finished) if value]
                if refill:
                    for index in refill:
                        self._row_cursor.pop((index, 0), None)
                        self._row_cursor.pop((index, 1), None)
                    new_handles, new_teams, new_roles = self._reset_slots(len(refill), rng)
                    for slot, handle, team, role in zip(refill, new_handles, new_teams, new_roles):
                        handles[slot] = handle
                        teams[slot] = team
                        roles[slot] = role
                        match_ids[slot] = self._match_counter
                        self._match_counter += 1
                        request_index[slot] = 0
                        turn[slot] = 0
                        env_rounds[slot] = 0
                        finished[slot] = False
            round_started = time.perf_counter()
            rounds += 1
            finished_before = list(finished)
            progressed = self._collect_round(
                handles, roles, teams, match_ids, request_index, turn, finished
            )
            if not progressed:
                raise RuntimeError("no actionable request while rolling slots are open")
            open_games = sum(1 for value in finished if not value)
            self.stats.max_open_games = max(self.stats.max_open_games, open_games)
            if telemetry:
                round_seconds = time.perf_counter() - round_started
                finished_count_before = sum(1 for value in finished_before if value)
                newly_finished = [
                    index for index, value in enumerate(finished)
                    if value and not finished_before[index]
                ]
                for index in newly_finished:
                    self.telemetry["game_rounds"].append(env_rounds[index] + 1)
                for index, value in enumerate(finished):
                    if not value:
                        env_rounds[index] += 1
                self.telemetry["rounds"] += 1
                self.telemetry["round_seconds"].append(round_seconds)
                self.telemetry["active_envs"].append(int(self._round_active_envs))
                self.telemetry["open_envs"].append(size - open_games)
                self.telemetry["idle_slot_seconds"] += finished_count_before * round_seconds
                self.telemetry["slot_seconds"] += size * round_seconds
                if t50 is None and (self.stats.games - initial_games) * 2 >= target_games:
                    t50 = time.perf_counter() - loop_started
        if telemetry:
            loop_wall = time.perf_counter() - loop_started
            self.telemetry["cohorts"].append(
                {
                    "size": size,
                    "rounds": rounds,
                    "wall_seconds": loop_wall,
                    "games_finished": self.stats.games - initial_games,
                    "tail_seconds_50_to_100": (
                        loop_wall - t50 if t50 is not None else None
                    ),
                    "mode": "rolling",
                }
            )
