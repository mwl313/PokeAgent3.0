"""Compact, engine-independent rollout schema.

One row is one *complete player request*: the observation (or a reference to
it), the branch candidate tree that was legal at collection time, the selected
prefix, the behaviour-policy log-probability, the value estimate and the
terminal reward.  Encoder activations are never stored by default; the learner
recomputes them.

The buffer supports

* overshoot rows (whatever a collection iteration actually produced),
* padded/masked final minibatches,
* both sides of a current-policy self-play match, and
* honest exclusion of historical-opponent rows from current-policy learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np
import torch

from agent.types.actions import BRANCH_CAPACITY, RequestKind
from agent.types.observation import ObservationBatch, ObservationLayout
from agent.types.requests import BranchCandidatesBatch, RequestRow


class ObservationStore:
    """Storage for the typed observation of each row.

    A real collector may instead keep an engine-side reference (handle,
    generation, side, snapshot id) and resolve it lazily; the learner only
    requires ``get(index) -> ObservationBatch``.
    """

    def add(self, observation: ObservationBatch) -> int:  # pragma: no cover - protocol
        raise NotImplementedError

    def get(self, index: int) -> ObservationBatch:  # pragma: no cover - protocol
        raise NotImplementedError

    def __len__(self) -> int:  # pragma: no cover - protocol
        raise NotImplementedError


class InlineObservationStore(ObservationStore):
    """Compact in-process store (the default; no activations, no byte blobs)."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def add(self, observation: ObservationBatch) -> int:
        if len(observation) != 1:
            raise ValueError("store one observation row at a time")
        # Compact low-precision arrays with a leading batch dimension of one.
        self._rows.append(observation.to_compact_numpy())
        return len(self._rows) - 1

    def add_compact(self, row: dict[str, Any]) -> int:
        """Store an already-compact row (same dtypes as ``to_compact_numpy``)."""
        self._rows.append(row)
        return len(self._rows) - 1

    def get(self, index: int) -> ObservationBatch:
        row = self._rows[index]
        return ObservationBatch.from_compact_numpy(row)

    def __len__(self) -> int:
        return len(self._rows)

    def stacked(self) -> ObservationBatch:
        if not self._rows:
            raise ValueError("observation store is empty")
        if len(self._rows) == 1:
            return ObservationBatch.from_compact_numpy(self._rows[0])
        import numpy as _np

        combined = {
            key: _np.concatenate([row[key] for row in self._rows], axis=0)
            for key in self._rows[0]
        }
        return ObservationBatch.from_compact_numpy(combined)

    def stacked_indices(self, indices) -> ObservationBatch:
        """Stack only the selected rows (streaming minibatch materialization)."""
        import numpy as _np

        chosen = [self._rows[int(index)] for index in indices]
        if not chosen:
            raise ValueError("observation selection is empty")
        if len(chosen) == 1:
            return ObservationBatch.from_compact_numpy(chosen[0])
        combined = {
            key: _np.concatenate([row[key] for row in chosen], axis=0)
            for key in chosen[0]
        }
        return ObservationBatch.from_compact_numpy(combined)


class ColumnarObservationStore(ObservationStore):
    """SoA (columnar) observation store: one growing array per field.

    `add_compact_at` appends row `index` of a compact decision-batch block
    directly into the per-field slabs, so the collector never builds a per-row
    dict (v5b T1). `stacked_indices` is one numpy fancy-index gather per field
    instead of `np.concatenate` over thousands of one-row arrays (v5b T2
    observations). Values, order and dtype are identical to
    `InlineObservationStore`; the row-SHA gate proves it.
    """

    _INITIAL_CAPACITY = 2048

    def __init__(self) -> None:
        self._arrays: dict[str, Any] = {}
        self._count = 0

    def _ensure(self, key: str, trailing: tuple, dtype) -> None:
        array = self._arrays.get(key)
        if array is None:
            self._arrays[key] = np.zeros(
                (self._INITIAL_CAPACITY,) + tuple(trailing), dtype=dtype
            )
            return
        if array.dtype != np.dtype(dtype) or array.shape[1:] != tuple(trailing):
            raise ValueError(f"columnar field {key!r} changed dtype/shape mid-collection")
        if self._count >= array.shape[0]:
            grown = np.zeros((array.shape[0] * 2,) + array.shape[1:], dtype=array.dtype)
            grown[: array.shape[0]] = array
            self._arrays[key] = grown

    def _append(self, row: dict) -> int:
        index = self._count
        for key, value in row.items():
            self._ensure(key, value.shape[1:], value.dtype)
            self._arrays[key][index] = value[0]
        self._count += 1
        return index

    def add(self, observation: ObservationBatch) -> int:
        return self.add_compact(observation.to_compact_numpy())

    def add_compact(self, row: dict) -> int:
        return self._append(row)

    def add_compact_at(self, block: dict, index: int) -> int:
        """Append row `index` of a compact batch block without per-row copies."""
        row = {key: value[index:index + 1] for key, value in block.items()}
        return self._append(row)

    def get(self, index: int) -> ObservationBatch:
        return ObservationBatch.from_compact_numpy(
            {key: value[index:index + 1] for key, value in self._arrays.items()}
        )

    def __len__(self) -> int:
        return self._count

    def stacked(self) -> ObservationBatch:
        if self._count == 0:
            raise ValueError("observation store is empty")
        return ObservationBatch.from_compact_numpy(
            {key: value[: self._count] for key, value in self._arrays.items()}
        )

    def stacked_indices(self, indices) -> ObservationBatch:
        selection = np.asarray(list(indices), dtype=np.int64)
        if selection.size == 0:
            raise ValueError("observation selection is empty")
        return ObservationBatch.from_compact_numpy(
            {key: value[selection] for key, value in self._arrays.items()}
        )


@dataclass
class RolloutRow:
    """One complete player request recorded during collection."""

    # -- identity / provenance
    match_id: int
    side: int  # 0 = P1, 1 = P2
    policy_id: str
    opponent_policy_id: str
    team_ids: tuple[int, int]
    request_index: int
    turn: int
    request_kind: RequestKind

    # -- observation reference
    observation_ref: int

    # -- action tree (candidate ids + masks + preserved selected prefix)
    branch_slots: tuple[int, ...]
    action_ids: tuple[tuple[tuple[int, ...], ...], ...]
    candidate_mask: tuple[tuple[bool, ...], ...]
    entity_token: tuple[tuple[int, ...], ...]
    move_token: tuple[tuple[int, ...], ...]
    selected: tuple[int, ...]
    selected_actions: tuple[tuple[int, ...], ...]

    # -- behaviour policy statistics
    old_logprob: float
    value: float

    # -- outcome
    reward: float = 0.0
    done: bool = False
    actor_active: bool = True

    # -- optional debug reference (RNG/seed pointer, not a weight snapshot)
    seed_ref: Optional[int] = None

    # -- filled by GAE / normalization
    advantage: float = 0.0
    return_: float = 0.0

    @property
    def branch_count(self) -> int:
        return len(self.branch_slots)

    @property
    def legal_counts(self) -> tuple[int, ...]:
        return tuple(sum(1 for flag in mask if flag) for mask in self.candidate_mask)

    def selected_logged(self) -> bool:
        return bool(self.selected) and all(
            index >= 0 and index < len(mask) for index, mask in zip(self.selected, self.candidate_mask)
        )

    def to_request_row(self, observation: ObservationBatch) -> RequestRow:
        from agent.types.actions import AtomicAction, CandidateSet
        from agent.types.requests import ActionRef

        branches = tuple(
            CandidateSet.from_tuples(actions, mask)
            for actions, mask in zip(self.action_ids, self.candidate_mask)
        )
        refs = tuple(
            tuple(
                ActionRef(
                    action=AtomicAction.from_tuple(action),
                    entity_token=int(entity),
                    move_token=int(move),
                )
                for action, entity, move in zip(actions, entities, moves)
            )
            for actions, entities, moves in zip(
                self.action_ids, self.entity_token, self.move_token
            )
        )
        return RequestRow(
            observation=observation,
            kind=RequestKind(self.request_kind),
            branch_slots=tuple(self.branch_slots),
            branches=branches,
            refs=refs,
        )


@dataclass
class RolloutBuffer:
    """Row storage plus minibatch/tensor admission for the PPO learner."""

    observation_store: ObservationStore = field(default_factory=InlineObservationStore)
    rows: list[RolloutRow] = field(default_factory=list)
    layout: ObservationLayout = field(default_factory=ObservationLayout)
    _stacked: Optional[ObservationBatch] = field(default=None, repr=False)

    # -- collection ------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[RolloutRow]:
        return iter(self.rows)

    def add_row(self, row: RolloutRow) -> RolloutRow:
        self.rows.append(row)
        self._stacked = None
        return row

    def extend(self, rows: Iterable[RolloutRow]) -> None:
        for row in rows:
            self.add_row(row)

    def record(
        self,
        observation: ObservationBatch,
        request: RequestRow,
        *,
        selected: Sequence[int],
        old_logprob: float,
        value: float,
        match_id: int,
        side: int,
        policy_id: str,
        opponent_policy_id: str,
        team_ids: tuple[int, int],
        request_index: int,
        turn: int,
        reward: float = 0.0,
        done: bool = False,
        seed_ref: Optional[int] = None,
    ) -> RolloutRow:
        """Append a row from a sampled request (the collector's main entry)."""
        selected = tuple(int(i) for i in selected)
        if len(selected) != request.branch_count:
            raise ValueError(
                f"selected prefix has {len(selected)} entries for "
                f"{request.branch_count} branches"
            )
        for index, branch in zip(selected, request.branches):
            if not 0 <= index < branch.size:
                raise ValueError("selected prefix index out of range")
            if not branch.mask[index]:
                raise ValueError("selected prefix points at an illegal candidate")
        observation_ref = self.observation_store.add(observation)
        row = RolloutRow(
            match_id=match_id,
            side=side,
            policy_id=str(policy_id),
            opponent_policy_id=str(opponent_policy_id),
            team_ids=(int(team_ids[0]), int(team_ids[1])),
            request_index=int(request_index),
            turn=int(turn),
            request_kind=RequestKind(request.kind),
            observation_ref=observation_ref,
            branch_slots=tuple(request.branch_slots),
            action_ids=tuple(branch.tuples() for branch in request.branches),
            candidate_mask=tuple(tuple(bool(m) for m in branch.mask) for branch in request.branches),
            entity_token=tuple(
                tuple(int(ref.entity_token) for ref in refs) for refs in request.refs
            ),
            move_token=tuple(
                tuple(int(ref.move_token) for ref in refs) for refs in request.refs
            ),
            selected=selected,
            selected_actions=tuple(
                branch.actions[index].as_tuple()
                for index, branch in zip(selected, request.branches)
            ),
            old_logprob=float(old_logprob),
            value=float(value),
            reward=float(reward),
            done=bool(done),
            actor_active=bool(request.actor_active),
            seed_ref=seed_ref,
        )
        return self.add_row(row)

    def record_packed(
        self,
        observation_compact: dict,
        *,
        branch_records: Sequence,
        entity_token: Sequence[Sequence[int]],
        move_token: Sequence[Sequence[int]],
        selected: Sequence[int],
        old_logprob: float,
        value: float,
        match_id: int,
        side: int,
        policy_id: str,
        opponent_policy_id: str,
        team_ids: tuple[int, int],
        request_index: int,
        turn: int,
        request_kind: RequestKind,
        branch_slots: Sequence[int],
        observation_index: Optional[int] = None,
        reward: float = 0.0,
        done: bool = False,
        actor_active: bool = True,
        seed_ref: Optional[int] = None,
    ) -> RolloutRow:
        """Record one row from packed candidate records without typed objects.

        ``branch_records[j]`` is the `[n_j, 6]` record array of branch `j`
        exactly as the engine returned it at the sampled prefix, so the learner's
        recomputation sees the same candidate tables and masks.
        """
        selected = tuple(int(i) for i in selected)
        action_ids = []
        masks = []
        for level in branch_records:
            records = tuple(map(tuple, level.tolist())) if hasattr(level, "tolist") else tuple(
                tuple(int(v) for v in record) for record in level
            )
            if not records:
                raise ValueError("packed branch without a legal candidate")
            action_ids.append(records)
            masks.append(tuple([True] * len(records)))
        if len(selected) != len(action_ids):
            raise ValueError(
                f"selected prefix has {len(selected)} entries for {len(action_ids)} branches"
            )
        for index, mask in zip(selected, masks):
            if not 0 <= index < len(mask):
                raise ValueError("selected prefix index out of range")
        if observation_index is not None:
            if hasattr(self.observation_store, "add_compact_at"):
                # SoA fast path: append row `observation_index` straight into
                # the per-field slabs without a per-row dict (v5b T1).
                observation_ref = self.observation_store.add_compact_at(
                    observation_compact, int(observation_index)
                )
            else:
                observation_ref = self.observation_store.add_compact(
                    {
                        key: value[observation_index:observation_index + 1].copy()
                        for key, value in observation_compact.items()
                    }
                )
        else:
            observation_ref = self.observation_store.add_compact(observation_compact)
        row = RolloutRow(
            match_id=int(match_id),
            side=int(side),
            policy_id=str(policy_id),
            opponent_policy_id=str(opponent_policy_id),
            team_ids=(int(team_ids[0]), int(team_ids[1])),
            request_index=int(request_index),
            turn=int(turn),
            request_kind=RequestKind(request_kind),
            observation_ref=observation_ref,
            branch_slots=tuple(int(slot) for slot in branch_slots),
            action_ids=tuple(action_ids),
            candidate_mask=tuple(masks),
            entity_token=tuple(tuple(int(v) for v in level) for level in entity_token),
            move_token=tuple(tuple(int(v) for v in level) for level in move_token),
            selected=selected,
            selected_actions=tuple(action_ids[level][selected[level]] for level in range(len(selected))),
            old_logprob=float(old_logprob),
            value=float(value),
            reward=float(reward),
            done=bool(done),
            actor_active=bool(actor_active),
            seed_ref=seed_ref,
        )
        return self.add_row(row)

    # -- filters ---------------------------------------------------------
    def current_policy_rows(self, policy_id: str) -> list[RolloutRow]:
        """Rows whose *acting* policy is the current one.

        Historical opponents' actions are not current-policy PPO rows; they are
        simply absent from this filter.
        """
        return [row for row in self.rows if row.policy_id == policy_id]

    def actor_rows(self) -> list[RolloutRow]:
        return [row for row in self.rows if row.actor_active]

    def value_rows(self) -> list[RolloutRow]:
        """All rows learn a value; the actor loss excludes all-singleton rows."""
        return list(self.rows)

    def match_ids(self) -> set[int]:
        return {row.match_id for row in self.rows}

    def policy_ids(self) -> set[str]:
        return {row.policy_id for row in self.rows}

    def natural_match_count(self) -> int:
        """A match counts once even when both sides were collected."""
        return len(self.match_ids())

    # -- observation access ----------------------------------------------
    def observation(self, index: int) -> ObservationBatch:
        return self.observation_store.get(index)

    def stacked_observations(self) -> ObservationBatch:
        if self._stacked is None:
            if not isinstance(
                self.observation_store, (InlineObservationStore, ColumnarObservationStore)
            ):
                raise TypeError(
                    "this observation store cannot be stacked; materialize the "
                    "typed observation per row instead"
                )
            self._stacked = self.observation_store.stacked()
        return self._stacked

    # -- batches -----------------------------------------------------------
    def to_batch(
        self,
        rows: Optional[Sequence[RolloutRow]] = None,
        device: torch.device | str | None = None,
    ) -> "RolloutBatch":
        import time as _time

        self.profile = {"observations": 0.0, "candidates": 0.0, "columns": 0.0, "move": 0.0}
        _t = _time.perf_counter()
        rows = list(self.rows if rows is None else rows)
        if not rows:
            raise ValueError("cannot build a batch from an empty rollout")
        if len(rows) == len(self.rows) and all(
            row.observation_ref == index for index, row in enumerate(rows)
        ):
            observations = self.stacked_observations()
        elif hasattr(self.observation_store, "stacked_indices"):
            # Streaming path: materialize only the selected rows instead of
            # stacking the whole iteration and slicing it.
            observations = self.observation_store.stacked_indices(
                [row.observation_ref for row in rows]
            )
        else:
            observations = self.observation(rows[0].observation_ref)
            if len(rows) > 1:
                observations = observations.cat(
                    [self.observation(row.observation_ref) for row in rows[1:]]
                )
        self.profile["observations"] = _time.perf_counter() - _t
        _t = _time.perf_counter()
        # Build the candidate table straight from the stored rows: the typed
        # ActionRef/CandidateSet round trip is equivalent but costs one Python
        # object per candidate and was the dominant learner-side cost.
        candidates = BranchCandidatesBatch.from_rows(
            rows,
            candidate_padding=64,
            branch_capacity=BRANCH_CAPACITY,
            device=device,
        )
        self.profile["candidates"] = _time.perf_counter() - _t
        _t = _time.perf_counter()

        def _column(name: str, dtype: torch.dtype) -> torch.Tensor:
            values = [getattr(row, name) for row in rows]
            if dtype == torch.float32:
                values = [float(value) for value in values]
            elif dtype == torch.bool:
                values = [bool(value) for value in values]
            else:
                values = [int(value) for value in values]
            return torch.tensor(values, dtype=dtype)

        batch = RolloutBatch(
            observation=observations,
            candidates=candidates,
            old_logprob=_column("old_logprob", torch.float32),
            values=_column("value", torch.float32),
            raw_advantages=_column("advantage", torch.float32),
            returns=_column("return_", torch.float32),
            rewards=_column("reward", torch.float32),
            dones=_column("done", torch.bool),
            actor_mask=_column("actor_active", torch.bool),
            row_valid=torch.ones(len(rows), dtype=torch.bool),
            match_ids=_column("match_id", torch.long),
            sides=_column("side", torch.long),
            request_index=_column("request_index", torch.long),
            turns=_column("turn", torch.long),
            request_kind=_column("request_kind", torch.long),
            policy_ids=[row.policy_id for row in rows],
        )
        # Columns + the row-table selection inside the candidate builder are the
        # CPU-side costs; the device move is measured separately.
        self.profile["columns"] = _time.perf_counter() - _t
        _t = _time.perf_counter()
        if device is not None:
            batch = batch.to(device)
            self.profile["move"] = _time.perf_counter() - _t
        return batch

    def iter_minibatches(
        self,
        batch_size: int,
        shuffle: bool = False,
        generator: Optional[torch.Generator] = None,
        drop_last: bool = False,
    ) -> Iterator["RolloutBatch"]:
        """Minibatches with a padded/masked final chunk (never dropped)."""
        if len(self) == 0:
            raise ValueError("cannot iterate an empty rollout")
        batch = self.to_batch()
        yield from batch.iter_minibatches(
            batch_size,
            shuffle=shuffle,
            generator=generator,
            drop_last=drop_last,
        )

    def to_batch_indices(
        self, indices: torch.Tensor, row_valid: Optional[torch.Tensor] = None
    ) -> "RolloutBatch":
        rows = [self.rows[int(i)] for i in indices.tolist()]
        batch = self.to_batch(rows)
        if row_valid is None:
            row_valid = torch.ones(len(rows), dtype=torch.bool)
        return replace(batch, row_valid=row_valid.to(torch.bool))

    # -- GAE ---------------------------------------------------------------
    def compute_gae(
        self,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        rows: Optional[Sequence[RolloutRow]] = None,
        bootstrap_by_sequence: Optional[dict] = None,
    ) -> None:
        from agent.ppo.gae import compute_gae_for_rows

        compute_gae_for_rows(
            list(self.rows if rows is None else rows),
            gamma=gamma,
            gae_lambda=gae_lambda,
            bootstrap_by_sequence=bootstrap_by_sequence,
        )

    # -- statistics ---------------------------------------------------------
    def stats(self) -> dict:
        return {
            "rows": len(self.rows),
            "actor_rows": sum(1 for row in self.rows if row.actor_active),
            "value_rows": len(self.rows),
            "matches": self.natural_match_count(),
            "policies": sorted(self.policy_ids()),
            "branches": sum(row.branch_count for row in self.rows),
            "candidates": sum(
                sum(len(mask) for mask in row.candidate_mask) for row in self.rows
            ),
        }


@dataclass
class RolloutBatch:
    """Tensor view of rollout rows ready for a PPO update."""

    observation: ObservationBatch
    candidates: BranchCandidatesBatch
    old_logprob: torch.Tensor  # [B]
    values: torch.Tensor  # [B]
    raw_advantages: torch.Tensor  # [B]
    returns: torch.Tensor  # [B]
    rewards: torch.Tensor  # [B]
    dones: torch.Tensor  # [B] bool
    actor_mask: torch.Tensor  # [B] bool
    row_valid: torch.Tensor  # [B] bool (minibatch padding mask)
    match_ids: torch.Tensor  # [B]
    sides: torch.Tensor  # [B]
    request_index: torch.Tensor  # [B]
    turns: torch.Tensor  # [B]
    request_kind: torch.Tensor  # [B]
    policy_ids: list[str] = field(default_factory=list)
    advantages: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return int(self.old_logprob.shape[0])

    def to(self, device: torch.device | str) -> "RolloutBatch":
        def _move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            return value

        return replace(
            self,
            observation=self.observation.to(device),
            candidates=self.candidates.to(device),
            old_logprob=_move(self.old_logprob),
            values=_move(self.values),
            raw_advantages=_move(self.raw_advantages),
            returns=_move(self.returns),
            rewards=_move(self.rewards),
            dones=_move(self.dones),
            actor_mask=_move(self.actor_mask),
            row_valid=_move(self.row_valid),
            match_ids=_move(self.match_ids),
            sides=_move(self.sides),
            request_index=_move(self.request_index),
            turns=_move(self.turns),
            request_kind=_move(self.request_kind),
            advantages=_move(self.advantages) if self.advantages is not None else None,
        )

    def select(self, indices: torch.Tensor, row_valid: Optional[torch.Tensor] = None) -> "RolloutBatch":
        # Minibatch indices are produced on the CPU while the batch may live on
        # a GPU; index_select requires both on the same device.
        # Keep the host indices for Python-only provenance. Converting `idx`
        # back to a list after moving it to CUDA would synchronise every cached
        # minibatch gather unnecessarily.
        policy_ids = (
            [self.policy_ids[int(i)] for i in indices.tolist()]
            if self.policy_ids else self.policy_ids
        )
        idx = indices.to(device=self.old_logprob.device, dtype=torch.long)
        return RolloutBatch(
            observation=self.observation.select(idx),
            candidates=self.candidates.select(idx),
            old_logprob=self.old_logprob.index_select(0, idx),
            values=self.values.index_select(0, idx),
            raw_advantages=self.raw_advantages.index_select(0, idx),
            returns=self.returns.index_select(0, idx),
            rewards=self.rewards.index_select(0, idx),
            dones=self.dones.index_select(0, idx),
            actor_mask=self.actor_mask.index_select(0, idx),
            row_valid=(
                self.row_valid.index_select(0, idx)
                if row_valid is None
                else row_valid.to(device=self.old_logprob.device, dtype=torch.bool)
            ),
            match_ids=self.match_ids.index_select(0, idx),
            sides=self.sides.index_select(0, idx),
            request_index=self.request_index.index_select(0, idx),
            turns=self.turns.index_select(0, idx),
            request_kind=self.request_kind.index_select(0, idx),
            policy_ids=policy_ids,
            advantages=(
                self.advantages.index_select(0, idx)
                if self.advantages is not None
                else None
            ),
        )

    def with_advantages(self, advantages: torch.Tensor) -> "RolloutBatch":
        return replace(self, advantages=advantages)

    def narrow(self, begin: int, end: int) -> "RolloutBatch":
        """Zero-copy contiguous row slice (the ordered minibatch case).

        `select(arange(begin, end))` materialises a gather copy of every row
        tensor; the micro loops process contiguous chunks, so `narrow` gives the
        same rows with views instead of copies (P5.4).
        """
        count = end - begin
        return RolloutBatch(
            observation=self.observation.narrow(begin, end),
            candidates=self.candidates.narrow(begin, end),
            old_logprob=self.old_logprob.narrow(0, begin, count),
            values=self.values.narrow(0, begin, count),
            raw_advantages=self.raw_advantages.narrow(0, begin, count),
            returns=self.returns.narrow(0, begin, count),
            rewards=self.rewards.narrow(0, begin, count),
            dones=self.dones.narrow(0, begin, count),
            actor_mask=self.actor_mask.narrow(0, begin, count),
            row_valid=self.row_valid.narrow(0, begin, count),
            match_ids=self.match_ids.narrow(0, begin, count),
            sides=self.sides.narrow(0, begin, count),
            request_index=self.request_index.narrow(0, begin, count),
            turns=self.turns.narrow(0, begin, count),
            request_kind=self.request_kind.narrow(0, begin, count),
            policy_ids=self.policy_ids[begin:end] if self.policy_ids else self.policy_ids,
            advantages=(
                self.advantages.narrow(0, begin, count)
                if self.advantages is not None
                else None
            ),
        )

    def iter_microbatches(self, microbatch_size: int) -> Iterator["RolloutBatch"]:
        for start in range(0, len(self), microbatch_size):
            stop = min(start + microbatch_size, len(self))
            yield self.narrow(start, stop)

    def iter_minibatches(
        self,
        batch_size: int,
        shuffle: bool = False,
        generator: Optional[torch.Generator] = None,
        drop_last: bool = False,
    ) -> Iterator["RolloutBatch"]:
        """Minibatches with a padded/masked final chunk (never dropped).

        Overshoot rows from a collection iteration are all present in the
        batch; the final chunk repeats row 0 as padding and marks it invalid
        instead of dropping real experience.
        """
        total = len(self)
        if total == 0:
            raise ValueError("cannot iterate an empty batch")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        order = torch.arange(total, dtype=torch.long)
        if shuffle:
            order = torch.randperm(total, generator=generator)
        limit = (total // batch_size) * batch_size if drop_last else total
        for start in range(0, limit, batch_size):
            index = order[start : start + batch_size]
            pad = batch_size - index.numel()
            if pad > 0:
                padded = torch.cat([index, index[:1].repeat(pad)])
                valid = torch.cat(
                    [
                        torch.ones(index.numel(), dtype=torch.bool),
                        torch.zeros(pad, dtype=torch.bool),
                    ]
                )
                # Spread the real rows over the minibatch so that a microbatch
                # split can never isolate all real rows into one chunk (and so
                # never hand the learner a padding-only microbatch).
                permutation = torch.randperm(batch_size, generator=generator)
                padded = padded[permutation]
                valid = valid[permutation]
                yield self.select(padded, row_valid=valid)
            else:
                yield self.select(index)

    @property
    def sample_weight(self) -> float:
        """Fraction of real rows in a padded minibatch (DDP sample weighting)."""
        return float(self.row_valid.float().mean())
