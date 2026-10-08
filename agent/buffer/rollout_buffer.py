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
            if not isinstance(self.observation_store, InlineObservationStore):
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
        rows = list(self.rows if rows is None else rows)
        if not rows:
            raise ValueError("cannot build a batch from an empty rollout")
        if len(rows) == len(self.rows) and all(
            row.observation_ref == index for index, row in enumerate(rows)
        ):
            observations = self.stacked_observations()
        else:
            observations = self.observation(rows[0].observation_ref)
            if len(rows) > 1:
                observations = observations.cat(
                    [self.observation(row.observation_ref) for row in rows[1:]]
                )
        requests = [
            row.to_request_row(observations.select([index]))
            for index, row in enumerate(rows)
        ]
        candidates = BranchCandidatesBatch.from_requests(
            requests,
            candidate_padding=64,
            branch_capacity=BRANCH_CAPACITY,
            device=device,
        )
        selected = torch.tensor(
            [
                list(row.selected) + [-1] * (BRANCH_CAPACITY - row.branch_count)
                for row in rows
            ],
            dtype=torch.long,
        )
        candidates = candidates.with_selected(selected)

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
            observation=observations if device is None else observations.to(device),
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
        if device is not None:
            batch = batch.to(device)
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
            policy_ids=[self.policy_ids[int(i)] for i in idx.tolist()],
            advantages=(
                self.advantages.index_select(0, idx)
                if self.advantages is not None
                else None
            ),
        )

    def with_advantages(self, advantages: torch.Tensor) -> "RolloutBatch":
        return replace(self, advantages=advantages)

    def iter_microbatches(self, microbatch_size: int) -> Iterator["RolloutBatch"]:
        for start in range(0, len(self), microbatch_size):
            stop = min(start + microbatch_size, len(self))
            index = torch.arange(start, stop, dtype=torch.long, device=self.old_logprob.device)
            yield self.select(index)

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
            else:
                padded, valid = index, torch.ones(index.numel(), dtype=torch.bool)
            yield self.select(padded, row_valid=valid)

    @property
    def sample_weight(self) -> float:
        """Fraction of real rows in a padded minibatch (DDP sample weighting)."""
        return float(self.row_valid.float().mean())
