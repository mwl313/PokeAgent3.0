"""Batched branch-candidate container used by the scorer and the buffer.

A request is a *tree of at most four conditional branches*.  Branch ``j`` is
scored with the GRU hidden state produced by the branches selected before it,
so a batch must carry, per branch, exactly the legal candidate set that the
engine mask exposes for that prefix.

Candidate tables are padded to ``candidate_padding`` (64) rows per branch and
to ``BRANCH_CAPACITY`` (4) branches per request.  Padding is always masked; the
learner never truncates a legal candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

import torch

from agent.types.actions import (
    ACTION_FIELDS,
    BRANCH_CAPACITY,
    ActionKind,
    AtomicAction,
    CandidateSet,
    RequestKind,
    branch_slots_for_request,
    is_singleton_request,
)
from agent.types.observation import ObservationBatch, ObservationLayout

NO_TOKEN = -1
"""Candidate field marking "this action has no referenced move token"."""


def resolve_action_tokens(
    layout: ObservationLayout, action: AtomicAction
) -> tuple[int, int]:
    """Token indices an action refers to inside its own observation.

    * ``PICK``   -> the picked party member (preview order)
    * ``MOVE``   -> the acting Pokemon and its current move slot
    * ``SWITCH`` -> the acting Pokemon (destination is a structured field)
    * ``PASS``   -> the acting Pokemon

    ``NO_TOKEN`` is returned for the move reference of non-move actions; the
    scorer substitutes a learned null embedding instead of a token gather.
    """
    kind = int(action.kind)
    if kind == ActionKind.PICK:
        return layout.self_pokemon_token(int(action.switch_destination)), NO_TOKEN
    if kind == ActionKind.MOVE:
        own_slot = int(action.own_slot)
        move_slot = int(action.move_slot)
        entity = layout.self_pokemon_token(own_slot) if 0 <= own_slot < 6 else NO_TOKEN
        # Struggle (and engine-internal move encodings) carry NO_SLOT for the
        # move slot; the scorer then substitutes its learned null move embedding
        # instead of gathering a token that does not exist.
        if 0 <= move_slot < 4 and 0 <= own_slot < 6:
            return entity, layout.move_token(own_slot, move_slot)
        return entity, NO_TOKEN
    if kind in (ActionKind.SWITCH, ActionKind.PASS):
        return layout.self_pokemon_token(int(action.own_slot)), NO_TOKEN
    raise ValueError(f"unsupported action kind {action.kind!r}")


@dataclass(frozen=True)
class ActionRef:
    """A candidate action plus its resolved observation token references."""

    action: AtomicAction
    entity_token: int
    move_token: int

    @classmethod
    def resolve(
        cls, layout: ObservationLayout, action: AtomicAction
    ) -> "ActionRef":
        entity, move = resolve_action_tokens(layout, action)
        return cls(action=action, entity_token=entity, move_token=move)

    @property
    def tuple(self) -> tuple[int, int, int, int, int, int]:
        return self.action.as_tuple()


@dataclass
class RequestRow:
    """One player request: an observation plus its branch candidate tree."""

    observation: ObservationBatch
    kind: RequestKind
    branch_slots: tuple[int, ...]
    branches: tuple[CandidateSet, ...]
    refs: tuple[tuple[ActionRef, ...], ...] = ()
    layout: ObservationLayout = field(default_factory=ObservationLayout)

    def __post_init__(self) -> None:
        if len(self.observation) != 1:
            raise ValueError("RequestRow holds exactly one observation")
        self.kind = RequestKind(self.kind)
        if not self.branch_slots:
            self.branch_slots = branch_slots_for_request(self.kind)
        if len(self.branches) != len(self.branch_slots):
            raise ValueError(
                f"{len(self.branches)} branch candidate sets for "
                f"{len(self.branch_slots)} branch slots"
            )
        if not self.refs:
            self.refs = tuple(
                tuple(ActionRef.resolve(self.layout, action) for action in branch.actions)
                for branch in self.branches
            )
        for branch, refs in zip(self.branches, self.refs):
            if len(branch.actions) != len(refs):
                raise ValueError("branch candidates and references must align")

    # -- convenience -----------------------------------------------------
    @property
    def branch_count(self) -> int:
        return len(self.branch_slots)

    @property
    def legal_counts(self) -> tuple[int, ...]:
        return tuple(branch.legal_count for branch in self.branches)

    @property
    def actor_active(self) -> bool:
        """False when every branch is a singleton/pass: no actor signal."""
        return not is_singleton_request(self.legal_counts)


@dataclass
class BranchCandidatesBatch:
    """Padded ``[B, K, P]`` candidate tables for a batch of requests."""

    action_ids: torch.Tensor  # long [B, K, P, 6]
    mask: torch.Tensor  # bool [B, K, P]  (legal candidate)
    entity_token: torch.Tensor  # long [B, K, P]
    move_token: torch.Tensor  # long [B, K, P]
    branch_valid: torch.Tensor  # bool [B, K]
    selected: torch.Tensor  # long [B, K]  (-1 when not recorded)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.action_ids.shape[:3])  # type: ignore[return-value]

    @property
    def branch_k(self) -> torch.Tensor:
        """Legal candidate count per branch (``[B, K]``)."""
        return self.mask.sum(dim=-1).to(torch.long)

    @property
    def actor_active(self) -> torch.Tensor:
        """Per-request flag: at least one branch has a real choice."""
        k = self.branch_k
        valid = self.branch_valid & (k >= 2)
        return valid.any(dim=-1)

    def to(self, device: torch.device | str) -> "BranchCandidatesBatch":
        return BranchCandidatesBatch(
            action_ids=self.action_ids.to(device),
            mask=self.mask.to(device),
            entity_token=self.entity_token.to(device),
            move_token=self.move_token.to(device),
            branch_valid=self.branch_valid.to(device),
            selected=self.selected.to(device),
        )

    def select(self, index: torch.Tensor | Sequence[int]) -> "BranchCandidatesBatch":
        idx = index if isinstance(index, torch.Tensor) else torch.as_tensor(list(index), dtype=torch.long)
        idx = idx.to(self.action_ids.device)
        return BranchCandidatesBatch(
            action_ids=self.action_ids.index_select(0, idx),
            mask=self.mask.index_select(0, idx),
            entity_token=self.entity_token.index_select(0, idx),
            move_token=self.move_token.index_select(0, idx),
            branch_valid=self.branch_valid.index_select(0, idx),
            selected=self.selected.index_select(0, idx),
        )

    def cat(self, others: Sequence["BranchCandidatesBatch"]) -> "BranchCandidatesBatch":
        parts = [self, *others]
        return BranchCandidatesBatch(
            action_ids=torch.cat([p.action_ids for p in parts], dim=0),
            mask=torch.cat([p.mask for p in parts], dim=0),
            entity_token=torch.cat([p.entity_token for p in parts], dim=0),
            move_token=torch.cat([p.move_token for p in parts], dim=0),
            branch_valid=torch.cat([p.branch_valid for p in parts], dim=0),
            selected=torch.cat([p.selected for p in parts], dim=0),
        )

    # -- builders ---------------------------------------------------------
    @classmethod
    def from_rows(
        cls,
        rows: Sequence,
        candidate_padding: int = 64,
        branch_capacity: int = BRANCH_CAPACITY,
        device: torch.device | str | None = None,
    ) -> "BranchCandidatesBatch":
        """Build the candidate table straight from stored rollout rows.

        `RolloutRow` already holds the packed action tuples, masks and the
        entity/move token references, so the learner does not need to
        reconstruct `ActionRef`/`CandidateSet` objects per candidate. The output
        is identical to `from_requests(row.to_request_row(...))`.
        """
        import numpy as np

        batch = len(rows)
        if batch == 0:
            raise ValueError("cannot build an empty candidate batch")
        action_ids = np.zeros((batch, branch_capacity, candidate_padding, ACTION_FIELDS), dtype=np.int64)
        mask = np.zeros((batch, branch_capacity, candidate_padding), dtype=bool)
        entity = np.zeros((batch, branch_capacity, candidate_padding), dtype=np.int64)
        move = np.full((batch, branch_capacity, candidate_padding), NO_TOKEN, dtype=np.int64)
        branch_valid = np.zeros((batch, branch_capacity), dtype=bool)
        selected = np.full((batch, branch_capacity), -1, dtype=np.int64)
        for index, row in enumerate(rows):
            count = row.branch_count
            if count > branch_capacity:
                raise ValueError(
                    f"request has {count} branches, capacity is {branch_capacity}"
                )
            for level in range(count):
                actions = row.action_ids[level]
                if len(actions) > candidate_padding:
                    raise ValueError(
                        f"branch {level} has {len(actions)} candidates, padding is "
                        f"{candidate_padding}; candidate capacity must be resolved "
                        "from the full scope instead of truncating"
                    )
                branch_valid[index, level] = True
                if actions:
                    action_ids[index, level, : len(actions)] = np.asarray(actions, dtype=np.int64)
                    mask[index, level, : len(actions)] = np.asarray(
                        row.candidate_mask[level], dtype=bool
                    )
                    entity[index, level, : len(actions)] = np.asarray(
                        row.entity_token[level], dtype=np.int64
                    )
                    move[index, level, : len(actions)] = np.asarray(
                        row.move_token[level], dtype=np.int64
                    )
            for level, pick in enumerate(row.selected):
                selected[index, level] = int(pick)
        out = cls(
            action_ids=torch.as_tensor(action_ids),
            mask=torch.as_tensor(mask),
            entity_token=torch.as_tensor(entity),
            move_token=torch.as_tensor(move),
            branch_valid=torch.as_tensor(branch_valid),
            selected=torch.as_tensor(selected),
        )
        if device is not None:
            out = out.to(device)
        return out

    @classmethod
    def from_requests(
        cls,
        rows: Sequence[RequestRow],
        candidate_padding: int = 64,
        branch_capacity: int = BRANCH_CAPACITY,
        device: torch.device | str | None = None,
    ) -> "BranchCandidatesBatch":
        if not rows:
            raise ValueError("cannot build an empty candidate batch")
        batch = len(rows)
        action_ids = torch.zeros(
            (batch, branch_capacity, candidate_padding, ACTION_FIELDS), dtype=torch.long
        )
        mask = torch.zeros((batch, branch_capacity, candidate_padding), dtype=torch.bool)
        entity = torch.zeros((batch, branch_capacity, candidate_padding), dtype=torch.long)
        move = torch.full(
            (batch, branch_capacity, candidate_padding), NO_TOKEN, dtype=torch.long
        )
        branch_valid = torch.zeros((batch, branch_capacity), dtype=torch.bool)
        selected = torch.full((batch, branch_capacity), -1, dtype=torch.long)

        for row_index, row in enumerate(rows):
            if row.branch_count > branch_capacity:
                raise ValueError(
                    f"request has {row.branch_count} branches, capacity is {branch_capacity}"
                )
            for branch_index, (branch, refs) in enumerate(zip(row.branches, row.refs)):
                if branch.size > candidate_padding:
                    raise ValueError(
                        f"branch {branch_index} has {branch.size} candidates, "
                        f"padding is {candidate_padding}; candidate capacity must be "
                        "resolved from the full scope instead of truncating"
                    )
                branch_valid[row_index, branch_index] = True
                for cand_index, ref in enumerate(refs):
                    action_ids[row_index, branch_index, cand_index] = torch.tensor(
                        ref.tuple, dtype=torch.long
                    )
                    mask[row_index, branch_index, cand_index] = bool(branch.mask[cand_index])
                    entity[row_index, branch_index, cand_index] = int(ref.entity_token)
                    move[row_index, branch_index, cand_index] = int(ref.move_token)
        batch_out = cls(
            action_ids=action_ids,
            mask=mask,
            entity_token=entity,
            move_token=move,
            branch_valid=branch_valid,
            selected=selected,
        )
        if device is not None:
            batch_out = batch_out.to(device)
        return batch_out

    def with_selected(self, selected: torch.Tensor) -> "BranchCandidatesBatch":
        if selected.shape != self.selected.shape:
            raise ValueError(
                f"selected shape {tuple(selected.shape)} != {tuple(self.selected.shape)}"
            )
        return replace(self, selected=selected.to(self.action_ids.device))
