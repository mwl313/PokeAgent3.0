"""Shared builders for synthetic (mock-only) learner tests."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch

from agent.buffer.rollout_buffer import RolloutBuffer
from agent.types.actions import (
    ActionKind,
    AtomicAction,
    CandidateSet,
    RequestKind,
    Resource,
    branch_slots_for_request,
)
from agent.types.observation import ObservationBatch
from agent.types.requests import BranchCandidatesBatch, RequestRow

NO_SLOT = 255


def move_action(slot: int, move_slot: int, target: int = 0, mega: bool = False) -> tuple:
    return (
        int(ActionKind.MOVE),
        slot,
        move_slot,
        target,
        NO_SLOT,
        int(Resource.MEGA if mega else Resource.NONE),
    )


def switch_action(slot: int, destination: int) -> tuple:
    return (
        int(ActionKind.SWITCH),
        slot,
        NO_SLOT,
        0,
        destination,
        int(Resource.NONE),
    )


def pass_action(slot: int) -> tuple:
    return (int(ActionKind.PASS), slot, NO_SLOT, 0, NO_SLOT, int(Resource.NONE))


def pick_action(slot: int, destination: int) -> tuple:
    return (
        int(ActionKind.PICK),
        slot,
        NO_SLOT,
        0,
        destination,
        int(Resource.NONE),
    )


def normal_branches(
    first_count: int = 3,
    second_count: int = 2,
    first_mask: Optional[Sequence[bool]] = None,
    second_mask: Optional[Sequence[bool]] = None,
) -> tuple[CandidateSet, ...]:
    first = CandidateSet.from_tuples(
        [move_action(0, index, index % 3) for index in range(first_count)],
        first_mask,
    )
    second = CandidateSet.from_tuples(
        [move_action(1, index, index, mega=index == second_count - 1) for index in range(second_count)],
        second_mask,
    )
    return (first, second)


def singleton_branches() -> tuple[CandidateSet, ...]:
    return (
        CandidateSet.from_tuples([pass_action(0)], [True]),
        CandidateSet.from_tuples([pass_action(1)], [True]),
    )


def preview_branches() -> tuple[CandidateSet, ...]:
    return (
        CandidateSet.from_tuples([pick_action(0, index) for index in range(6)]),
        CandidateSet.from_tuples([pick_action(1, index) for index in range(1, 6)]),
        CandidateSet.from_tuples([pick_action(2, index) for index in range(4)]),
        CandidateSet.from_tuples([pick_action(3, index) for index in range(3)]),
    )


def make_request(
    observation: ObservationBatch,
    branches: Optional[Sequence[CandidateSet]] = None,
    kind: RequestKind = RequestKind.NORMAL,
    branch_slots: Optional[Sequence[int]] = None,
    seed: int = 0,
) -> RequestRow:
    if branches is None:
        branches = normal_branches()
    if branch_slots is not None:
        slots = tuple(branch_slots)
    else:
        slots = branch_slots_for_request(kind)
        if len(slots) != len(branches):
            # Test helper: a caller-supplied branch tree wins over the default
            # slot layout of the request kind.
            slots = tuple(range(len(branches)))
    return RequestRow(
        observation=observation,
        kind=kind,
        branch_slots=tuple(slots),
        branches=tuple(branches),
    )


def make_candidate_batch(
    models: Sequence[RequestRow], device: str | torch.device = "cpu"
) -> BranchCandidatesBatch:
    return BranchCandidatesBatch.from_requests(list(models), device=device)


def synthetic_rows(
    count: int = 6,
    *,
    match_ids: Optional[Sequence[int]] = None,
    sides: Optional[Sequence[int]] = None,
    policy_ids: Optional[Sequence[str]] = None,
    opponent_policy_id: str = "current",
    singleton: Optional[Sequence[bool]] = None,
    rewards: Optional[Sequence[float]] = None,
    dones: Optional[Sequence[bool]] = None,
    values: Optional[Sequence[float]] = None,
    seed: int = 0,
) -> RolloutBuffer:
    """Deterministic fake rollout rows (no model, no engine)."""
    buffer = RolloutBuffer()
    for index in range(count):
        observation = ObservationBatch.dummy(1, seed=seed + index)
        branches = (
            singleton_branches() if singleton and singleton[index] else normal_branches()
        )
        request = make_request(observation, branches)
        buffer.record(
            observation=observation,
            request=request,
            selected=(0, 0),
            old_logprob=-1.0 - 0.01 * index,
            value=float(values[index]) if values else 0.1 * index,
            match_id=int(match_ids[index]) if match_ids else index // 2,
            side=int(sides[index]) if sides else index % 2,
            policy_id=(policy_ids[index] if policy_ids else "current"),
            opponent_policy_id=opponent_policy_id,
            team_ids=(index % 4, (index + 1) % 4),
            request_index=index % 3,
            turn=index + 1,
            reward=float(rewards[index]) if rewards else 0.0,
            done=bool(dones[index]) if dones else False,
        )
    return buffer
