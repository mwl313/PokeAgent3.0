"""Typed contracts shared by the model, the buffer and any engine adapter."""

from agent.types.actions import (
    ACTION_FIELDS,
    ActionKind,
    AtomicAction,
    BRANCH_CAPACITY,
    CandidateSet,
    RequestKind,
    Resource,
    branch_slots_for_request,
    is_singleton_request,
)
from agent.types.observation import (
    ObservationBatch,
    ObservationLayout,
)
from agent.types.requests import (
    ActionRef,
    BranchCandidatesBatch,
    RequestRow,
)

__all__ = [
    "ACTION_FIELDS",
    "ActionKind",
    "ActionRef",
    "AtomicAction",
    "BRANCH_CAPACITY",
    "BranchCandidatesBatch",
    "CandidateSet",
    "ObservationBatch",
    "ObservationLayout",
    "RequestKind",
    "RequestRow",
    "Resource",
    "branch_slots_for_request",
    "is_singleton_request",
]
