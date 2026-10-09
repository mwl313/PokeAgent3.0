"""Action and request vocabulary for the PA3 branch-structured policy.

The integer encodings mirror the future native engine contract
(`engine/src/actions.rs`, `engine/src/python.rs::parse_action`) so that a
rollout recorded from the engine can be replayed by the learner without a
translation layer that could drift:

* request kinds: ``0=preview, 1=normal, 2=replacement, 3=wait, 4=finished``
* action kinds:  ``0=pick, 1=move, 2=switch, 3=pass``
* resource:      ``0=none, 1=mega``

An atomic action is the packed tuple
``(kind, own_slot, move_slot, target_location, switch_destination, resource)``.
The model never invents free text options; every candidate is such a tuple.

Note on ``wait``/``finished``: the engine emits them as protocol states, not as
player decisions.  They are represented here so a rollout can record them
honestly, but they must never be forged into PPO actor rows (see
``is_singleton_request`` and :mod:`agent.buffer.rollout_buffer`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

ACTION_FIELDS = 6
"""Length of the packed atomic action tuple."""

BRANCH_CAPACITY = 4
"""Maximum number of within-request branches (preview lead A/B/reserve/…)."""

NO_SLOT = 255
"""``engine/src/actions.rs::NO_SLOT``; used for unpresent tuple fields."""

TARGET_LOCATION_MIN = -1
TARGET_LOCATION_MAX = 4
"""Player-visible target locations: -1 self/field, 0..3 opposing slots."""


class RequestKind(IntEnum):
    """Player request kind.  Values are the native ``request_kind_id``."""

    PREVIEW = 0
    NORMAL = 1
    REPLACEMENT = 2
    WAIT = 3
    FINISHED = 4

    @classmethod
    def from_native(cls, value: int) -> "RequestKind":
        return cls(int(value))

    @property
    def is_decision(self) -> bool:
        """Whether the engine expects the player to submit a choice."""
        return self in (RequestKind.PREVIEW, RequestKind.NORMAL, RequestKind.REPLACEMENT)


class ActionKind(IntEnum):
    PICK = 0
    MOVE = 1
    SWITCH = 2
    PASS = 3


class Resource(IntEnum):
    NONE = 0
    MEGA = 1


def normalize_target_location(value: int) -> int:
    """Clamp an ``i8`` target location into the model's embedding range."""
    value = int(value)
    if value < TARGET_LOCATION_MIN:
        return TARGET_LOCATION_MIN
    if value > TARGET_LOCATION_MAX:
        return TARGET_LOCATION_MAX
    return value


@dataclass(frozen=True)
class AtomicAction:
    """One complete slot-level action in the native packed representation."""

    kind: int
    own_slot: int = NO_SLOT
    move_slot: int = NO_SLOT
    target_location: int = 0
    switch_destination: int = NO_SLOT
    resource: int = int(Resource.NONE)

    def __post_init__(self) -> None:
        if int(self.kind) not in {int(k) for k in ActionKind}:
            raise ValueError(f"unknown action kind {self.kind!r}")
        if int(self.resource) not in {int(r) for r in Resource}:
            raise ValueError(f"unknown action resource {self.resource!r}")
        if not 0 <= int(self.own_slot) <= NO_SLOT:
            raise ValueError(f"own_slot out of range: {self.own_slot!r}")

    @classmethod
    def from_tuple(cls, values: Sequence[int]) -> "AtomicAction":
        if len(values) != ACTION_FIELDS:
            raise ValueError(
                f"atomic action needs {ACTION_FIELDS} fields, got {len(values)}"
            )
        return cls(
            kind=int(values[0]),
            own_slot=int(values[1]),
            move_slot=int(values[2]),
            target_location=normalize_target_location(int(values[3])),
            switch_destination=int(values[4]),
            resource=int(values[5]),
        )

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (
            int(self.kind),
            int(self.own_slot),
            int(self.move_slot),
            normalize_target_location(int(self.target_location)),
            int(self.switch_destination),
            int(self.resource),
        )

    def with_resource(self, resource: int) -> "AtomicAction":
        return AtomicAction(
            kind=self.kind,
            own_slot=self.own_slot,
            move_slot=self.move_slot,
            target_location=self.target_location,
            switch_destination=self.switch_destination,
            resource=int(resource),
        )


@dataclass(frozen=True)
class CandidateSet:
    """The legal completion candidates of a single within-request branch."""

    actions: tuple[AtomicAction, ...]
    mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.actions) != len(self.mask):
            raise ValueError("candidate actions and mask must have equal length")
        if not self.actions:
            raise ValueError("a decision branch needs at least one candidate")

    @property
    def size(self) -> int:
        return len(self.actions)

    @property
    def legal_count(self) -> int:
        return sum(1 for flag in self.mask if flag)

    @property
    def is_singleton(self) -> bool:
        return self.legal_count <= 1

    def tuples(self) -> tuple[tuple[int, int, int, int, int, int], ...]:
        return tuple(action.as_tuple() for action in self.actions)

    @classmethod
    def from_tuples(
        cls,
        values: Iterable[Sequence[int]],
        mask: Iterable[bool] | None = None,
    ) -> "CandidateSet":
        actions = tuple(AtomicAction.from_tuple(v) for v in values)
        flags = tuple(bool(m) for m in mask) if mask is not None else tuple(True for _ in actions)
        return cls(actions=actions, mask=flags)


def branch_slots_for_request(
    kind: RequestKind,
    requires_replacement: Sequence[bool] = (False, False),
) -> tuple[int, ...]:
    """Within-request branch slots, mirroring ``Request::branch_slots_inline``.

    * ``PREVIEW``     -> four picks: lead A, lead B, reserve 1, reserve 2
    * ``NORMAL``      -> slot A, then slot B conditioned on A
    * ``REPLACEMENT`` -> only the slots that actually require a replacement
    * ``WAIT``/``FINISHED`` -> no branch
    """
    if kind is RequestKind.PREVIEW:
        return (0, 1, 2, 3)
    if kind is RequestKind.NORMAL:
        return (0, 1)
    if kind is RequestKind.REPLACEMENT:
        if len(requires_replacement) != 2:
            raise ValueError("replacement request needs two slot flags")
        return tuple(slot for slot in range(2) if requires_replacement[slot])
    return ()


def is_singleton_request(legal_counts: Sequence[int]) -> bool:
    """True when no branch gives the actor a real choice.

    A request whose branches all have at most one legal candidate carries no
    actor learning signal.  It is excluded from the actor loss (and from
    advantage normalization) while remaining a valid value-learning row.
    """
    if not legal_counts:
        return True
    return all(int(count) <= 1 for count in legal_counts)
