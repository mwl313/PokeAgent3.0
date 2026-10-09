"""Typed, engine-independent observation contract.

The model consumes :class:`ObservationBatch` only.  The future native engine
packs one blob per player view (`engine/python/pa3_engine/observation.py`); a
thin adapter (:meth:`ObservationBatch.from_native_payload`) converts that
payload into the typed contract, and :mod:`agent.mock_engine` produces the same
payload shape from fake data.  The model therefore never depends on the final
engine implementation, and a real engine rollout is a pure data hand-off.

Layout (``observation.schema = pa3-entity-observation-v1``):

===========  ====  =========================================
role         n     contents
===========  ====  =========================================
global       1     request kind, turn, side, regulation
field        1     public weather/terrain/room state
side         2     both sides' public side conditions
pokemon      12    own six then opponent six, player view
moves        48    four current move slots per Pokemon
events       24    recently observed semantic events
===========  ====  =========================================
active       88    padded to 96 fixed rows

Each token carries 32 integer category features (with a known mask), 50
normalized continuous features (known mask), and 40 boolean flags (known mask).
Unobserved or unknown entries are masked, never imputed with an invented value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

try:  # numpy is only needed for the native-payload adapter
    import numpy as np
except Exception:  # pragma: no cover - numpy is a hard dependency of the packer
    np = None  # type: ignore[assignment]


def _as_numpy(value: Any):
    if np is None:  # pragma: no cover - defensive
        raise RuntimeError("numpy is required to adapt native observation payloads")
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


class ObservationLayout:
    """Fixed token layout of ``pa3-entity-observation-v1``.

    The layout is data, not hard-coded index math inside the model: an engine
    adapter may hand over an alternative mapping as long as it produces the
    same number of padded rows.  Every index accessor validates that the index
    stays inside the padded token block.
    """

    TOKENS = 96
    ACTIVE_TOKENS = 88
    CATEGORY_SLOTS = 32
    FLOAT_SLOTS = 50
    FLAG_SLOTS = 40

    GLOBAL = 0
    FIELD = 1
    SIDE_START = 2
    POKEMON_START = 4
    MOVES_START = 16
    EVENTS_START = 64

    POKEMON_COUNT = 12
    MOVE_PER_POKEMON = 4

    ROLE_PAD = 0
    ROLE_GLOBAL = 1
    ROLE_FIELD = 2
    ROLE_SIDE_SELF = 3
    ROLE_SIDE_OPP = 4
    ROLE_POKEMON_SELF = 5
    ROLE_POKEMON_OPP = 6
    ROLE_MOVE_SELF = 7
    ROLE_MOVE_OPP = 8
    ROLE_EVENT = 9
    ROLE_VOCAB = 12

    SIDE_NEUTRAL = 0
    SIDE_SELF = 1
    SIDE_OPPONENT = 2
    SIDE_VOCAB = 3

    def __post_init__(self) -> None:
        if self.MOVES_START + self.POKEMON_COUNT * self.MOVE_PER_POKEMON > self.ACTIVE_TOKENS:
            raise ValueError("move tokens overflow the active token block")

    # -- index helpers -------------------------------------------------
    def self_pokemon_token(self, slot: int) -> int:
        self._check_slot(slot)
        return self.POKEMON_START + slot

    def opponent_pokemon_token(self, slot: int) -> int:
        self._check_slot(slot)
        return self.POKEMON_START + 6 + slot

    def move_token(self, pokemon_index: int, move_slot: int) -> int:
        if not 0 <= pokemon_index < self.POKEMON_COUNT:
            raise ValueError(f"pokemon index out of range: {pokemon_index}")
        if not 0 <= move_slot < self.MOVE_PER_POKEMON:
            raise ValueError(f"move slot out of range: {move_slot}")
        return self.MOVES_START + pokemon_index * self.MOVE_PER_POKEMON + move_slot

    @staticmethod
    def _check_slot(slot: int) -> None:
        if not 0 <= int(slot) < 6:
            raise ValueError(f"party slot out of range: {slot}")

    # -- default role / side ids ---------------------------------------
    def default_roles(self) -> torch.Tensor:
        roles = torch.zeros(self.TOKENS, dtype=torch.long)
        roles[self.GLOBAL] = self.ROLE_GLOBAL
        roles[self.FIELD] = self.ROLE_FIELD
        roles[self.SIDE_START] = self.ROLE_SIDE_SELF
        roles[self.SIDE_START + 1] = self.ROLE_SIDE_OPP
        for slot in range(6):
            roles[self.POKEMON_START + slot] = self.ROLE_POKEMON_SELF
            roles[self.POKEMON_START + 6 + slot] = self.ROLE_POKEMON_OPP
        for index in range(self.POKEMON_COUNT):
            role = self.ROLE_MOVE_SELF if index < 6 else self.ROLE_MOVE_OPP
            for move_slot in range(self.MOVE_PER_POKEMON):
                roles[self.move_token(index, move_slot)] = role
        roles[self.EVENTS_START : self.ACTIVE_TOKENS] = self.ROLE_EVENT
        return roles

    def default_sides(self) -> torch.Tensor:
        sides = torch.zeros(self.TOKENS, dtype=torch.long)
        sides[self.SIDE_START] = self.SIDE_SELF
        sides[self.SIDE_START + 1] = self.SIDE_OPPONENT
        sides[self.POKEMON_START : self.POKEMON_START + 6] = self.SIDE_SELF
        sides[self.POKEMON_START + 6 : self.POKEMON_START + 12] = self.SIDE_OPPONENT
        sides[self.MOVES_START : self.MOVES_START + 24] = self.SIDE_SELF
        sides[self.MOVES_START + 24 : self.EVENTS_START] = self.SIDE_OPPONENT
        return sides

    def default_token_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.TOKENS, dtype=torch.bool)
        mask[: self.ACTIVE_TOKENS] = True
        return mask


@dataclass
class ObservationBatch:
    """A batch of padded observation token blocks.

    All tensors share the leading batch dimension ``B`` and the padded token
    dimension ``T`` (96).  Feature tensors are never produced for tokens whose
    ``token_mask`` entry is false; those rows stay zero after embedding.
    """

    token_mask: torch.Tensor  # bool   [B, T]
    categories: torch.Tensor  # long   [B, T, C]
    category_known: torch.Tensor  # bool [B, T, C]
    floats: torch.Tensor  # float32 [B, T, F]
    float_known: torch.Tensor  # bool  [B, T, F]
    flags: torch.Tensor  # bool   [B, T, G]
    flag_known: torch.Tensor  # bool  [B, T, G]
    role_ids: torch.Tensor  # long   [B, T]
    side_ids: torch.Tensor  # long   [B, T]
    layout: ObservationLayout = field(default_factory=ObservationLayout)
    schema_version: int = 1

    # -- construction ---------------------------------------------------
    @classmethod
    def from_native_payload(
        cls,
        payload: Mapping[str, Any],
        layout: ObservationLayout | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "ObservationBatch":
        """Adapt one engine payload (see ``parse_view``) into the typed contract.

        ``payload`` may describe a single view or a batch with a leading row
        dimension.  Unknown payload keys are ignored; missing role/side vectors
        fall back to the fixed layout.
        """
        layout = layout or ObservationLayout()
        tokens = int(_as_numpy(payload["token_mask"]).shape[-1])
        if tokens != layout.TOKENS:
            raise ValueError(
                f"native payload has {tokens} tokens, contract expects {layout.TOKENS}"
            )

        def _tensor(name: str, dims: int, kind: str) -> torch.Tensor:
            value = _as_numpy(payload[name])
            if value.ndim == dims:
                value = value[None, ...]
            if value.ndim != dims + 1:
                raise ValueError(f"payload[{name!r}] has unexpected rank {value.ndim}")
            # A field sliced out of the fixed-stride structured batch keeps the
            # struct stride, which torch cannot wrap. numpy still calls a
            # single-request batch "C contiguous" (the stride of a size-1
            # dimension is irrelevant to it) while torch rejects it, so test the
            # stride condition torch actually enforces and copy once per field.
            itemsize = value.itemsize
            if itemsize and any(stride % itemsize for stride in value.strides):
                value = np.array(value, dtype=value.dtype, order="C", copy=True)
            tensor = torch.as_tensor(value)
            if kind == "bool":
                tensor = tensor.to(torch.bool)
            elif kind == "long":
                tensor = tensor.to(torch.long)
            else:
                tensor = tensor.to(dtype)
            return tensor.to(device) if device is not None else tensor

        token_mask = _tensor("token_mask", 1, "bool")
        categories = _tensor("categories", 2, "long")
        category_known = _tensor("category_known", 2, "bool")
        floats = _tensor("floats", 2, "float")
        float_known = _tensor("float_known", 2, "bool")
        flags = _tensor("flags", 2, "bool")
        flag_known = _tensor("flag_known", 2, "bool")

        batch = token_mask.shape[0]
        if "role_ids" in payload:
            role_ids = _tensor("role_ids", 1, "long")
        else:
            role_ids = layout.default_roles().expand(batch, -1).contiguous().clone()
            role_ids[~token_mask] = layout.ROLE_PAD
        if "side_ids" in payload:
            side_ids = _tensor("side_ids", 1, "long")
        else:
            side_ids = layout.default_sides().expand(batch, -1).contiguous().clone()

        return cls(
            token_mask=token_mask,
            categories=categories,
            category_known=category_known,
            floats=floats,
            float_known=float_known,
            flags=flags,
            flag_known=flag_known,
            role_ids=role_ids,
            side_ids=side_ids,
            layout=layout,
            schema_version=int(_as_numpy(payload.get("schema_version", 1)).reshape(-1)[0]),
        )

    @classmethod
    def dummy(
        cls,
        batch_size: int = 1,
        layout: ObservationLayout | None = None,
        seed: int = 0,
        device: torch.device | str | None = None,
    ) -> "ObservationBatch":
        """Deterministic synthetic batch for tests and shape checks."""
        layout = layout or ObservationLayout()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        shape_t = (batch_size, layout.TOKENS)
        token_mask = layout.default_token_mask().expand(batch_size, -1).clone()
        categories = torch.randint(
            0, 64, (batch_size, layout.TOKENS, layout.CATEGORY_SLOTS), generator=generator
        )
        floats = torch.randn(
            (batch_size, layout.TOKENS, layout.FLOAT_SLOTS), generator=generator
        )
        flags = torch.rand((batch_size, layout.TOKENS, layout.FLAG_SLOTS), generator=generator) < 0.5
        batch = cls(
            token_mask=token_mask,
            categories=categories,
            category_known=torch.ones(
                (batch_size, layout.TOKENS, layout.CATEGORY_SLOTS), dtype=torch.bool
            ),
            floats=floats,
            float_known=torch.ones((batch_size, layout.TOKENS, layout.FLOAT_SLOTS), dtype=torch.bool),
            flags=flags,
            flag_known=torch.ones((batch_size, layout.TOKENS, layout.FLAG_SLOTS), dtype=torch.bool),
            role_ids=layout.default_roles().expand(batch_size, -1).clone(),
            side_ids=layout.default_sides().expand(batch_size, -1).clone(),
            layout=layout,
        )
        if device is not None:
            batch = batch.to(device)
        return batch

    # -- basic protocol -------------------------------------------------
    def __len__(self) -> int:
        return int(self.token_mask.shape[0])

    @property
    def device(self) -> torch.device:
        return self.token_mask.device

    def to(self, device: torch.device | str) -> "ObservationBatch":
        return ObservationBatch(
            token_mask=self.token_mask.to(device),
            categories=self.categories.to(device),
            category_known=self.category_known.to(device),
            floats=self.floats.to(device),
            float_known=self.float_known.to(device),
            flags=self.flags.to(device),
            flag_known=self.flag_known.to(device),
            role_ids=self.role_ids.to(device),
            side_ids=self.side_ids.to(device),
            layout=self.layout,
            schema_version=self.schema_version,
        )

    def select(self, index: torch.Tensor | Sequence[int]) -> "ObservationBatch":
        """Gather rows; ``index`` is a 1-D long tensor or a sequence."""
        idx = index if isinstance(index, torch.Tensor) else torch.as_tensor(list(index), dtype=torch.long)
        idx = idx.to(self.device)
        return ObservationBatch(
            token_mask=self.token_mask.index_select(0, idx),
            categories=self.categories.index_select(0, idx),
            category_known=self.category_known.index_select(0, idx),
            floats=self.floats.index_select(0, idx),
            float_known=self.float_known.index_select(0, idx),
            flags=self.flags.index_select(0, idx),
            flag_known=self.flag_known.index_select(0, idx),
            role_ids=self.role_ids.index_select(0, idx),
            side_ids=self.side_ids.index_select(0, idx),
            layout=self.layout,
            schema_version=self.schema_version,
        )

    def narrow(self, begin: int, end: int) -> "ObservationBatch":
        """Zero-copy contiguous slice of the leading (row) dimension."""
        count = end - begin
        return ObservationBatch(
            token_mask=self.token_mask.narrow(0, begin, count),
            categories=self.categories.narrow(0, begin, count),
            category_known=self.category_known.narrow(0, begin, count),
            floats=self.floats.narrow(0, begin, count),
            float_known=self.float_known.narrow(0, begin, count),
            flags=self.flags.narrow(0, begin, count),
            flag_known=self.flag_known.narrow(0, begin, count),
            role_ids=self.role_ids.narrow(0, begin, count),
            side_ids=self.side_ids.narrow(0, begin, count),
            layout=self.layout,
            schema_version=self.schema_version,
        )

    def cat(self, others: Sequence["ObservationBatch"]) -> "ObservationBatch":
        parts = [self, *others]
        return ObservationBatch(
            token_mask=torch.cat([p.token_mask for p in parts], dim=0),
            categories=torch.cat([p.categories for p in parts], dim=0),
            category_known=torch.cat([p.category_known for p in parts], dim=0),
            floats=torch.cat([p.floats for p in parts], dim=0),
            float_known=torch.cat([p.float_known for p in parts], dim=0),
            flags=torch.cat([p.flags for p in parts], dim=0),
            flag_known=torch.cat([p.flag_known for p in parts], dim=0),
            role_ids=torch.cat([p.role_ids for p in parts], dim=0),
            side_ids=torch.cat([p.side_ids for p in parts], dim=0),
            layout=self.layout,
            schema_version=self.schema_version,
        )

    def clone(self) -> "ObservationBatch":
        return self.select(torch.arange(len(self), dtype=torch.long, device=self.device))

    # -- packed representation for the rollout buffer --------------------
    def to_compact_numpy(self):
        """Compact (low precision) per-row arrays for the rollout buffer.

        Only the typed observation is stored; encoder activations are never
        part of this representation.
        """
        if np is None:  # pragma: no cover - defensive
            raise RuntimeError("numpy is required for the compact representation")
        out = {}
        cpu = {
            "token_mask": self.token_mask,
            "categories": self.categories,
            "category_known": self.category_known,
            "floats": self.floats,
            "float_known": self.float_known,
            "flags": self.flags,
            "flag_known": self.flag_known,
            "role_ids": self.role_ids,
            "side_ids": self.side_ids,
        }
        for name, tensor in cpu.items():
            value = tensor.detach().cpu().numpy()
            if value.dtype == np.int64:
                value = value.astype(np.uint16)
            elif value.dtype == np.float32:
                value = value.astype(np.float16)
            out[name] = value
        return out

    @classmethod
    def from_compact_numpy(
        cls,
        payload: Mapping[str, Any],
        layout: ObservationLayout | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "ObservationBatch":
        """Rebuild a typed batch from the buffer's compact arrays.

        Every array carries a leading batch dimension; the token/feature ranks
        are validated so a corrupted row cannot silently reshape into the model.
        """
        layout = layout or ObservationLayout()
        arrays = {name: _as_numpy(value) for name, value in payload.items()}

        expected_rank = {
            "token_mask": 2,
            "role_ids": 2,
            "side_ids": 2,
            "categories": 3,
            "category_known": 3,
            "floats": 3,
            "float_known": 3,
            "flags": 3,
            "flag_known": 3,
        }

        def _tensor(name: str) -> torch.Tensor:
            value = arrays[name]
            if value.ndim != expected_rank[name]:
                raise ValueError(
                    f"compact observation {name!r} has rank {value.ndim}, "
                    f"expected {expected_rank[name]}"
                )
            return torch.as_tensor(value.copy())

        return cls(
            token_mask=_tensor("token_mask").to(torch.bool),
            categories=_tensor("categories").to(torch.long),
            category_known=_tensor("category_known").to(torch.bool),
            floats=_tensor("floats").to(dtype),
            float_known=_tensor("float_known").to(torch.bool),
            flags=_tensor("flags").to(torch.bool),
            flag_known=_tensor("flag_known").to(torch.bool),
            role_ids=_tensor("role_ids").to(torch.long),
            side_ids=_tensor("side_ids").to(torch.long),
            layout=layout,
        )


def observation_shape(layout: ObservationLayout | None = None) -> dict[str, tuple[int, ...]]:
    """Shape contract of one observation row (used by the shape tests)."""
    layout = layout or ObservationLayout()
    return {
        "token_mask": (layout.TOKENS,),
        "role_ids": (layout.TOKENS,),
        "side_ids": (layout.TOKENS,),
        "categories": (layout.TOKENS, layout.CATEGORY_SLOTS),
        "floats": (layout.TOKENS, layout.FLOAT_SLOTS),
        "flags": (layout.TOKENS, layout.FLAG_SLOTS),
    }
