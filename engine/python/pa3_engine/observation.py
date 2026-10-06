"""Parser for the native packed observation blob (pa3-observation-v1).

The Rust binding returns one little-endian `bytes` per player view; this module
turns it into numpy array views without per-element Python work. The layout is
versioned by `pa3_engine.SCHEMA_VERSION`; keep this table in sync with
`engine/src/python.rs::pack_observation`.
"""
import numpy as np

TOKENS = 96
CATEGORY_COUNT = 32
FLOAT_COUNT = 50
FLAG_COUNT = 40

# Fixed header: version, token mask, six token blocks, five ragged length rows.
FIXED = np.dtype([
    ("schema_version", "<u2"),
    ("token_mask", "u1", (TOKENS,)),
    ("categories", "<u2", (TOKENS, CATEGORY_COUNT)),
    ("category_known", "u1", (TOKENS, CATEGORY_COUNT)),
    ("floats", "<f4", (TOKENS, FLOAT_COUNT)),
    ("float_known", "u1", (TOKENS, FLOAT_COUNT)),
    ("flags", "u1", (TOKENS, FLAG_COUNT)),
    ("flag_known", "u1", (TOKENS, FLAG_COUNT)),
    ("effect_counts", "<u2", (TOKENS,)),
    ("repertoire_counts", "<u2", (TOKENS,)),
    ("type_counts", "<u2", (TOKENS,)),
    ("base_move_counts", "<u2", (TOKENS,)),
    ("move_effect_counts", "<u2", (TOKENS,)),
], align=False)

EFFECT = np.dtype([
    ("id", "<u2"),
    ("present", "u1"),
    ("duration", "<f4"),
    ("duration_known", "u1"),
    ("stacks", "<f4"),
    ("stacks_known", "u1"),
    ("source", "u1"),
    ("source_known", "u1"),
], align=False)

REPERTOIRE = np.dtype([("id", "<u2")], align=False)
TYPE = np.dtype([("id", "<u2")], align=False)

BASE_MOVE = np.dtype([
    ("id", "<u2"),
    ("pp", "<f4"),
    ("max_pp", "<f4"),
    ("disabled", "u1"),
    ("used", "u1"),
], align=False)

MOVE_EFFECT = np.dtype([
    ("kind", "u1"),
    ("chance", "<f4"),
    ("status", "<u2"),
    ("volatile", "<u2"),
    ("boosts", "<f4", (7,)),
    ("heal", "<f4"),
    ("heal_known", "u1"),
], align=False)


def _read(blob: bytes, offset: int, counts, dtype: np.dtype) -> np.ndarray:
    total = int(counts.sum())
    size = total * dtype.itemsize
    if offset + size > len(blob):
        raise ValueError(f"observation section truncated at offset {offset}")
    return np.frombuffer(blob, dtype=dtype, count=total, offset=offset)


def parse_view(blob: bytes) -> dict:
    """Decode one Rust `pack_observation` blob into named array views."""
    if len(blob) < FIXED.itemsize:
        raise ValueError(f"observation blob truncated: {len(blob)} < {FIXED.itemsize}")
    header = np.frombuffer(blob, dtype=FIXED, count=1)[0]
    offset = FIXED.itemsize
    effect_rows = _read(blob, offset, header["effect_counts"], EFFECT)
    offset += effect_rows.nbytes
    repertoire_rows = _read(blob, offset, header["repertoire_counts"], REPERTOIRE)
    offset += repertoire_rows.nbytes
    type_rows = _read(blob, offset, header["type_counts"], TYPE)
    offset += type_rows.nbytes
    base_move_rows = _read(blob, offset, header["base_move_counts"], BASE_MOVE)
    offset += base_move_rows.nbytes
    move_effect_rows = _read(blob, offset, header["move_effect_counts"], MOVE_EFFECT)
    offset += move_effect_rows.nbytes
    if offset != len(blob):
        raise ValueError(f"observation blob has {len(blob) - offset} trailing bytes")
    return {
        "schema_version": int(header["schema_version"]),
        "token_mask": header["token_mask"].astype(bool),
        "categories": header["categories"],
        "category_known": header["category_known"].astype(bool),
        "floats": header["floats"],
        "float_known": header["float_known"].astype(bool),
        "flags": header["flags"].astype(bool),
        "flag_known": header["flag_known"].astype(bool),
        "effect_counts": header["effect_counts"],
        "repertoire_counts": header["repertoire_counts"],
        "type_counts": header["type_counts"],
        "base_move_counts": header["base_move_counts"],
        "move_effect_counts": header["move_effect_counts"],
        "effects": effect_rows,
        "repertoire": repertoire_rows,
        "types": type_rows,
        "base_moves": base_move_rows,
        "move_effects": move_effect_rows,
    }


def split_rows(flat: np.ndarray, counts) -> list:
    """Split a flat ragged section into one array view per token."""
    out = []
    start = 0
    for count in counts:
        count = int(count)
        out.append(flat[start:start + count])
        start += count
    return out


def parse_batch(fixed: bytes, ragged: bytes, count: int) -> dict:
    """Zero-copy batch decode for `NativeEngine.observe_fixed_batch`.

    `fixed` holds `count` fixed-stride token blocks; `ragged` concatenates the
    five ragged sections of every view in request order. Every array returned
    below is a numpy view over the input buffers — no per-view copy.
    """
    if count < 0 or len(fixed) != count * FIXED.itemsize:
        raise ValueError(
            f"fixed batch size {len(fixed)} != {count} * {FIXED.itemsize}"
        )
    header = np.frombuffer(fixed, dtype=FIXED, count=count)
    view = {
        "header": header,
        "schema_version": header["schema_version"],
        "token_mask": header["token_mask"].astype(bool),
        "categories": header["categories"],
        "category_known": header["category_known"].astype(bool),
        "floats": header["floats"],
        "float_known": header["float_known"].astype(bool),
        "flags": header["flags"].astype(bool),
        "flag_known": header["flag_known"].astype(bool),
        "effect_counts": header["effect_counts"],
        "repertoire_counts": header["repertoire_counts"],
        "type_counts": header["type_counts"],
        "base_move_counts": header["base_move_counts"],
        "move_effect_counts": header["move_effect_counts"],
        "ragged": ragged,
    }
    expected = (
        int(view["effect_counts"].sum()) * EFFECT.itemsize
        + int(view["repertoire_counts"].sum()) * REPERTOIRE.itemsize
        + int(view["type_counts"].sum()) * TYPE.itemsize
        + int(view["base_move_counts"].sum()) * BASE_MOVE.itemsize
        + int(view["move_effect_counts"].sum()) * MOVE_EFFECT.itemsize
    )
    if len(ragged) != expected:
        raise ValueError(f"ragged batch size {len(ragged)} != {expected}")
    return view


def split_batch_ragged(batch: dict, index: int) -> dict:
    """Slice one view's ragged rows out of a `parse_batch` result.

    Layout is per view (all five sections) in request order, so the offset of
    view `index` is the sum of the complete byte sizes of the preceding views.
    """
    ragged = batch["ragged"]
    sections = (
        ("effects", EFFECT, "effect_counts"),
        ("repertoire", REPERTOIRE, "repertoire_counts"),
        ("types", TYPE, "type_counts"),
        ("base_moves", BASE_MOVE, "base_move_counts"),
        ("move_effects", MOVE_EFFECT, "move_effect_counts"),
    )
    per_view_bytes = sum(
        batch[key].astype(np.int64).sum(axis=1) * dtype.itemsize
        for _, dtype, key in sections
    )
    offset = int(per_view_bytes[:index].sum())
    out = {}
    for name, dtype, key in sections:
        total = int(batch[key][index].sum())
        out[name] = np.frombuffer(ragged, dtype=dtype, count=total, offset=offset)
        offset += total * dtype.itemsize
    return out
