"""PokeAgent 3.0 native engine Python package.

The compiled PyO3 extension lives beside this file as `pa3_engine.so` (built by
`scripts/build_python.sh`, gitignored). Import order matters: this module
re-exports the native batch API and the packed-observation parser together.
"""
from . import pa3_engine as _native
from .observation import (
    BASE_MOVE,
    EFFECT,
    FIXED,
    MOVE_EFFECT,
    PACKED_CANDIDATE_FIELDS,
    REPERTOIRE,
    TYPE,
    packed_candidate_rows,
    parse_batch,
    parse_packed_candidates,
    parse_view,
    split_batch_ragged,
    split_rows,
)

NativeEngine = _native.NativeEngine
SCHEMA_VERSION = _native.SCHEMA_VERSION
OBSERVATION_TOKENS = _native.OBSERVATION_TOKENS
CATEGORY_COUNT = _native.CATEGORY_COUNT
FLOAT_COUNT = _native.FLOAT_COUNT
FLAG_COUNT = _native.FLAG_COUNT
OBSERVATION_FIXED_BYTES = _native.OBSERVATION_FIXED_BYTES

__all__ = [
    "NativeEngine",
    "SCHEMA_VERSION",
    "OBSERVATION_TOKENS",
    "CATEGORY_COUNT",
    "FLOAT_COUNT",
    "FLAG_COUNT",
    "OBSERVATION_FIXED_BYTES",
    "parse_view",
    "parse_batch",
    "split_batch_ragged",
    "split_rows",
    "FIXED",
    "EFFECT",
    "REPERTOIRE",
    "TYPE",
    "BASE_MOVE",
    "MOVE_EFFECT",
    "PACKED_CANDIDATE_FIELDS",
    "parse_packed_candidates",
    "packed_candidate_rows",
]
