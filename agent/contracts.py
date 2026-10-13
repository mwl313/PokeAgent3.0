"""Frozen V2 identities. Legacy policies cannot be silently reinterpreted."""
import hashlib
import json
from pathlib import Path

SPEC = json.loads((Path(__file__).resolve().parents[1] / "configs/pa3_v2_contract.json").read_text())
CONTRACT_HASH = hashlib.sha256(json.dumps(SPEC, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
EXPECTED_HASH = "8a0e225e3f657f1016b1dfb39d0e683425185bf14b5a6c2a540930da55f6db43"
if CONTRACT_HASH != EXPECTED_HASH:
    raise ValueError("V2 semantic contract changed without a coordinated wire/schema migration")
SCHEMA_VERSION = 2
OBSERVATION_SCHEMA = SPEC["observation_schema"]
ACTION_SCHEMA = SPEC["action_schema"]
