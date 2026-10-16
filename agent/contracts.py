"""Frozen V2 identities. Legacy policies cannot be silently reinterpreted."""
import hashlib
import json
from pathlib import Path

SPEC = json.loads((Path(__file__).resolve().parents[1] / "configs/pa3_v2_contract.json").read_text())
CONTRACT_HASH = hashlib.sha256(json.dumps(SPEC, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
EXPECTED_HASH = "dd723dc687a60b0568302fa0b09cfb110cbe67eac2c7022f2b4ffe4dff373e35"
if CONTRACT_HASH != EXPECTED_HASH:
    raise ValueError("V2 semantic contract changed without a coordinated wire/schema migration")
SCHEMA_VERSION = 2
OBSERVATION_SCHEMA = SPEC["observation_schema"]
ACTION_SCHEMA = SPEC["action_schema"]


def execution_identity(mode):
    mode=str(mode).upper()
    if mode not in ('A','B','C','D','E'):
        raise ValueError('unknown V2 precision mode')
    return {'observation_schema':OBSERVATION_SCHEMA,'action_schema':ACTION_SCHEMA,
            'semantic_hash':CONTRACT_HASH,'ragged_contract_hash':CONTRACT_HASH,
            'precision_contract':('pa3-precision-E-v1' if mode=='E' else f'pa3-precision-{mode}-v2')}


def require_checkpoint_identity(state,mode=None):
    identity=state.get('contracts') if isinstance(state,dict) else None
    if not isinstance(identity,dict):
        raise ValueError('checkpoint has no V2 contract identity; V1/raw weights are incompatible')
    precision=identity.get('precision_contract','')
    valid={f'pa3-precision-{m}-v2':m for m in ('A','B','C','D')}
    valid['pa3-precision-E-v1']='E'
    if precision not in valid or identity!=execution_identity(mode or valid[precision]):
        raise ValueError('checkpoint observation/action/ragged/precision contract mismatch')
    return valid[precision]
