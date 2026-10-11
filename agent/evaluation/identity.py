"""Versioned, canonical identities for frozen evaluation artifacts."""
from __future__ import annotations

import hashlib
import os

MODEL_DIGEST_ALGORITHM = "pa3-model-state-sha256-sorted-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_model_digest(state_dict):
    """Stable state identity independent of Python dictionary insertion order."""
    digest = hashlib.sha256(MODEL_DIGEST_ALGORITHM.encode())
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def path_digest(path):
    path = os.path.abspath(os.fspath(path))
    if not os.path.isfile(path):
        raise ValueError(f"identity input is not a file: {path}")
    return sha256_file(path)
