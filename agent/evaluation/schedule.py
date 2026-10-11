"""Resumable, informational evaluation schedule keyed by committed matches."""
from __future__ import annotations

import json
import os
import tempfile


def due_evaluation(state_path, committed_matches, *, anchor=352256):
    """Return lightweight/full due once, without ever blocking training."""
    if os.path.exists(state_path):
        state = json.load(open(state_path))
    else:
        state = {"anchor": anchor, "completed": {}}
    offset = max(0, int(committed_matches) - int(state.get("anchor", anchor)))
    full = offset // 200000
    light = offset // 50000
    key = f"full:{full}" if full else f"light:{light}"
    if (full and full > 0 and key not in state["completed"]) or (not full and light > 0 and key not in state["completed"]):
        return {"kind": "full" if full else "light", "key": key, "committed_matches": int(committed_matches), "pending": True}
    return {"kind": None, "key": None, "committed_matches": int(committed_matches), "pending": False}


def mark_completed(state_path, key, receipt):
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"anchor": 352256, "completed": {}}
    state.setdefault("completed", {})[key] = receipt
    os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".schedule-", dir=os.path.dirname(os.path.abspath(state_path)))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, state_path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
