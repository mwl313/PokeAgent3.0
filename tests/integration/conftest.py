"""Integration-test path setup (real Rust engine + Python agent)."""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT, REPO_ROOT / "engine" / "python"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

torch = pytest.importorskip("torch")  # noqa: F841


@pytest.fixture(scope="session")
def native_engine_data():
    data = REPO_ROOT / "engine" / "data"
    teams = data / "training-teams.json"
    if not data.exists() or not teams.exists():
        pytest.skip("native engine data is not present")
    return str(data), str(teams)
