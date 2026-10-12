"""Crash-safe, bounded storage for PPO checkpoint generations.

This module is deliberately independent of the learner and launcher.  A caller
hands it already-validated rank state; the manager only publishes complete
generations and never makes a model-only file resumable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping


GB = 1_000_000_000
GIB = 1 << 30


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as f:
        os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class StorageBudgetError(RuntimeError):
    pass


class CheckpointGenerationStore:
    """Publish and retain complete two-rank checkpoint generations.

    Files are written beneath ``root/generations`` and become resumable only
    after both ranks, hashes, schema, and caller-supplied consistency metadata
    have been written to a manifest and the ``LATEST`` pointer is atomically
    replaced.  Incomplete directories are ignored by discovery.
    """

    def __init__(self, root: str | Path, *, budget_bytes: int | None = None,
                 reserve_bytes: int = 15 * GIB, concurrent_write_allowance_bytes: int = 0,
                 keep_generations: int = 3):
        self.root = Path(root).resolve()
        self.generations = self.root / "generations"
        self.registry_path = self.root / "protected_artifacts.json"
        self.latest_path = self.root / "LATEST"
        self.budget_bytes = None if budget_bytes is None else int(budget_bytes)
        self.concurrent_write_allowance_bytes = int(concurrent_write_allowance_bytes)
        self.reserve_bytes = int(reserve_bytes)
        self.keep_generations = int(keep_generations)
        if self.keep_generations < 3:
            raise ValueError("at least three complete generations are required")
        self.generations.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_json(self.registry_path, {"protected": [], "pins": {}})

    def _write_json(self, path: Path, value: Any) -> None:
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(value, f, sort_keys=True, indent=2)
            f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)

    def disk_report(self, *, projected_write_bytes: int = 0) -> dict:
        usage = shutil.disk_usage(self.root)
        owned = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        checkpoint = sum(p.stat().st_size for p in self.generations.rglob("*") if p.is_file())
        projected = int(projected_write_bytes) * 2 + self.concurrent_write_allowance_bytes  # generation plus temporary copy and external writer allowance
        return {
            "path": str(self.root), "available_bytes": usage.free,
            "filesystem_total_bytes": usage.total, "filesystem_used_bytes": usage.used,
            "project_owned_bytes": owned, "checkpoint_bytes": checkpoint,
            "project_budget_bytes": self.budget_bytes,
            "project_budget_remaining_bytes": (max(0, self.budget_bytes - owned) if self.budget_bytes is not None else None),
            "projected_peak_write_bytes": projected,
            "safe_to_write": usage.free - projected >= self.reserve_bytes,
        }

    def require_capacity(self, projected_write_bytes: int) -> dict:
        report = self.disk_report(projected_write_bytes=projected_write_bytes)
        if not report["safe_to_write"]:
            raise StorageBudgetError(json.dumps(report, sort_keys=True))
        return report

    def protected(self) -> dict:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def protect(self, path: str | Path, *, reason: str) -> None:
        data = self.protected(); item = {"path": str(Path(path).resolve()), "reason": reason}
        if item not in data["protected"]:
            data["protected"].append(item)
        self._write_json(self.registry_path, data)

    def pin(self, policy_id: str, path: str | Path, *, reason: str = "active evaluation") -> None:
        data = self.protected(); data.setdefault("pins", {})[policy_id] = {"path": str(Path(path).resolve()), "reason": reason}
        self._write_json(self.registry_path, data)

    def _protected_paths(self) -> set[Path]:
        data = self.protected(); paths = {Path(x["path"]) for x in data.get("protected", [])}
        paths.update(Path(x["path"]) for x in data.get("pins", {}).values())
        return paths

    def publish(self, generation: str, rank_files: Mapping[int, bytes], metadata: Mapping[str, Any]) -> dict:
        if set(rank_files) != {0, 1}:
            raise ValueError("a complete DDP generation requires rank 0 and rank 1")
        if not metadata.get("schema") or not metadata.get("iteration_complete", False):
            raise ValueError("generation metadata must prove a completed iteration")
        if int(metadata.get("world_size", 0)) != 2:
            raise ValueError("storage policy currently requires two DDP ranks")
        self.require_capacity(sum(len(x) for x in rank_files.values()))
        stamp = f"{generation}-{time.time_ns()}"
        tmp = self.generations / f".incomplete-{stamp}"
        final = self.generations / stamp
        tmp.mkdir(parents=True)
        files = {}
        try:
            for rank, payload in rank_files.items():
                path = tmp / f"rank{rank}.pt"
                with path.open("wb") as f:
                    f.write(payload); f.flush(); os.fsync(f.fileno())
                files[str(rank)] = {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            manifest = dict(metadata)
            manifest.update({"generation": generation, "published_at": time.time(), "ranks": files, "complete": True})
            self._write_json(tmp / "manifest.json", manifest)
            _fsync_dir(tmp)
            os.replace(tmp, final); _fsync_dir(self.generations)
            pointer = self.latest_path.with_name(f".LATEST.tmp-{os.getpid()}-{time.time_ns()}")
            pointer.write_text(final.name + "\n", encoding="utf-8"); _fsync_file(pointer); os.replace(pointer, self.latest_path); _fsync_dir(self.root)
            self.rotate()
            return manifest
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def generations_ready(self) -> list[Path]:
        found = []
        for p in self.generations.iterdir():
            if not p.is_dir() or p.name.startswith(".incomplete-") or not (p / "manifest.json").exists():
                continue
            try:
                m = json.loads((p / "manifest.json").read_text())
                if m.get("complete") is True and set(m.get("ranks", {})) == {"0", "1"} and all((p / v["path"]).exists() and _sha256(p / v["path"]) == v["sha256"] for v in m["ranks"].values()):
                    found.append(p)
            except (OSError, ValueError, KeyError):
                continue
        return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)

    def latest(self) -> Path | None:
        ready = self.generations_ready()
        if not ready:
            return None
        if self.latest_path.exists():
            target = self.generations / self.latest_path.read_text().strip()
            if target in ready:
                return target
        # A damaged pointer or newest generation must never make an older
        # verified generation unusable.  The caller can resume this fallback;
        # publication of a replacement pointer remains an explicit write.
        return ready[0]

    def rotate(self) -> list[str]:
        ready = self.generations_ready(); keep = {p.resolve() for p in ready[:self.keep_generations]}
        protected = self._protected_paths(); removed = []
        for p in ready[self.keep_generations:]:
            if p.resolve() in protected or (p / "manifest.json").resolve() in protected: continue
            shutil.rmtree(p); removed.append(str(p))
        return removed

    def cleanup_incomplete(self) -> list[str]:
        removed = []
        for p in self.generations.glob(".incomplete-*"):
            if p.is_dir(): shutil.rmtree(p); removed.append(str(p))
        return removed
