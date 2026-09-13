"""Lightweight task snapshot and rollback mechanism for NovaCLI workspace.

Allows creating file checkpoints before agent tasks and inspecting diffs or
restoring previous snapshots safely.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nova.workspace.files import Workspace


@dataclass
class SnapshotEntry:
    path: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotEntry:
        return cls(path=str(data.get("path", "")), content=str(data.get("content", "")))


@dataclass
class Snapshot:
    id: str
    timestamp: float
    description: str
    files: list[SnapshotEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "description": self.description,
            "files": [f.to_dict() for f in self.files],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Snapshot:
        return cls(
            id=str(data.get("id", "")),
            timestamp=float(data.get("timestamp") or 0.0),
            description=str(data.get("description", "")),
            files=[
                SnapshotEntry.from_dict(f)
                for f in (data.get("files") or [])
                if isinstance(f, dict)
            ],
        )


class SnapshotManager:
    """Manages file checkpoints under <project_root>/.nova/snapshots/."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    @property
    def snapshots_dir(self) -> Path:
        return self.workspace.root / ".nova" / "snapshots"

    def create_snapshot(self, description: str = "task_checkpoint", max_files: int = 500) -> Snapshot:
        snapshot_id = f"snap_{int(time.time() * 1000)}"
        entries: list[SnapshotEntry] = []

        for file_path in self.workspace.iter_files(limit=max_files):
            rel_path = self.workspace.relative(file_path)
            try:
                content = self.workspace.read_text(file_path, max_bytes=200_000)
                entries.append(SnapshotEntry(path=rel_path, content=content))
            except Exception:
                pass

        snapshot = Snapshot(
            id=snapshot_id,
            timestamp=time.time(),
            description=description,
            files=entries,
        )

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        snap_file = self.snapshots_dir / f"{snapshot_id}.json"
        snap_file.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
        return snapshot

    def list_snapshots(self) -> list[Snapshot]:
        if not self.snapshots_dir.exists():
            return []

        snapshots: list[Snapshot] = []
        for file in sorted(self.snapshots_dir.glob("snap_*.json"), reverse=True):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    snapshots.append(Snapshot.from_dict(data))
            except Exception:
                pass
        return snapshots

    def latest_snapshot(self) -> Snapshot | None:
        snaps = self.list_snapshots()
        return snaps[0] if snaps else None

    def restore_snapshot(self, snapshot_id: str | None = None) -> Snapshot | None:
        if snapshot_id:
            snap_file = self.snapshots_dir / f"{snapshot_id}.json"
            if not snap_file.exists():
                return None
            data = json.loads(snap_file.read_text(encoding="utf-8"))
            snapshot = Snapshot.from_dict(data)
        else:
            snapshot = self.latest_snapshot()

        if not snapshot:
            return None

        for entry in snapshot.files:
            try:
                self.workspace.write_text(entry.path, entry.content)
            except Exception:
                pass
        return snapshot
