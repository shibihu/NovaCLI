"""Tests for Workspace Snapshot / Undo System."""

from __future__ import annotations

from pathlib import Path
from nova.workspace.files import Workspace
from nova.workspace.snapshot import SnapshotManager


def test_snapshot_create_and_restore(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("v1", encoding="utf-8")
    workspace = Workspace(tmp_path)
    mgr = SnapshotManager(workspace)

    # 1. Create checkpoint
    snap = mgr.create_snapshot("initial state")
    assert snap.id.startswith("snap_")
    assert len(snap.files) >= 1

    # 2. Modify workspace
    (tmp_path / "main.py").write_text("v2", encoding="utf-8")
    assert (tmp_path / "main.py").read_text() == "v2"

    # 3. Restore snapshot
    restored = mgr.restore_snapshot(snap.id)
    assert restored is not None
    assert (tmp_path / "main.py").read_text() == "v1"


def test_list_and_latest_snapshot(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("code", encoding="utf-8")
    workspace = Workspace(tmp_path)
    mgr = SnapshotManager(workspace)

    snap1 = mgr.create_snapshot("first")
    snaps = mgr.list_snapshots()
    assert len(snaps) == 1
    assert mgr.latest_snapshot().id == snap1.id
