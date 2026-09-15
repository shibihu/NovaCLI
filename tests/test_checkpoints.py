"""Unit tests for CheckpointManager in nova.core.checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from nova.core.checkpoints import CheckpointManager
from nova.workspace.files import Workspace


def test_checkpoint_creation_and_inspection(tmp_path: Path):
    (tmp_path / "hello.py").write_text("print('hello')", encoding="utf-8")
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    cp = cpm.create(task_id="task_123")
    assert cp.id.startswith("cp_")
    assert cp.task_id == "task_123"

    # Modify hello.py and create new.py
    (tmp_path / "hello.py").write_text("print('hello world')", encoding="utf-8")
    (tmp_path / "new.py").write_text("new_content", encoding="utf-8")

    info = cpm.inspect(cp.id)
    assert info["total_changes"] == 2
    paths = {c["path"]: c["status"] for c in info["changed_files"]}
    assert paths["hello.py"] == "modified"
    assert paths["new.py"] == "created"


def test_checkpoint_rollback_agent_changes(tmp_path: Path):
    (tmp_path / "app.py").write_text("v1", encoding="utf-8")
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    cp = cpm.create(task_id="task_rollback")

    # Agent modifies app.py and creates gen.py
    (tmp_path / "app.py").write_text("v2", encoding="utf-8")
    (tmp_path / "gen.py").write_text("generated", encoding="utf-8")

    res = cpm.rollback(cp.id)
    assert res.ok is True
    assert "app.py" in res.restored
    assert "gen.py" in res.removed
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "v1"
    assert not (tmp_path / "gen.py").exists()


def test_checkpoint_protects_user_dirty_files(tmp_path: Path):
    (tmp_path / "user_edit.py").write_text("user_v1", encoding="utf-8")

    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    cp = cpm.create(task_id="task_user")
    manifest_path = cpm.checkpoints_dir / cp.id / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["metadata"]["user_dirty_files"] = ["user_edit.py"]
    for f in data["metadata"]["files"]:
        if f["path"] == "user_edit.py":
            f["is_user_owned"] = True
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # User edit changes further
    (tmp_path / "user_edit.py").write_text("user_v2", encoding="utf-8")

    res = cpm.rollback(cp.id)
    assert any("user_edit.py" in p for p in res.preserved)
    assert (tmp_path / "user_edit.py").read_text(encoding="utf-8") == "user_v2"
