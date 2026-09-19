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


def test_checkpoint_redo_basic(tmp_path: Path):
    (tmp_path / "main.py").write_text("v1", encoding="utf-8")
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    cp = cpm.create(task_id="task_redo")
    (tmp_path / "main.py").write_text("v2", encoding="utf-8")

    res_undo = cpm.rollback(cp.id)
    assert res_undo.ok is True
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == "v1"

    res_redo = cpm.redo(cp.id)
    assert res_redo.ok is True
    assert "main.py" in res_redo.restored
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == "v2"


def test_checkpoint_redo_created_and_deleted(tmp_path: Path):
    (tmp_path / "old.py").write_text("old_content", encoding="utf-8")
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    cp = cpm.create(task_id="task_create_del")

    # Agent creates new.py and deletes old.py
    (tmp_path / "new.py").write_text("new_content", encoding="utf-8")
    (tmp_path / "old.py").unlink()

    # Undo
    cpm.rollback(cp.id)
    assert not (tmp_path / "new.py").exists()
    assert (tmp_path / "old.py").read_text(encoding="utf-8") == "old_content"

    # Redo
    res = cpm.redo(cp.id)
    assert res.ok is True
    assert "new.py" in res.restored
    assert "old.py" in res.removed
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "new_content"
    assert not (tmp_path / "old.py").exists()


def test_checkpoint_redo_conflict_user_modified_after_undo(tmp_path: Path):
    (tmp_path / "file.py").write_text("v1", encoding="utf-8")
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    cp = cpm.create(task_id="task_conflict")
    (tmp_path / "file.py").write_text("v2_agent", encoding="utf-8")

    # Undo
    cpm.rollback(cp.id)
    assert (tmp_path / "file.py").read_text(encoding="utf-8") == "v1"

    # User modifies file.py after Undo
    (tmp_path / "file.py").write_text("v3_user_edit", encoding="utf-8")

    # Redo must preserve user edit
    res = cpm.redo(cp.id)
    assert res.ok is True
    assert any("file.py" in p for p in res.preserved)
    assert (tmp_path / "file.py").read_text(encoding="utf-8") == "v3_user_edit"


def test_checkpoint_redo_consumed_and_undo_redo_cycle(tmp_path: Path):
    (tmp_path / "cycle.py").write_text("v1", encoding="utf-8")
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    cp = cpm.create(task_id="task_cycle")
    (tmp_path / "cycle.py").write_text("v2", encoding="utf-8")

    cpm.rollback(cp.id)
    assert (tmp_path / "cycle.py").read_text(encoding="utf-8") == "v1"

    cpm.redo(cp.id)
    assert (tmp_path / "cycle.py").read_text(encoding="utf-8") == "v2"

    # Redo again without Undo fails cleanly
    with pytest.raises(KeyError):
        cpm.redo(cp.id)

    # Undo again creates fresh redo state
    cpm.rollback(cp.id)
    assert (tmp_path / "cycle.py").read_text(encoding="utf-8") == "v1"

    cpm.redo(cp.id)
    assert (tmp_path / "cycle.py").read_text(encoding="utf-8") == "v2"
