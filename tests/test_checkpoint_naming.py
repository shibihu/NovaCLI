"""Tests for human-readable checkpoint naming, manual rename API, legacy fallback, and rollback/redo integration."""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from nova.config import load_settings
from nova.web.app import create_app
from nova.core.checkpoints import CheckpointManager, generate_checkpoint_name
from nova.workspace.files import Workspace


def _make_client(tmp_path: Path) -> TestClient:
    settings = load_settings(
        project_root=tmp_path,
        env={"NOVA_WEB_TOKEN": "", "GROQ_API_KEY": "gsk_test"},
    )
    return TestClient(create_app(settings))


def test_checkpoint_autonaming(tmp_path: Path):
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    # Create dummy file to snapshot
    (tmp_path / "hello.txt").write_text("initial content", encoding="utf-8")

    cp = cpm.create(task_id="Remove the OllamaFolder directory")
    assert cp.name == "Remove the OllamaFolder directory"
    assert cp.id.startswith("cp_")

    loaded = cpm.get(cp.id)
    assert loaded is not None
    assert loaded.name == "Remove the OllamaFolder directory"


def test_checkpoint_rename_api(tmp_path: Path):
    client = _make_client(tmp_path)
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    (tmp_path / "test.py").write_text("# test file", encoding="utf-8")
    cp = cpm.create(task_id="Initial task")

    # Rename via API
    res = client.patch(
        f"/api/agent/checkpoint/{cp.id}",
        json={"name": "Refactor Authentication Logic"},
    )
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert res.json()["checkpoint"]["name"] == "Refactor Authentication Logic"

    # Verify loaded
    loaded = cpm.get(cp.id)
    assert loaded.name == "Refactor Authentication Logic"


def test_checkpoint_rename_validation(tmp_path: Path):
    client = _make_client(tmp_path)
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    (tmp_path / "file.txt").write_text("data", encoding="utf-8")
    cp = cpm.create(task_id="Valid Task")

    # Empty name
    res_empty = client.patch(f"/api/agent/checkpoint/{cp.id}", json={"name": ""})
    assert res_empty.status_code in (400, 422)

    # Invalid control characters / newline
    res_ctrl = client.patch(f"/api/agent/checkpoint/{cp.id}", json={"name": "Bad\nName"})
    assert res_ctrl.status_code == 400

    # Path traversal attempt in name
    res_trav = client.patch(f"/api/agent/checkpoint/{cp.id}", json={"name": "../../../etc/passwd"})
    assert res_trav.status_code == 400


def test_checkpoint_legacy_fallback(tmp_path: Path):
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    (tmp_path / "doc.md").write_text("docs", encoding="utf-8")
    cp = cpm.create(task_id="Legacy Task")

    # Remove 'name' key from manifest.json to simulate an older checkpoint
    manifest_path = tmp_path / ".nova" / "checkpoints" / cp.id / "manifest.json"
    mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "name" in mdata["metadata"]:
        del mdata["metadata"]["name"]
    manifest_path.write_text(json.dumps(mdata, indent=2), encoding="utf-8")

    # Load legacy checkpoint
    legacy_cp = cpm.get(cp.id)
    assert legacy_cp is not None
    assert legacy_cp.name == "Legacy Task"


def test_checkpoint_rename_integrity_with_rollback_and_redo(tmp_path: Path):
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    f = tmp_path / "main.py"
    f.write_text("v1", encoding="utf-8")

    cp = cpm.create(task_id="Initial state")
    f.write_text("v2 modified by agent", encoding="utf-8")

    # Rename checkpoint
    cpm.rename(cp.id, "Custom Checkpoint Title")

    # Perform Rollback
    rb_res = cpm.rollback(cp.id)
    assert rb_res.ok
    assert f.read_text(encoding="utf-8") == "v1"

    # Perform Redo
    redo_res = cpm.redo(cp.id)
    assert redo_res.ok
    assert f.read_text(encoding="utf-8") == "v2 modified by agent"
