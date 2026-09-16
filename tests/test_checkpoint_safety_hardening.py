"""Comprehensive security hardening regression tests for Checkpoints and Rollback."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from nova.config import load_settings
from nova.core.checkpoints import CheckpointManager
from nova.web.app import create_app
from nova.workspace.files import Workspace


@pytest.fixture
def test_app(tmp_path: Path):
    settings = load_settings(
        project_root=tmp_path,
        env={
            "GROQ_API_KEY": "gsk_live_secret_key_123456789012345",
            "NOVA_WEB_TOKEN": "secret-token-123",
            "NOVA_SAFETY_MODE": "smart",
        },
        user_config_path=tmp_path / "_no_user_config.json",
    )
    app = create_app(settings)
    return app


@pytest.fixture
def auth_client(test_app):
    client = TestClient(test_app)
    client.headers["Authorization"] = "Bearer secret-token-123"
    return client


# ---------------------------------------------------------------------------
# 1. User Ownership Protection & Sentinel Verification
# ---------------------------------------------------------------------------


def test_pre_existing_user_dirty_file_never_destroyed(tmp_path: Path):
    user_sentinel = tmp_path / "user_work.py"
    user_sentinel.write_text("USER_SENTINEL_V1", encoding="utf-8")

    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    # Create checkpoint
    cp = cpm.create(task_id="agent_task_1")

    # Force user_work.py into user_dirty_files in metadata
    manifest_path = cpm.checkpoints_dir / cp.id / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["metadata"]["user_dirty_files"] = ["user_work.py"]
    for f in data["metadata"]["files"]:
        if f["path"] == "user_work.py":
            f["is_user_owned"] = True
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # User modifies file further after checkpoint creation
    user_sentinel.write_text("USER_SENTINEL_V2", encoding="utf-8")

    # Agent creates a new file
    (tmp_path / "agent_created.py").write_text("agent_code", encoding="utf-8")

    # Perform rollback
    res = cpm.rollback(cp.id)

    # Invariant checks
    assert user_sentinel.read_text(encoding="utf-8") == "USER_SENTINEL_V2", "User work was overwritten!"
    assert any("user_work.py" in p for p in res.preserved)
    assert not (tmp_path / "agent_created.py").exists(), "Agent file was not removed"


# ---------------------------------------------------------------------------
# 2. Cross-Session Isolation
# ---------------------------------------------------------------------------


def test_cross_session_checkpoint_access_blocked(auth_client, tmp_path: Path):
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    cp_A = cpm.create(task_id="task_A", session_id="sess_A")

    # Session B attempting to inspect Session A checkpoint via API
    res_inspect = auth_client.get(
        f"/api/agent/checkpoint/{cp_A.id}", params={"session_id": "sess_B"}
    )
    assert res_inspect.status_code == 403

    # Session B attempting to rollback Session A checkpoint via API
    res_rb = auth_client.post(
        f"/api/agent/checkpoint/{cp_A.id}/rollback", json={"session_id": "sess_B"}
    )
    assert res_rb.status_code == 403

    # Session B attempting to delete Session A checkpoint via API
    res_del = auth_client.delete(
        f"/api/agent/checkpoint/{cp_A.id}", params={"session_id": "sess_B"}
    )
    assert res_del.status_code in (403, 404)


# ---------------------------------------------------------------------------
# 3. Cross-Workspace Isolation
# ---------------------------------------------------------------------------


def test_cross_workspace_checkpoint_access_blocked(tmp_path: Path):
    ws_A = Workspace(tmp_path / "ws_A", create_root=True)
    ws_B = Workspace(tmp_path / "ws_B", create_root=True)

    cpm_A = CheckpointManager(ws_A)
    cpm_B = CheckpointManager(ws_B)

    cp_A = cpm_A.create(task_id="task_A")

    # Copy cp_A into ws_B's checkpoint directory
    cp_B_dir = cpm_B.checkpoints_dir / cp_A.id
    import shutil
    shutil.copytree(cpm_A.checkpoints_dir / cp_A.id, cp_B_dir)

    # cpm_B must reject cp_A because root does not match ws_B
    assert cpm_B.get(cp_A.id) is None


# ---------------------------------------------------------------------------
# 4. Checkpoint ID & Path Traversal Security
# ---------------------------------------------------------------------------


def test_checkpoint_id_traversal_attempts_rejected(auth_client):
    traversal_ids = [
        "../../etc/passwd",
        "../cp_123",
        "/etc/passwd",
        "cp_123; rm -rf /",
    ]

    for cp_id in traversal_ids:
        res = auth_client.get(f"/api/agent/checkpoint/{cp_id}")
        assert res.status_code in (400, 404)

        res_rb = auth_client.post(f"/api/agent/checkpoint/{cp_id}/rollback", json={})
        assert res_rb.status_code in (400, 404)


# ---------------------------------------------------------------------------
# 5. Static Inspection Security Audits
# ---------------------------------------------------------------------------


def test_checkpoints_file_has_no_direct_subprocess_calls():
    cp_path = Path(__file__).resolve().parent.parent / "nova" / "core" / "checkpoints.py"
    text = cp_path.read_text(encoding="utf-8")
    assert "subprocess" not in text, "checkpoints.py must not directly invoke subprocess"
    assert "os.system" not in text, "checkpoints.py must not invoke os.system"


def test_checkpoints_and_routes_have_no_check_safety_false():
    for rel_path in ("nova/core/checkpoints.py", "nova/web/routes.py"):
        fpath = Path(__file__).resolve().parent.parent / rel_path
        text = fpath.read_text(encoding="utf-8")
        assert "check_safety=False" not in text, f"{rel_path} must not contain check_safety=False"
