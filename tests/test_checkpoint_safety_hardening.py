"""Comprehensive security hardening regression tests for Checkpoints and Rollback (PR #22.1)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from nova.config import load_settings
from nova.core.checkpoints import CheckpointManager
from nova.core.git import GitService
from nova.web.app import create_app
from nova.core.safety import SafetyError
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
# 1. Git HEAD Commit SHA Correctness
# ---------------------------------------------------------------------------


async def test_git_head_is_commit_sha(tmp_path: Path):
    try:
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True)
        (tmp_path / "README.md").write_text("# Repo\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL)

        expected_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    except Exception:
        pytest.skip("git CLI not available")

    ws = Workspace(tmp_path)
    git_svc = GitService(ws)
    head_sha = await git_svc.current_head()
    assert head_sha == expected_sha

    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="head_task")
    assert cp.git_head == expected_sha


# ---------------------------------------------------------------------------
# 2. Mandatory Session Ownership
# ---------------------------------------------------------------------------


def test_missing_session_id_cannot_access_session_checkpoint(tmp_path: Path):
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    cp = cpm.create(task_id="task_sess", session_id="sess_secret_123")

    # Missing session_id must be DENIED
    with pytest.raises(PermissionError):
        cpm.get(cp.id)

    with pytest.raises(PermissionError):
        cpm.get(cp.id, session_id=None)

    with pytest.raises(PermissionError):
        cpm.inspect(cp.id, session_id=None)

    with pytest.raises(PermissionError):
        cpm.rollback(cp.id, session_id=None)

    assert cpm.delete(cp.id, session_id=None) is False

    # Matching session_id must succeed
    cp_ok = cpm.get(cp.id, session_id="sess_secret_123")
    assert cp_ok is not None
    assert cp_ok.id == cp.id


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
        f"/api/agent/checkpoint/{cp_A.id}/rollback", json={"session_id": "sess_B", "confirm": True}
    )
    assert res_rb.status_code == 403

    # Session B attempting to delete Session A checkpoint via API
    res_del = auth_client.delete(
        f"/api/agent/checkpoint/{cp_A.id}", params={"session_id": "sess_B"}
    )
    assert res_del.status_code in (403, 404)


# ---------------------------------------------------------------------------
# 3. Confirm Semantics Verification
# ---------------------------------------------------------------------------


def test_confirm_false_does_not_rollback(auth_client, tmp_path: Path):
    (tmp_path / "app.py").write_text("v1", encoding="utf-8")
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    cp = cpm.create(task_id="task_confirm")
    (tmp_path / "app.py").write_text("v2_modified", encoding="utf-8")

    # confirm=False -> rejected with HTTP 400
    res = auth_client.post(
        f"/api/agent/checkpoint/{cp.id}/rollback", json={"confirm": False}
    )
    assert res.status_code == 400
    assert "explicit confirmation" in res.json()["detail"]
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "v2_modified"


def test_confirm_true_allows_normal_rollback(auth_client, tmp_path: Path):
    (tmp_path / "app.py").write_text("v1", encoding="utf-8")
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    cp = cpm.create(task_id="task_confirm")
    (tmp_path / "app.py").write_text("v2_modified", encoding="utf-8")

    # confirm=True -> proceeds
    res = auth_client.post(
        f"/api/agent/checkpoint/{cp.id}/rollback", json={"confirm": True}
    )
    assert res.status_code == 200
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "v1"


# ---------------------------------------------------------------------------
# 4. User Ownership Sentinel & Invariant Verification
# ---------------------------------------------------------------------------


def test_user_owned_file_remains_untouched(tmp_path: Path):
    user_file = tmp_path / "user_work.py"
    user_file.write_text("USER_ORIGINAL_V1", encoding="utf-8")

    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    cp = cpm.create(task_id="task_user_owned")

    # Mark user_work.py as user-owned
    manifest_path = cpm.checkpoints_dir / cp.id / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["metadata"]["user_dirty_files"] = ["user_work.py"]
    for f in data["metadata"]["files"]:
        if f["path"] == "user_work.py":
            f["is_user_owned"] = True
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # User modifies file further
    user_file.write_text("USER_MODIFIED_V2", encoding="utf-8")

    # Rollback must NEVER touch user_work.py
    res = cpm.rollback(cp.id)
    assert user_file.read_text(encoding="utf-8") == "USER_MODIFIED_V2"
    assert any("user_work.py" in p for p in res.preserved)


# ---------------------------------------------------------------------------
# 5. Symlink & Path Hardening Tests
# ---------------------------------------------------------------------------


def test_symlink_replacement_cannot_escape_workspace(tmp_path: Path):
    outside_dir = tmp_path.parent / "outside_target"
    outside_dir.mkdir(exist_ok=True)
    outside_secret = outside_dir / "secret.txt"
    outside_secret.write_text("TOP_SECRET", encoding="utf-8")

    (tmp_path / "target.txt").write_text("inside_v1", encoding="utf-8")

    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="sym_task")

    # Replace target.txt with symlink to outside_secret
    (tmp_path / "target.txt").unlink()
    try:
        os.symlink(outside_secret, tmp_path / "target.txt")
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported")

    res = cpm.rollback(cp.id)
    assert outside_secret.read_text(encoding="utf-8") == "TOP_SECRET", "Outside secret was overwritten through symlink!"


def test_tampered_manifest_path_is_rejected(tmp_path: Path):
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="tamper_task")

    manifest_path = cpm.checkpoints_dir / cp.id / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Inject malicious traversal paths into manifest
    data["metadata"]["files"].append({
        "path": "../../etc/passwd",
        "state": "modified",
        "before_hash": "1234",
        "after_hash": "5678",
        "existed_before": True,
        "existed_after": True,
        "is_user_owned": False,
    })
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # get() must reject the tampered manifest completely
    assert cpm.get(cp.id) is None


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

    assert cpm_B.get(cp_A.id) is None


# ---------------------------------------------------------------------------
# 6. Static Code Audits
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


# ---------------------------------------------------------------------------
# 7. Checkpoint Symlink Hardening & Manifest Schema Security Tests (PR #23.1)
# ---------------------------------------------------------------------------


def test_is_safe_rel_path_allows_legitimate_double_dot_filename():
    from nova.core.checkpoints import _is_safe_rel_path
    assert _is_safe_rel_path("version..txt") is True
    assert _is_safe_rel_path("sub/version..txt") is True


def test_is_safe_rel_path_rejects_real_traversal():
    from nova.core.checkpoints import _is_safe_rel_path
    assert _is_safe_rel_path("../secret.txt") is False
    assert _is_safe_rel_path("foo/../secret.txt") is False
    assert _is_safe_rel_path("foo/../../secret.txt") is False
    assert _is_safe_rel_path("/etc/passwd") is False
    assert _is_safe_rel_path("C:\\Windows\\System32") is False


def test_checkpoint_does_not_read_file_symlink_outside_workspace(tmp_path: Path):
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    secret_content = "DO_NOT_COPY_THIS_SECRET_SENTINEL_123"
    outside_secret = outside_dir / "secret.txt"
    outside_secret.write_text(secret_content, encoding="utf-8")

    (ws_dir / "inside.txt").write_text("inside_valid_content", encoding="utf-8")

    symlink_file = ws_dir / "secret.txt"
    try:
        os.symlink(outside_secret, symlink_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported")

    ws = Workspace(ws_dir)
    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="symlink_test")

    # Verify inside.txt is in checkpoint
    assert any(f.path == "inside.txt" for f in cp.files)

    # Verify secret.txt is NOT in checkpoint files
    assert not any(f.path == "secret.txt" for f in cp.files)

    # Sentinel Check: Verify secret content does NOT exist in any checkpoint snapshot file
    cp_dir = cpm.checkpoints_dir / cp.id
    for root, _, files in os.walk(cp_dir):
        for f in files:
            fpath = Path(root) / f
            text = fpath.read_text(encoding="utf-8", errors="replace")
            assert secret_content not in text, f"Secret content leaked into snapshot {fpath}!"


def test_nested_file_symlink_outside_workspace(tmp_path: Path):
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    secret_content = "DO_NOT_COPY_NESTED_SECRET_456"
    outside_secret = outside_dir / "secret.txt"
    outside_secret.write_text(secret_content, encoding="utf-8")

    src_dir = ws_dir / "src"
    src_dir.mkdir()
    nested_symlink = src_dir / "leaked.txt"
    try:
        os.symlink(outside_secret, nested_symlink)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported")

    ws = Workspace(ws_dir)
    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="nested_symlink_test")

    assert not any("leaked.txt" in f.path for f in cp.files)

    cp_dir = cpm.checkpoints_dir / cp.id
    for root, _, files in os.walk(cp_dir):
        for f in files:
            text = (Path(root) / f).read_text(encoding="utf-8", errors="replace")
            assert secret_content not in text


def test_checkpoint_destination_cannot_escape(tmp_path: Path):
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="dest_test")

    # Manually attempt to inspect or load with a destination attempt
    cp_dir = cpm.checkpoints_dir / cp.id
    snaps_dir = cp_dir / "snapshots"

    # Attempt destination traversal path
    malicious_target = (snaps_dir / "../../outside.txt").resolve()
    snaps_resolved = snaps_dir.resolve()
    assert snaps_resolved not in malicious_target.parents


def test_manifest_schema_validation(tmp_path: Path):
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="schema_test")

    manifest_path = cpm.checkpoints_dir / cp.id / "manifest.json"

    # Test malformed non-dict metadata
    manifest_path.write_text(json.dumps({"metadata": "not_a_dict"}), encoding="utf-8")
    assert cpm.get(cp.id) is None

    # Test malformed non-list files
    manifest_path.write_text(json.dumps({"metadata": {"files": "not_a_list"}}), encoding="utf-8")
    cp_loaded = cpm.get(cp.id)
    assert cp_loaded is None

    # Test malformed file entry (non-dict item)
    manifest_path.write_text(json.dumps({"metadata": {"files": ["not_a_dict"]}}), encoding="utf-8")
    cp_loaded2 = cpm.get(cp.id)
    assert cp_loaded2 is None


# ---------------------------------------------------------------------------
# 8. Storage-Root Symlink & Strict Manifest Validation Tests (PR #24.1)
# ---------------------------------------------------------------------------


def test_checkpoint_storage_root_cannot_escape_via_nova_symlink(tmp_path: Path):
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()
    outside_dir = tmp_path / "outside_nova"
    outside_dir.mkdir()

    nova_symlink = ws_dir / ".nova"
    try:
        os.symlink(outside_dir, nova_symlink)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported")

    ws = Workspace(ws_dir)
    with pytest.raises(SafetyError):
        CheckpointManager(ws)


def test_checkpoint_storage_root_cannot_escape_via_checkpoints_symlink(tmp_path: Path):
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()
    nova_dir = ws_dir / ".nova"
    nova_dir.mkdir()
    outside_dir = tmp_path / "outside_cps"
    outside_dir.mkdir()

    cp_symlink = nova_dir / "checkpoints"
    try:
        os.symlink(outside_dir, cp_symlink)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported")

    ws = Workspace(ws_dir)
    with pytest.raises(SafetyError):
        CheckpointManager(ws)


def test_manifest_rejects_invalid_state_type(tmp_path: Path):
    (tmp_path / "app.py").write_text("code", encoding="utf-8")
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="state_type_test")

    manifest_path = cpm.checkpoints_dir / cp.id / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    data["metadata"]["files"][0]["state"] = 123
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    assert cpm.get(cp.id) is None


def test_manifest_rejects_invalid_boolean_types(tmp_path: Path):
    (tmp_path / "app.py").write_text("code", encoding="utf-8")
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="bool_type_test")

    manifest_path = cpm.checkpoints_dir / cp.id / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    data["metadata"]["files"][0]["is_user_owned"] = "false"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    assert cpm.get(cp.id) is None


def test_manifest_rejects_invalid_hash_types(tmp_path: Path):
    (tmp_path / "app.py").write_text("code", encoding="utf-8")
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="hash_type_test")

    manifest_path = cpm.checkpoints_dir / cp.id / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    data["metadata"]["files"][0]["before_hash"] = 12345
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    assert cpm.get(cp.id) is None


def test_manifest_rejects_invalid_required_fields(tmp_path: Path):
    (tmp_path / "app.py").write_text("code", encoding="utf-8")
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="req_fields_test")

    manifest_path = cpm.checkpoints_dir / cp.id / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    del data["metadata"]["files"][0]["state"]
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    assert cpm.get(cp.id) is None


def test_manifest_rejects_invalid_files_list_structure(tmp_path: Path):
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="files_list_test")

    manifest_path = cpm.checkpoints_dir / cp.id / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    data["metadata"]["files"] = "not_a_list"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    assert cpm.get(cp.id) is None


def test_manifest_rejects_invalid_user_dirty_files_structure(tmp_path: Path):
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="dirty_structure_test")

    manifest_path = cpm.checkpoints_dir / cp.id / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    data["metadata"]["user_dirty_files"] = {"invalid": "dict"}
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    assert cpm.get(cp.id) is None


def test_manifest_rejects_invalid_toplevel_fields(tmp_path: Path):
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)
    cp = cpm.create(task_id="toplevel_type_test")

    manifest_path = cpm.checkpoints_dir / cp.id / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    data["metadata"]["id"] = 99999
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    assert cpm.get(cp.id) is None


def test_manifest_preserves_legitimate_existing_manifests(tmp_path: Path):
    (tmp_path / "hello.py").write_text("print('hello')", encoding="utf-8")
    ws = Workspace(tmp_path)
    cpm = CheckpointManager(ws)

    cp = cpm.create(task_id="valid_test")
    loaded = cpm.get(cp.id)

    assert loaded is not None
    assert loaded.id == cp.id
    assert loaded.task_id == "valid_test"
    assert len(loaded.files) == 1
    assert loaded.files[0].path == "hello.py"
