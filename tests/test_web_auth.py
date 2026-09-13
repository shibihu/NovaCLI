"""Tests for Web API Authentication."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient
from nova.web.app import create_app
from nova.config import load_settings


def test_localhost_unauthenticated_allowed_when_no_token_set(tmp_path: Path) -> None:
    settings = load_settings(project_root=tmp_path, env={"GROQ_API_KEY": "gsk_test"})
    app = create_app(settings)
    client = TestClient(app)

    # Localhost requests pass through TestClient
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_web_token_required_when_configured(tmp_path: Path) -> None:
    settings = load_settings(
        project_root=tmp_path,
        env={"GROQ_API_KEY": "gsk_test", "NOVA_WEB_TOKEN": "secret-token-123"},
    )
    app = create_app(settings)
    client = TestClient(app)

    # Missing token -> 401
    res_no_auth = client.get("/api/health")
    assert res_no_auth.status_code == 401

    # Invalid token -> 401
    res_bad_auth = client.get("/api/health", headers={"Authorization": "Bearer wrong-token"})
    assert res_bad_auth.status_code == 401

    # Valid token via Bearer header -> 200
    res_ok = client.get("/api/health", headers={"Authorization": "Bearer secret-token-123"})
    assert res_ok.status_code == 200

    # Valid token via query param -> 200
    res_query_ok = client.get("/api/health?token=secret-token-123")
    assert res_query_ok.status_code == 200
