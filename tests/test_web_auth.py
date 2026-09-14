"""Tests for Web API authentication layer (NOVA_WEB_TOKEN and localhost/LAN controls)."""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from nova.config import load_settings
from nova.web.app import create_app

SECRET_TOKEN = "secret_web_token_123456"


@pytest.fixture
def auth_app(tmp_path: Path):
    settings = load_settings(
        project_root=tmp_path,
        env={"NOVA_WEB_TOKEN": SECRET_TOKEN, "GROQ_API_KEY": "gsk_test"},
    )
    return create_app(settings)


@pytest.fixture
def client(auth_app):
    return TestClient(auth_app)


def test_public_endpoints_accessible_without_token(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/health").status_code == 200


def test_api_endpoint_blocks_missing_token(client):
    resp = client.get("/api/config")
    assert resp.status_code == 401
    assert "Invalid or missing authentication token" in resp.json()["detail"]


def test_api_endpoint_blocks_invalid_token(client):
    headers = {"Authorization": "Bearer wrong_token"}
    resp = client.get("/api/config", headers=headers)
    assert resp.status_code == 401


def test_api_endpoint_allows_valid_bearer_token(client):
    headers = {"Authorization": f"Bearer {SECRET_TOKEN}"}
    resp = client.get("/api/config", headers=headers)
    assert resp.status_code == 200
    assert "provider" in resp.json()


def test_api_endpoint_allows_valid_x_header_token(client):
    headers = {"X-Nova-Web-Token": SECRET_TOKEN}
    resp = client.get("/api/config", headers=headers)
    assert resp.status_code == 200


def test_rest_api_endpoint_rejects_query_param_token(client):
    # Query string tokens are deprecated and disabled for REST APIs
    resp = client.get(f"/api/config?token={SECRET_TOKEN}")
    assert resp.status_code == 401


def test_sse_endpoint_protected(client):
    resp = client.get("/api/agent/stream?session_id=unknown")
    assert resp.status_code == 401

    resp_authed = client.get(f"/api/agent/stream?session_id=unknown&token={SECRET_TOKEN}")
    assert resp_authed.status_code == 404  # Passes auth, fails at unknown session ID


def test_protected_endpoints_matrix(client):
    headers = {"Authorization": f"Bearer {SECRET_TOKEN}"}
    endpoints = [
        ("/api/project", "GET"),
        ("/api/project/intelligence", "GET"),
        ("/api/tree", "GET"),
        ("/api/files", "GET"),
        ("/api/config/status", "GET"),
    ]
    for url, method in endpoints:
        # Unauthed
        res_unauthed = client.request(method, url)
        assert res_unauthed.status_code == 401, f"Endpoint {url} failed auth check"

        # Authed
        res_authed = client.request(method, url, headers=headers)
        assert res_authed.status_code in (200, 400), f"Endpoint {url} failed with auth"


def test_options_preflight_bypasses_auth(client):
    resp = client.options("/api/config", headers={"Origin": "http://example.com"})
    assert resp.status_code == 200


def test_lan_non_localhost_access_requires_token(tmp_path: Path):
    # App without NOVA_WEB_TOKEN configured
    settings = load_settings(project_root=tmp_path, env={})
    app = create_app(settings)
    # Simulate non-localhost client
    client_lan = TestClient(app, client=("192.168.1.50", 54321))

    resp = client_lan.get("/api/config")
    assert resp.status_code == 401
    assert "non-localhost" in resp.json()["detail"]


def test_localhost_without_token_allowed_when_no_token_set(tmp_path: Path):
    settings = load_settings(project_root=tmp_path, env={})
    app = create_app(settings)
    client_local = TestClient(app, client=("127.0.0.1", 54321))

    resp = client_local.get("/api/config")
    assert resp.status_code == 200
