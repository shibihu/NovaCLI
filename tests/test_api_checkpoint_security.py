"""Security tests for Checkpoint and Rollback endpoints."""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from nova.config import load_settings
from nova.web.app import create_app


@pytest.fixture
def api_client(tmp_path: Path):
    settings = load_settings(
        project_root=tmp_path,
        env={"NOVA_WEB_TOKEN": "token-123"},
        user_config_path=tmp_path / "_no_user_config.json",
    )
    app = create_app(settings)
    client = TestClient(app)
    client.headers["Authorization"] = "Bearer token-123"
    return client


def test_checkpoint_id_path_traversal_rejected(api_client):
    res_get = api_client.get("/api/agent/checkpoint/../../etc/passwd")
    assert res_get.status_code in (400, 404)

    res_rb = api_client.post("/api/agent/checkpoint/../../etc/passwd/rollback")
    assert res_rb.status_code in (400, 404)


def test_checkpoint_never_snapshots_sensitive_files(api_client, tmp_path: Path):
    (tmp_path / ".env").write_text("SECRET=gsk_secret_123", encoding="utf-8")
    (tmp_path / "safe.py").write_text("print('ok')", encoding="utf-8")

    res_create = api_client.post("/api/agent/checkpoint", json={"task_id": "secret_task"})
    assert res_create.status_code == 200
    cp_id = res_create.json()["checkpoint"]["id"]

    cp_dir = tmp_path / ".nova" / "checkpoints" / cp_id
    snap_env = cp_dir / "snapshots" / ".env"
    assert not snap_env.exists(), ".env must never be snapshotted in checkpoint storage"
