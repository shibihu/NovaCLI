"""Integration tests for Web API Git and Checkpoint endpoints."""

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


def test_api_checkpoints_crud_and_rollback(api_client, tmp_path: Path):
    (tmp_path / "main.py").write_text("v1", encoding="utf-8")

    # Create checkpoint
    res_create = api_client.post("/api/agent/checkpoint", json={"task_id": "test_task"})
    assert res_create.status_code == 200
    cp_id = res_create.json()["checkpoint"]["id"]

    # List checkpoints
    res_list = api_client.get("/api/agent/checkpoints")
    assert res_list.status_code == 200
    assert any(c["id"] == cp_id for c in res_list.json()["checkpoints"])

    # Modify file
    (tmp_path / "main.py").write_text("v2", encoding="utf-8")

    # Inspect
    res_inspect = api_client.get(f"/api/agent/checkpoint/{cp_id}")
    assert res_inspect.status_code == 200
    assert res_inspect.json()["total_changes"] == 1

    # Rollback
    res_rb = api_client.post(f"/api/agent/checkpoint/{cp_id}/rollback", json={"confirm": True})
    assert res_rb.status_code == 200
    assert "main.py" in res_rb.json()["restored"]
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == "v1"

    # Delete
    res_del = api_client.delete(f"/api/agent/checkpoint/{cp_id}")
    assert res_del.status_code == 200
