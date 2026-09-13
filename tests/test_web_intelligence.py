"""Tests for Project Intelligence endpoints in Web API."""

from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient
from nova.web.app import create_app
from nova.config import load_settings


def test_web_project_intelligence_endpoints(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")
    settings = load_settings(project_root=tmp_path, env={"GROQ_API_KEY": "gsk_test"})
    app = create_app(settings)
    client = TestClient(app)

    # 1. GET /api/project/intelligence
    res_intel = client.get("/api/project/intelligence")
    assert res_intel.status_code == 200
    data = res_intel.json()
    assert data["name"] == tmp_path.name
    assert "gsk_test" not in str(data)

    # 2. POST /api/project/refresh
    res_refresh = client.post("/api/project/refresh")
    assert res_refresh.status_code == 200
    data_refresh = res_refresh.json()
    assert data_refresh["name"] == tmp_path.name

    # 3. GET /api/config/status
    res_config = client.get("/api/config/status")
    assert res_config.status_code == 200
    cfg = res_config.json()
    assert cfg["has_api_key"] is True
    assert "gsk_test" not in str(cfg)
