"""Tests for WebSocket PTY terminal transport and authentication."""

from __future__ import annotations

import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from nova.config import load_settings
from nova.web.app import create_app

SECRET_TOKEN = "secret_web_token_789012"


@pytest.fixture
def auth_app(tmp_path: Path):
    settings = load_settings(
        project_root=tmp_path,
        env={"NOVA_WEB_TOKEN": SECRET_TOKEN, "GROQ_API_KEY": "gsk_test"},
    )
    return create_app(settings)


def test_websocket_terminal_unauthorized(auth_app):
    client = TestClient(auth_app)
    with client.websocket_connect("/ws/terminal") as websocket:
        msg = websocket.receive_json()
        assert msg.get("type") == "error"
        assert "Invalid or missing" in msg.get("message", "")


def test_websocket_terminal_authorized_and_executes_cmd(tmp_path: Path):
    settings = load_settings(
        project_root=tmp_path,
        env={"NOVA_WEB_TOKEN": SECRET_TOKEN, "GROQ_API_KEY": "gsk_test"},
    )
    app = create_app(settings)
    client = TestClient(app)

    headers = {"Authorization": f"Bearer {SECRET_TOKEN}"}
    with client.websocket_connect("/ws/terminal", headers=headers) as websocket:
        websocket.send_json({"type": "input", "data": "echo terminal_hello\n"})

        output_text = ""
        for _ in range(20):
            try:
                data = websocket.receive_json()
                if data.get("type") == "output":
                    output_text += data.get("data", "")
                    if "terminal_hello" in output_text:
                        break
            except Exception:
                break

        assert "terminal_hello" in output_text


def test_websocket_terminal_resize_message(tmp_path: Path):
    settings = load_settings(project_root=tmp_path, env={})
    app = create_app(settings)
    client = TestClient(app)

    with client.websocket_connect("/ws/terminal") as websocket:
        websocket.send_json({"type": "resize", "cols": 110, "rows": 42})
        websocket.send_json({"type": "input", "data": "stty size\n"})

        output_text = ""
        for _ in range(20):
            try:
                data = websocket.receive_json()
                if data.get("type") == "output":
                    output_text += data.get("data", "")
                    if "42 110" in output_text:
                        break
            except Exception:
                break

        assert "42 110" in output_text
