"""Tests for persistent chat sessions storage, server restart persistence, and REST APIs."""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from nova.config import load_settings
from nova.web.app import create_app
from nova.core.sessions import generate_chat_title, SessionStorageManager
from nova.workspace.files import Workspace


def _make_client(tmp_path: Path) -> TestClient:
    settings = load_settings(
        project_root=tmp_path,
        env={"NOVA_WEB_TOKEN": "", "GROQ_API_KEY": "gsk_test"},
    )
    return TestClient(create_app(settings))


def test_title_generator():
    assert generate_chat_title("Fix Ollama tool calling") == "Fix Ollama tool calling"
    assert generate_chat_title("   ```python\nprint('hello')\n```   ") == "Print('hello')"
    assert generate_chat_title("") == "New Chat"
    long_task = "Fix Ollama tool calling with invalid json args and update system prompt"
    title = generate_chat_title(long_task)
    assert len(title) > 30  # Proves 20-char premature truncation is fixed!
    assert "Fix Ollama tool calling" in title


def test_session_create_list_get(tmp_path: Path):
    client = _make_client(tmp_path)

    # Create session
    res = client.post("/api/agent/sessions", json={"task": "Add checkpoint redo support"})
    assert res.status_code == 200
    data = res.json()
    assert data["ok"] is True
    sess = data["session"]
    sid = sess["id"]
    assert sess["title"] == "Add checkpoint redo support"

    # List sessions
    res_list = client.get("/api/agent/sessions")
    assert res_list.status_code == 200
    sessions = res_list.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["id"] == sid

    # Get session
    res_get = client.get(f"/api/agent/sessions/{sid}")
    assert res_get.status_code == 200
    sess_detail = res_get.json()["session"]
    assert sess_detail["id"] == sid
    assert sess_detail["title"] == "Add checkpoint redo support"


def test_session_persists_across_server_restart(tmp_path: Path):
    # App instance 1
    client1 = _make_client(tmp_path)

    res1 = client1.post("/api/agent/sessions", json={"task": "Fix Gemini signatures"})
    sid = res1.json()["session"]["id"]

    # Save messages to session via manager
    ws = Workspace(tmp_path)
    mgr = SessionStorageManager(ws)
    s_obj = mgr.get(sid)
    assert s_obj is not None
    s_obj.messages.append({"role": "user", "content": "Fix Gemini signatures"})
    s_obj.messages.append({"role": "assistant", "content": "I have updated gemini.py"})
    mgr.save(s_obj)

    # Simulate server restart by instantiating new client/app instance 2
    client2 = _make_client(tmp_path)

    # Verify session list on restarted server
    res_list = client2.get("/api/agent/sessions")
    assert res_list.status_code == 200
    sessions = res_list.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["id"] == sid
    assert sessions[0]["title"] == "Fix Gemini signatures"

    # Verify session message history loaded on restarted server
    res_detail = client2.get(f"/api/agent/sessions/{sid}")
    assert res_detail.status_code == 200
    messages = res_detail.json()["session"]["messages"]
    assert len(messages) == 2
    assert messages[1]["content"] == "I have updated gemini.py"


def test_session_rename_delete(tmp_path: Path):
    client = _make_client(tmp_path)

    # Create session
    res = client.post("/api/agent/sessions", json={"task": "Initial task"})
    sid = res.json()["session"]["id"]

    # Rename session
    res_rename = client.patch(f"/api/agent/sessions/{sid}", json={"title": "Updated Title"})
    assert res_rename.status_code == 200
    assert res_rename.json()["session"]["title"] == "Updated Title"

    # Verify rename persisted
    res_get = client.get(f"/api/agent/sessions/{sid}")
    assert res_get.json()["session"]["title"] == "Updated Title"

    # Delete session
    res_del = client.delete(f"/api/agent/sessions/{sid}")
    assert res_del.status_code == 200
    assert res_del.json()["ok"] is True

    # Verify session is deleted
    res_get_deleted = client.get(f"/api/agent/sessions/{sid}")
    assert res_get_deleted.status_code == 404


def test_session_workspace_isolation(tmp_path: Path):
    client = _make_client(tmp_path)

    # Non-existent session
    res = client.get("/api/agent/sessions/nonexistent_session_999")
    assert res.status_code == 404

    # Invalid session ID with path traversal
    res_traversal = client.get("/api/agent/sessions/.._.._etc_passwd")
    assert res_traversal.status_code == 404
