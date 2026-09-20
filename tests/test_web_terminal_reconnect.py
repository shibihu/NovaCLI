"""Tests for terminal WebSocket auto-reconnect backend logic and security boundaries."""

import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import nova.web.terminal as terminal_module
from nova.config import load_settings
from nova.web.app import create_app


def _make_client(tmp_path: Path) -> TestClient:
    settings = load_settings(
        project_root=tmp_path,
        env={"NOVA_WEB_TOKEN": "", "GROQ_API_KEY": "gsk_test"},
    )
    return TestClient(create_app(settings))


def test_terminal_reconnect_reuses_session(tmp_path: Path):
    terminal_module.pty_manager.clear()
    client = _make_client(tmp_path)

    # 1. First connection creates a new session
    session_id = None
    with client.websocket_connect("/ws/terminal") as ws1:
        msg = ws1.receive_json()
        assert msg.get("type") == "connected"
        session_id = msg.get("session_id")
        assert session_id is not None

        # Ping to verify active connection
        ws1.send_json({"type": "ping"})
        msg_pong = ws1.receive_json()
        assert msg_pong.get("type") == "pong"

    # 2. Check that the session is still alive in pty_manager after ws1 disconnect
    alive_session = terminal_module.pty_manager.get(session_id, project_root=tmp_path)
    assert alive_session is not None
    assert alive_session.is_alive
    assert alive_session.disconnected_at is not None

    # 3. Reconnect passing session_id
    with client.websocket_connect(f"/ws/terminal?session_id={session_id}") as ws2:
        msg = ws2.receive_json()
        assert msg.get("type") == "connected"
        assert msg.get("session_id") == session_id

        # Send command to confirm reconnected session works
        ws2.send_json({"type": "input", "data": "echo reconnect_ok\r"})
        output_text = ""
        for _ in range(20):
            ws2.send_json({"type": "ping"})
            m = ws2.receive_json()
            if m.get("type") == "output":
                output_text += m.get("data", "")
                if "reconnect_ok" in output_text:
                    break
            elif m.get("type") == "pong":
                time.sleep(0.05)

        assert "reconnect_ok" in output_text

    # Cleanup session
    terminal_module.pty_manager.close(session_id)


def test_terminal_reconnect_cross_workspace_access_denied(tmp_path: Path):
    terminal_module.pty_manager.clear()
    client = _make_client(tmp_path)

    # Create session in tmp_path
    with client.websocket_connect("/ws/terminal") as ws:
        msg = ws.receive_json()
        sid = msg.get("session_id")

    # Attempt to access session from a different workspace root
    other_dir = tmp_path / "other_workspace"
    other_dir.mkdir(parents=True, exist_ok=True)

    denied_session = terminal_module.pty_manager.get(sid, project_root=other_dir)
    assert denied_session is None

    # Cleanup
    terminal_module.pty_manager.close(sid)


def test_terminal_orphan_grace_period_cleanup(tmp_path: Path):
    terminal_module.pty_manager.clear()
    client = _make_client(tmp_path)

    with client.websocket_connect("/ws/terminal") as ws:
        msg = ws.receive_json()
        sid = msg.get("session_id")

    sess = terminal_module.pty_manager.sessions.get(sid)
    assert sess is not None
    # Simulate orphan exceeding grace period
    sess.disconnected_at = time.time() - 400.0

    # Cleanup orphans with 300s threshold
    removed = terminal_module.pty_manager.cleanup_orphans(grace_period_seconds=300.0)
    assert removed == 1
    assert sid not in terminal_module.pty_manager.sessions


def test_terminal_reconnect_nonexistent_session_creates_new(tmp_path: Path):
    client = _make_client(tmp_path)

    with client.websocket_connect("/ws/terminal?session_id=fake_nonexistent_1234") as ws:
        msg = ws.receive_json()
        assert msg.get("type") == "connected"
        new_id = msg.get("session_id")
        assert new_id is not None
        assert new_id != "fake_nonexistent_1234"

    # Cleanup
    terminal_module.pty_manager.close(new_id)


def test_terminal_output_buffer_replay_on_reconnect(tmp_path: Path):
    terminal_module.pty_manager.clear()
    client = _make_client(tmp_path)

    # 1. Connect and write output
    session_id = None
    with client.websocket_connect("/ws/terminal") as ws1:
        msg = ws1.receive_json()
        assert msg.get("type") == "connected"
        session_id = msg.get("session_id")

        ws1.send_json({"type": "input", "data": "echo replay_marker_12345\r"})
        output_text = ""
        for _ in range(20):
            ws1.send_json({"type": "ping"})
            m = ws1.receive_json()
            if m.get("type") == "output":
                output_text += m.get("data", "")
                if "replay_marker_12345" in output_text:
                    break
            elif m.get("type") == "pong":
                time.sleep(0.05)
        assert "replay_marker_12345" in output_text

    # 2. Reconnect and verify buffered output is replayed
    with client.websocket_connect(f"/ws/terminal?session_id={session_id}") as ws2:
        m1 = ws2.receive_json()
        assert m1.get("type") == "connected"

        # Next frame is buffered output replay
        m2 = ws2.receive_json()
        assert m2.get("type") == "output"
        assert "replay_marker_12345" in m2.get("data", "")

    terminal_module.pty_manager.close(session_id)
