"""Tests for WebSocket PTY terminal transport and authentication.

Every read loop is bounded: the client polls the server with ``ping`` frames, so
a broken terminal fails the test instead of hanging the suite (the previous
version of this file could deadlock forever). Output assertions stay loose
because a shell is an external process — but the echo assertion is precise
enough to catch the "typing produces no characters" regression.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

import nova.web.terminal as terminal_module
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


def _make_client(tmp_path: Path) -> TestClient:
    settings = load_settings(
        project_root=tmp_path,
        env={"NOVA_WEB_TOKEN": "", "GROQ_API_KEY": "gsk_test"},
    )
    return TestClient(create_app(settings))


def _drain(ws, predicate=None, attempts: int = 30, delay: float = 0.1) -> tuple[str, bool]:
    """Collect output frames, polling with ``ping`` so the wait is bounded.

    Returns ``(output_text, matched)``. ``matched`` is True when *predicate*
    matched, when the shell reported ``exit``, or — for a predicate-less call —
    after one clean ping/pong round trip. Any ``error`` frame fails the test.
    """
    text = ""

    for _ in range(attempts):
        if predicate is not None and predicate(text):
            return text, True
        time.sleep(delay)
        ws.send_json({"type": "ping"})
        for _ in range(500):
            try:
                msg = ws.receive_json()
            except WebSocketDisconnect:
                return text, predicate is None or bool(predicate(text))
            mtype = msg.get("type")
            if mtype == "output":
                text += msg.get("data", "")
                if predicate is not None and predicate(text):
                    return text, True
            elif mtype == "error":
                raise AssertionError(f"server error frame: {msg.get('message')!r}")
            elif mtype == "exit":
                return text, predicate is None or bool(predicate(text))
            elif mtype == "pong":
                break

    return text, predicate is None or bool(predicate(text))


def _wait_for_exit(ws, limit: int = 500) -> bool:
    """Read frames until the server reports the shell's exit code.

    The endpoint closes the socket as soon as the shell exits, so the loop is
    bounded by that close (which surfaces as ``WebSocketDisconnect``). No further
    messages are sent, so a closed socket can never block the test.
    """
    for _ in range(limit):
        try:
            msg = ws.receive_json()
        except WebSocketDisconnect:
            return False
        mtype = msg.get("type")
        if mtype == "exit":
            return isinstance(msg.get("code"), int)
        if mtype == "error":
            raise AssertionError(f"server error frame: {msg.get('message')!r}")
    return False


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_websocket_terminal_unauthorized(auth_app):
    client = TestClient(auth_app)
    with client.websocket_connect("/ws/terminal") as websocket:
        msg = websocket.receive_json()
        assert msg.get("type") == "error"
        assert "Invalid or missing" in msg.get("message", "")


def test_websocket_terminal_authorized_header(auth_app):
    """A valid Authorization header is accepted instead of being rejected."""
    client = TestClient(auth_app)
    headers = {"Authorization": f"Bearer {SECRET_TOKEN}"}
    with client.websocket_connect("/ws/terminal", headers=headers) as websocket:
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json().get("type") == "pong"


# ---------------------------------------------------------------------------
# Core transport
# ---------------------------------------------------------------------------


def test_websocket_terminal_executes_input(tmp_path: Path):
    client = _make_client(tmp_path)

    with client.websocket_connect("/ws/terminal") as websocket:
        websocket.send_json({"type": "input", "data": "echo terminal_hello\r"})
        output_text, matched = _drain(
            websocket, lambda text: "terminal_hello" in text
        )

    assert matched, f"expected command output, got: {output_text!r}"


def test_websocket_terminal_echoes_typed_characters(tmp_path: Path):
    """Typed keys must be echoed immediately, before Enter is pressed.

    With the old pipe-only Windows backend nothing appeared until the whole line
    was submitted, which is the bug this guards against.
    """
    client = _make_client(tmp_path)

    with client.websocket_connect("/ws/terminal") as websocket:
        # Drain the shell banner/prompt so the echo assertion is unambiguous.
        _drain(websocket)

        websocket.send_json({"type": "input", "data": "nova_marker"})
        output_text, matched = _drain(
            websocket, lambda text: "nova_marker" in text
        )

        # Clear the pending line so the shell does not run it.
        websocket.send_json({"type": "input", "data": "\x03"})

    assert "nova" in output_text and "marker" in output_text, f"typed characters were not echoed back, got: {output_text!r}"


def test_websocket_terminal_handles_resize(tmp_path: Path):
    """A resize message is accepted and does not error (all platforms)."""
    client = _make_client(tmp_path)

    with client.websocket_connect("/ws/terminal") as websocket:
        websocket.send_json({"type": "resize", "cols": 110, "rows": 42})
        _, matched = _drain(websocket)

    assert matched, "server stopped responding after a resize message"


@pytest.mark.skipif(os.name == "nt", reason="'stty' is POSIX-only")
def test_websocket_terminal_resize_reaches_shell(tmp_path: Path):
    """On a POSIX shell the resize must actually reach the terminal window."""
    client = _make_client(tmp_path)

    with client.websocket_connect("/ws/terminal") as websocket:
        websocket.send_json({"type": "resize", "cols": 110, "rows": 42})
        websocket.send_json({"type": "input", "data": "stty size\r"})
        output_text, matched = _drain(websocket, lambda text: "42 110" in text)

    assert matched, f"expected '42 110' from stty, got: {output_text!r}"


def test_websocket_terminal_ping_pong(tmp_path: Path):
    client = _make_client(tmp_path)

    with client.websocket_connect("/ws/terminal") as websocket:
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json().get("type") == "pong"


def test_websocket_terminal_malformed_message_is_treated_as_input(tmp_path: Path):
    """Non-JSON text is forwarded as literal keystrokes, not a crash."""
    client = _make_client(tmp_path)

    with client.websocket_connect("/ws/terminal") as websocket:
        websocket.send_text("this is not json")
        _, matched = _drain(websocket)

    assert matched, "server stopped responding after a malformed frame"


# ---------------------------------------------------------------------------
# Failure handling and lifecycle
# ---------------------------------------------------------------------------


def test_websocket_terminal_reports_pty_creation_failure(tmp_path: Path, monkeypatch):
    client = _make_client(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated PTY failure")

    monkeypatch.setattr(terminal_module.pty_manager, "create", boom)

    with client.websocket_connect("/ws/terminal") as websocket:
        msg = websocket.receive_json()

    assert msg.get("type") == "error"
    assert "Failed to start terminal" in msg.get("message", "")
    assert "simulated PTY failure" in msg.get("message", "")


def test_websocket_terminal_redacts_secrets_in_errors(tmp_path: Path, monkeypatch):
    """A failure message must never echo configured secret values."""
    secret = "super_secret_token_value"
    settings = load_settings(
        project_root=tmp_path,
        env={"NOVA_WEB_TOKEN": secret, "GROQ_API_KEY": "gsk_test"},
    )
    client = TestClient(create_app(settings))

    def boom(*args, **kwargs):
        raise RuntimeError(f"boom for token {secret}")

    monkeypatch.setattr(terminal_module.pty_manager, "create", boom)

    headers = {"Authorization": f"Bearer {secret}"}
    with client.websocket_connect("/ws/terminal", headers=headers) as websocket:
        msg = websocket.receive_json()

    assert msg.get("type") == "error"
    assert secret not in msg.get("message", "")
    assert "Failed to start terminal" in msg.get("message", "")


def test_websocket_terminal_cleans_up_session_on_disconnect(tmp_path: Path):
    terminal_module.pty_manager.clear()
    client = _make_client(tmp_path)

    with client.websocket_connect("/ws/terminal") as websocket:
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json().get("type") == "pong"
        assert terminal_module.pty_manager.sessions

    # The server closes the PTY in its finally block once the socket drops.
    for _ in range(50):
        if not terminal_module.pty_manager.sessions:
            break
        time.sleep(0.05)

    assert terminal_module.pty_manager.sessions == {}


def test_websocket_terminal_reports_exit_when_shell_ends(tmp_path: Path):
    client = _make_client(tmp_path)

    with client.websocket_connect("/ws/terminal") as websocket:
        websocket.send_json({"type": "input", "data": "exit\r"})
        assert _wait_for_exit(websocket), "server never reported the shell exit"



def test_websocket_terminal_uses_dotenv_nova_terminal_shell(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("NOVA_TERMINAL_SHELL=custom_shell_from_dotenv\nGROQ_API_KEY=gsk_test\n", encoding="utf-8")

    captured_args = {}

    def fake_create(cwd, cols=80, rows=24, shell_path=None, env=None):
        captured_args["cwd"] = cwd
        captured_args["env"] = env
        raise RuntimeError("simulated creation stop")

    monkeypatch.setattr(terminal_module.pty_manager, "create", fake_create)

    settings = load_settings(project_root=tmp_path, env={})
    app = create_app(settings)
    client = TestClient(app)

    with client.websocket_connect("/ws/terminal") as websocket:
        msg = websocket.receive_json()
        assert msg.get("type") == "error"

    assert captured_args.get("env", {}).get("NOVA_TERMINAL_SHELL") == "custom_shell_from_dotenv"
