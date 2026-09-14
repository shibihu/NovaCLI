"""Tests for PTY session management and interactive terminal engine."""

from __future__ import annotations

import time
from pathlib import Path
import pytest

from nova.core.pty import PTYManager, PTYSession


def test_pty_creation_and_isatty(tmp_path: Path):
    manager = PTYManager()
    session = manager.create(tmp_path, cols=80, rows=24)
    assert session.is_alive is True

    # Send command to check TTY status
    session.write('python3 -c "import sys; print(sys.stdin.isatty(), sys.stdout.isatty())"\n')

    output = ""
    for _ in range(20):
        time.sleep(0.1)
        chunk = session.read().decode("utf-8", errors="replace")
        output += chunk
        if "True True" in output:
            break

    assert "True True" in output
    session.close()
    assert session.is_alive is False


def test_pty_stty_size_and_resize(tmp_path: Path):
    session = PTYSession("s1", tmp_path, cols=120, rows=35)
    assert session.cols == 120
    assert session.rows == 35

    # Check stty size
    session.write("stty size\n")
    output = ""
    for _ in range(20):
        time.sleep(0.1)
        chunk = session.read().decode("utf-8", errors="replace")
        output += chunk
        if "35 120" in output:
            break

    assert "35 120" in output

    # Resize PTY
    session.resize(cols=90, rows=40)
    assert session.cols == 90
    assert session.rows == 40

    session.write("stty size\n")
    output2 = ""
    for _ in range(20):
        time.sleep(0.1)
        chunk = session.read().decode("utf-8", errors="replace")
        output2 += chunk
        if "40 90" in output2:
            break

    assert "40 90" in output2
    session.close()


def test_pty_env_scrubs_secrets(tmp_path: Path):
    custom_env = {
        "GROQ_API_KEY": "gsk_secret_key_12345",
        "NOVA_WEB_TOKEN": "web_secret_token_12345",
        "SAFE_VAR": "safe_value",
    }
    session = PTYSession("s2", tmp_path, env=custom_env)

    assert "GROQ_API_KEY" not in session.env
    assert "NOVA_WEB_TOKEN" not in session.env
    assert session.env.get("SAFE_VAR") == "safe_value"
    assert session.env.get("TERM") == "xterm-256color"

    session.close()


def test_pty_process_cleanup_on_close(tmp_path: Path):
    manager = PTYManager()
    session = manager.create(tmp_path)
    pid = session.pid

    manager.close(session.id)
    assert session.is_alive is False
    assert manager.get(session.id) is None
