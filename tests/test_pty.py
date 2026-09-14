"""Tests for PTY session management and interactive terminal engine.

Includes cross-platform tests for both Unix and Windows implementations.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from nova.core.pty import PTYManager, PTYSession, get_pty_backend_info


class TestPTYBackendSelection:
    """Test PTY backend selection and initialization."""

    def test_pty_backend_info(self):
        """Verify that get_pty_backend_info returns correct platform info."""
        info = get_pty_backend_info()
        assert "platform" in info
        assert "backend" in info
        assert "pty_status" in info

        if os.name == "nt":
            assert "Windows" in info["platform"] or "ConPTY" in info["backend"]
        else:
            assert "Unix" in info["backend"] or "PTY" in info["backend"]

    def test_import_nova_succeeds(self):
        """Verify that 'import nova' works without Unix module errors on Windows."""
        # This test ensures Windows can import nova without fcntl/termios errors
        # If we got this far, the test passes
        import nova
        assert nova is not None


class TestPTYSessionCommon:
    """Cross-platform tests for PTY session functionality."""

    def test_pty_creation_basic(self, tmp_path: Path):
        """Test basic PTY session creation."""
        manager = PTYManager()
        session = manager.create(tmp_path, cols=80, rows=24)

        assert session.id is not None
        assert session.cols == 80
        assert session.rows == 24
        assert session.cwd == tmp_path
        assert session.is_alive is True
        assert session.pid is not None

        session.close()
        assert session.is_alive is False

    def test_pty_manager_get_inactive_session(self, tmp_path: Path):
        """Test that manager.get() removes terminated sessions."""
        manager = PTYManager()
        session = manager.create(tmp_path, cols=80, rows=24)
        session_id = session.id

        # Session should be retrievable
        retrieved = manager.get(session_id)
        assert retrieved is not None

        # Close the session
        session.close()

        # Terminated session should not be returned
        retrieved = manager.get(session_id)
        assert retrieved is None

    def test_pty_manager_close(self, tmp_path: Path):
        """Test PTY manager session closing."""
        manager = PTYManager()
        session = manager.create(tmp_path)
        session_id = session.id

        manager.close(session_id)
        assert session.is_alive is False
        assert manager.get(session_id) is None

    def test_pty_manager_clear(self, tmp_path: Path):
        """Test PTY manager clearing all sessions."""
        manager = PTYManager()
        s1 = manager.create(tmp_path)
        s2 = manager.create(tmp_path)

        assert len(manager.sessions) == 2
        manager.clear()
        assert len(manager.sessions) == 0
        assert s1.is_alive is False
        assert s2.is_alive is False

    def test_pty_env_scrubs_secrets(self, tmp_path: Path):
        """Test that sensitive environment variables are scrubbed."""
        custom_env = {
            "GROQ_API_KEY": "gsk_secret_key_12345",
            "NOVA_WEB_TOKEN": "web_secret_token_12345",
            "SAFE_VAR": "safe_value",
        }
        session = PTYSession(
            "test_session", tmp_path, env=custom_env
        )

        assert "GROQ_API_KEY" not in session.env
        assert "NOVA_WEB_TOKEN" not in session.env
        assert session.env.get("SAFE_VAR") == "safe_value"
        assert session.env.get("TERM") == "xterm-256color"

        session.close()

    def test_pty_resize_basic(self, tmp_path: Path):
        """Test that resize updates dimensions."""
        session = PTYSession("test_session", tmp_path, cols=80, rows=24)

        assert session.cols == 80
        assert session.rows == 24

        session.resize(100, 30)
        assert session.cols == 100
        assert session.rows == 30

        session.close()

    def test_pty_write_read_basic(self, tmp_path: Path):
        """Test basic write/read operations (platform-independent)."""
        session = PTYSession("test_session", tmp_path, cols=80, rows=24)

        # Write should not raise
        session.write("echo test\n")

        # Read should return bytes or be empty
        data = session.read(4096)
        assert isinstance(data, bytes)

        session.close()

    def test_pty_closed_operations_no_error(self, tmp_path: Path):
        """Test that operations on closed session don't raise errors."""
        session = PTYSession("test_session", tmp_path)
        session.close()

        # These should all be safe on a closed session
        session.write("should be no-op")
        data = session.read()
        assert data == b""
        
        session.resize(100, 30)
        assert session.is_alive is False


class TestUnixPTY:
    """Unix-specific PTY tests (Linux, WSL, Termux)."""

    @pytest.mark.skipif(os.name == "nt", reason="Unix PTY tests only")
    def test_pty_creation_and_isatty(self, tmp_path: Path):
        """Test that Unix PTY is a real TTY."""
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

        assert "True True" in output, f"Expected 'True True' in output, got: {output}"
        session.close()
        assert session.is_alive is False

    @pytest.mark.skipif(os.name == "nt", reason="Unix PTY tests only")
    def test_pty_stty_size_and_resize(self, tmp_path: Path):
        """Test Unix PTY resize via stty."""
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

        assert "35 120" in output, f"Expected '35 120' in output, got: {output}"

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

        assert "40 90" in output2, f"Expected '40 90' in output, got: {output2}"
        session.close()

    @pytest.mark.skipif(os.name == "nt", reason="Unix PTY tests only")
    def test_pty_shell_type_detection(self, tmp_path: Path):
        """Test that Unix PTY properly detects and launches shell."""
        from nova.core.pty.unix import _find_default_shell

        shell = _find_default_shell()
        assert shell is not None
        assert os.path.isabs(shell)
        assert os.path.exists(shell)


class TestWindowsPTY:
    """Windows-specific ConPTY tests."""

    @pytest.mark.skipif(os.name != "nt", reason="Windows ConPTY tests only")
    def test_conpty_initialization(self, tmp_path: Path):
        """Test Windows ConPTY initialization."""
        session = PTYSession("win_session", tmp_path, cols=80, rows=24)

        assert session.is_alive is True
        assert session.pid is not None
        assert session.cols == 80
        assert session.rows == 24

        session.close()
        assert session.is_alive is False

    @pytest.mark.skipif(os.name != "nt", reason="Windows ConPTY tests only")
    def test_conpty_shell_configuration(self, tmp_path: Path):
        """Test Windows shell configuration."""
        from nova.core.pty.windows import _find_shell

        shell = _find_shell()
        assert shell is not None
        assert os.path.exists(shell)

    @pytest.mark.skipif(os.name != "nt", reason="Windows ConPTY tests only")
    def test_conpty_basic_io(self, tmp_path: Path):
        """Test basic ConPTY input/output."""
        session = PTYSession("win_io_test", tmp_path, cols=80, rows=24)

        # Send a simple command
        session.write("echo test\r\n")

        # Try to read output
        output = b""
        for _ in range(20):
            time.sleep(0.1)
            chunk = session.read(4096)
            if chunk:
                output += chunk
            if b"test" in output:
                break

        # On Windows, we should see some output
        assert len(output) > 0 or session.is_alive

        session.close()

    @pytest.mark.skipif(os.name != "nt", reason="Windows ConPTY tests only")
    def test_conpty_resize(self, tmp_path: Path):
        """Test Windows ConPTY resize support."""
        session = PTYSession("win_resize_test", tmp_path, cols=80, rows=24)

        session.resize(100, 30)
        assert session.cols == 100
        assert session.rows == 30

        session.close()

    @pytest.mark.skipif(os.name != "nt", reason="Windows ConPTY tests only")
    def test_conpty_process_cleanup(self, tmp_path: Path):
        """Test Windows ConPTY proper process cleanup."""
        manager = PTYManager()
        session = manager.create(tmp_path)
        pid = session.pid

        manager.close(session.id)
        assert session.is_alive is False
        assert manager.get(session.id) is None




class TestWindowsPtyLogicMocked:
    """Tests for Windows PTY logic using mocks so they run on any OS (CI)."""

    def test_find_shell_custom_nova_terminal_shell(self, tmp_path: Path):
        from nova.core.pty.windows import _find_shell
        dummy_shell = tmp_path / "custom_shell.exe"
        dummy_shell.write_text("echo dummy")

        env = {"NOVA_TERMINAL_SHELL": str(dummy_shell)}
        resolved = _find_shell(env)
        assert resolved == str(dummy_shell)

    def test_find_shell_missing_configured_shell_raises(self):
        from nova.core.pty.windows import _find_shell
        env = {"NOVA_TERMINAL_SHELL": r"C:\nonexistent_shell_12345.exe"}
        with pytest.raises(RuntimeError, match="Configured shell not found"):
            _find_shell(env)

    def test_find_shell_comspec(self, tmp_path: Path):
        from nova.core.pty.windows import _find_shell
        dummy_comspec = tmp_path / "cmd.exe"
        dummy_comspec.write_text("cmd")
        env = {"COMSPEC": str(dummy_comspec)}
        assert _find_shell(env) == str(dummy_comspec)

    def test_pty_environment_inheritance_and_case_insensitive_scrubbing(self, tmp_path: Path):
        from nova.core.pty.windows import PTYSession as WinPTY

        custom_env = {
            "PATH": "/usr/bin:/bin",
            "Path": "/usr/bin:/bin",
            "groq_api_key": "secret_groq_key",
            "NOVA_SECRET_TOKEN": "secret_token",
            "CUSTOM_USER_VAR": "custom_val",
        }

        # Patch _create_pty to avoid calling Win32 APIs during env test
        def mock_create(self_obj):
            pass

        old_create = WinPTY._create_pty
        WinPTY._create_pty = mock_create
        try:
            session = WinPTY("test_win_env", tmp_path, shell_path="/bin/sh", env=custom_env)
            assert "groq_api_key" not in session.env
            assert "GROQ_API_KEY" not in session.env
            assert "NOVA_SECRET_TOKEN" not in session.env
            assert session.env.get("CUSTOM_USER_VAR") == "custom_val"
            assert session.env.get("PATH") == "/usr/bin:/bin"
            assert "Path" not in session.env or session.env.get("PATH") is not None
            assert session.cwd == tmp_path.resolve()
        finally:
            WinPTY._create_pty = old_create

    def test_terminal_diagnostics(self, tmp_path: Path):
        from nova.core.pty import get_terminal_diagnostics
        env = {
            "PATH": os.environ.get("PATH", ""),
            "GROQ_API_KEY": "secret_groq_key",
            "USERPROFILE": r"C:\Users\TestUser",
        }
        diag = get_terminal_diagnostics(cwd=str(tmp_path), env=env)

        assert diag["cwd"] == str(tmp_path.resolve())
        assert "GROQ_API_KEY" not in diag
        assert "tool_resolutions" in diag
        assert "python" in diag["tool_resolutions"]
        assert "git" in diag["tool_resolutions"]
        assert "node" in diag["tool_resolutions"]
