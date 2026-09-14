"""POSIX PTY (Pseudo-Terminal) manager for interactive web terminal sessions."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import signal
import struct
import subprocess
import termios
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from nova.core.runner import SCRUBBED_ENV_KEYS

logger = logging.getLogger(__name__)

_SHELL_CANDIDATES = (
    "/bin/bash",
    "/usr/bin/bash",
    "/bin/sh",
    "/system/bin/sh",
)


def _find_default_shell() -> str:
    """Find a valid interactive POSIX shell."""
    env_shell = os.environ.get("SHELL")
    if env_shell and os.path.isabs(env_shell) and os.access(env_shell, os.X_OK):
        return env_shell
    for candidate in _SHELL_CANDIDATES:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return "/bin/sh"


def set_pty_size(fd: int, cols: int, rows: int) -> None:
    """Set the window size of a PTY master file descriptor."""
    cols = max(10, min(500, int(cols)))
    rows = max(5, min(200, int(rows)))
    size = struct.pack("HHHH", rows, cols, 0, 0)
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, size)
    except (OSError, ValueError) as exc:
        logger.debug("Failed to set PTY size (%dx%d): %s", cols, rows, exc)


class PTYSession:
    """Manages one interactive PTY child shell process."""

    def __init__(
        self,
        session_id: str,
        cwd: Path,
        *,
        cols: int = 80,
        rows: int = 24,
        shell_path: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.id = session_id
        self.cwd = Path(cwd).resolve()
        self.cols = cols
        self.rows = rows
        self.created_at = time.time()
        self.shell_path = shell_path or _find_default_shell()

        # Build secret-scrubbed environment
        base_env = dict(env if env is not None else os.environ)
        for key in list(base_env):
            if key in SCRUBBED_ENV_KEYS or key.startswith("NOVA_SECRET"):
                base_env.pop(key, None)
        base_env.update(
            {
                "TERM": "xterm-256color",
                "COLORTERM": "truecolor",
                "NOVA_WORKSPACE": str(self.cwd),
            }
        )
        self.env = base_env

        # Create master and slave pseudo-terminals
        self.master_fd, slave_fd = os.openpty()
        set_pty_size(self.master_fd, self.cols, self.rows)

        # Set master_fd to non-blocking mode
        flags = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
        fcntl.fcntl(self.master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        # Spawn child shell attached to slave PTY
        try:
            self.process = subprocess.Popen(
                [self.shell_path],
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=str(self.cwd),
                env=self.env,
                preexec_fn=os.setsid,  # Start process in a new session group
                close_fds=True,
            )
        finally:
            os.close(slave_fd)

        self.pid = self.process.pid
        self._closed = False

    @property
    def is_alive(self) -> bool:
        return not self._closed and self.process.poll() is None

    def resize(self, cols: int, rows: int) -> None:
        """Resize the terminal window dimensions."""
        if self._closed:
            return
        self.cols = cols
        self.rows = rows
        set_pty_size(self.master_fd, cols, rows)

    def write(self, data: bytes | str) -> None:
        """Write user input bytes or text to the PTY stdin."""
        if self._closed:
            return
        raw = data.encode("utf-8") if isinstance(data, str) else data
        if not raw:
            return
        try:
            os.write(self.master_fd, raw)
        except (OSError, ValueError) as exc:
            logger.debug("PTY write error on session %s: %s", self.id, exc)

    def read(self, max_bytes: int = 4096) -> bytes:
        """Read output bytes from PTY master fd (non-blocking)."""
        if self._closed:
            return b""
        try:
            return os.read(self.master_fd, max_bytes)
        except (BlockingIOError, InterruptedError):
            return b""
        except OSError:
            # EIO is expected when child process exits
            self.close()
            return b""

    def close(self) -> None:
        """Terminate child process and close PTY file descriptors."""
        if self._closed:
            return
        self._closed = True

        try:
            os.close(self.master_fd)
        except OSError:
            pass

        if self.process.poll() is None:
            try:
                # Terminate process group to prevent zombies
                os.killpg(os.getpgid(self.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError, PermissionError):
                try:
                    self.process.terminate()
                except OSError:
                    pass

            # Wait briefly then force kill if needed
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError, PermissionError):
                    try:
                        self.process.kill()
                    except OSError:
                        pass
                try:
                    self.process.wait(timeout=1.0)
                except Exception:
                    pass


class PTYManager:
    """Manages active PTY sessions."""

    def __init__(self) -> None:
        self.sessions: dict[str, PTYSession] = {}

    def create(
        self,
        cwd: str | Path,
        *,
        cols: int = 80,
        rows: int = 24,
        env: Mapping[str, str] | None = None,
    ) -> PTYSession:
        session_id = uuid.uuid4().hex[:16]
        session = PTYSession(
            session_id, Path(cwd), cols=cols, rows=rows, env=env
        )
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str) -> PTYSession | None:
        session = self.sessions.get(session_id)
        if session and not session.is_alive:
            self.close(session_id)
            return None
        return session

    def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session:
            session.close()

    def clear(self) -> None:
        for session_id in list(self.sessions):
            self.close(session_id)


__all__ = ["PTYManager", "PTYSession", "set_pty_size"]
