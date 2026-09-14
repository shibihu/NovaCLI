"""Unix/POSIX PTY (Pseudo-Terminal) implementation for Linux, WSL, and Termux."""

from __future__ import annotations

import fcntl
import logging
import os
import signal
import struct
import subprocess
import termios
from pathlib import Path
from typing import Mapping

from nova.core.pty.base import PTYSession as PTYSessionBase
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


class PTYSession(PTYSessionBase):
    """Unix/POSIX PTY session managing one interactive shell process."""

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
        """Initialize Unix PTY session."""
        super().__init__(session_id, cwd, cols=cols, rows=rows, shell_path=shell_path, env=env)
        
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
        shell_cmd = [self.shell_path]
        if self.shell_path.endswith(("bash", "sh", "zsh")) and "-i" not in shell_cmd:
            shell_cmd.append("-i")

        try:
            self.process = subprocess.Popen(
                shell_cmd,
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

        self._pid = self.process.pid

    @property
    def pid(self) -> int | None:
        """Get the process ID of the shell."""
        return self._pid

    @property
    def returncode(self) -> int | None:
        """Get the process exit code, or None if still running."""
        return self.process.returncode

    @property
    def is_alive(self) -> bool:
        """Check if the PTY session is still running."""
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
                os.killpg(os.getpgid(self._pid), signal.SIGTERM)
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
                    os.killpg(os.getpgid(self._pid), signal.SIGKILL)
                except (ProcessLookupError, OSError, PermissionError):
                    try:
                        self.process.kill()
                    except OSError:
                        pass
                try:
                    self.process.wait(timeout=1.0)
                except Exception:
                    pass


__all__ = ["PTYSession", "set_pty_size"]
