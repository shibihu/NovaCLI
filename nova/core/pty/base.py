"""Abstract base class for platform-specific PTY implementations."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping


class PTYSession(ABC):
    """Abstract base class for PTY session implementations.
    
    Subclasses must implement platform-specific PTY creation and I/O.
    The interface is identical across Unix and Windows backends.
    """

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
        """Initialize PTY session with common attributes.
        
        Args:
            session_id: Unique session identifier.
            cwd: Working directory for shell.
            cols: Terminal width in columns.
            rows: Terminal height in rows.
            shell_path: Path to shell executable (platform-specific default if None).
            env: Environment variables for shell (secret keys are scrubbed).
        """
        self.id = session_id
        self.cwd = Path(cwd).resolve()
        self.cols = cols
        self.rows = rows
        self.created_at = time.time()
        self.shell_path = shell_path
        self.env = dict(env) if env else {}
        self._closed = False

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """Check if PTY session and child process are still running."""

    @property
    @abstractmethod
    def pid(self) -> int | None:
        """Get the process ID of the shell, or None if not running."""

    @property
    @abstractmethod
    def returncode(self) -> int | None:
        """Get the process exit code, or None if still running."""

    @abstractmethod
    def resize(self, cols: int, rows: int) -> None:
        """Resize the terminal window to specified dimensions."""

    @abstractmethod
    def write(self, data: bytes | str) -> None:
        """Write user input to the PTY stdin."""

    @abstractmethod
    def read(self, max_bytes: int = 4096) -> bytes:
        """Read output from the PTY (non-blocking)."""

    @abstractmethod
    def close(self) -> None:
        """Terminate the child process and clean up resources."""
