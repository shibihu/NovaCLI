"""PTY session manager for creating and managing PTY sessions across all platforms."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Mapping

from nova.core.pty.base import PTYSession


class PTYManager:
    """Manages active PTY sessions across all platforms.
    
    This manager is platform-agnostic and works with both Unix and Windows
    PTY implementations. It provides session creation, retrieval, and cleanup.
    """

    def __init__(self) -> None:
        """Initialize the PTY manager."""
        self.sessions: dict[str, PTYSession] = {}

    def create(
        self,
        cwd: str | Path,
        *,
        cols: int = 80,
        rows: int = 24,
        shell_path: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> PTYSession:
        """Create a new PTY session.
        
        Args:
            cwd: Working directory for the shell.
            cols: Terminal width in columns.
            rows: Terminal height in rows.
            shell_path: Path to shell executable (uses platform default if None).
            env: Environment variables for the shell.

        Returns:
            A new PTYSession instance.

        Raises:
            RuntimeError: If the PTY backend cannot be initialized.
        """
        session_id = uuid.uuid4().hex[:16]
        
        # Import the appropriate backend (deferred to avoid platform-specific imports at module load)
        from nova.core.pty import _get_pty_session_class
        pty_session_class = _get_pty_session_class()
        
        session = pty_session_class(
            session_id, Path(cwd), cols=cols, rows=rows, shell_path=shell_path, env=env
        )
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str) -> PTYSession | None:
        """Get an active PTY session by ID.
        
        Args:
            session_id: The session ID.

        Returns:
            The PTYSession if active, or None if not found or terminated.
        """
        session = self.sessions.get(session_id)
        if session and not session.is_alive:
            self.close(session_id)
            return None
        return session

    def close(self, session_id: str) -> None:
        """Close a PTY session and clean up resources.
        
        Args:
            session_id: The session ID to close.
        """
        session = self.sessions.pop(session_id, None)
        if session:
            session.close()

    def clear(self) -> None:
        """Close all active PTY sessions."""
        for session_id in list(self.sessions):
            self.close(session_id)


__all__ = ["PTYManager"]
