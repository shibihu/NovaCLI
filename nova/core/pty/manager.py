"""PTY session manager for creating and managing PTY sessions across all platforms."""

from __future__ import annotations

import secrets
import time

DEFAULT_GRACE_PERIOD_SECONDS = 300.0


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
        session_id = secrets.token_hex(16)
        
        # Import the appropriate backend (deferred to avoid platform-specific imports at module load)
        from nova.core.pty import _get_pty_session_class
        pty_session_class = _get_pty_session_class()
        
        session = pty_session_class(
            session_id, Path(cwd), cols=cols, rows=rows, shell_path=shell_path, env=env
        )
        self.sessions[session_id] = session
        return session

    def cleanup_orphans(self, grace_period_seconds: float = DEFAULT_GRACE_PERIOD_SECONDS) -> int:
        """Prune PTY sessions that have been disconnected longer than grace_period_seconds."""
        now = time.time()
        removed = 0
        for sid, session in list(self.sessions.items()):
            if not session.is_alive:
                self.close(sid)
                removed += 1
            elif session.disconnected_at is not None and (now - session.disconnected_at) > grace_period_seconds:
                self.close(sid)
                removed += 1
        return removed

    def get(self, session_id: str, project_root: str | Path | None = None) -> PTYSession | None:
        """Get an active PTY session by ID with workspace containment validation.
        
        Args:
            session_id: The session ID.
            project_root: Optional workspace root path to enforce boundary containment.

        Returns:
            The PTYSession if active, or None if not found, terminated, or out-of-bounds.
        """
        self.cleanup_orphans()
        session = self.sessions.get(session_id)
        if not session:
            return None
        if not session.is_alive:
            self.close(session_id)
            return None
        if project_root is not None:
            expected_root = Path(project_root).resolve()
            sess_root = session.cwd.resolve()
            if sess_root != expected_root and expected_root not in sess_root.parents:
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
