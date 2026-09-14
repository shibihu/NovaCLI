"""Windows ConPTY (Pseudoconsole) implementation for interactive terminal sessions."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import io
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Mapping

from nova.core.pty.base import PTYSession as PTYSessionBase
from nova.core.runner import SCRUBBED_ENV_KEYS

logger = logging.getLogger(__name__)

# Windows API constants
CREATE_NEW_PROCESS_GROUP = 0x00000200
PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE = 0x00020016

# Handle ctypes
kernel32 = ctypes.windll.kernel32


class COORD(ctypes.Structure):
    """Console coordinate structure."""
    _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]


def _find_shell() -> str:
    """Find the Windows shell executable."""
    shell_env = os.environ.get("NOVA_TERMINAL_SHELL")
    if shell_env:
        # User explicitly configured a shell
        if os.path.exists(shell_env):
            return shell_env
        # Try resolving from PATH
        resolved = shutil.which(shell_env)
        if resolved:
            return resolved
        raise RuntimeError(f"Configured shell not found: {shell_env}")

    # Try COMSPEC (standard Windows shell path)
    comspec = os.environ.get("COMSPEC")
    if comspec and os.path.exists(comspec):
        return comspec

    # Fallback candidates
    candidates = [
        "C:\\Windows\\System32\\cmd.exe",
        "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise RuntimeError("No Windows shell found (cmd.exe or powershell.exe)")


class PTYSession(PTYSessionBase):
    """Windows ConPTY session managing one interactive shell process.
    
    This implementation uses subprocess with pipe-based I/O rather than
    the low-level ConPTY API, which provides better compatibility and
    simpler handle management.
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
        """Initialize Windows PTY session."""
        super().__init__(session_id, cwd, cols=cols, rows=rows, shell_path=shell_path, env=env)

        self.shell_path = shell_path or _find_shell()

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

        # Initialize output buffer
        self.output_buffer = io.BytesIO()
        self.output_lock = threading.Lock()

        # Spawn shell with pipes
        self._create_pty()

    def _create_pty(self) -> None:
        """Create and spawn shell process with pipes for I/O."""
        try:
            # Spawn the shell process with pipes
            self.process = subprocess.Popen(
                [self.shell_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self.cwd),
                env=self.env,
                creationflags=CREATE_NEW_PROCESS_GROUP,
                text=False,
                bufsize=0,
            )
            self._pid = self.process.pid

            # Start background thread to read process output
            self._reader_thread = threading.Thread(
                target=self._reader_thread_func, daemon=True
            )
            self._reader_thread.start()

        except Exception as e:
            logger.error(f"Windows PTY initialization failed: {e}")
            raise

    def _reader_thread_func(self) -> None:
        """Background thread that reads process output and stores it in buffer."""
        if not self.process or not self.process.stdout:
            return

        try:
            while self.process.poll() is None:
                try:
                    chunk = self.process.stdout.read(4096)
                    if chunk:
                        with self.output_lock:
                            self.output_buffer.write(chunk)
                    else:
                        break
                except (IOError, OSError):
                    break
        except Exception as e:
            logger.debug(f"Reader thread error: {e}")

    @property
    def pid(self) -> int | None:
        """Get the process ID of the shell."""
        return self._pid if hasattr(self, "_pid") else None

    @property
    def returncode(self) -> int | None:
        """Get the process exit code, or None if still running."""
        if hasattr(self, "process") and self.process:
            return self.process.returncode
        return None

    @property
    def is_alive(self) -> bool:
        """Check if the PTY session is still running."""
        if self._closed:
            return False
        if not hasattr(self, "process"):
            return False
        return self.process.poll() is None

    def resize(self, cols: int, rows: int) -> None:
        """Resize the terminal window dimensions.
        
        Note: On Windows, we track the dimensions but cannot actually
        resize the ConPTY without more complex API calls. This is a
        limitation of the pipe-based approach.
        """
        if self._closed:
            return
        
        cols = max(10, min(500, int(cols)))
        rows = max(5, min(200, int(rows)))
        
        self.cols = cols
        self.rows = rows
        
        # On Windows with pipes, we can't easily resize the underlying console
        logger.debug(f"Terminal resized to {cols}x{rows} (reflected in UI only)")

    def write(self, data: bytes | str) -> None:
        """Write user input to the shell's stdin."""
        if self._closed or not self.process or not self.process.stdin:
            return
        
        raw = data.encode("utf-8") if isinstance(data, str) else data
        if not raw:
            return

        try:
            self.process.stdin.write(raw)
            self.process.stdin.flush()
        except (OSError, ValueError, BrokenPipeError) as exc:
            logger.debug(f"Windows PTY write error on session {self.id}: {exc}")
            self.close()

    def read(self, max_bytes: int = 4096) -> bytes:
        """Read available output from the shell.
        
        This reads from our internal buffer which is filled by the
        background reader thread.
        """
        if self._closed:
            return b""

        try:
            with self.output_lock:
                # Get current buffer position
                current_pos = self.output_buffer.tell()
                self.output_buffer.seek(0, 2)  # Seek to end
                end_pos = self.output_buffer.tell()
                
                # If there's no new data, return empty
                if current_pos >= end_pos:
                    return b""
                
                # Read new data from current position to end
                self.output_buffer.seek(current_pos)
                data = self.output_buffer.read(max_bytes)
                
                # If we've read all the data, trim the buffer
                if self.output_buffer.tell() >= end_pos:
                    # Reset buffer for next read cycle
                    remaining = self.output_buffer.read()
                    self.output_buffer = io.BytesIO()
                    if remaining:
                        self.output_buffer.write(remaining)
                
                return data
        except Exception as e:
            logger.debug(f"Windows PTY read error on session {self.id}: {e}")
            return b""

    def close(self) -> None:
        """Terminate the shell process and clean up resources."""
        if self._closed:
            return
        self._closed = True

        # Close the shell process
        if hasattr(self, "process") and self.process:
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except Exception:
                    pass

            if self.process.stdout:
                try:
                    self.process.stdout.close()
                except Exception:
                    pass

            if self.process.poll() is None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    try:
                        self.process.kill()
                        self.process.wait(timeout=1.0)
                    except Exception as e:
                        logger.debug(f"Error killing shell process: {e}")
                except Exception as e:
                    logger.debug(f"Error terminating shell process: {e}")


__all__ = ["PTYSession"]

