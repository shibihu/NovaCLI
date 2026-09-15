"""Platform-aware PTY abstraction layer.

This module provides a unified PTY interface for both Unix (Linux, Termux, WSL)
and Windows (ConPTY) systems. The appropriate backend is selected based on
the platform detected at import time.

This design ensures that:
1. Windows can import nova without loading Unix-only modules
2. Unix systems get the optimized POSIX PTY implementation
3. The rest of NovaCLI works identically on all platforms
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova.core.pty.base import PTYSession

__all__ = ["PTYSession", "PTYManager", "get_pty_backend_info"]


def _get_pty_session_class() -> type[PTYSession]:
    """Get the appropriate PTYSession class for the current platform.
    
    This function is called at runtime to select the platform-specific
    implementation. Unix-only imports only happen on Unix systems.
    
    Returns:
        The PTYSession class appropriate for the current platform.
    """
    if os.name == "nt":
        # Windows: use ConPTY implementation
        from nova.core.pty.windows import PTYSession
        return PTYSession
    else:
        # Unix-like (Linux, WSL, Termux): use POSIX PTY implementation
        from nova.core.pty.unix import PTYSession
        return PTYSession


# Create a factory function for PTYSession that defers class selection
def PTYSession(*args, **kwargs):  # noqa: N802
    """Factory function that creates a platform-appropriate PTYSession.
    
    This factory defers platform detection and class selection until the first
    PTYSession is created, avoiding Unix module imports on Windows at the time
    of 'import nova'.
    
    Args:
        *args: Positional arguments passed to the PTYSession constructor.
        **kwargs: Keyword arguments passed to the PTYSession constructor.
    
    Returns:
        A PTYSession instance appropriate for the current platform.
    """
    pty_class = _get_pty_session_class()
    return pty_class(*args, **kwargs)


def get_pty_backend_info() -> dict:
    """Get information about the active PTY backend.
    
    Returns:
        A dictionary with backend information including platform, name, shell, etc.
    """
    if os.name == "nt":
        import os as os_module
        from nova.core.pty.windows import _find_shell
        
        try:
            shell_path = _find_shell()
            pty_status = "Available"
            pty_error = None
        except RuntimeError as e:
            shell_path = None
            pty_status = "Error"
            pty_error = str(e)
        
        return {
            "platform": "Windows",
            "backend": "Windows ConPTY",
            "shell": shell_path,
            "pty_status": pty_status,
            "pty_error": pty_error,
        }
    else:
        import os as os_module
        from nova.core.pty.unix import _find_default_shell
        
        try:
            shell_path = _find_default_shell()
            pty_status = "Available"
            pty_error = None
        except Exception as e:
            shell_path = None
            pty_status = "Error"
            pty_error = str(e)
        
        return {
            "platform": os_module.uname().sysname,
            "backend": "Unix PTY",
            "shell": shell_path,
            "pty_status": pty_status,
            "pty_error": pty_error,
        }


# Import the manager - this is platform-agnostic
from nova.core.pty.manager import PTYManager

__all__.append("PTYManager")


def get_terminal_diagnostics(cwd: str | None = None, env: dict | None = None) -> dict:
    """Gather safe, secret-scrubbed diagnostics for command resolution and shell environment."""
    import shutil
    from pathlib import Path
    from nova.core.runner import SCRUBBED_ENV_KEYS

    effective_cwd = str(Path(cwd or os.getcwd()).resolve())
    effective_env = dict(env if env is not None else os.environ)

    # Scrub secrets
    scrubbed_upper = {k.upper() for k in SCRUBBED_ENV_KEYS}
    for k in list(effective_env.keys()):
        if k.upper() in scrubbed_upper or k.upper().startswith("NOVA_SECRET"):
            effective_env.pop(k, None)

    backend_info = get_pty_backend_info()
    shell_path = backend_info.get("shell")

    path_var = effective_env.get("PATH") or effective_env.get("Path") or os.environ.get("PATH")

    def resolve_tool(name: str) -> str | None:
        return shutil.which(name, path=path_var)

    return {
        "cwd": effective_cwd,
        "backend": backend_info.get("backend"),
        "platform": backend_info.get("platform"),
        "shell": shell_path,
        "COMSPEC": effective_env.get("COMSPEC") or os.environ.get("COMSPEC"),
        "PATHEXT": effective_env.get("PATHEXT") or os.environ.get("PATHEXT"),
        "USERPROFILE": effective_env.get("USERPROFILE") or os.environ.get("USERPROFILE"),
        "TEMP": effective_env.get("TEMP") or os.environ.get("TEMP"),
        "TMP": effective_env.get("TMP") or os.environ.get("TMP"),
        "PATH": path_var,
        "tool_resolutions": {
            "python": resolve_tool("python") or resolve_tool("python3"),
            "git": resolve_tool("git"),
            "node": resolve_tool("node"),
            "npm": resolve_tool("npm"),
            "ls": resolve_tool("ls"),
        },
    }

__all__.append('get_terminal_diagnostics')
