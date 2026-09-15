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
