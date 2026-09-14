"""Windows ConPTY (Pseudoconsole) implementation for interactive terminal sessions.

This backend attaches the shell to a real Windows pseudoconsole via the ConPTY
API (``CreatePseudoConsole``). Using a pseudoconsole — rather than anonymous
pipes — is what makes the web terminal genuinely interactive on Windows:

* the console line discipline echoes typed characters as you type,
* backspace / arrow keys / Tab / Ctrl+C / Ctrl+D work,
* ANSI/VT escape sequences emitted by the shell are preserved,
* interactive programs (``python``, ``more``, sub-shells, ...) can be driven.

The API is bound directly with :mod:`ctypes`, so no extra dependency (and no C
toolchain) is required. Windows 10 1809+ is required; on older builds a clear
``RuntimeError`` is raised so the WebSocket layer can report it.
"""

from __future__ import annotations

import collections
import ctypes
import logging
import os
import shutil
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Mapping

from nova.core.pty.base import PTYSession as PTYSessionBase
from nova.core.runner import SCRUBBED_ENV_KEYS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Win32 constants
# ---------------------------------------------------------------------------

EXTENDED_STARTUPINFO_PRESENT = 0x00080000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE = 0x00020016

# Without STARTF_USESTDHANDLES, CreateProcess duplicates the *parent's* standard
# handles into a console child even when handle inheritance is disabled. That
# makes the shell talk to the parent console instead of the pseudoconsole, which
# is exactly the "typing produces no output" failure mode. Setting the flag with
# NULL std handles forces the child onto the ConPTY.
# See microsoft/terminal discussion #15814.
STARTF_USESTDHANDLES = 0x00000100
STILL_ACTIVE = 259
WAIT_TIMEOUT = 0x00000102
WAIT_OBJECT_0 = 0x00000000


# ---------------------------------------------------------------------------
# Win32 structures
# ---------------------------------------------------------------------------


class COORD(ctypes.Structure):
    """Console coordinate (used for the pseudoconsole size)."""

    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class SECURITY_ATTRIBUTES(ctypes.Structure):
    """Win32 ``SECURITY_ATTRIBUTES`` for inheritable pipe handles."""

    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    ]


class STARTUPINFOW(ctypes.Structure):
    """Win32 ``STARTUPINFOW``."""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class STARTUPINFOEXW(ctypes.Structure):
    """Win32 ``STARTUPINFOEXW`` (``STARTUPINFOW`` plus an attribute list)."""

    _fields_ = [("StartupInfo", STARTUPINFOW), ("lpAttributeList", wintypes.LPVOID)]


class PROCESS_INFORMATION(ctypes.Structure):
    """Win32 ``PROCESS_INFORMATION``."""

    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


_kernel32 = None
_api_lock = threading.Lock()


def _kernel32_api():
    """Load and bind the ConPTY/kernel32 entry points exactly once.

    Raises:
        RuntimeError: If the ConPTY API is unavailable (pre-1809 Windows).
    """
    global _kernel32
    with _api_lock:
        if _kernel32 is not None:
            return _kernel32

        if os.name != "nt":
            raise RuntimeError("ConPTY is only available on Windows.")

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        try:
            kernel32.CreatePseudoConsole
            kernel32.ResizePseudoConsole
            kernel32.ClosePseudoConsole
        except AttributeError as exc:  # pragma: no cover - old Windows only
            raise RuntimeError(
                "ConPTY is unavailable on this Windows version "
                "(Windows 10 1809 or newer is required)."
            ) from exc

        kernel32.CreatePseudoConsole.argtypes = [
            COORD,
            wintypes.HANDLE,
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        kernel32.CreatePseudoConsole.restype = ctypes.c_long  # HRESULT

        kernel32.ResizePseudoConsole.argtypes = [wintypes.HANDLE, COORD]
        kernel32.ResizePseudoConsole.restype = ctypes.c_long  # HRESULT

        kernel32.ClosePseudoConsole.argtypes = [wintypes.HANDLE]
        kernel32.ClosePseudoConsole.restype = None

        kernel32.CreatePipe.argtypes = [
            ctypes.POINTER(wintypes.HANDLE),
            ctypes.POINTER(wintypes.HANDLE),
            ctypes.POINTER(SECURITY_ATTRIBUTES),
            wintypes.DWORD,
        ]
        kernel32.CreatePipe.restype = wintypes.BOOL

        kernel32.InitializeProcThreadAttributeList.argtypes = [
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL

        kernel32.UpdateProcThreadAttribute.argtypes = [
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.c_size_t,
            wintypes.LPVOID,
            ctypes.c_size_t,
            wintypes.LPVOID,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL

        kernel32.DeleteProcThreadAttributeList.argtypes = [wintypes.LPVOID]
        kernel32.DeleteProcThreadAttributeList.restype = None

        kernel32.CreateProcessW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPCWSTR,
            wintypes.LPVOID,
            ctypes.POINTER(PROCESS_INFORMATION),
        ]
        kernel32.CreateProcessW.restype = wintypes.BOOL

        for name in ("ReadFile", "WriteFile"):
            func = getattr(kernel32, name)
            func.argtypes = [
                wintypes.HANDLE,
                wintypes.LPVOID,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                wintypes.LPVOID,
            ]
            func.restype = wintypes.BOOL

        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD

        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL

        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL

        _kernel32 = kernel32
        return _kernel32


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


def _build_env_block(env: Mapping[str, str]) -> ctypes.Array:
    """Serialise *env* into the UTF-16 environment block Windows expects."""
    entries = "".join(f"{key}={value}\0" for key, value in env.items())
    return ctypes.create_unicode_buffer(entries + "\0")


class PTYSession(PTYSessionBase):
    """Windows ConPTY session managing one interactive shell process.

    The shell runs attached to a real pseudoconsole. A background reader thread
    drains console output into a thread-safe buffer, mirroring the blocking
    behaviour of the POSIX backend.
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
        """Initialize Windows ConPTY session."""
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

        # Output buffer (thread-safe) filled by the reader thread
        self.output_buffer: collections.deque[bytes] = collections.deque(maxlen=10000)
        self.output_lock = threading.Lock()
        self.reader_thread: threading.Thread | None = None

        # Win32 handles / state
        self._hpc: int | None = None
        self._h_in_write: int | None = None
        self._h_out_read: int | None = None
        self._h_process: int | None = None
        self._h_thread: int | None = None
        self._pid: int | None = None
        self._returncode: int | None = None

        self._create_pty()

    # -- Creation ----------------------------------------------------------

    def _create_pty(self) -> None:
        """Create the pseudoconsole and spawn the shell attached to it."""
        kernel32 = _kernel32_api()

        # Two anonymous pipes: one carrying our input to the console, one
        # carrying the console's output back to us.
        sa = SECURITY_ATTRIBUTES()
        sa.nLength = ctypes.sizeof(SECURITY_ATTRIBUTES)
        sa.bInheritHandle = True

        in_read = wintypes.HANDLE()
        in_write = wintypes.HANDLE()
        out_read = wintypes.HANDLE()
        out_write = wintypes.HANDLE()

        if not kernel32.CreatePipe(
            ctypes.byref(in_read), ctypes.byref(in_write), ctypes.byref(sa), 0
        ):
            raise OSError(ctypes.get_last_error(), "CreatePipe (stdin) failed")
        if not kernel32.CreatePipe(
            ctypes.byref(out_read), ctypes.byref(out_write), ctypes.byref(sa), 0
        ):
            kernel32.CloseHandle(in_read)
            kernel32.CloseHandle(in_write)
            raise OSError(ctypes.get_last_error(), "CreatePipe (stdout) failed")

        # Create the pseudoconsole. It duplicates the handles it is given.
        hpc = wintypes.HANDLE()
        cols = max(10, min(500, int(self.cols)))
        rows = max(5, min(200, int(self.rows)))
        hr = kernel32.CreatePseudoConsole(
            COORD(cols, rows), in_read, out_write, 0, ctypes.byref(hpc)
        )
        if hr != 0:
            for handle in (in_read, in_write, out_read, out_write):
                kernel32.CloseHandle(handle)
            raise OSError(f"CreatePseudoConsole failed (HRESULT 0x{hr & 0xFFFFFFFF:08x})")

        self._hpc = hpc
        self._h_in_write = in_write
        self._h_out_read = out_read
        self.cols = cols
        self.rows = rows

        attribute_list = None
        try:
            # Prepare the pseudo-console attribute for the child process.
            size = ctypes.c_size_t(0)
            kernel32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
            attribute_list = ctypes.create_string_buffer(size.value)
            attr_ptr = ctypes.cast(attribute_list, wintypes.LPVOID)
            if not kernel32.InitializeProcThreadAttributeList(
                attr_ptr, 1, 0, ctypes.byref(size)
            ):
                raise OSError(ctypes.get_last_error(), "InitializeProcThreadAttributeList failed")
            if not kernel32.UpdateProcThreadAttribute(
                attr_ptr,
                0,
                PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
                hpc,
                ctypes.sizeof(wintypes.HANDLE),
                None,
                None,
            ):
                raise OSError(ctypes.get_last_error(), "UpdateProcThreadAttribute failed")

            si = STARTUPINFOEXW()
            si.StartupInfo.cb = ctypes.sizeof(STARTUPINFOEXW)
            si.StartupInfo.dwFlags = STARTF_USESTDHANDLES
            si.lpAttributeList = attr_ptr

            env_block = _build_env_block(self.env)
            command_line = ctypes.create_unicode_buffer(f'"{self.shell_path}"')
            pi = PROCESS_INFORMATION()

            created = kernel32.CreateProcessW(
                self.shell_path,
                command_line,
                None,
                None,
                False,
                EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
                ctypes.cast(env_block, wintypes.LPVOID),
                str(self.cwd),
                ctypes.byref(si),
                ctypes.byref(pi),
            )
            if not created:
                raise OSError(ctypes.get_last_error(), "CreateProcessW failed")

            self._h_process = pi.hProcess
            self._h_thread = pi.hThread
            self._pid = int(pi.dwProcessId)
        except Exception:
            # Roll back the pseudoconsole on any failure so nothing leaks.
            if self._hpc:
                kernel32.ClosePseudoConsole(self._hpc)
                self._hpc = None
            self._close_handles()
            raise
        finally:
            if attribute_list is not None:
                kernel32.DeleteProcThreadAttributeList(
                    ctypes.cast(attribute_list, wintypes.LPVOID)
                )
            # The pseudoconsole owns its copies of these handles now.
            kernel32.CloseHandle(in_read)
            kernel32.CloseHandle(out_write)

        # Start draining console output on a background thread.
        self.reader_thread = threading.Thread(
            target=self._reader_thread_func,
            name=f"nova-conpty-reader-{self.id}",
            daemon=True,
        )
        self.reader_thread.start()

    # -- Reader thread -----------------------------------------------------

    def _reader_thread_func(self) -> None:
        """Block on ``ReadFile`` and buffer every console output chunk."""
        kernel32 = _kernel32_api()
        handle = self._h_out_read
        if not handle:
            return

        buffer = ctypes.create_string_buffer(4096)
        read = wintypes.DWORD(0)

        try:
            while True:
                ok = kernel32.ReadFile(handle, buffer, 4096, ctypes.byref(read), None)
                if not ok or read.value == 0:
                    break
                chunk = buffer.raw[: read.value]
                with self.output_lock:
                    self.output_buffer.append(chunk)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("ConPTY reader thread stopped: %s", exc)

    # -- PTYSession interface ---------------------------------------------

    @property
    def pid(self) -> int | None:
        """Get the process ID of the shell, or None if not running."""
        return self._pid

    @property
    def returncode(self) -> int | None:
        """Get the process exit code, or None if still running."""
        if self._returncode is not None:
            return self._returncode
        if not self._h_process:
            return None
        code = wintypes.DWORD(0)
        if not _kernel32_api().GetExitCodeProcess(self._h_process, ctypes.byref(code)):
            return None
        if code.value == STILL_ACTIVE:
            return None
        self._returncode = int(code.value)
        return self._returncode

    @property
    def is_alive(self) -> bool:
        """Check if the PTY session and child shell are still running."""
        if self._closed:
            return False
        if not self._h_process:
            return False
        return self.returncode is None

    def resize(self, cols: int, rows: int) -> None:
        """Resize the pseudoconsole window so the shell sees the new size."""
        if self._closed:
            return

        cols = max(10, min(500, int(cols)))
        rows = max(5, min(200, int(rows)))

        self.cols = cols
        self.rows = rows

        if not self._hpc:
            return
        hr = _kernel32_api().ResizePseudoConsole(self._hpc, COORD(cols, rows))
        if hr != 0:
            logger.warning(
                "ResizePseudoConsole failed for session %s (HRESULT 0x%08x)",
                self.id,
                hr & 0xFFFFFFFF,
            )

    def write(self, data: bytes | str) -> None:
        """Write user input bytes or text to the pseudoconsole input."""
        if self._closed or not self._h_in_write:
            return

        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        if not raw:
            return

        written = wintypes.DWORD(0)
        ok = _kernel32_api().WriteFile(
            self._h_in_write, raw, len(raw), ctypes.byref(written), None
        )
        if not ok:
            logger.debug("ConPTY write failed on session %s", self.id)
            self.close()

    def read(self, max_bytes: int = 4096) -> bytes:
        """Drain buffered console output (never blocks)."""
        if self._closed:
            return b""

        with self.output_lock:
            if not self.output_buffer:
                return b""

            result = b""
            while self.output_buffer and len(result) < max_bytes:
                result += self.output_buffer.popleft()
            return result

    def close(self) -> None:
        """Close the pseudoconsole, terminate the shell and free all handles."""
        if self._closed:
            return
        self._closed = True

        kernel32 = _kernel32_api()

        # Closing the pseudoconsole ends the console session for the child.
        if self._hpc:
            kernel32.ClosePseudoConsole(self._hpc)
            self._hpc = None

        # Wait for the shell to exit, forcing it if it lingers.
        if self._h_process:
            if kernel32.WaitForSingleObject(self._h_process, 1000) == WAIT_TIMEOUT:
                kernel32.TerminateProcess(self._h_process, 1)
                kernel32.WaitForSingleObject(self._h_process, 1000)
            code = wintypes.DWORD(0)
            if kernel32.GetExitCodeProcess(self._h_process, ctypes.byref(code)):
                self._returncode = int(code.value)

        # Let the reader thread observe the console close and finish.
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)

        self._close_handles()

    def _close_handles(self) -> None:
        """Close every Win32 handle this session still owns."""
        kernel32 = _kernel32_api()
        for attr in ("_h_in_write", "_h_out_read"):
            handle = getattr(self, attr, None)
            if handle:
                kernel32.CloseHandle(handle)
                setattr(self, attr, None)
        for attr in ("_h_thread", "_h_process"):
            handle = getattr(self, attr, None)
            if handle:
                kernel32.CloseHandle(handle)
                setattr(self, attr, None)


__all__ = ["PTYSession", "_find_shell"]
