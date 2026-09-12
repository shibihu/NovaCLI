"""Sandboxed shell command execution.

Every command the agent (or the user, through the CLI/web terminal) runs goes
through :class:`CommandRunner`, which:

* consults the safety policy before spawning anything,
* runs inside the workspace root,
* enforces a wall-clock timeout and kills the whole process *group* so
  background children cannot survive a timeout,
* scrubs NovaCLI's own secrets from the child environment,
* truncates output so a runaway ``find /`` cannot blow up memory or the
  model's context window.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path
from typing import Mapping

from .models import RunResult
from .safety import SafetyMode, SafetyPolicy

DEFAULT_MAX_OUTPUT_BYTES = 200_000
_GRACE_PERIOD_SECONDS = 2.0

# Environment variables NovaCLI injects and must never hand to a child process.
SCRUBBED_ENV_KEYS: tuple[str, ...] = ("GROQ_API_KEY", "NOVA_GROQ_API_KEY")

# Basenames that indicate a shell is available.
_SHELL_CANDIDATES = ("/bin/bash", "/usr/bin/bash", "/bin/sh", "/system/bin/sh")


def _find_shell() -> str | None:
    """Best available POSIX shell, or ``None`` to let the OS decide."""
    for candidate in _SHELL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    removed = len(text) - limit
    return f"{head}\n... [{removed} bytes truncated] ...\n{tail}"


class CommandRunner:
    """Runs shell commands asynchronously, safely and with a timeout."""

    def __init__(
        self,
        project_root: str | Path,
        timeout: int = 30,
        *,
        safety: SafetyPolicy | None = None,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        scrub_env: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.root = Path(project_root).expanduser().resolve()
        self.timeout = max(1, int(timeout))
        self.safety = safety or SafetyPolicy(self.root, SafetyMode.SMART)
        self.max_output_bytes = max_output_bytes
        self.scrub_env = scrub_env
        self._base_env = dict(env) if env is not None else None

    # -- Public API ------------------------------------------------------

    def build_env(self) -> dict[str, str]:
        """Child environment: inherited, minus NovaCLI's own secrets."""
        base = dict(self._base_env) if self._base_env is not None else dict(os.environ)
        if self.scrub_env:
            for key in list(base):
                if key in SCRUBBED_ENV_KEYS or key.startswith("NOVA_SECRET"):
                    base.pop(key, None)
        base.setdefault("NOVA_WORKSPACE", str(self.root))
        return base

    def resolve_cwd(self, cwd: str | Path | None) -> Path:
        """Resolve a working directory, clamped to the workspace root."""
        if cwd is None:
            return self.root
        candidate = Path(cwd).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        if candidate != self.root and self.root not in candidate.parents:
            return self.root
        return candidate if candidate.is_dir() else self.root

    async def run(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        timeout: int | None = None,
        approved: bool = False,
        check_safety: bool = True,
    ) -> RunResult:
        """Execute ``command`` and capture its result.

        Never raises for command failures — a non-zero exit, a timeout or a
        policy block are all reported through the returned
        :class:`RunResult`. Only genuinely unexpected OS errors propagate.
        """
        command = (command or "").strip()
        effective_timeout = self.timeout if timeout is None else max(1, int(timeout))

        if check_safety:
            verdict = self.safety.check_command(command)
            if not verdict.allowed:
                return RunResult(
                    command=command,
                    exit_code=None,
                    blocked=True,
                    reason=verdict.reason or "blocked by safety policy",
                )
            # `allowed` means "permitted in principle"; approval is the
            # separate gate that decides whether it runs unattended.
            if verdict.requires_approval and not approved:
                return RunResult(
                    command=command,
                    exit_code=None,
                    blocked=True,
                    reason=f"requires approval ({verdict.reason})",
                )

        workdir = self.resolve_cwd(cwd)
        started = time.perf_counter()

        popen_kwargs: dict[str, object] = {}
        if os.name == "posix":
            # New session => the child gets its own process group, so a timeout
            # can kill the whole tree instead of just the shell.
            popen_kwargs["start_new_session"] = True

        shell = _find_shell()
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(workdir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                executable=shell,
                env=self.build_env(),
                **popen_kwargs,  # type: ignore[arg-type]
            )
        except (OSError, ValueError) as exc:
            return RunResult(
                command=command,
                exit_code=None,
                stderr=str(exc),
                duration_ms=int((time.perf_counter() - started) * 1000),
                blocked=True,
                reason=f"could not start command: {exc}",
            )

        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(), timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            await self._terminate(process)
            stdout_b, stderr_b = await self._drain(process)
        except asyncio.CancelledError:
            await self._terminate(process)
            raise

        duration_ms = int((time.perf_counter() - started) * 1000)
        stdout = _truncate(self._decode(stdout_b), self.max_output_bytes)
        stderr = _truncate(self._decode(stderr_b), self.max_output_bytes)

        exit_code: int | None = process.returncode
        if timed_out:
            note = f"command timed out after {effective_timeout}s and was terminated"
            stderr = f"{stderr}\n{note}".strip()
            exit_code = None

        return RunResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )

    @staticmethod
    def _decode(raw: bytes | None) -> str:
        if not raw:
            return ""
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    async def _drain(process: asyncio.subprocess.Process) -> tuple[bytes, bytes]:
        """Collect whatever output a killed process already produced."""
        try:
            return await asyncio.wait_for(process.communicate(), timeout=_GRACE_PERIOD_SECONDS)
        except (asyncio.TimeoutError, ProcessLookupError, ValueError):
            return b"", b""

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        """SIGTERM the process group, then SIGKILL if it refuses to die."""
        if process.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:  # pragma: no cover - Windows only
                process.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.terminate()
            except (ProcessLookupError, OSError):
                return

        try:
            await asyncio.wait_for(process.wait(), timeout=_GRACE_PERIOD_SECONDS)
            return
        except asyncio.TimeoutError:
            pass

        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:  # pragma: no cover - Windows only
                process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=_GRACE_PERIOD_SECONDS)
        except asyncio.TimeoutError:  # pragma: no cover - extreme edge case
            pass
