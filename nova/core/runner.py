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
from typing import Any, Mapping, Protocol, runtime_checkable

from .models import RunResult
from .safety import SafetyMode, SafetyPolicy

DEFAULT_MAX_OUTPUT_BYTES = 200_000
_GRACE_PERIOD_SECONDS = 2.0

# Environment variables NovaCLI injects and must never hand to a child process.
SCRUBBED_ENV_KEYS: tuple[str, ...] = ("GROQ_API_KEY", "NOVA_GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY", "NOVA_WEB_TOKEN")

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


@runtime_checkable
class ExecutionBackend(Protocol):
    """Abstraction for command execution backends (Local host vs Docker sandbox)."""

    async def run(
        self,
        command: str,
        cwd: Path,
        env: dict[str, str],
        timeout: float,
    ) -> tuple[int | None, str, str, bool]:
        """Execute command and return (exit_code, stdout, stderr, timed_out)."""
        ...


class LocalBackend:
    """Local host execution backend."""

    async def run(
        self,
        command: str,
        cwd: Path,
        env: dict[str, str],
        timeout: float,
    ) -> tuple[int | None, str, str, bool]:
        popen_kwargs: dict[str, Any] = {}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True

        shell = _find_shell()
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            executable=shell,
            env=env,
            **popen_kwargs,
        )

        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            await self._terminate(process)
            stdout_b, stderr_b = await self._drain(process)
        except asyncio.CancelledError:
            await self._terminate(process)
            raise

        stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
        stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""
        exit_code = process.returncode if not timed_out else None

        return exit_code, stdout, stderr, timed_out

    @staticmethod
    async def _drain(process: asyncio.subprocess.Process) -> tuple[bytes, bytes]:
        try:
            return await asyncio.wait_for(process.communicate(), timeout=_GRACE_PERIOD_SECONDS)
        except (asyncio.TimeoutError, ProcessLookupError, ValueError):
            return b"", b""

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
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
            else:
                process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=_GRACE_PERIOD_SECONDS)
        except asyncio.TimeoutError:
            pass


class DockerBackend:
    """Docker container sandboxed execution backend.

    Applies security restrictions (cap-drop, no-new-privileges, resource limits,
    tmpfs mounts) and ensures container termination on timeout.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        network_disabled: bool = False,
        cap_drop_all: bool = True,
        no_new_privileges: bool = True,
        pids_limit: int = 256,
        memory_limit: str | None = "512m",
        cpu_limit: str | None = "1.5",
    ) -> None:
        self.image = image
        self.network_disabled = network_disabled
        self.cap_drop_all = cap_drop_all
        self.no_new_privileges = no_new_privileges
        self.pids_limit = pids_limit
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit

    async def run(
        self,
        command: str,
        cwd: Path,
        env: dict[str, str],
        timeout: float,
    ) -> tuple[int | None, str, str, bool]:
        import uuid
        container_name = f"nova_sandbox_{uuid.uuid4().hex[:12]}"

        docker_args = [
            "docker", "run",
            "--name", container_name,
            "--rm", "-i",
            "-v", f"{cwd}:/workspace",
            "-w", "/workspace",
            "--tmpfs", "/tmp:exec,mode=1777",
        ]

        if self.cap_drop_all:
            docker_args.extend(["--cap-drop", "ALL"])
        if self.no_new_privileges:
            docker_args.extend(["--security-opt", "no-new-privileges"])
        if self.pids_limit:
            docker_args.extend(["--pids-limit", str(self.pids_limit)])
        if self.memory_limit:
            docker_args.extend(["--memory", str(self.memory_limit)])
        if self.cpu_limit:
            docker_args.extend(["--cpus", str(self.cpu_limit)])
        if self.network_disabled:
            docker_args.extend(["--network", "none"])

        for k, v in env.items():
            docker_args.extend(["-e", f"{k}={v}"])

        docker_args.append(self.image)
        docker_args.extend(["sh", "-c", command])

        process = await asyncio.create_subprocess_exec(
            *docker_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )

        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            await self._cleanup_container(container_name, process)
            stdout_b, stderr_b = b"", b"Docker execution timed out"
        except asyncio.CancelledError:
            await self._cleanup_container(container_name, process)
            raise

        stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
        stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""
        exit_code = process.returncode if not timed_out else None

        return exit_code, stdout, stderr, timed_out

    @staticmethod
    async def _cleanup_container(container_name: str, process: asyncio.subprocess.Process) -> None:
        try:
            process.terminate()
        except Exception:
            pass

        # Force kill container via docker CLI to prevent orphaned processes
        try:
            kill_proc = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(kill_proc.wait(), timeout=5.0)
        except Exception:
            pass


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
        backend: ExecutionBackend | None = None,
    ) -> None:
        self.root = Path(project_root).expanduser().resolve()
        self.timeout = max(1, int(timeout))
        self.safety = safety or SafetyPolicy(self.root, SafetyMode.SMART)
        self.max_output_bytes = max_output_bytes
        self.scrub_env = scrub_env
        self._base_env = dict(env) if env is not None else None
        self.backend = backend or LocalBackend()

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
        """Execute ``command`` and capture its result."""
        command = (command or "").strip()
        effective_timeout = float(self.timeout if timeout is None else max(1, int(timeout)))

        if check_safety:
            verdict = self.safety.check_command(command)
            if not verdict.allowed:
                return RunResult(
                    command=command,
                    exit_code=None,
                    blocked=True,
                    reason=verdict.reason or "blocked by safety policy",
                )
            if verdict.requires_approval and not approved:
                return RunResult(
                    command=command,
                    exit_code=None,
                    blocked=True,
                    reason=f"requires approval ({verdict.reason})",
                )

        workdir = self.resolve_cwd(cwd)
        started = time.perf_counter()

        try:
            exit_code, raw_stdout, raw_stderr, timed_out = await self.backend.run(
                command, cwd=workdir, env=self.build_env(), timeout=effective_timeout
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

        duration_ms = int((time.perf_counter() - started) * 1000)
        stdout = _truncate(raw_stdout, self.max_output_bytes)
        stderr = _truncate(raw_stderr, self.max_output_bytes)

        if timed_out:
            note = f"command timed out after {effective_timeout:.0f}s and was terminated"
            stderr = f"{stderr}\n{note}".strip() if stderr else note
            exit_code = None

        return RunResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )
