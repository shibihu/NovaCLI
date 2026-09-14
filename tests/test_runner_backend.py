"""Tests for ExecutionBackend abstraction (LocalBackend and DockerBackend)."""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest
import shutil
from unittest.mock import AsyncMock, patch

from nova.core.runner import CommandRunner, LocalBackend, DockerBackend, ExecutionBackend


def is_docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        import subprocess
        res = subprocess.run(
            ["docker", "run", "--rm", "python:3.12-slim", "echo", "test"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5
        )
        return res.returncode == 0
    except Exception:
        return False


@pytest.mark.asyncio
async def test_local_backend_executes_command(tmp_path: Path):
    backend = LocalBackend()
    assert isinstance(backend, ExecutionBackend)

    exit_code, stdout, stderr, timed_out = await backend.run(
        "echo hello_local", cwd=tmp_path, env={}, timeout=5.0
    )
    assert exit_code == 0
    assert "hello_local" in stdout
    assert timed_out is False


@pytest.mark.asyncio
async def test_command_runner_uses_configured_backend(tmp_path: Path):
    mock_backend = AsyncMock()
    mock_backend.run.return_value = (0, "mock_out", "", False)

    runner = CommandRunner(tmp_path, backend=mock_backend)
    result = await runner.run("echo test")

    assert result.ok is True
    assert result.stdout == "mock_out"
    assert mock_backend.run.called is True


@pytest.mark.asyncio
async def test_docker_backend_constructs_cmd_with_security_flags(tmp_path: Path):
    docker_backend = DockerBackend(
        image="python:3.12-slim",
        network_disabled=True,
        cap_drop_all=True,
        no_new_privileges=True,
        memory_limit="256m",
        cpu_limit="1.0"
    )

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"docker_stdout", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        exit_code, stdout, stderr, timed_out = await docker_backend.run(
            "pytest -q", cwd=tmp_path, env={"TEST_ENV": "1"}, timeout=10.0
        )

        assert exit_code == 0
        assert stdout == "docker_stdout"
        assert mock_exec.called is True
        args = list(mock_exec.call_args[0])
        assert "docker" in args
        assert "run" in args
        assert "--cap-drop" in args
        assert "ALL" in args
        assert "--security-opt" in args
        assert "no-new-privileges" in args
        assert "--memory" in args
        assert "256m" in args
        assert "--cpus" in args
        assert "1.0" in args
        assert "--network" in args
        assert "none" in args
        assert "python:3.12-slim" in args


@pytest.mark.asyncio
async def test_docker_backend_timeout_and_cleanup(tmp_path: Path):
    docker_backend = DockerBackend()

    with patch("asyncio.create_subprocess_exec") as mock_exec, \
         patch.object(DockerBackend, "_cleanup_container", new_callable=AsyncMock) as mock_cleanup:
        mock_proc = AsyncMock()
        mock_proc.communicate.side_effect = asyncio.TimeoutError()
        mock_exec.return_value = mock_proc

        exit_code, stdout, stderr, timed_out = await docker_backend.run(
            "sleep 100", cwd=tmp_path, env={}, timeout=0.1
        )

        assert exit_code is None
        assert timed_out is True
        assert b"timed out" in stderr.encode() or "timed out" in stderr
        assert mock_cleanup.called is True


@pytest.mark.asyncio
async def test_docker_real_execution(tmp_path: Path):
    if not is_docker_available():
        pytest.skip("Docker daemon is not available")

    backend = DockerBackend(network_disabled=True)
    exit_code, stdout, stderr, timed_out = await backend.run(
        "echo docker_real", cwd=tmp_path, env={}, timeout=10.0
    )
    assert exit_code == 0
    assert "docker_real" in stdout
    assert timed_out is False
