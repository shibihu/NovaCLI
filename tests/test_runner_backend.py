"""Tests for ExecutionBackend abstraction (LocalBackend and DockerBackend)."""

from __future__ import annotations

from pathlib import Path
import pytest
from unittest.mock import AsyncMock, patch

from nova.core.runner import CommandRunner, LocalBackend, DockerBackend, ExecutionBackend


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
async def test_docker_backend_constructs_cmd(tmp_path: Path):
    docker_backend = DockerBackend(image="python:3.12-slim")

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
        args = mock_exec.call_args[0]
        assert "docker" in args
        assert "run" in args
        assert "python:3.12-slim" in args
