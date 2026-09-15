"""Tests for :mod:`nova.core.runner`."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nova.core.models import SafetyMode
from nova.core.runner import SCRUBBED_ENV_KEYS, CommandRunner
from nova.core.safety import SafetyPolicy


@pytest.fixture
def safe_runner(tmp_path: Path) -> CommandRunner:
    return CommandRunner(tmp_path, timeout=10, safety=SafetyPolicy(tmp_path, SafetyMode.SMART))


# --- Construction / environment --------------------------------------------


def test_root_is_resolved(tmp_path: Path) -> None:
    runner = CommandRunner(tmp_path, safety=SafetyPolicy(tmp_path))
    assert runner.root == tmp_path.resolve()


def test_build_env_scrubs_the_api_key(tmp_path: Path) -> None:
    runner = CommandRunner(
        tmp_path,
        safety=SafetyPolicy(tmp_path),
        env={"GROQ_API_KEY": "gsk_secret", "PATH": "/usr/bin", "NOVA_SECRET_TOKEN": "x"},
    )
    env = runner.build_env()
    assert "GROQ_API_KEY" not in env
    assert "NOVA_SECRET_TOKEN" not in env
    assert env["PATH"] == "/usr/bin"
    assert env["NOVA_WORKSPACE"] == str(runner.root)


def test_build_env_can_be_disabled(tmp_path: Path) -> None:
    runner = CommandRunner(
        tmp_path, safety=SafetyPolicy(tmp_path), scrub_env=False, env={"GROQ_API_KEY": "k"}
    )
    assert runner.build_env()["GROQ_API_KEY"] == "k"


def test_scrubbed_keys_are_documented() -> None:
    assert "GROQ_API_KEY" in SCRUBBED_ENV_KEYS


# --- cwd handling -----------------------------------------------------------


def test_resolve_cwd_defaults_to_root(tmp_path: Path) -> None:
    runner = CommandRunner(tmp_path, safety=SafetyPolicy(tmp_path))
    assert runner.resolve_cwd(None) == runner.root


def test_resolve_cwd_accepts_a_subdirectory(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    runner = CommandRunner(tmp_path, safety=SafetyPolicy(tmp_path))
    assert runner.resolve_cwd("sub") == runner.root / "sub"


def test_resolve_cwd_clamps_escapes(tmp_path: Path) -> None:
    runner = CommandRunner(tmp_path, safety=SafetyPolicy(tmp_path))
    assert runner.resolve_cwd("../..") == runner.root


def test_resolve_cwd_falls_back_for_missing_dir(tmp_path: Path) -> None:
    runner = CommandRunner(tmp_path, safety=SafetyPolicy(tmp_path))
    assert runner.resolve_cwd("does-not-exist") == runner.root


# --- Execution --------------------------------------------------------------


async def test_run_captures_stdout(safe_runner: CommandRunner) -> None:
    result = await safe_runner.run("echo hello")
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert result.ok is True


async def test_run_captures_nonzero_exit(safe_runner: CommandRunner) -> None:
    result = await safe_runner.run("echo oops >&2; exit 3")
    assert result.exit_code == 3
    assert "oops" in result.stderr
    assert result.ok is False


async def test_run_records_duration(safe_runner: CommandRunner) -> None:
    result = await safe_runner.run("true")
    assert result.duration_ms >= 0


async def test_run_in_workspace(safe_runner: CommandRunner) -> None:
    result = await safe_runner.run("pwd")
    assert str(safe_runner.root) in result.stdout


async def test_run_respects_cwd(safe_runner: CommandRunner) -> None:
    (safe_runner.root / "sub").mkdir()
    result = await safe_runner.run("pwd", cwd="sub")
    assert result.stdout.strip().endswith("sub")


async def test_run_times_out_and_kills(safe_runner: CommandRunner) -> None:
    result = await safe_runner.run("sleep 30", timeout=1)
    assert result.timed_out is True
    assert result.exit_code is None
    assert "timed out" in result.stderr.lower()


async def test_run_with_python_interpreter(safe_runner: CommandRunner) -> None:
    result = await safe_runner.run(f'{sys.executable} -c "print(2+2)"')
    assert "4" in result.stdout


async def test_output_is_truncated(tmp_path: Path) -> None:
    runner = CommandRunner(
        tmp_path, timeout=15, max_output_bytes=200, safety=SafetyPolicy(tmp_path)
    )
    result = await runner.run(f'{sys.executable} -c "print(\'x\'*5000)"')
    assert "truncated" in result.stdout
    assert len(result.stdout) < 2_000


async def test_child_env_has_no_api_key(tmp_path: Path) -> None:
    runner = CommandRunner(
        tmp_path,
        timeout=15,
        safety=SafetyPolicy(tmp_path),
        env={"GROQ_API_KEY": "gsk_leak", "PATH": "/usr/bin:/bin"},
    )
    result = await runner.run('echo "${GROQ_API_KEY:-MISSING}"')
    assert "gsk_leak" not in result.stdout
    assert "MISSING" in result.stdout


# --- Safety integration -----------------------------------------------------


async def test_forbidden_command_is_blocked(tmp_path: Path) -> None:
    runner = CommandRunner(tmp_path, safety=SafetyPolicy(tmp_path))
    result = await runner.run("rm -rf /")
    assert result.blocked is True
    assert result.exit_code is None
    assert result.reason


async def test_blocked_command_does_not_execute(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    runner = CommandRunner(tmp_path, safety=SafetyPolicy(tmp_path))
    result = await runner.run(f"touch {marker}; rm -rf /")
    assert result.blocked is True
    assert not marker.exists()


async def test_dangerous_command_needs_approval(tmp_path: Path) -> None:
    strict = CommandRunner(tmp_path, safety=SafetyPolicy(tmp_path, SafetyMode.STRICT))
    result = await strict.run("rm somefile.txt")
    assert result.blocked is True
    assert "approval" in result.reason


async def test_approved_command_runs(tmp_path: Path) -> None:
    (tmp_path / "gone.txt").write_text("x", encoding="utf-8")
    strict = CommandRunner(tmp_path, safety=SafetyPolicy(tmp_path, SafetyMode.STRICT))
    result = await strict.run("rm gone.txt", approved=True)
    assert result.blocked is False
    assert not (tmp_path / "gone.txt").exists()


async def test_safety_check_can_be_skipped(tmp_path: Path) -> None:
    runner = CommandRunner(tmp_path, safety=SafetyPolicy(tmp_path))
    result = await runner.run("echo direct", check_safety=False)
    assert result.exit_code == 0
    assert "direct" in result.stdout


# --- Rendering --------------------------------------------------------------


async def test_render_includes_exit_code(safe_runner: CommandRunner) -> None:
    rendered = (await safe_runner.run("echo hi")).render()
    assert "exit_code: 0" in rendered
    assert "hello" not in rendered


async def test_render_marks_blocked(tmp_path: Path) -> None:
    runner = CommandRunner(tmp_path, safety=SafetyPolicy(tmp_path))
    rendered = (await runner.run("rm -rf /")).render()
    assert "blocked" in rendered.lower()


async def test_render_handles_empty_output(safe_runner: CommandRunner) -> None:
    assert "(no output)" in (await safe_runner.run("true")).render()


async def test_result_to_dict_is_json_safe(safe_runner: CommandRunner) -> None:
    import json

    data = (await safe_runner.run("echo json")).to_dict()
    json.dumps(data)
    assert data["exit_code"] == 0


# --- run_args direct execution ----------------------------------------------


async def test_run_args_executes_direct_args_without_shell(safe_runner: CommandRunner) -> None:
    args = [
        sys.executable,
        "-c",
        "import sys; print(repr(sys.argv[1]))",
        "hello; echo PWNED",
    ]
    result = await safe_runner.run_args(args)
    assert result.exit_code == 0
    assert "'hello; echo PWNED'" in result.stdout


async def test_run_args_respects_safety_policy(tmp_path: Path) -> None:
    runner = CommandRunner(tmp_path, safety=SafetyPolicy(tmp_path))
    result = await runner.run_args(["rm", "-rf", "/"])
    assert result.blocked is True
