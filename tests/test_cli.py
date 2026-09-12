"""Tests for :mod:`nova.cli.commands`.

These drive ``main()`` directly, so they cover argument parsing and rendering
without spawning a subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nova.cli.commands import Console, build_parser, main
from nova.config import API_KEY_HINT

from conftest import TEST_API_KEY, FakeProvider


# --- Parser -----------------------------------------------------------------


def test_parser_lists_every_command() -> None:
    parser = build_parser()
    actions = [
        a for a in parser._subparsers._group_actions  # type: ignore[attr-defined]
    ]
    names = set(actions[0].choices)
    assert {
        "ask", "chat", "serve", "run", "summary", "tree", "ls",
        "read", "search", "config", "doctor", "init", "version",
    } <= names


def test_parser_rejects_unknown_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nonsense"])


def test_version_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0


def test_main_without_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


# --- Console ----------------------------------------------------------------


def test_console_can_disable_colour() -> None:
    assert Console(color=False).paint("hi", "red") == "hi"


def test_console_paint_adds_ansi_when_enabled() -> None:
    assert "\033[" in Console(color=True).paint("hi", "red")


def test_console_warn_and_error_prefix(capsys: pytest.CaptureFixture[str]) -> None:
    console = Console(color=False)
    console.warn("careful")
    console.error("bad")
    out = capsys.readouterr().out
    assert "careful" in out
    assert "bad" in out


# --- version ----------------------------------------------------------------


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert "NovaCLI v" in capsys.readouterr().out


def test_version_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "NovaCLI"


# --- config -----------------------------------------------------------------


def test_config_masks_the_key(
    monkeypatch: pytest.MonkeyPatch, tmp_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", TEST_API_KEY)
    assert main(["config", "-C", str(tmp_project)]) == 0
    out = capsys.readouterr().out
    assert TEST_API_KEY not in out
    assert "configured" in out


def test_config_json_has_no_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", TEST_API_KEY)
    assert main(["config", "-C", str(tmp_project), "--json"]) == 0
    assert TEST_API_KEY not in capsys.readouterr().out


def test_config_without_key_explains_setup(
    monkeypatch: pytest.MonkeyPatch, bare_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("nova.config.DEFAULT_USER_CONFIG_PATH", bare_project / "absent.json")
    assert main(["config", "-C", str(bare_project)]) == 0
    out = capsys.readouterr().out
    assert "missing" in out
    assert "console.groq.com" in out


# --- workspace commands -----------------------------------------------------


def test_summary_command(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["summary", "-C", str(tmp_project)]) == 0
    out = capsys.readouterr().out
    assert "Project:" in out
    assert "Python" in out


def test_summary_json(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["summary", "-C", str(tmp_project), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["languages"]["Python"] >= 3


def test_tree_command(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tree", "-C", str(tmp_project)]) == 0
    assert "main.py" in capsys.readouterr().out


def test_tree_hides_ignored_dirs(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["tree", "-C", str(tmp_project)])
    assert "__pycache__" not in capsys.readouterr().out


def test_ls_command(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ls", "-C", str(tmp_project)]) == 0
    assert "main.py" in capsys.readouterr().out


def test_read_command(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["read", "main.py", "-C", str(tmp_project)]) == 0
    assert "def greet" in capsys.readouterr().out


def test_read_refuses_secrets(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["read", ".env", "-C", str(tmp_project)]) == 1
    assert "Refused" in capsys.readouterr().out


def test_read_missing_file_errors(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["read", "ghost.py", "-C", str(tmp_project)]) == 1
    assert "does not exist" in capsys.readouterr().out


def test_search_command(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["search", "def greet", "-C", str(tmp_project)]) == 0
    assert "main.py" in capsys.readouterr().out


def test_search_no_matches(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["search", "zzznotfound", "-C", str(tmp_project)]) == 0
    assert "No matches" in capsys.readouterr().out


def test_search_with_glob(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["search", "greet", "-g", "**/*.md", "-C", str(tmp_project)]) == 0


# --- run --------------------------------------------------------------------


# NOTE: `nova run` uses REMAINDER, so its own flags must precede the command.
def test_run_executes_a_safe_command(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "-C", str(tmp_project), "echo", "cli-works"]) == 0
    assert "cli-works" in capsys.readouterr().out


def test_run_passes_flags_to_the_command(
    tmp_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Flags after the command belong to the command, not to NovaCLI."""
    assert main(["run", "-C", str(tmp_project), "ls", "-1"]) == 0
    assert "main.py" in capsys.readouterr().out


def test_run_refuses_a_forbidden_command(
    tmp_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "-C", str(tmp_project), "rm", "-rf", "/"]) == 4
    assert "Refused" in capsys.readouterr().out


def test_run_requires_approval_in_strict_mode(
    tmp_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_project / "notes.txt").write_text("x", encoding="utf-8")
    code = main(["run", "-C", str(tmp_project), "--safety", "strict", "rm", "notes.txt"])
    assert code == 4
    assert "approval" in capsys.readouterr().out.lower()


def test_run_with_yes_approves(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_project / "notes.txt").write_text("x", encoding="utf-8")
    code = main(["run", "-C", str(tmp_project), "--yes", "rm", "notes.txt"])
    assert code == 0
    assert not (tmp_project / "notes.txt").exists()


def test_run_reports_nonzero_exit(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "-C", str(tmp_project), "false"]) == 1


def test_run_requires_a_command(tmp_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "-C", str(tmp_project)]) == 2
    assert "No command given" in capsys.readouterr().out


# --- init / doctor ----------------------------------------------------------


def test_init_creates_env_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "fresh"
    assert main(["init", str(target)]) == 0
    env_file = target / ".env"
    assert env_file.exists()
    assert "GROQ_API_KEY=" in env_file.read_text()


def test_init_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "fresh"
    main(["init", str(target)])
    assert main(["init", str(target)]) == 1
    assert "--force" in capsys.readouterr().out


def test_init_force_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "fresh"
    main(["init", str(target)])
    assert main(["init", str(target), "--force"]) == 0


def test_doctor_reports_missing_key(
    monkeypatch: pytest.MonkeyPatch, tmp_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    code = main(["doctor", "-C", str(tmp_project)])
    out = capsys.readouterr().out
    assert "Python" in out
    assert code in (0, 1)


def test_doctor_reports_configured_key(
    monkeypatch: pytest.MonkeyPatch, tmp_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", TEST_API_KEY)
    main(["doctor", "-C", str(tmp_project)])
    assert "API key configured" in capsys.readouterr().out


# --- ask --------------------------------------------------------------------


def test_ask_without_key_gives_setup_error_and_exit_3(
    monkeypatch: pytest.MonkeyPatch, bare_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("nova.config.DEFAULT_USER_CONFIG_PATH", bare_project / "absent.json")
    assert main(["ask", "hello", "-C", str(bare_project)]) == 3
    out = capsys.readouterr().out
    assert "GROQ_API_KEY" in out
    assert "console.groq.com" in out


def test_ask_requires_a_task(
    monkeypatch: pytest.MonkeyPatch, tmp_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", TEST_API_KEY)
    assert main(["ask", "  ", "-C", str(tmp_project)]) == 2


def test_ask_streams_progress_and_prints_the_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project: Path,
    capsys: pytest.CaptureFixture[str],
    make_agent,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", TEST_API_KEY)
    provider = FakeProvider(
        [
            FakeProvider.action("read_file", path="main.py"),
            FakeProvider.final("It prints a greeting."),
        ]
    )
    agent = make_agent(provider)
    monkeypatch.setattr("nova.cli.commands.build_agent", lambda settings, **kw: agent)

    assert main(["ask", "what does main.py do?", "-C", str(tmp_project), "--yes"]) == 0
    out = capsys.readouterr().out
    assert "It prints a greeting." in out
    assert "read_file" in out  # the tool call was reported


def test_ask_reports_agent_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project: Path,
    capsys: pytest.CaptureFixture[str],
    make_agent,
) -> None:
    from nova.ai import AIProviderError

    monkeypatch.setenv("GROQ_API_KEY", TEST_API_KEY)
    agent = make_agent(FakeProvider([AIProviderError("upstream exploded")]))
    monkeypatch.setattr("nova.cli.commands.build_agent", lambda settings, **kw: agent)

    main(["ask", "hello", "-C", str(tmp_project)])
    assert "upstream exploded" in capsys.readouterr().out


def test_ask_json_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project: Path,
    capsys: pytest.CaptureFixture[str],
    make_agent,
) -> None:
    """`--json` must emit nothing but valid JSON on stdout."""
    monkeypatch.setenv("GROQ_API_KEY", TEST_API_KEY)
    agent = make_agent(FakeProvider([FakeProvider.final("json answer")]))
    monkeypatch.setattr("nova.cli.commands.build_agent", lambda settings, **kw: agent)

    assert main(["ask", "hi", "-C", str(tmp_project), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"] == "json answer"


def test_ask_uses_the_project_root_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project: Path,
    capsys: pytest.CaptureFixture[str],
    make_agent,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", TEST_API_KEY)
    provider = FakeProvider([FakeProvider.final("ok")])
    agent = make_agent(provider)
    monkeypatch.setattr("nova.cli.commands.build_agent", lambda settings, **kw: agent)

    main(["ask", "hi", "-C", str(tmp_project)])
    assert str(tmp_project) in provider.calls[0][-1]["content"]


# --- api key hint -----------------------------------------------------------


def test_api_key_hint_is_actionable() -> None:
    assert "https://console.groq.com/keys" in API_KEY_HINT
