"""Shared pytest fixtures.

Everything the tests need is injected — no real API keys, no network, no
writes outside the temporary project directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from nova.ai import AIProviderError
from nova.config import load_settings
from nova.core.agent import NovaAgent, ToolBox
from nova.core.context import ContextBuilder
from nova.core.runner import CommandRunner
from nova.core.safety import SafetyPolicy
from nova.workspace.files import Workspace
from nova.workspace.projects import ProjectAnalyzer

TEST_API_KEY = "gsk_test0000000000000000000000000000000000"


# ---------------------------------------------------------------------------
# Fake provider
# ---------------------------------------------------------------------------


class FakeProvider:
    """Scripted stand-in for :class:`~nova.ai.groq.GroqProvider`.

    Satisfies the :class:`~nova.ai.AIProvider` protocol without any network
    access. Replies are consumed in order; an ``Exception`` instance in the
    script is raised instead of returned.
    """

    def __init__(
        self,
        replies: Sequence[Any] | None = None,
        *,
        model: str = "fake-model",
        configured: bool = True,
    ) -> None:
        self.replies: list[Any] = list(replies or [])
        self.model_name = model
        self.configured = configured
        self.calls: list[list[dict[str, str]]] = []
        self.closed = False

    async def complete(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str:
        self.calls.append(list(messages))
        if not self.replies:
            raise AIProviderError("FakeProvider ran out of scripted replies.")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return str(reply)

    async def aclose(self) -> None:
        self.closed = True

    # Convenience builders for readable tests.
    @staticmethod
    def action(tool: str, thought: str = "thinking", **arguments: Any) -> str:
        return json.dumps(
            {"thought": thought, "action": tool, "action_input": arguments}
        )

    @staticmethod
    def final(answer: str, thought: str = "done") -> str:
        return json.dumps({"thought": thought, "final_answer": answer})


# ---------------------------------------------------------------------------
# Project fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """A small but realistic Python project."""
    (tmp_path / "README.md").write_text(
        "# Demo Project\n\nA tiny sample project used by the NovaCLI test suite.\n",
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text("requests>=2.0\n", encoding="utf-8")
    (tmp_path / "main.py").write_text(
        "def greet(name: str) -> str:\n"
        '    """Return a greeting for name."""\n'
        '    return f"hello {name}"\n'
        "\n"
        '\nif __name__ == "__main__":\n'
        '    print(greet("world"))\n',
        encoding="utf-8",
    )
    (tmp_path / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n"
        "    return left + right\n"
        "\n"
        "\n"
        "def divide(left: int, right: int) -> float:\n"
        "    return left / right\n",
        encoding="utf-8",
    )
    (tmp_path / "nova_notes.md").write_text(
        "# Notes\n\nThe parser lives in calculator.py and handles arithmetic.\n",
        encoding="utf-8",
    )

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_main.py").write_text(
        "from main import greet\n\n\ndef test_greet():\n"
        '    assert greet("x") == "hello x"\n',
        encoding="utf-8",
    )

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "helper.py").write_text("VALUE = 42\n", encoding="utf-8")

    # Noise that the workspace must ignore.
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_bytes(b"\x00\x01\x02")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    # Secrets that must never be readable.
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk_real_secret_value_abcdefghijklmnop\n", encoding="utf-8")

    return tmp_path


@pytest.fixture
def bare_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project with no ``.env`` — for testing the unconfigured path."""
    root = tmp_path_factory.mktemp("bare")
    (root / "main.py").write_text("x = 1\n", encoding="utf-8")
    return root


@pytest.fixture
def settings(tmp_project: Path):
    """Settings pointed at the temp project with a fake key."""
    return load_settings(
        project_root=tmp_project,
        env={"GROQ_API_KEY": TEST_API_KEY, "GROQ_MODEL": "test-model"},
        user_config_path=tmp_project / "_no_such_user_config.json",
    )


@pytest.fixture
def safety(settings):
    return SafetyPolicy(settings.project_root, settings.safety_mode, secret_values=[TEST_API_KEY])


@pytest.fixture
def workspace(settings, safety) -> Workspace:
    return Workspace(settings.project_root, safety=safety)


@pytest.fixture
def analyzer(workspace) -> ProjectAnalyzer:
    return ProjectAnalyzer(workspace)


@pytest.fixture
def context_builder(workspace, analyzer, safety) -> ContextBuilder:
    return ContextBuilder(workspace, analyzer=analyzer, safety=safety)


@pytest.fixture
def runner(settings, safety) -> CommandRunner:
    return CommandRunner(settings.project_root, settings.command_timeout, safety=safety)


@pytest.fixture
def toolbox(workspace, runner, analyzer, safety) -> ToolBox:
    return ToolBox(workspace, runner, analyzer=analyzer, safety=safety)


@pytest.fixture
def make_agent(settings, workspace, safety, runner, analyzer, context_builder, toolbox):
    """Factory returning a NovaAgent wired to the temp project."""

    def factory(provider: FakeProvider, **kwargs: Any) -> NovaAgent:
        return NovaAgent(
            provider,
            settings,
            workspace=workspace,
            safety=safety,
            runner=runner,
            analyzer=analyzer,
            context_builder=context_builder,
            toolbox=toolbox,
            **kwargs,
        )

    return factory


@pytest.fixture
def fake_provider() -> type[FakeProvider]:
    return FakeProvider
