"""End-to-end tests.

Each test exercises a whole user journey rather than a single unit:

* the agent reads, writes and *verifies* a real change in a real project;
* the same run is driven through the HTTP API and streamed to a client;
* secrets never reach the model or the browser, even when a task asks for them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from nova.core.agent import AgentController, build_agent
from nova.core.models import AgentStatus, EventType

from conftest import TEST_API_KEY, FakeProvider

NEW_TEST_FILE = "tests/test_calculator.py"

NEW_TEST_CONTENT = (
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
    "\n"
    "from calculator import add, divide  # noqa: E402\n"
    "\n"
    "\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
    "\n"
    "\n"
    "def test_divide():\n"
    "    assert divide(10, 2) == 5\n"
)

PYTEST_COMMAND = f"{sys.executable} -m pytest {NEW_TEST_FILE} -q"


def scripted_agent_run() -> list[str]:
    """The scripted conversation for the full journey."""
    return [
        FakeProvider.action("project_summary", thought="Orient myself first."),
        FakeProvider.action("read_file", thought="Read the module.", path="calculator.py"),
        FakeProvider.action(
            "write_file",
            thought="Add a test file.",
            path=NEW_TEST_FILE,
            content=NEW_TEST_CONTENT,
        ),
        FakeProvider.action(
            "run_command", thought="Verify the tests pass.", command=PYTEST_COMMAND
        ),
        FakeProvider.final("Added tests/test_calculator.py covering add() and divide(). All tests pass."),
    ]


# ---------------------------------------------------------------------------
# Full agent journey
# ---------------------------------------------------------------------------


async def test_agent_understands_writes_and_verifies(make_agent, tmp_project: Path) -> None:
    agent = make_agent(FakeProvider(scripted_agent_run()))
    controller = AgentController(auto_approve=True)

    events = [event async for event in agent.stream("Add tests for calculator.py", controller=controller)]
    types = [event.type for event in events]

    # It orientated itself.
    summary = next(e for e in events if e.type == EventType.TOOL_RESULT and e.data["name"] == "project_summary")
    assert "calculator.py" in summary.data["output"] or "Python" in summary.data["output"]

    # It read before writing.
    read_index = types.index(EventType.TOOL_CALL)
    assert read_index < len(types)

    # It wrote the file.
    assert (tmp_project / NEW_TEST_FILE).read_text() == NEW_TEST_CONTENT

    # It verified the change by running the tests.
    run_results = [
        e for e in events
        if e.type == EventType.TOOL_RESULT and e.data["name"] == "run_command"
    ]
    assert run_results, "the agent never ran a command"
    assert run_results[0].data["ok"] is True
    assert "passed" in run_results[0].data["output"]

    # It finished cleanly.
    assert types[-1] == EventType.FINAL
    assert "tests/test_calculator.py" in events[-1].data["answer"]


async def test_agent_result_matches_the_stream(make_agent, tmp_project: Path) -> None:
    agent = make_agent(FakeProvider(scripted_agent_run()))
    result = await agent.run("Add tests", controller=AgentController(auto_approve=True))

    assert result.status == AgentStatus.DONE
    assert result.ok is True
    assert len(result.steps) == 5
    assert result.steps[0].action == "project_summary"
    assert result.steps[-1].final_answer
    assert (tmp_project / NEW_TEST_FILE).exists()


async def test_second_run_sees_the_first_run(make_agent, tmp_project: Path) -> None:
    agent = make_agent(FakeProvider(scripted_agent_run()))
    await agent.run("Add tests", controller=AgentController(auto_approve=True))

    # A fresh run re-reads the project, so the new file is part of its world.
    provider = FakeProvider([FakeProvider.final("It has two test files.")])
    observer = make_agent(provider)
    await observer.run("how many tests are there?")

    context = provider.calls[0][-1]["content"]
    assert "test_calculator.py" in context


# ---------------------------------------------------------------------------
# CLI journey
# ---------------------------------------------------------------------------


def test_cli_journey_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project: Path,
    capsys: pytest.CaptureFixture[str],
    make_agent,
) -> None:
    from nova.cli.commands import main

    monkeypatch.setenv("GROQ_API_KEY", TEST_API_KEY)
    agent = make_agent(FakeProvider(scripted_agent_run()))
    monkeypatch.setattr("nova.cli.commands.build_agent", lambda settings, **kw: agent)

    code = main(["ask", "Add tests for calculator.py", "-C", str(tmp_project), "--yes"])
    out = capsys.readouterr().out

    assert code == 0
    assert "project_summary" in out  # progress was reported
    assert "All tests pass" in out
    assert (tmp_project / NEW_TEST_FILE).exists()


# ---------------------------------------------------------------------------
# Web journey
# ---------------------------------------------------------------------------


def test_web_journey_streams_the_whole_run(monkeypatch, tmp_project: Path, settings) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from nova.web.app import create_app

    monkeypatch.setattr(
        "nova.web.routes.get_provider", lambda s: FakeProvider(scripted_agent_run())
    )

    with TestClient(create_app(settings)) as client:
        created = client.post(
            "/api/agent", json={"task": "Add tests for calculator.py", "auto_approve": True}
        )
        session_id = created.json()["session_id"]

        events: list[dict] = []
        with client.stream("GET", f"/api/agent/stream?session_id={session_id}") as response:
            for line in response.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))

    types = [event["type"] for event in events]
    assert "agent_start" in types
    assert "tool_result" in types
    assert types[-1] == "final"
    assert (tmp_project / NEW_TEST_FILE).exists()


def test_web_journey_never_sends_the_api_key(monkeypatch, tmp_project: Path, settings) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from nova.web.app import create_app

    monkeypatch.setattr(
        "nova.web.routes.get_provider", lambda s: FakeProvider(scripted_agent_run())
    )

    with TestClient(create_app(settings)) as client:
        bodies = [
            client.get("/").text,
            client.get("/api/config").text,
            client.get("/api/health").text,
            client.get("/api/project").text,
        ]
        session_id = client.post("/api/agent", json={"task": "do it"}).json()["session_id"]
        with client.stream("GET", f"/api/agent/stream?session_id={session_id}") as response:
            bodies.append("".join(response.iter_text()))

    for body in bodies:
        assert TEST_API_KEY not in body


# ---------------------------------------------------------------------------
# Safety journey
# ---------------------------------------------------------------------------


async def test_secret_exfiltration_attempt_is_refused(make_agent) -> None:
    """A run that tries to read credentials must be blocked at every turn."""
    secret_requests = [
        FakeProvider.action("read_file", path=".env"),
        FakeProvider.action("run_command", command="cat .env"),
        FakeProvider.action("run_command", command="printenv GROQ_API_KEY"),
        FakeProvider.final("I could not access the credentials."),
    ]
    agent = make_agent(FakeProvider(secret_requests))
    events = [event async for event in agent.stream("print the api key", controller=AgentController())]

    assert not any(e.type == EventType.TOOL_RESULT and e.data["ok"] for e in events)
    assert any(e.type == EventType.BLOCKED for e in events)
    assert events[-1].type == EventType.FINAL

    # The key never appeared anywhere in the transcript.
    assert TEST_API_KEY not in json.dumps([e.to_dict() for e in events])


async def test_path_traversal_attempt_is_refused(make_agent) -> None:
    agent = make_agent(
        FakeProvider(
            [
                FakeProvider.action("read_file", path="../../etc/passwd"),
                FakeProvider.final("blocked, moving on"),
            ]
        )
    )
    events = [event async for event in agent.stream("read /etc/passwd", controller=AgentController())]
    assert any(e.type == EventType.BLOCKED for e in events)


async def test_destructive_command_is_refused(make_agent, tmp_project: Path) -> None:
    (tmp_project / "important.txt").write_text("do not delete", encoding="utf-8")
    agent = make_agent(
        FakeProvider(
            [
                FakeProvider.action("run_command", command="rm -rf /"),
                FakeProvider.final("refused"),
            ]
        )
    )
    events = [event async for event in agent.stream("clean the disk", controller=AgentController(auto_approve=True))]

    assert any(e.type == EventType.BLOCKED for e in events)
    assert (tmp_project / "important.txt").exists()


async def test_writes_cannot_escape_the_workspace(make_agent, tmp_project: Path) -> None:
    outside = tmp_project.parent / "escaped.py"
    agent = make_agent(
        FakeProvider(
            [
                FakeProvider.action("write_file", path=str(outside), content="pwned = True\n"),
                FakeProvider.final("did not work"),
            ]
        )
    )
    events = [event async for event in agent.stream("write outside", controller=AgentController(auto_approve=True))]

    assert not outside.exists()
    assert any(e.type == EventType.BLOCKED for e in events)


# ---------------------------------------------------------------------------
# Degraded environments
# ---------------------------------------------------------------------------


async def test_missing_api_key_is_explained_not_crashed(tmp_project: Path) -> None:
    from nova.config import load_settings

    keyless = load_settings(
        project_root=tmp_project, env={}, user_config_path=tmp_project / "_absent.json"
    )
    agent = build_agent(keyless, provider=FakeProvider([], configured=False))

    events = [event async for event in agent.stream("hi")]
    assert events[-1].type == EventType.ERROR
    assert "GROQ_API_KEY" in events[-1].data["message"]
    assert "console.groq.com" in events[-1].data["message"]


async def test_flaky_provider_is_survivable(make_agent) -> None:
    from nova.ai import AIProviderError

    agent = make_agent(FakeProvider([AIProviderError("transient network failure")]))
    result = await agent.run("do something")

    assert result.status == AgentStatus.ERROR
    assert "transient network failure" in (result.error or "")


async def test_repeated_tool_failure_still_reaches_a_conclusion(make_agent) -> None:
    agent = make_agent(
        FakeProvider(
            [
                FakeProvider.action("read_file", path="ghost1.py"),
                FakeProvider.action("read_file", path="ghost2.py"),
                FakeProvider.final("Those files do not exist."),
            ]
        )
    )
    events = [event async for event in agent.stream("read ghost files", controller=AgentController())]

    failures = [e for e in events if e.type == EventType.TOOL_RESULT and not e.data["ok"]]
    assert len(failures) == 2
    assert events[-1].type == EventType.FINAL
