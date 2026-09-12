"""Tests for :mod:`nova.core.agent`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nova.ai import AIProviderError
from nova.core.agent import (
    TOOL_NAMES,
    AgentController,
    AgentDecision,
    NovaAgent,
    extract_json_object,
    parse_agent_response,
    render_tool_catalog,
)
from nova.core.models import AgentStatus, EventType, Message

from conftest import FakeProvider


# --- JSON extraction --------------------------------------------------------


def test_extract_simple_object() -> None:
    assert extract_json_object('{"a": 1}') == '{"a": 1}'


def test_extract_ignores_leading_prose() -> None:
    assert extract_json_object('Sure! {"a": 1} done') == '{"a": 1}'


def test_extract_handles_braces_inside_strings() -> None:
    found = extract_json_object('{"code": "if (x) { y(); }"}')
    assert json.loads(found)["code"] == "if (x) { y(); }"


def test_extract_handles_escaped_quotes() -> None:
    found = extract_json_object('{"text": "say \\"hi\\" now"}')
    assert json.loads(found)["text"] == 'say "hi" now'


def test_extract_nested_objects() -> None:
    found = extract_json_object('{"a": {"b": {"c": 1}}}')
    assert json.loads(found) == {"a": {"b": {"c": 1}}}


def test_extract_returns_none_without_braces() -> None:
    assert extract_json_object("no json here") is None


def test_extract_returns_none_for_unbalanced() -> None:
    assert extract_json_object('{"a": 1') is None


def test_extract_empty_input() -> None:
    assert extract_json_object("") is None


# --- Response parsing -------------------------------------------------------


def test_parse_action() -> None:
    decision = parse_agent_response(
        '{"thought": "look", "action": "read_file", "action_input": {"path": "main.py"}}'
    )
    assert decision.is_action
    assert decision.action == "read_file"
    assert decision.action_input == {"path": "main.py"}
    assert decision.thought == "look"


def test_parse_final_answer() -> None:
    decision = parse_agent_response('{"thought": "done", "final_answer": "It is a demo."}')
    assert decision.is_final
    assert decision.final == "It is a demo."


def test_parse_strips_markdown_fences() -> None:
    decision = parse_agent_response('```json\n{"final_answer": "hi"}\n```')
    assert decision.final == "hi"


def test_parse_accepts_tool_alias() -> None:
    decision = parse_agent_response('{"tool": "list_files", "input": {"path": "."}}')
    assert decision.action == "list_files"


def test_parse_accepts_singular_aliases() -> None:
    decision = parse_agent_response('{"action": "search", "args": {"query": "x"}}')
    assert decision.action == "search"
    assert decision.action_input == {"query": "x"}


def test_parse_accepts_answer_alias() -> None:
    assert parse_agent_response('{"answer": "done"}').final == "done"


def test_parse_splits_a_bare_command_string() -> None:
    decision = parse_agent_response('{"action": "run_command pytest -q"}')
    assert decision.action == "run_command"
    assert decision.action_input["command"] == "pytest -q"


def test_parse_plain_text_becomes_the_final_answer() -> None:
    decision = parse_agent_response("I could not find the file.")
    assert decision.is_final
    assert decision.final == "I could not find the file."


def test_parse_empty_response_records_an_error() -> None:
    decision = parse_agent_response("")
    assert decision.parse_error == "empty response"
    assert decision.final is None


def test_parse_object_without_action_or_answer_is_retried() -> None:
    decision = parse_agent_response('{"thought": "hmm"}')
    assert decision.is_final is False
    assert decision.is_action is False
    assert decision.parse_error


def test_parse_final_with_dict_body_is_json_encoded() -> None:
    decision = parse_agent_response('{"final_answer": {"a": 1}}')
    assert '"a": 1' in decision.final


def test_tool_catalog_lists_every_tool() -> None:
    catalog = render_tool_catalog()
    for name in TOOL_NAMES:
        assert name in catalog


def test_agent_decision_flags() -> None:
    assert AgentDecision(action="read_file").is_action is True
    assert AgentDecision(final="x").is_final is True


# --- Prompt assembly --------------------------------------------------------


def test_build_messages_includes_system_context_and_task(make_agent) -> None:
    agent = make_agent(FakeProvider())
    messages, files = agent.build_messages("explain auth")
    assert messages[0].role == "system"
    assert "read_file" in messages[0].content
    assert messages[-1].role == "user"
    assert "explain auth" in messages[-1].content


def test_build_messages_includes_history(make_agent) -> None:
    agent = make_agent(FakeProvider())
    history = [Message.user("earlier"), Message.assistant("reply")]
    messages, _ = agent.build_messages("now", history)
    contents = [m.content for m in messages]
    assert "earlier" in contents
    assert "reply" in contents


# --- The loop ---------------------------------------------------------------


async def test_final_answer_first_turn(make_agent) -> None:
    provider = FakeProvider([FakeProvider.final("All good.")])
    agent = make_agent(provider)

    events = [event async for event in agent.stream("say hi")]
    types = [event.type for event in events]

    assert types[0] == EventType.AGENT_START
    assert types[-1] == EventType.FINAL
    assert events[-1].data["answer"] == "All good."


async def test_read_file_tool_is_executed(make_agent) -> None:
    provider = FakeProvider(
        [
            FakeProvider.action("read_file", path="main.py"),
            FakeProvider.final("It greets people."),
        ]
    )
    agent = make_agent(provider)
    events = [event async for event in agent.stream("what does main.py do?")]

    calls = [e for e in events if e.type == EventType.TOOL_CALL]
    results = [e for e in events if e.type == EventType.TOOL_RESULT]
    assert calls[0].data["tool"] == "read_file"
    assert results[0].data["ok"] is True
    assert "def greet" in results[0].data["output"]


async def test_observation_is_fed_back_to_the_model(make_agent) -> None:
    provider = FakeProvider(
        [FakeProvider.action("read_file", path="main.py"), FakeProvider.final("done")]
    )
    agent = make_agent(provider)
    [event async for event in agent.stream("read main.py")]

    second_call = provider.calls[1]
    assert any("Observation" in m["content"] for m in second_call)


async def test_unknown_tool_is_reported_to_the_model(make_agent) -> None:
    provider = FakeProvider(
        [FakeProvider.action("teleport", path="x"), FakeProvider.final("gave up")]
    )
    agent = make_agent(provider)
    events = [event async for event in agent.stream("do something")]

    results = [e for e in events if e.type == EventType.TOOL_RESULT]
    assert results[0].data["ok"] is False
    assert "Unknown tool" in results[0].data["error"]


async def test_forbidden_action_is_blocked_without_running(make_agent, tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            FakeProvider.action("run_command", command="rm -rf /"),
            FakeProvider.final("understood"),
        ]
    )
    agent = make_agent(provider)
    events = [event async for event in agent.stream("clean up")]

    blocked = [e for e in events if e.type == EventType.BLOCKED]
    assert blocked
    assert blocked[0].data["tool"] == "run_command"


async def test_reading_a_secret_file_is_blocked(make_agent) -> None:
    provider = FakeProvider(
        [FakeProvider.action("read_file", path=".env"), FakeProvider.final("cannot")]
    )
    agent = make_agent(provider)
    events = [event async for event in agent.stream("read the env")]
    assert any(e.type == EventType.BLOCKED for e in events)


async def test_write_file_creates_the_file_with_approval(make_agent, tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            FakeProvider.action("write_file", path="created.py", content="x = 1\n"),
            FakeProvider.final("written"),
        ]
    )
    controller = AgentController(auto_approve=True)
    agent = make_agent(provider)
    [event async for event in agent.stream("create a file", controller=controller)]

    assert (tmp_path / "created.py").read_text() == "x = 1\n"


async def test_run_command_tool_executes(make_agent) -> None:
    provider = FakeProvider(
        [
            FakeProvider.action("run_command", command="echo from-tool"),
            FakeProvider.final("ran it"),
        ]
    )
    agent = make_agent(provider)
    events = [event async for event in agent.stream("echo something")]
    results = [e for e in events if e.type == EventType.TOOL_RESULT]
    assert "from-tool" in results[0].data["output"]


async def test_project_summary_tool(make_agent) -> None:
    provider = FakeProvider(
        [FakeProvider.action("project_summary"), FakeProvider.final("ok")]
    )
    agent = make_agent(provider)
    events = [event async for event in agent.stream("what is this?")]
    results = [e for e in events if e.type == EventType.TOOL_RESULT]
    assert "Project:" in results[0].data["output"]


async def test_search_tool(make_agent) -> None:
    provider = FakeProvider(
        [FakeProvider.action("search", query="def greet"), FakeProvider.final("ok")]
    )
    agent = make_agent(provider)
    events = [event async for event in agent.stream("find greet")]
    results = [e for e in events if e.type == EventType.TOOL_RESULT]
    assert "main.py" in results[0].data["output"]


async def test_list_files_tool(make_agent) -> None:
    provider = FakeProvider(
        [FakeProvider.action("list_files", path="."), FakeProvider.final("ok")]
    )
    agent = make_agent(provider)
    events = [event async for event in agent.stream("list files")]
    results = [e for e in events if e.type == EventType.TOOL_RESULT]
    assert "main.py" in results[0].data["output"]


async def test_max_steps_is_enforced(make_agent) -> None:
    provider = FakeProvider([FakeProvider.action("list_files", path=".") for _ in range(10)])
    agent = make_agent(provider, max_steps=3)
    events = [event async for event in agent.stream("loop forever")]

    assert events[-1].type == EventType.FINAL
    assert events[-1].data["limit_reached"] is True
    assert len([e for e in events if e.type == EventType.TOOL_CALL]) == 3


async def test_provider_error_ends_the_run(make_agent) -> None:
    provider = FakeProvider([AIProviderError("boom")])
    agent = make_agent(provider)
    events = [event async for event in agent.stream("hi")]

    assert events[-1].type == EventType.ERROR
    assert "boom" in events[-1].data["message"]


async def test_unconfigured_provider_reports_setup_error(make_agent) -> None:
    provider = FakeProvider([], configured=False)
    agent = make_agent(provider)
    events = [event async for event in agent.stream("hi")]

    assert events[-1].type == EventType.ERROR
    assert "GROQ_API_KEY" in events[-1].data["message"]


async def test_malformed_json_is_retried(make_agent) -> None:
    # Valid JSON, but it names neither a tool nor an answer.
    provider = FakeProvider(['{"thought": "hmm"}', FakeProvider.final("recovered")])
    agent = make_agent(provider)
    events = [event async for event in agent.stream("hi")]

    assert events[-1].data["answer"] == "recovered"
    nudge = provider.calls[1][-1]["content"]
    assert "unusable" in nudge


async def test_prose_reply_is_accepted_as_an_answer(make_agent) -> None:
    agent = make_agent(FakeProvider(["The file does not exist."]))
    events = [event async for event in agent.stream("hi")]
    assert events[-1].type == EventType.FINAL
    assert events[-1].data["answer"] == "The file does not exist."


async def test_progress_events_are_emitted(make_agent) -> None:
    provider = FakeProvider([FakeProvider.final("done")])
    agent = make_agent(provider)
    events = [event async for event in agent.stream("hi")]
    assert any(e.type == EventType.PROGRESS for e in events)


async def test_events_are_terminal_at_the_end(make_agent) -> None:
    provider = FakeProvider([FakeProvider.final("done")])
    agent = make_agent(provider)
    events = [event async for event in agent.stream("hi")]
    assert events[-1].is_terminal


async def test_events_serialise_to_json(make_agent) -> None:
    provider = FakeProvider([FakeProvider.final("done")])
    agent = make_agent(provider)
    events = [event async for event in agent.stream("hi")]
    json.dumps(events[-1].to_dict())


# --- run() ------------------------------------------------------------------


async def test_run_returns_a_result(make_agent) -> None:
    agent = make_agent(FakeProvider([FakeProvider.final("the answer")]))
    result = await agent.run("question")

    assert result.status == AgentStatus.DONE
    assert result.answer == "the answer"
    assert result.ok is True
    assert result.finished_at is not None


async def test_run_records_steps(make_agent) -> None:
    provider = FakeProvider(
        [FakeProvider.action("read_file", path="main.py"), FakeProvider.final("done")]
    )
    agent = make_agent(provider)
    result = await agent.run("read it")

    assert len(result.steps) == 2
    assert result.steps[0].action == "read_file"
    assert result.steps[0].result.ok is True
    assert result.steps[-1].final_answer == "done"


async def test_run_marks_errors(make_agent) -> None:
    agent = make_agent(FakeProvider([AIProviderError("nope")]))
    result = await agent.run("question")
    assert result.status == AgentStatus.ERROR
    assert result.error == "nope"


async def test_run_result_serialises(make_agent) -> None:
    agent = make_agent(FakeProvider([FakeProvider.final("x")]))
    data = (await agent.run("y")).to_dict()
    json.dumps(data)
    assert data["status"] == "done"


def test_run_sync_wrapper(make_agent) -> None:
    agent = make_agent(FakeProvider([FakeProvider.final("sync answer")]))
    assert agent.run_sync("q").answer == "sync answer"


# --- Secrets ----------------------------------------------------------------


async def test_tool_output_is_redacted(make_agent, tmp_path: Path) -> None:
    secret = "gsk_leakedkeyvalue_abcdefghijklmnopqrst"
    (tmp_path / "leaky.py").write_text(f'KEY = "{secret}"\n', encoding="utf-8")

    provider = FakeProvider(
        [FakeProvider.action("read_file", path="leaky.py"), FakeProvider.final("seen")]
    )
    agent = make_agent(provider)
    events = [event async for event in agent.stream("read leaky.py")]

    results = [e for e in events if e.type == EventType.TOOL_RESULT]
    assert secret not in json.dumps(results[0].data)


async def test_prompt_never_contains_the_session_key(make_agent) -> None:
    provider = FakeProvider([FakeProvider.final("ok")])
    agent = make_agent(provider)
    await agent.run("hello")

    from conftest import TEST_API_KEY

    assert TEST_API_KEY not in json.dumps(provider.calls)
