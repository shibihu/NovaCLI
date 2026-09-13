"""Tests for native tool calling, raw argument preservation, multi-step tool calls, and GPT-OSS tool choice regression."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from nova.ai.groq import GroqProvider
from nova.core.agent import AgentController, NovaAgent
from nova.core.models import AIResponse, ToolCall
from conftest import FakeProvider


async def test_regression_tool_choice_is_not_none_when_tools_provided() -> None:
    """Verify that tool_choice is 'auto' (never 'none') when tools are provided."""
    from tests.test_groq import FakeClient, SECRET

    client = FakeClient()
    provider = GroqProvider(SECRET, client=client)

    tools = [{"type": "function", "function": {"name": "search"}}]
    await provider.complete([{"role": "user", "content": "find test"}], tools=tools)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call.get("tools") == tools
    assert call.get("tool_choice") == "auto"
    assert call.get("tool_choice") != "none"


async def test_raw_arguments_preserves_exact_string_and_roundtrips() -> None:
    """Verify raw function.arguments string is preserved and roundtripped in assistant messages."""
    raw_str = '{"query": "main.py", "glob": "*.py"}'
    tool_call = SimpleNamespace(
        id="call_raw1",
        function=SimpleNamespace(name="search", arguments=raw_str)
    )

    provider = FakeProvider([
        AIResponse(text=None, tool_calls=[ToolCall(id="call_raw1", name="search", arguments={"query": "main.py", "glob": "*.py"}, raw_arguments=raw_str)]),
        AIResponse(text="Found matches.")
    ])

    agent = NovaAgent(provider)
    result = await agent.run("Search main.py")

    assert result.ok is True
    assert result.answer == "Found matches."

    # Check conversation history sent on turn 2
    last_messages = provider.calls[-1]
    assistant_msg = [m for m in last_messages if m.get("role") == "assistant"][0]
    assert assistant_msg["tool_calls"][0]["id"] == "call_raw1"
    assert assistant_msg["tool_calls"][0]["function"]["arguments"] == raw_str

    tool_msg = [m for m in last_messages if m.get("role") == "tool"][0]
    assert tool_msg["tool_call_id"] == "call_raw1"
    assert tool_msg["name"] == "search"


async def test_malformed_json_arguments_safety(make_agent) -> None:
    """Verify malformed JSON arguments are preserved as raw string and do NOT execute tool."""
    raw_malformed = '{"path": "unclosed_string'
    provider = FakeProvider([
        AIResponse(text=None, tool_calls=[ToolCall(id="call_bad_json", name="read_file", arguments=None, raw_arguments=raw_malformed)]),
        AIResponse(text="I apologize for the bad JSON.")
    ])

    agent = make_agent(provider)
    result = await agent.run("Read unclosed string file")

    assert result.ok is True
    assert result.answer == "I apologize for the bad JSON."

    last_messages = provider.calls[-1]
    tool_msg = [m for m in last_messages if m.get("role") == "tool"][0]
    assert tool_msg["tool_call_id"] == "call_bad_json"
    assert "Tool argument parsing failed" in tool_msg["content"]


async def test_unknown_tool_fails_safely_and_returns_error(make_agent) -> None:
    """Verify requesting an unknown tool returns a structured tool error with original ID."""
    provider = FakeProvider([
        AIResponse(text=None, tool_calls=[ToolCall(id="call_unknown1", name="magic_teleport", arguments={"dest": "moon"}, raw_arguments='{"dest": "moon"}')]),
        AIResponse(text="Cannot teleport.")
    ])

    agent = make_agent(provider)
    result = await agent.run("Teleport to moon")

    assert result.ok is True
    assert result.answer == "Cannot teleport."

    last_messages = provider.calls[-1]
    tool_msg = [m for m in last_messages if m.get("role") == "tool"][0]
    assert tool_msg["tool_call_id"] == "call_unknown1"
    assert "Tool execution failed: unknown tool" in tool_msg["content"]


async def test_agent_multiple_sequential_native_tool_calls(make_agent, tmp_path: Path) -> None:
    """Test sequential native tool calls: list_files -> write_file -> final answer."""
    provider = FakeProvider([
        AIResponse(
            text="Checking directory first...",
            tool_calls=[ToolCall(id="call_step1", name="list_files", arguments={"path": "."}, raw_arguments='{"path": "."}')]
        ),
        AIResponse(
            text="Now creating test.js...",
            tool_calls=[ToolCall(id="call_step2", name="write_file", arguments={"path": "test.js", "content": "console.log('hello');"}, raw_arguments='{"path": "test.js", "content": "console.log(\'hello\');"}')]
        ),
        AIResponse(text="Successfully created test.js with JS test code.")
    ])

    controller = AgentController(auto_approve=True)
    agent = make_agent(provider)
    result = await agent.run("Can you create a test.js inside this directory?", controller=controller)

    assert result.ok is True
    assert "created test.js" in result.answer.lower()
    assert (tmp_path / "test.js").exists()
    assert "console.log('hello');" in (tmp_path / "test.js").read_text()

    last_call = provider.calls[-1]
    tool_messages = [m for m in last_call if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0]["tool_call_id"] == "call_step1"
    assert tool_messages[0]["name"] == "list_files"
    assert tool_messages[1]["tool_call_id"] == "call_step2"
    assert tool_messages[1]["name"] == "write_file"
