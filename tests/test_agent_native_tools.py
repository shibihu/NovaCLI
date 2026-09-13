"""Tests for native tool calling, multi-step tool calls, and GPT-OSS tool choice regression."""

from __future__ import annotations

import json
from pathlib import Path
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


async def test_agent_multiple_sequential_native_tool_calls(make_agent, tmp_path: Path) -> None:
    """Test sequential native tool calls: list_files -> write_file -> final answer."""
    provider = FakeProvider([
        AIResponse(
            text="Checking directory first...",
            tool_calls=[ToolCall(id="call_step1", name="list_files", arguments={"path": "."})]
        ),
        AIResponse(
            text="Now creating test.js...",
            tool_calls=[ToolCall(id="call_step2", name="write_file", arguments={"path": "test.js", "content": "console.log('hello');"})]
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

    # Check conversation message history passed to the provider
    last_call = provider.calls[-1]

    # Verify tool results contain matching tool_call_id
    tool_messages = [m for m in last_call if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[0]["tool_call_id"] == "call_step1"
    assert tool_messages[0]["name"] == "list_files"
    assert tool_messages[1]["tool_call_id"] == "call_step2"
    assert tool_messages[1]["name"] == "write_file"


async def test_agent_handles_malformed_tool_args_gracefully(make_agent) -> None:
    """Verify that if a model supplies bad tool arguments, it receives an error observation rather than crashing."""
    provider = FakeProvider([
        AIResponse(
            text="Searching...",
            tool_calls=[ToolCall(id="call_bad", name="search", arguments={})]  # missing 'query'
        ),
        AIResponse(text="Recovered from search error.")
    ])

    agent = make_agent(provider)
    result = await agent.run("search for something")

    assert result.ok is True
    assert result.answer == "Recovered from search error."
