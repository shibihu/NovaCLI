"""Tests for rate-limit recovery logic, cancellation, and mascot status rendering."""

import asyncio
from pathlib import Path
import pytest

from nova.ai import AIProviderError
from nova.core.agent import AgentController, NovaAgent, parse_rate_limit_info
from nova.core.models import AIResponse, EventType
from nova.ui.mascot import get_safe_status, render_mascot
from nova.workspace.files import Workspace


def test_parse_rate_limit_info():
    # Minute rate limit with retry after
    e1 = AIProviderError("Groq rate limit reached (429). Retry after 45s.")
    is_rl, retry_after, reason, is_perm = parse_rate_limit_info(e1)
    assert is_rl is True
    assert retry_after == 45
    assert is_perm is False

    # Minute rate limit without retry delay
    e2 = AIProviderError("429 Too Many Requests")
    is_rl, retry_after, reason, is_perm = parse_rate_limit_info(e2)
    assert is_rl is True
    assert retry_after == 0
    assert is_perm is False

    # Permanent daily quota exhausted
    e3 = AIProviderError("429 Daily quota exhausted for model")
    is_rl, retry_after, reason, is_perm = parse_rate_limit_info(e3)
    assert is_perm is True

    # Permanent auth error
    e4 = AIProviderError("401 Unauthorized: Invalid API key")
    is_rl, retry_after, reason, is_perm = parse_rate_limit_info(e4)
    assert is_perm is True


def test_mascot_rendering_and_status():
    assert render_mascot("working", use_unicode=True) == "✦ Nova ⚙"
    assert render_mascot("working", use_unicode=False) == "[N] Nova *"
    assert render_mascot("waiting", use_unicode=True) == "✦ Nova ⏳"

    status_tool = get_safe_status("tool_call", {"tool": "write_file", "input": {"path": "main.py"}})
    assert status_tool == "Nova is editing main.py…"

    status_rl = get_safe_status("rate_limit_wait", {"retry_after": 30})
    assert status_rl == "Nova is waiting for API rate limit (30s)…"


@pytest.mark.asyncio
async def test_rate_limit_retry_recovery(tmp_path: Path):
    class MockRateLimitProvider:
        model_name = "mock/model"
        configured = True
        calls = 0

        async def complete(self, messages, tools=None, tool_choice=None):
            self.calls += 1
            if self.calls == 1:
                raise AIProviderError("Groq rate limit reached (429). Retry after 1s.")
            return AIResponse(text='{"final": "Task completed successfully!"}')

        async def aclose(self):
            pass

    ws = Workspace(tmp_path)
    provider = MockRateLimitProvider()
    agent = NovaAgent(provider, workspace=ws)

    events = []
    async for evt in agent.stream("Simple task"):
        events.append(evt)

    event_types = [e.type for e in events]
    assert EventType.RATE_LIMIT_WAIT in event_types
    assert EventType.FINAL in event_types
    assert provider.calls == 2

    final_evt = next(e for e in events if e.type == EventType.FINAL)
    assert final_evt.data["answer"] == "Task completed successfully!"


@pytest.mark.asyncio
async def test_rate_limit_cancellation_during_wait(tmp_path: Path):
    class MockEndlessRateLimitProvider:
        model_name = "mock/model"
        configured = True

        async def complete(self, messages, tools=None, tool_choice=None):
            raise AIProviderError("429 Too Many Requests. Try again in 60s.")

        async def aclose(self):
            pass

    ws = Workspace(tmp_path)
    provider = MockEndlessRateLimitProvider()
    agent = NovaAgent(provider, workspace=ws)
    controller = AgentController()

    async def stream_task():
        events = []
        async for evt in agent.stream("Task", controller=controller):
            events.append(evt)
            if evt.type == EventType.RATE_LIMIT_WAIT:
                # Cancel immediately upon receiving rate limit wait
                controller.cancel()
        return events

    events = await stream_task()
    event_types = [e.type for e in events]
    assert EventType.RATE_LIMIT_WAIT in event_types
    assert EventType.CANCELLED in event_types
