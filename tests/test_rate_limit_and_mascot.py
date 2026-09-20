"""Tests for provider-aware rate-limit recovery logic, cancellation, and mascot status rendering."""

import asyncio
from pathlib import Path
import pytest

from nova.ai import AIProviderError, parse_rate_limit_info
from nova.core.agent import AgentController, NovaAgent
from nova.core.models import AIResponse, EventType
from nova.ui.mascot import get_safe_status, render_mascot
from nova.workspace.files import Workspace


def test_parse_rate_limit_info_providers():
    # Groq 429 with retry after
    e1 = AIProviderError("Groq rate limit reached (429). Retry after 17s.")
    rl1 = parse_rate_limit_info(e1, provider_name="groq")
    assert rl1.is_rate_limit is True
    assert rl1.retry_after == 17
    assert rl1.is_permanent is False
    assert rl1.provider == "groq"

    # Gemini temporary resource exhausted
    e2 = AIProviderError("Gemini returned RESOURCE_EXHAUSTED: Rate limit reached. Try again in 25 seconds.")
    rl2 = parse_rate_limit_info(e2, provider_name="gemini")
    assert rl2.is_rate_limit is True
    assert rl2.retry_after == 25
    assert rl2.is_permanent is False

    # Gemini daily quota limit exhausted (permanent)
    e3 = AIProviderError("Gemini returned RESOURCE_EXHAUSTED: Daily quota limit exceeded for model gemini-2.5-flash.")
    rl3 = parse_rate_limit_info(e3, provider_name="gemini")
    assert rl3.is_rate_limit is True
    assert rl3.is_permanent is True
    assert rl3.limit_type == "DAILY_QUOTA"

    # OpenRouter 429
    e4 = AIProviderError("OpenRouter 429 Too Many Requests: resets in 12s")
    rl4 = parse_rate_limit_info(e4, provider_name="openrouter")
    assert rl4.is_rate_limit is True
    assert rl4.retry_after == 12
    assert rl4.is_permanent is False

    # Permanent auth error
    e5 = AIProviderError("401 Unauthorized: Invalid API key")
    rl5 = parse_rate_limit_info(e5, provider_name="groq")
    assert rl5.is_permanent is True


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
                controller.cancel()
        return events

    events = await stream_task()
    event_types = [e.type for e in events]
    assert EventType.RATE_LIMIT_WAIT in event_types
    assert EventType.CANCELLED in event_types


def test_aiprovidererror_metadata():
    err = AIProviderError(
        "Groq 429", status_code=429, retry_after=17, provider="groq", model="openai/gpt-oss-20b"
    )
    rl = parse_rate_limit_info(err)
    assert rl.is_rate_limit is True
    assert rl.retry_after == 17
    assert rl.provider == "groq"
    assert rl.is_permanent is False
