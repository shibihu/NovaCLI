"""Structured ``Retry-After`` normalisation across the HTTP providers.

Covers the forms providers actually emit (integer seconds, an RFC 7231
HTTP-date, Groq's ``2m59s`` reset hints) and the rule that permanent
authentication/quota failures are surfaced instead of retried.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from nova.ai import (
    AIProviderError,
    RateLimitInfo,
    parse_rate_limit_info,
    parse_retry_after,
    retry_after_from_headers,
)
from nova.ai.cerebras import CerebrasProvider
from nova.ai.gemini import GeminiProvider
from nova.ai.groq import GroqProvider
from nova.ai.openrouter import OpenRouterProvider
from nova.core.agent import AgentController, NovaAgent
from nova.core.models import AIResponse, EventType
from nova.workspace.files import Workspace

# ---------------------------------------------------------------------------
# Parsing primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("60", 60),
        (60, 60),
        ("12.5", 12),
        ("0", 0),
        ("59s", 59),
        ("2m59.56s", 180),
        ("1m", 60),
        (None, None),
        ("", None),
        ("not-a-delay", None),
        ("-5", None),
    ],
)
def test_parse_retry_after_forms(value, expected) -> None:
    assert parse_retry_after(value) == expected


def test_parse_retry_after_accepts_http_date() -> None:
    soon = datetime.now(timezone.utc) + timedelta(seconds=120)
    parsed = parse_retry_after(format_datetime(soon))
    assert parsed is not None
    assert 110 <= parsed <= 121


def test_parse_retry_after_treats_a_past_date_as_zero() -> None:
    past = datetime.now(timezone.utc) - timedelta(seconds=90)
    assert parse_retry_after(format_datetime(past)) == 0


def test_retry_after_from_headers_prefers_retry_after() -> None:
    headers = httpx.Headers(
        {"retry-after": "30", "x-ratelimit-reset-requests": "7m"}
    )
    assert retry_after_from_headers(headers) == 30
    assert retry_after_from_headers(httpx.Headers({"x-ratelimit-reset-requests": "7m"})) == 420
    assert retry_after_from_headers(httpx.Headers({})) is None
    assert retry_after_from_headers(None) is None


# ---------------------------------------------------------------------------
# Per-provider HTTP error normalisation
# ---------------------------------------------------------------------------


def _error_response(status: int, headers: dict[str, str], body: dict) -> httpx.Response:
    return httpx.Response(status, headers=headers, json=body)


@pytest.mark.asyncio
async def test_gemini_normalizes_retry_after_seconds() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        return _error_response(
            429,
            {"retry-after": "42"},
            {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    provider = GeminiProvider(api_key="gemini_test_key", model="gemini-2.5-flash", client=client)

    with pytest.raises(AIProviderError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])

    error = caught.value
    assert error.status_code == 429
    assert error.retry_after == 42
    assert error.provider == "gemini"
    assert error.model == "gemini-2.5-flash"
    assert error.error_code == "RESOURCE_EXHAUSTED"
    assert error.is_permanent is False

    info = parse_rate_limit_info(error)
    assert info.is_rate_limit is True
    assert info.retry_after == 42
    assert info.provider == "gemini"
    assert info.model == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_gemini_normalizes_retry_after_http_date() -> None:
    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=90))

    def transport(request: httpx.Request) -> httpx.Response:
        return _error_response(
            429, {"Retry-After": when}, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    provider = GeminiProvider(api_key="gemini_test_key", client=client)

    with pytest.raises(AIProviderError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])

    assert caught.value.retry_after is not None
    assert 80 <= caught.value.retry_after <= 91


@pytest.mark.asyncio
async def test_openrouter_normalizes_retry_after() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        return _error_response(
            429,
            {"retry-after": "17"},
            {"error": {"code": "rate_limited", "message": "slow down"}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    provider = OpenRouterProvider(api_key="or_test_key", model="openai/gpt-oss-20b", client=client)

    with pytest.raises(AIProviderError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])

    error = caught.value
    assert (error.status_code, error.retry_after) == (429, 17)
    assert error.provider == "openrouter"
    assert error.model == "openai/gpt-oss-20b"
    assert error.error_code == "rate_limited"


@pytest.mark.asyncio
async def test_openrouter_uses_the_groq_style_reset_header() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        return _error_response(
            429, {"x-ratelimit-reset-requests": "1m30s"}, {"error": {"code": "rate_limited"}}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    provider = OpenRouterProvider(api_key="or_test_key", client=client)

    with pytest.raises(AIProviderError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])

    assert caught.value.retry_after == 90


@pytest.mark.asyncio
async def test_cerebras_normalizes_retry_after() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        return _error_response(
            429, {"retry-after": "7"}, {"error": {"code": "too_many_requests"}}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    provider = CerebrasProvider(api_key="csk_test_key", model="llama3.1-8b", client=client)

    with pytest.raises(AIProviderError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])

    error = caught.value
    assert (error.status_code, error.retry_after) == (429, 7)
    assert error.provider == "cerebras"
    assert error.model == "llama3.1-8b"
    assert error.error_code == "too_many_requests"


@pytest.mark.asyncio
async def test_gemini_invalid_key_is_permanent_and_never_retried() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"status": "UNAUTHENTICATED"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    provider = GeminiProvider(api_key="gemini_test_key", client=client)

    with pytest.raises(AIProviderError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])

    assert caught.value.is_permanent is True
    info = parse_rate_limit_info(caught.value)
    assert info.is_permanent is True
    assert info.is_rate_limit is False


@pytest.mark.asyncio
async def test_openrouter_daily_quota_is_permanent_not_a_rate_limit() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"code": "insufficient_quota", "message": "daily quota exhausted"}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    provider = OpenRouterProvider(api_key="or_test_key", client=client)

    with pytest.raises(AIProviderError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])

    info = parse_rate_limit_info(caught.value)
    assert info.is_permanent is True


# ---------------------------------------------------------------------------
# Groq (SDK based)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class _FakeGroqError(Exception):
    def __init__(self, message: str, status_code: int, headers: dict[str, str]) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = _FakeResponse(headers)


class _FakeGroqClient:
    def __init__(self, error: Exception) -> None:
        self.chat = self
        self.completions = self
        self._error = error

    async def create(self, **kwargs):
        raise self._error


@pytest.mark.asyncio
async def test_groq_normalizes_retry_after_header() -> None:
    error = _FakeGroqError("429 Too Many Requests", 429, {"retry-after": "30"})
    provider = GroqProvider(api_key="gsk_test", model="openai/gpt-oss-20b", client=_FakeGroqClient(error))

    with pytest.raises(AIProviderError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])

    result = caught.value
    assert result.status_code == 429
    assert result.retry_after == 30
    assert result.provider == "groq"
    assert result.model == "openai/gpt-oss-20b"
    assert result.is_permanent is False

    info = parse_rate_limit_info(result)
    assert info.is_rate_limit is True
    assert info.retry_after == 30
    assert info.provider == "groq"
    assert info.model == "openai/gpt-oss-20b"


@pytest.mark.asyncio
async def test_groq_normalizes_http_date_retry_after() -> None:
    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=45))
    error = _FakeGroqError("429 Too Many Requests", 429, {"retry-after": when})
    provider = GroqProvider(api_key="gsk_test", client=_FakeGroqClient(error))

    with pytest.raises(AIProviderError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])

    assert caught.value.retry_after is not None
    assert 35 <= caught.value.retry_after <= 46


@pytest.mark.asyncio
async def test_groq_auth_failure_is_marked_permanent() -> None:
    error = _FakeGroqError("401 Unauthorized: invalid api key", 401, {})
    provider = GroqProvider(api_key="gsk_test", client=_FakeGroqClient(error))

    with pytest.raises(AIProviderError) as caught:
        await provider.complete([{"role": "user", "content": "hi"}])

    assert caught.value.is_permanent is True
    assert parse_rate_limit_info(caught.value).is_permanent is True


def test_provider_and_model_stay_separate() -> None:
    info = RateLimitInfo(is_rate_limit=True, provider="gemini", model="gemini-2.5-flash")
    assert info.provider == "gemini"
    assert info.model == "gemini-2.5-flash"
    assert "/" not in info.provider


# ---------------------------------------------------------------------------
# Agent-level retry behaviour
# ---------------------------------------------------------------------------


def _agent(provider, tmp_path: Path) -> NovaAgent:
    agent = NovaAgent(provider, workspace=Workspace(tmp_path))
    # Keep the retry loop instant instead of waiting out the real backoff.
    agent.settings = agent.settings.with_overrides(
        rate_limit_retry=True, rate_limit_fallback_seconds=0
    )
    return agent


@pytest.mark.asyncio
async def test_retry_count_is_bounded(tmp_path: Path) -> None:
    class AlwaysRateLimited:
        model_name = "mock/model"
        provider_id = "mock"
        configured = True

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tools=None, tool_choice=None):
            self.calls += 1
            raise AIProviderError(
                "429 Too Many Requests",
                status_code=429,
                provider="groq",
                model="openai/gpt-oss-20b",
            )

        async def aclose(self) -> None:
            pass

    provider = AlwaysRateLimited()
    agent = _agent(provider, tmp_path)

    events = [event async for event in agent.stream("task")]

    waits = [e for e in events if e.type == EventType.RATE_LIMIT_WAIT]
    assert 0 < len(waits) <= 3
    assert any(e.type == EventType.ERROR for e in events)
    assert provider.calls == len(waits) + 1

    # provider and model are reported separately.
    assert waits[0].data["provider"] == "mock"
    assert waits[0].data["model"] == "mock/model"


@pytest.mark.asyncio
async def test_permanent_auth_error_is_not_retried(tmp_path: Path) -> None:
    class AuthFailure:
        model_name = "mock/model"
        provider_id = "gemini"
        configured = True

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tools=None, tool_choice=None):
            self.calls += 1
            raise AIProviderError(
                "Gemini rejected the API key (HTTP 401).",
                status_code=401,
                provider="gemini",
                is_permanent=True,
            )

        async def aclose(self) -> None:
            pass

    provider = AuthFailure()
    agent = _agent(provider, tmp_path)

    events = [event async for event in agent.stream("task")]

    assert provider.calls == 1
    assert not any(e.type == EventType.RATE_LIMIT_WAIT for e in events)
    assert any(e.type == EventType.ERROR for e in events)


@pytest.mark.asyncio
async def test_permanent_quota_error_is_not_retried(tmp_path: Path) -> None:
    class QuotaExhausted:
        model_name = "mock/model"
        configured = True

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tools=None, tool_choice=None):
            self.calls += 1
            raise AIProviderError("429 daily quota exhausted for this account")

        async def aclose(self) -> None:
            pass

    provider = QuotaExhausted()
    agent = _agent(provider, tmp_path)

    events = [event async for event in agent.stream("task")]

    assert provider.calls == 1
    assert not any(e.type == EventType.RATE_LIMIT_WAIT for e in events)
    assert any(e.type == EventType.ERROR for e in events)


@pytest.mark.asyncio
async def test_a_retried_request_does_not_execute_the_tool_twice(
    tmp_path: Path, monkeypatch
) -> None:
    from nova.core.agent import ToolBox

    executions: list[str] = []
    original_execute = ToolBox.execute

    async def counting_execute(self, name, arguments=None):
        executions.append(name)
        return await original_execute(self, name, arguments)

    monkeypatch.setattr(ToolBox, "execute", counting_execute)

    class RetryThenTool:
        model_name = "mock/model"
        configured = True

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tools=None, tool_choice=None):
            self.calls += 1
            if self.calls == 1:
                raise AIProviderError("429 Too Many Requests")
            if self.calls == 2:
                from nova.core.models import ToolCall

                return AIResponse(
                    text=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="list_files",
                            arguments={"path": "."},
                            raw_arguments='{"path": "."}',
                        )
                    ],
                )
            return AIResponse(text='{"final": "done"}')

        async def aclose(self) -> None:
            pass

    provider = RetryThenTool()
    agent = _agent(provider, tmp_path)

    events = [event async for event in agent.stream("list the files")]

    assert provider.calls == 3  # one rejected attempt plus the retry
    assert executions == ["list_files"]  # executed exactly once
    assert any(e.type == EventType.FINAL for e in events)


@pytest.mark.asyncio
async def test_rate_limit_wait_respects_cancellation(tmp_path: Path) -> None:
    class Endless429:
        model_name = "mock/model"
        configured = True

        async def complete(self, messages, tools=None, tool_choice=None):
            raise AIProviderError("429 Too Many Requests. Try again in 1s.")

        async def aclose(self) -> None:
            pass

    agent = _agent(Endless429(), tmp_path)
    controller = AgentController()

    seen: list[str] = []
    async for event in agent.stream("task", controller=controller):
        seen.append(event.type)
        if event.type == EventType.RATE_LIMIT_WAIT:
            controller.cancel()

    assert EventType.RATE_LIMIT_WAIT in seen
    assert EventType.CANCELLED in seen
