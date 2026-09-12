"""Tests for :mod:`nova.ai.groq`.

All SDK interaction is faked, so these tests never touch the network.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from nova.ai import AIProviderError, AIProvider, MissingAPIKeyError, get_provider, provider_names
from nova.ai.groq import DEFAULT_MODEL, GroqProvider

SECRET = "gsk_thisisaverysecretkeyvalue_1234567890"


# --- Fake SDK ---------------------------------------------------------------


def make_response(content: str = "hello") -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class FakeClient:
    """Mimics the slice of ``AsyncGroq`` that the provider uses."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response if response is not None else make_response()
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def provider() -> GroqProvider:
    return GroqProvider(SECRET, "llama-3.3-70b-versatile", client=FakeClient())


# --- Protocol conformance ---------------------------------------------------


def test_satisfies_the_provider_protocol(provider: GroqProvider) -> None:
    assert isinstance(provider, AIProvider)


def test_provider_names_includes_groq() -> None:
    assert "groq" in provider_names()


def test_get_provider_builds_a_groq_provider() -> None:
    settings = SimpleNamespace(groq_api_key=SECRET, groq_model="m", command_timeout=30)
    assert isinstance(get_provider(settings), GroqProvider)


def test_get_provider_rejects_unknown_names() -> None:
    with pytest.raises(AIProviderError, match="Unknown provider"):
        get_provider(SimpleNamespace(groq_api_key=SECRET), name="gemini")


def test_default_model_constant() -> None:
    assert DEFAULT_MODEL == "llama-3.3-70b-versatile"


# --- Introspection ----------------------------------------------------------


def test_model_name(provider: GroqProvider) -> None:
    assert provider.model_name == "llama-3.3-70b-versatile"


def test_configured_is_true_with_a_key(provider: GroqProvider) -> None:
    assert provider.configured is True


def test_configured_is_false_without_a_key() -> None:
    assert GroqProvider(None).configured is False


def test_repr_hides_the_key(provider: GroqProvider) -> None:
    assert SECRET not in repr(provider)


def test_blank_key_is_treated_as_missing() -> None:
    assert GroqProvider("   ").configured is False


# --- complete() -------------------------------------------------------------


async def test_complete_returns_text() -> None:
    client = FakeClient(make_response("the answer"))
    provider = GroqProvider(SECRET, client=client)
    assert await provider.complete([{"role": "user", "content": "hi"}]) == "the answer"


async def test_complete_passes_model_and_messages() -> None:
    client = FakeClient()
    provider = GroqProvider(SECRET, "default-model", client=client)
    await provider.complete([{"role": "user", "content": "ping"}], model="override-model")

    call = client.calls[0]
    assert call["model"] == "override-model"
    assert call["messages"][-1] == {"role": "user", "content": "ping"}


async def test_complete_uses_default_model_when_unspecified() -> None:
    client = FakeClient()
    provider = GroqProvider(SECRET, "my-default", client=client)
    await provider.complete([{"role": "user", "content": "x"}])
    assert client.calls[0]["model"] == "my-default"


async def test_complete_injects_a_system_message() -> None:
    client = FakeClient()
    provider = GroqProvider(SECRET, client=client)
    await provider.complete([{"role": "user", "content": "x"}])
    assert client.calls[0]["messages"][0]["role"] == "system"


async def test_complete_preserves_an_existing_system_message() -> None:
    client = FakeClient()
    provider = GroqProvider(SECRET, client=client)
    await provider.complete(
        [{"role": "system", "content": "custom"}, {"role": "user", "content": "x"}]
    )
    assert client.calls[0]["messages"][0]["content"] == "custom"
    assert len(client.calls[0]["messages"]) == 2


async def test_complete_rejects_empty_message_list(provider: GroqProvider) -> None:
    with pytest.raises(AIProviderError, match="No messages"):
        await provider.complete([])


async def test_complete_strips_whitespace() -> None:
    provider = GroqProvider(SECRET, client=FakeClient(make_response("  spaced  ")))
    assert await provider.complete([{"role": "user", "content": "x"}]) == "spaced"


# --- Failure modes ----------------------------------------------------------


async def test_missing_key_raises_on_first_call() -> None:
    provider = GroqProvider(None)
    with pytest.raises(MissingAPIKeyError) as excinfo:
        await provider.complete([{"role": "user", "content": "hi"}])
    assert "GROQ_API_KEY" in str(excinfo.value)
    assert "console.groq.com" in str(excinfo.value)


async def test_auth_error_mentions_the_key_and_setup() -> None:
    client = FakeClient(error=Exception("401 Unauthorized: invalid api key"))
    provider = GroqProvider(SECRET, client=client)
    with pytest.raises(AIProviderError) as excinfo:
        await provider.complete([{"role": "user", "content": "hi"}])

    message = str(excinfo.value)
    assert "api key" in message.lower()
    assert "console.groq.com" in message
    assert SECRET not in message


async def test_rate_limit_error_is_explained() -> None:
    client = FakeClient(error=Exception("429 rate limit exceeded"))
    provider = GroqProvider(SECRET, client=client)
    with pytest.raises(AIProviderError, match="rate limit"):
        await provider.complete([{"role": "user", "content": "hi"}])


async def test_timeout_error_is_explained() -> None:
    client = FakeClient(error=asyncio.TimeoutError())
    provider = GroqProvider(SECRET, timeout=7, client=client)
    with pytest.raises(AIProviderError, match="timed out"):
        await provider.complete([{"role": "user", "content": "hi"}])


async def test_connection_error_is_explained() -> None:
    client = FakeClient(error=Exception("connection reset by peer"))
    provider = GroqProvider(SECRET, client=client)
    with pytest.raises(AIProviderError, match="Could not reach Groq"):
        await provider.complete([{"role": "user", "content": "hi"}])


async def test_model_not_found_error_is_actionable() -> None:
    client = FakeClient(error=Exception("model `bad-model` not found"))
    provider = GroqProvider(SECRET, "bad-model", client=client)
    with pytest.raises(AIProviderError, match="GROQ_MODEL"):
        await provider.complete([{"role": "user", "content": "hi"}])


async def test_empty_choices_raises() -> None:
    provider = GroqProvider(SECRET, client=FakeClient(SimpleNamespace(choices=[])))
    with pytest.raises(AIProviderError, match="no choices"):
        await provider.complete([{"role": "user", "content": "hi"}])


async def test_empty_content_raises() -> None:
    provider = GroqProvider(SECRET, client=FakeClient(make_response("")))
    with pytest.raises(AIProviderError, match="empty completion"):
        await provider.complete([{"role": "user", "content": "hi"}])


async def test_missing_message_raises() -> None:
    response = SimpleNamespace(choices=[SimpleNamespace(message=None)])
    provider = GroqProvider(SECRET, client=FakeClient(response))
    with pytest.raises(AIProviderError, match="no message"):
        await provider.complete([{"role": "user", "content": "hi"}])


async def test_refusal_is_surfaced() -> None:
    message = SimpleNamespace(content=None, refusal="I cannot help with that")
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    provider = GroqProvider(SECRET, client=FakeClient(response))
    with pytest.raises(AIProviderError, match="declined"):
        await provider.complete([{"role": "user", "content": "hi"}])


async def test_list_content_parts_are_joined() -> None:
    message = SimpleNamespace(content=[{"text": "part one "}, {"text": "part two"}])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    provider = GroqProvider(SECRET, client=FakeClient(response))
    assert await provider.complete([{"role": "user", "content": "hi"}]) == "part one part two"


async def test_errors_never_leak_the_key() -> None:
    client = FakeClient(error=Exception(f"boom with {SECRET} inside"))
    provider = GroqProvider(SECRET, client=client)
    with pytest.raises(AIProviderError) as excinfo:
        await provider.complete([{"role": "user", "content": "hi"}])
    assert SECRET not in str(excinfo.value)


# --- Lifecycle --------------------------------------------------------------


async def test_aclose_closes_an_owned_client() -> None:
    client = FakeClient()
    provider = GroqProvider(SECRET, client=client)
    await provider.aclose()
    assert client.closed is False  # injected clients are not ours to close
    assert provider._client is None


async def test_aclose_is_idempotent(provider: GroqProvider) -> None:
    await provider.aclose()
    await provider.aclose()


async def test_async_context_manager() -> None:
    client = FakeClient()
    async with GroqProvider(SECRET, client=client) as provider:
        assert provider.model_name


def test_get_provider_passes_settings_through() -> None:
    settings = SimpleNamespace(
        groq_api_key=SECRET, groq_model="llama-3.1-8b-instant", command_timeout=45
    )
    provider = get_provider(settings)
    assert provider.model_name == "llama-3.1-8b-instant"
    assert provider.timeout == 45.0
