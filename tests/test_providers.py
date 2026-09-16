"""Tests for provider registry, provider implementations, model resolution, and error routing."""

from __future__ import annotations

from types import SimpleNamespace
import pytest
import httpx

from nova.ai import AIProviderError, MissingAPIKeyError, get_provider, normalize_provider_name, provider_names
from nova.ai.groq import GroqProvider
from nova.ai.ollama import OllamaProvider
from nova.ai.gemini import GeminiProvider
from nova.ai.openrouter import OpenRouterProvider
from nova.ai.cerebras import CerebrasProvider
from nova.config import load_settings, get_api_key_hint


def test_provider_registry_contains_all_providers():
    names = provider_names()
    assert "groq" in names
    assert "gemini" in names
    assert "ollama" in names
    assert "openrouter" in names
    assert "cerebras" in names


def test_provider_name_normalization():
    assert normalize_provider_name("groq") == "groq"
    assert normalize_provider_name("Open-Router") == "openrouter"
    assert normalize_provider_name("google") == "gemini"
    assert normalize_provider_name("local") == "ollama"
    assert normalize_provider_name("cerebras") == "cerebras"


def test_get_provider_instantiates_correct_classes():
    s_groq = SimpleNamespace(provider="groq", groq_api_key="key", groq_model="m1", command_timeout=30)
    assert isinstance(get_provider(s_groq), GroqProvider)

    s_ollama = SimpleNamespace(provider="ollama", ollama_model="m2", ollama_base_url="http://localhost:11434", command_timeout=30)
    assert isinstance(get_provider(s_ollama), OllamaProvider)

    s_gemini = SimpleNamespace(provider="gemini", gemini_api_key="key", gemini_model="m3", command_timeout=30)
    assert isinstance(get_provider(s_gemini), GeminiProvider)

    s_openrouter = SimpleNamespace(provider="openrouter", openrouter_api_key="key", openrouter_model="m4", command_timeout=30)
    assert isinstance(get_provider(s_openrouter), OpenRouterProvider)

    s_cerebras = SimpleNamespace(provider="cerebras", cerebras_api_key="key", cerebras_model="m5", command_timeout=30)
    assert isinstance(get_provider(s_cerebras), CerebrasProvider)


def test_ollama_does_not_require_groq_or_api_key():
    s = load_settings(env={"NOVA_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen3:8b"})
    assert s.provider == "ollama"
    assert s.has_api_key is True
    assert s.require_api_key() == ""
    assert s.masked_api_key == "not required"


def test_gemini_missing_key_error_does_not_mention_groq():
    s = load_settings(env={"NOVA_PROVIDER": "gemini"})
    assert s.provider == "gemini"
    assert s.has_api_key is False
    with pytest.raises(Exception) as exc_info:
        s.require_api_key()
    msg = str(exc_info.value)
    assert "Gemini" in msg
    assert "Groq" not in msg


def test_openrouter_missing_key_error_does_not_mention_groq():
    s = load_settings(env={"NOVA_PROVIDER": "openrouter"})
    assert s.provider == "openrouter"
    assert s.has_api_key is False
    with pytest.raises(Exception) as exc_info:
        s.require_api_key()
    msg = str(exc_info.value)
    assert "OpenRouter" in msg
    assert "Groq" not in msg


def test_cerebras_missing_key_error_does_not_mention_groq():
    s = load_settings(env={"NOVA_PROVIDER": "cerebras"})
    assert s.provider == "cerebras"
    assert s.has_api_key is False
    with pytest.raises(Exception) as exc_info:
        s.require_api_key()
    msg = str(exc_info.value)
    assert "Cerebras" in msg
    assert "Groq" not in msg


@pytest.mark.asyncio
async def test_ollama_runtime_request_uses_configured_server():
    def mock_transport(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "localhost"
        assert request.url.port == 11434
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Hello from mock Ollama",
                    }
                }
            ]
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", model="qwen3:8b", client=client)
    res = await provider.complete([{"role": "user", "content": "hello"}])
    assert res.text == "Hello from mock Ollama"


@pytest.mark.asyncio
async def test_no_silent_failover():
    # If Ollama connection fails, it should raise AIProviderError without switching to Groq
    def mock_transport(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", model="qwen3:8b", client=client)
    with pytest.raises(AIProviderError, match="Cannot connect to Ollama"):
        await provider.complete([{"role": "user", "content": "hello"}])


# --- Timeout separation tests -----------------------------------------------


def test_get_provider_passes_llm_timeout():
    s_ollama = SimpleNamespace(
        provider="ollama",
        ollama_model="m2",
        ollama_base_url="http://localhost:11434",
        command_timeout=30,
        llm_timeout=1800,
    )
    prov = get_provider(s_ollama)
    assert isinstance(prov, OllamaProvider)
    assert prov.timeout == 1800.0


@pytest.mark.asyncio
async def test_ollama_timeout_error_message_format():
    def mock_transport(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Timeout")

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OllamaProvider(base_url="http://localhost:11434", model="qwen3:8b", timeout=1800, client=client)
    with pytest.raises(AIProviderError, match="Ollama request timed out after 1800s"):
        await provider.complete([{"role": "user", "content": "hello"}])
