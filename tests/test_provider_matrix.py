"""Comprehensive provider capability and test matrix for Groq, Gemini, Ollama, OpenRouter, and Cerebras."""

from __future__ import annotations

import os
from types import SimpleNamespace
import pytest
import httpx

from nova.ai import get_provider, provider_names, AIProviderError
from nova.ai.groq import GroqProvider
from nova.ai.gemini import GeminiProvider
from nova.ai.ollama import OllamaProvider
from nova.ai.openrouter import OpenRouterProvider
from nova.ai.cerebras import CerebrasProvider


@pytest.mark.parametrize("provider_name", ["groq", "gemini", "ollama", "openrouter", "cerebras"])
def test_provider_matrix_instantiation(provider_name):
    settings = SimpleNamespace(
        provider=provider_name,
        groq_api_key="key",
        gemini_api_key="key",
        openrouter_api_key="key",
        cerebras_api_key="key",
        ollama_base_url="http://localhost:11434",
        command_timeout=30,
    )
    provider = get_provider(settings)
    assert provider.model_name is not None
    assert provider.configured is True


@pytest.mark.asyncio
async def test_gemini_provider_mock_completion():
    def mock_transport(request: httpx.Request) -> httpx.Response:
        assert "generativelanguage.googleapis.com" in str(request.url)
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Hello from Gemini",
                    }
                }
            ]
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = GeminiProvider(api_key="gem_test_key", client=client)
    res = await provider.complete([{"role": "user", "content": "hi"}])
    assert res.text == "Hello from Gemini"


@pytest.mark.asyncio
async def test_openrouter_provider_mock_completion():
    def mock_transport(request: httpx.Request) -> httpx.Response:
        assert "openrouter.ai" in str(request.url)
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Hello from OpenRouter",
                    }
                }
            ]
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = OpenRouterProvider(api_key="or_test_key", client=client)
    res = await provider.complete([{"role": "user", "content": "hi"}])
    assert res.text == "Hello from OpenRouter"


@pytest.mark.asyncio
async def test_cerebras_provider_mock_completion():
    def mock_transport(request: httpx.Request) -> httpx.Response:
        assert "api.cerebras.ai" in str(request.url)
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Hello from Cerebras",
                    }
                }
            ]
        }
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
    provider = CerebrasProvider(api_key="csk_test_key", client=client)
    res = await provider.complete([{"role": "user", "content": "hi"}])
    assert res.text == "Hello from Cerebras"


# --- Optional Real Provider Integration Tests ---


@pytest.mark.asyncio
async def test_real_gemini_integration():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY is not configured in environment")
    provider = GeminiProvider(api_key=api_key)
    res = await provider.complete([{"role": "user", "content": "Say hello in one word"}])
    assert res.text is not None


@pytest.mark.asyncio
async def test_real_openrouter_integration():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY is not configured in environment")
    provider = OpenRouterProvider(api_key=api_key)
    res = await provider.complete([{"role": "user", "content": "Say hello in one word"}])
    assert res.text is not None


@pytest.mark.asyncio
async def test_real_cerebras_integration():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        pytest.skip("CEREBRAS_API_KEY is not configured in environment")
    provider = CerebrasProvider(api_key=api_key)
    res = await provider.complete([{"role": "user", "content": "Say hello in one word"}])
    assert res.text is not None
