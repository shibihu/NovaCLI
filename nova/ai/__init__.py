"""AI provider layer.

Nova Core never talks to a vendor SDK directly — it talks to the
:class:`AIProvider` protocol. Provider implementations:
- Groq (:mod:`nova.ai.groq`)
- Ollama (:mod:`nova.ai.ollama`)
- Gemini (:mod:`nova.ai.gemini`)
- OpenRouter (:mod:`nova.ai.openrouter`)
- Cerebras (:mod:`nova.ai.cerebras`)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from nova.core.models import AIResponse

__all__ = [
    "AIProvider",
    "AIProviderError",
    "MissingAPIKeyError",
    "get_provider",
    "normalize_provider_name",
    "provider_names",
]


class AIProviderError(Exception):
    """Any provider failure, already sanitised of secrets."""


class MissingAPIKeyError(AIProviderError):
    """No API key configured; the message explains how to fix it."""


@runtime_checkable
class AIProvider(Protocol):
    """Minimal contract every model backend must satisfy."""

    @property
    def model_name(self) -> str:
        """Identifier of the model currently in use."""
        ...

    @property
    def configured(self) -> bool:
        """True when the provider has everything it needs to run."""
        ...

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
    ) -> AIResponse:
        """Return the assistant's response (text and/or native tool calls)."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...


def provider_names() -> tuple[str, ...]:
    """Registered provider identifiers."""
    return ("groq", "gemini", "ollama", "openrouter", "cerebras")


def normalize_provider_name(name: str | None) -> str:
    """Normalize user input or aliases into canonical provider identifier."""
    if not name:
        return "groq"
    norm = str(name).strip().lower().replace("_", "").replace("-", "")
    if norm in ("groq",):
        return "groq"
    if norm in ("gemini", "google", "googlegemini"):
        return "gemini"
    if norm in ("ollama", "local"):
        return "ollama"
    if norm in ("openrouter", "openrouterai"):
        return "openrouter"
    if norm in ("cerebras", "cerebrasai"):
        return "cerebras"
    return str(name).strip().lower()


def get_provider(settings: object, name: str | None = None) -> AIProvider:
    """Build the configured provider implementation.

    The provider implementation is selected based on `name` (defaulting to `settings.provider`),
    and the model is resolved via `settings.model` or provider-specific configuration.
    """
    raw_name = name or getattr(settings, "provider", "groq")
    prov_name = normalize_provider_name(raw_name)

    if prov_name == "groq":
        from .groq import GroqProvider

        return GroqProvider(
            api_key=getattr(settings, "groq_api_key", None),
            model=getattr(settings, "groq_model", None) or getattr(settings, "model", "openai/gpt-oss-20b"),
            timeout=float(getattr(settings, "command_timeout", 30)),
            reasoning_effort=getattr(settings, "reasoning_effort", None),
        )

    if prov_name == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(
            base_url=getattr(settings, "ollama_base_url", "http://localhost:11434"),
            model=getattr(settings, "ollama_model", None) or getattr(settings, "model", "qwen3:4b"),
            timeout=float(getattr(settings, "command_timeout", 30)),
        )

    if prov_name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(
            api_key=getattr(settings, "gemini_api_key", None),
            model=getattr(settings, "gemini_model", None) or getattr(settings, "model", "gemini-2.5-flash"),
            timeout=float(getattr(settings, "command_timeout", 30)),
        )

    if prov_name == "openrouter":
        from .openrouter import OpenRouterProvider

        return OpenRouterProvider(
            api_key=getattr(settings, "openrouter_api_key", None),
            model=getattr(settings, "openrouter_model", None) or getattr(settings, "model", "openai/gpt-oss-20b"),
            timeout=float(getattr(settings, "command_timeout", 30)),
        )

    if prov_name == "cerebras":
        from .cerebras import CerebrasProvider

        return CerebrasProvider(
            api_key=getattr(settings, "cerebras_api_key", None),
            model=getattr(settings, "cerebras_model", None) or getattr(settings, "model", "llama3.1-8b"),
            timeout=float(getattr(settings, "command_timeout", 30)),
        )

    raise AIProviderError(
        f"Unknown provider {raw_name!r}. Available: {', '.join(provider_names())}"
    )
